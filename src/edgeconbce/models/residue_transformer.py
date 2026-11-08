from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ResidueTransformerConfig:
    input_dim: int
    hidden_dim: int = 192
    cnn_kernel_size: int = 5
    cnn_dilation_rates: tuple[int, ...] = (1, 2, 4)
    transformer_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.15


class MultiScaleMotifBlock(nn.Module):
    """Parallel dilated convolutions for short and regional residue motifs."""

    def __init__(self, hidden_dim: int, kernel_size: int, dilation_rates: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        if not dilation_rates:
            raise ValueError("dilation_rates must not be empty.")
        branches: list[nn.Module] = []
        for dilation in dilation_rates:
            padding = (kernel_size // 2) * int(dilation)
            branches.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=int(dilation)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.mix = nn.Sequential(
            nn.Conv1d(hidden_dim * len(branches), hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GELU(),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        channels_first = hidden.transpose(1, 2)
        branch_outputs = [branch(channels_first) for branch in self.branches]
        return self.mix(torch.cat(branch_outputs, dim=1)).transpose(1, 2)


class ResidueTransformer(nn.Module):
    """Residue-level PLM feature head with multi-scale motif and context blocks."""

    def __init__(self, config: ResidueTransformerConfig) -> None:
        super().__init__()
        if config.hidden_dim % config.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads.")
        self.config = config
        self.input_projection = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.local_motif = MultiScaleMotifBlock(
            hidden_dim=config.hidden_dim,
            kernel_size=config.cnn_kernel_size,
            dilation_rates=config.cnn_dilation_rates,
            dropout=config.dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        self.context_gate = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Sigmoid(),
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(features)
        motif = self.local_motif(hidden)
        hidden = hidden + motif
        encoded = self.context_encoder(hidden, src_key_padding_mask=~mask.bool())
        gate = self.context_gate(encoded)
        fused = hidden + (gate * (encoded - hidden))
        logits = self.output_head(fused).squeeze(-1)
        return logits.masked_fill(~mask.bool(), 0.0)

    def initialize_output_bias(self, positive_prior: float) -> None:
        positive_prior = min(max(float(positive_prior), 1e-4), 1.0 - 1e-4)
        bias = torch.logit(torch.as_tensor(positive_prior, dtype=self.output_head[-1].bias.dtype))
        nn.init.constant_(self.output_head[-1].bias, float(bias))


def masked_focal_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
    positive_weight: float | None = None,
) -> torch.Tensor:
    labels = labels.float()
    mask = mask.bool()
    if positive_weight is None:
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    else:
        pos_weight = torch.as_tensor(float(positive_weight), dtype=logits.dtype, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)
    probabilities = torch.sigmoid(logits)
    p_t = torch.where(labels > 0.5, probabilities, 1.0 - probabilities)
    alpha_t = torch.where(labels > 0.5, torch.full_like(labels, alpha), torch.full_like(labels, 1.0 - alpha))
    focal = alpha_t * torch.pow(1.0 - p_t, gamma) * bce
    return focal[mask].mean()


def masked_region_smoothness_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    valid_pairs = mask[:, 1:].bool() & mask[:, :-1].bool()
    if not torch.any(valid_pairs):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    probability_delta = torch.abs(probabilities[:, 1:] - probabilities[:, :-1])
    label_delta = torch.abs(labels[:, 1:].float() - labels[:, :-1].float())
    # Penalize fragmentation inside same-label stretches more than true boundaries.
    penalty = probability_delta * (1.0 - label_delta)
    return penalty[valid_pairs].mean()


def masked_tversky_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = 0.45,
    beta: float = 0.55,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    labels = labels.float()
    mask = mask.bool()
    probabilities = torch.sigmoid(logits)
    valid_probabilities = probabilities[mask]
    valid_labels = labels[mask]
    if valid_labels.numel() == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    true_positive = torch.sum(valid_probabilities * valid_labels)
    false_positive = torch.sum(valid_probabilities * (1.0 - valid_labels))
    false_negative = torch.sum((1.0 - valid_probabilities) * valid_labels)
    tversky = (true_positive + epsilon) / (
        true_positive + (float(alpha) * false_positive) + (float(beta) * false_negative) + epsilon
    )
    return 1.0 - tversky
