from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from edgeconbce.artifacts import write_json_artifact
from edgeconbce.models.residue_transformer import (
    ResidueTransformer,
    ResidueTransformerConfig,
    masked_focal_bce_loss,
    masked_region_smoothness_loss,
    masked_tversky_loss,
)
from edgeconbce.pipeline.bp3_modern_extension import AnnotatedSequence, _load_or_compute_features, _parse_fasta
from edgeconbce.pipeline.report import write_csv, write_markdown
from edgeconbce.utils.env import system_metadata


DEFAULT_THRESHOLD_GRID = tuple(round(float(value), 2) for value in np.linspace(0.05, 0.95, 91))


@dataclass(frozen=True)
class CarbonBceRConfig:
    train_fasta_path: str
    test_fasta_path: str
    output_dir: str
    seed: int = 42
    validation_fraction: float = 0.2
    epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    hidden_dim: int = 192
    cnn_dilation_rates: tuple[int, ...] = (1, 2, 4)
    transformer_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.15
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    smoothness_weight: float = 0.02
    tversky_weight: float = 0.15
    tversky_alpha: float = 0.45
    tversky_beta: float = 0.55
    prior_bias_init: bool = True
    decision_threshold_grid: tuple[float, ...] = DEFAULT_THRESHOLD_GRID
    model_name: str = "CARBON-BCE-R region-aware local"
    dataset_name: str = "BP3C50ID external test (15 antigens)"


class ResidueFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[AnnotatedSequence]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        bundle = _load_or_compute_features(record.sequence)
        features = _carbon_bce_r_features(bundle)
        return {
            "sequence_id": record.sequence_id,
            "features": torch.from_numpy(features.astype(np.float32)),
            "labels": torch.from_numpy(record.labels.astype(np.float32)),
        }


def _load_config(path: str | Path) -> CarbonBceRConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    threshold_grid = raw.get("decision_threshold_grid")
    if isinstance(threshold_grid, dict):
        threshold_grid = tuple(round(float(value), 2) for value in np.linspace(float(threshold_grid["start"]), float(threshold_grid["stop"]), int(threshold_grid["count"])))
    elif isinstance(threshold_grid, list):
        threshold_grid = tuple(float(value) for value in threshold_grid)
    else:
        threshold_grid = DEFAULT_THRESHOLD_GRID
    dilation_rates = raw.get("cnn_dilation_rates", (1, 2, 4))
    return CarbonBceRConfig(
        train_fasta_path=str(raw["train_fasta_path"]),
        test_fasta_path=str(raw["test_fasta_path"]),
        output_dir=str(raw["output_dir"]),
        seed=int(raw.get("seed", 42)),
        validation_fraction=float(raw.get("validation_fraction", 0.2)),
        epochs=int(raw.get("epochs", 30)),
        batch_size=int(raw.get("batch_size", 4)),
        learning_rate=float(raw.get("learning_rate", 0.0003)),
        weight_decay=float(raw.get("weight_decay", 0.0001)),
        hidden_dim=int(raw.get("hidden_dim", 192)),
        cnn_dilation_rates=tuple(int(value) for value in dilation_rates),
        transformer_layers=int(raw.get("transformer_layers", 2)),
        attention_heads=int(raw.get("attention_heads", 4)),
        dropout=float(raw.get("dropout", 0.15)),
        focal_alpha=float(raw.get("focal_alpha", 0.75)),
        focal_gamma=float(raw.get("focal_gamma", 2.0)),
        smoothness_weight=float(raw.get("smoothness_weight", 0.02)),
        tversky_weight=float(raw.get("tversky_weight", 0.15)),
        tversky_alpha=float(raw.get("tversky_alpha", 0.45)),
        tversky_beta=float(raw.get("tversky_beta", 0.55)),
        prior_bias_init=bool(raw.get("prior_bias_init", True)),
        decision_threshold_grid=threshold_grid,
        model_name=str(raw.get("model_name", "CARBON-BCE-R region-aware local")),
        dataset_name=str(raw.get("dataset_name", "BP3C50ID external test (15 antigens)")),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _carbon_bce_r_features(bundle: dict[str, np.ndarray]) -> np.ndarray:
    embeddings = bundle["embeddings"].astype(np.float32)
    length = embeddings.shape[0]
    positions = np.arange(length, dtype=np.float32)
    length_norm = np.full((length, 1), length / 1023.0, dtype=np.float32)
    pos_norm = (positions / max(1.0, float(length - 1))).reshape(-1, 1).astype(np.float32)
    edge_distance = (np.minimum(positions + 1.0, float(length) - positions) / max(1.0, float(length))).reshape(-1, 1).astype(np.float32)
    contact_features = np.column_stack(
        [
            bundle["contact_density"],
            bundle["contact_long_range"],
            bundle["contact_density_smooth"],
        ]
    ).astype(np.float32)
    return np.concatenate([embeddings, contact_features, length_norm, pos_norm, edge_distance], axis=1).astype(np.float32)


def _collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(int(item["features"].shape[0]) for item in items)
    feature_dim = int(items[0]["features"].shape[1])
    features = torch.zeros((len(items), max_len, feature_dim), dtype=torch.float32)
    labels = torch.zeros((len(items), max_len), dtype=torch.float32)
    mask = torch.zeros((len(items), max_len), dtype=torch.bool)
    sequence_ids: list[str] = []
    for index, item in enumerate(items):
        length = int(item["features"].shape[0])
        features[index, :length] = item["features"]
        labels[index, :length] = item["labels"]
        mask[index, :length] = True
        sequence_ids.append(str(item["sequence_id"]))
    return {"features": features, "labels": labels, "mask": mask, "sequence_ids": sequence_ids}


def _record_split(records: list[AnnotatedSequence], validation_fraction: float, seed: int) -> tuple[list[AnnotatedSequence], list[AnnotatedSequence]]:
    residue_labels = np.concatenate([record.labels for record in records], axis=0)
    groups = np.concatenate([np.full(len(record.labels), index, dtype=np.int32) for index, record in enumerate(records)], axis=0)
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    train_indices, validation_indices = next(splitter.split(np.zeros_like(residue_labels), residue_labels, groups=groups))
    train_group_ids = set(int(group_id) for group_id in np.unique(groups[train_indices]))
    validation_group_ids = set(int(group_id) for group_id in np.unique(groups[validation_indices]))
    return [record for index, record in enumerate(records) if index in train_group_ids], [record for index, record in enumerate(records) if index in validation_group_ids]


def _predict_records(model: ResidueTransformer, records: list[AnnotatedSequence], device: torch.device, batch_size: int) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    loader = DataLoader(ResidueFeatureDataset(records), batch_size=batch_size, shuffle=False, collate_fn=_collate_batch)
    rows: list[dict[str, Any]] = []
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            logits = model(features, mask)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            labels = batch["labels"].numpy()
            masks = batch["mask"].numpy()
            for index, sequence_id in enumerate(batch["sequence_ids"]):
                valid = masks[index].astype(bool)
                sequence_scores = probabilities[index][valid].astype(np.float64)
                sequence_labels = labels[index][valid].astype(np.int8)
                rows.append(
                    {
                        "sequence_id": sequence_id,
                        "length": int(valid.sum()),
                        "probabilities": sequence_scores.tolist(),
                        "labels": sequence_labels.astype(int).tolist(),
                    }
                )
                all_scores.append(sequence_scores)
                all_labels.append(sequence_labels)
    return rows, np.concatenate(all_labels), np.concatenate(all_scores)


def _metric_summary(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | None]:
    predictions = (scores >= threshold).astype(np.int8)
    summary: dict[str, float | None] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "roc_auc": None,
        "average_precision": None,
    }
    if len(np.unique(labels)) > 1:
        summary["roc_auc"] = float(roc_auc_score(labels, scores))
        summary["average_precision"] = float(average_precision_score(labels, scores))
    return summary


def _select_threshold(labels: np.ndarray, scores: np.ndarray, grid: tuple[float, ...]) -> tuple[float, dict[str, float | None]]:
    best_threshold = 0.5
    best_summary: dict[str, float | None] | None = None
    best_objective = (-1.0, -1.0, -1.0)
    for threshold in grid:
        summary = _metric_summary(labels, scores, float(threshold))
        objective = (float(summary["mcc"]), float(summary["f1"]), float(summary["average_precision"] or 0.0))
        if objective > best_objective:
            best_threshold = float(threshold)
            best_summary = summary
            best_objective = objective
    assert best_summary is not None
    return best_threshold, best_summary


def train_and_evaluate(config: CarbonBceRConfig) -> dict[str, Any]:
    _seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_records_all = _parse_fasta(config.train_fasta_path)
    train_records, validation_records = _record_split(train_records_all, config.validation_fraction, config.seed)
    test_records = _parse_fasta(config.test_fasta_path)

    sample_features = _carbon_bce_r_features(_load_or_compute_features(train_records[0].sequence))
    model = ResidueTransformer(
        ResidueTransformerConfig(
            input_dim=int(sample_features.shape[1]),
            hidden_dim=config.hidden_dim,
            cnn_dilation_rates=config.cnn_dilation_rates,
            transformer_layers=config.transformer_layers,
            attention_heads=config.attention_heads,
            dropout=config.dropout,
        )
    ).to(device)

    labels_for_weight = np.concatenate([record.labels for record in train_records], axis=0)
    positive_count = max(1, int(labels_for_weight.sum()))
    negative_count = max(1, int(len(labels_for_weight) - labels_for_weight.sum()))
    positive_weight = negative_count / positive_count
    positive_prior = positive_count / max(1, positive_count + negative_count)
    if config.prior_bias_init:
        model.initialize_output_bias(positive_prior)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = DataLoader(ResidueFeatureDataset(train_records), batch_size=config.batch_size, shuffle=True, collate_fn=_collate_batch)
    history: list[dict[str, Any]] = []
    best_checkpoint_path = output_dir / "carbon_bce_r_best.pt"
    best_validation_mcc = -999.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, mask)
            focal_loss = masked_focal_bce_loss(
                logits,
                labels,
                mask,
                alpha=config.focal_alpha,
                gamma=config.focal_gamma,
                positive_weight=positive_weight,
            )
            smoothness_loss = masked_region_smoothness_loss(logits, labels, mask)
            tversky_loss = masked_tversky_loss(
                logits,
                labels,
                mask,
                alpha=config.tversky_alpha,
                beta=config.tversky_beta,
            )
            loss = focal_loss + (config.smoothness_weight * smoothness_loss) + (config.tversky_weight * tversky_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        _, validation_labels, validation_scores = _predict_records(model, validation_records, device, config.batch_size)
        threshold, validation_summary = _select_threshold(validation_labels, validation_scores, config.decision_threshold_grid)
        history_row = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 6),
            "positive_prior": round(float(positive_prior), 6),
            "validation_threshold": round(float(threshold), 6),
            "validation_mcc": round(float(validation_summary["mcc"]), 6),
            "validation_f1": round(float(validation_summary["f1"]), 6),
            "validation_roc_auc": None if validation_summary["roc_auc"] is None else round(float(validation_summary["roc_auc"]), 6),
            "validation_average_precision": None if validation_summary["average_precision"] is None else round(float(validation_summary["average_precision"]), 6),
        }
        history.append(history_row)
        if float(validation_summary["mcc"]) > best_validation_mcc:
            best_validation_mcc = float(validation_summary["mcc"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model.config),
                    "threshold": threshold,
                    "training_config": asdict(config),
                    "validation_summary": validation_summary,
                },
                best_checkpoint_path,
            )

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    threshold = float(checkpoint["threshold"])
    test_rows, test_labels, test_scores = _predict_records(model, test_records, device, config.batch_size)
    test_summary = _metric_summary(test_labels, test_scores, threshold)

    baseline_row = {
        "model": config.model_name,
        "source_type": "local_rerun",
        "dataset": config.dataset_name,
        "roc_auc": None if test_summary["roc_auc"] is None else round(float(test_summary["roc_auc"]), 6),
        "mcc": round(float(test_summary["mcc"]), 6),
        "f1": round(float(test_summary["f1"]), 6),
        "average_precision": None if test_summary["average_precision"] is None else round(float(test_summary["average_precision"]), 6),
        "latency_ms": None,
        "total_gco2e": None,
        "raw_sequence_share_rate": 0.0,
        "notes": "CARBON-BCE-R region-aware PyTorch multi-scale CNN/Transformer residue predictor trained from frozen ESM/contact features with focal, smoothness, and Tversky overlap losses.",
        "source_url": None,
    }

    history_csv = output_dir / "training_history.csv"
    per_sequence_json = output_dir / "carbon_bce_r_test_predictions.json"
    baseline_json = output_dir / "carbon_bce_r_baseline_row.json"
    report_json = output_dir / "carbon_bce_r_report.json"
    report_md = output_dir / "carbon_bce_r_report.md"

    write_csv(history_csv, list(history[0].keys()) if history else ["epoch"], history)
    write_json_artifact(per_sequence_json, {"threshold": threshold, "rows": test_rows})
    write_json_artifact(baseline_json, {"rows": [baseline_row]})
    write_markdown(
        report_md,
        "\n".join(
            [
                "# CARBON-BCE-R Training Report",
                "",
                f"Device: `{device}`",
                "Algorithm: `region-aware multi-scale CNN/Transformer + Tversky overlap loss`",
                f"Best validation MCC: `{best_validation_mcc:.6f}`",
                f"Selected threshold: `{threshold:.6f}`",
                "",
                "| model | ROC AUC | MCC | F1 | AP |",
                "|---|---:|---:|---:|---:|",
                f"| {baseline_row['model']} | {baseline_row['roc_auc']} | {baseline_row['mcc']} | {baseline_row['f1']} | {baseline_row['average_precision']} |",
                "",
                "Use `carbon_bce_r_baseline_row.json` as `--extra-baselines` when regenerating baseline comparison.",
            ]
        ),
    )
    payload = {
        "project": "edgecon-bce",
        "analysis": "carbon_bce_r_train_eval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "device": str(device),
        "config": asdict(config),
        "checkpoint_path": str(best_checkpoint_path),
        "selected_threshold": threshold,
        "algorithm": {
            "name": "CARBON-BCE-R region-aware local",
            "motif_block": "parallel dilated CNN branches over frozen PLM/contact residue features",
            "context_block": "gated Transformer encoder fusion",
            "loss": "positive-weighted focal BCE + adjacent-region smoothness + Tversky overlap",
            "positive_prior": positive_prior,
        },
        "validation_summary": checkpoint["validation_summary"],
        "test_summary": test_summary,
        "baseline_row": baseline_row,
        "history": history,
        "reports": {
            "history_csv": str(history_csv),
            "test_predictions_json": str(per_sequence_json),
            "baseline_json": str(baseline_json),
            "markdown": str(report_md),
            "json": str(report_json),
        },
    }
    write_json_artifact(report_json, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate CARBON-BCE-R on Colab GPU or a local CUDA runtime.")
    parser.add_argument("--config", required=True, help="Path to CARBON-BCE-R YAML config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    report = train_and_evaluate(config)
    print(Path(report["reports"]["markdown"]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
