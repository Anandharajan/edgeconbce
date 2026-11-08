from __future__ import annotations

import unittest

import numpy as np
import torch

from edgeconbce.models.residue_transformer import (
    ResidueTransformer,
    ResidueTransformerConfig,
    masked_focal_bce_loss,
    masked_region_smoothness_loss,
    masked_tversky_loss,
)
from edgeconbce.pipeline.carbon_bce_r import _select_threshold


class CarbonBceRModelTests(unittest.TestCase):
    def test_residue_transformer_forward_and_losses(self) -> None:
        torch.manual_seed(7)
        model = ResidueTransformer(
            ResidueTransformerConfig(
                input_dim=8,
                hidden_dim=16,
                cnn_dilation_rates=(1, 2),
                transformer_layers=1,
                attention_heads=4,
                dropout=0.0,
            )
        )
        model.initialize_output_bias(positive_prior=0.25)
        features = torch.randn(2, 5, 8)
        mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
        labels = torch.tensor([[0, 1, 1, 0, 0], [0, 0, 1, 0, 1]], dtype=torch.float32)

        logits = model(features, mask)

        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertTrue(torch.isfinite(logits[mask]).all())
        self.assertTrue(torch.equal(logits[~mask], torch.zeros_like(logits[~mask])))

        focal_loss = masked_focal_bce_loss(logits, labels, mask, positive_weight=2.0)
        smoothness_loss = masked_region_smoothness_loss(logits, labels, mask)
        tversky_loss = masked_tversky_loss(logits, labels, mask)
        self.assertTrue(torch.isfinite(focal_loss))
        self.assertTrue(torch.isfinite(smoothness_loss))
        self.assertTrue(torch.isfinite(tversky_loss))
        self.assertGreaterEqual(float(tversky_loss), 0.0)
        self.assertLessEqual(float(tversky_loss), 1.0)

    def test_threshold_selection_prefers_best_mcc(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.int8)
        scores = np.array([0.1, 0.3, 0.7, 0.9], dtype=np.float64)

        threshold, summary = _select_threshold(labels, scores, (0.2, 0.5, 0.8))

        self.assertEqual(threshold, 0.5)
        self.assertEqual(summary["mcc"], 1.0)
        self.assertEqual(summary["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
