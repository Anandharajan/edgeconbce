from __future__ import annotations

from unittest import TestCase

import numpy as np

from edgeconbce.metrics.regions import positive_runs, summarize_region_predictions
from edgeconbce.pipeline.bp3_modern_extension import _region_constrained_scores


class RegionMetricTests(TestCase):
    def test_positive_runs_extracts_contiguous_regions(self) -> None:
        runs = positive_runs(np.asarray([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.int8))
        self.assertEqual(runs, [(1, 3), (4, 5), (6, 8)])

    def test_region_summary_reports_overlap_and_compactness(self) -> None:
        labels = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int8)
        predictions = np.asarray([0, 1, 0, 0, 1, 1], dtype=np.int8)
        summary = summarize_region_predictions(labels, predictions)
        self.assertEqual(summary["predicted_region_count"], 2)
        self.assertEqual(summary["true_region_count"], 2)
        self.assertEqual(summary["region_overlap_count"], 2)
        self.assertAlmostEqual(summary["region_recall"], 2 / 3, places=6)
        self.assertAlmostEqual(summary["region_precision"], 2 / 3, places=6)

    def test_region_constraint_suppresses_isolated_low_region_scores(self) -> None:
        probabilities = np.asarray([0.10, 0.82, 0.12, 0.20, 0.77, 0.79, 0.76, 0.15], dtype=np.float64)
        feature_bundle = {
            "contact_density_smooth": np.asarray([0.0, 0.1, 0.0, 0.2, 0.9, 1.0, 0.9, 0.1], dtype=np.float32)
        }
        constrained = _region_constrained_scores(
            probabilities,
            feature_bundle,
            threshold=0.5,
            radius=1,
            outside_scale=0.25,
        )
        self.assertLess(constrained[1], probabilities[1])
        self.assertGreaterEqual(constrained[5], probabilities[5])
