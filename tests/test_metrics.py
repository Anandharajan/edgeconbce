from __future__ import annotations

from unittest import TestCase

from edgeconbce.metrics.ranking import auc_score, average_precision, summarize_binary_metrics


class MetricTests(TestCase):
    def test_metric_bounds(self) -> None:
        labels = [True, False, True, False]
        scores = [0.9, 0.1, 0.8, 0.4]
        predictions = [score >= 0.5 for score in scores]
        summary = summarize_binary_metrics(labels, predictions)
        self.assertTrue(0.0 <= auc_score(labels, scores) <= 1.0)
        self.assertTrue(0.0 <= average_precision(labels, scores) <= 1.0)
        for key in ("accuracy", "precision", "recall", "f1"):
            self.assertTrue(0.0 <= summary[key] <= 1.0)

