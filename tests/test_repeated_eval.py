from __future__ import annotations

from unittest import TestCase

from edgeconbce.pipeline.repeated_eval import (
    _aggregate_comparison_rows,
    _aggregate_policy_rows,
    _comparison_significance_rows,
    _paired_delta_rows,
    _two_sided_sign_test_p_value,
)


class RepeatedEvalSummaryTests(TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "seed": 42,
                "policy": "local_only",
                "mcc": 0.1,
                "f1": 0.2,
                "roc_auc": 0.6,
                "average_precision": 0.3,
                "mean_latency_ms": 1.0,
                "mean_total_gco2e": 0.1,
                "route_rate": None,
            },
            {
                "seed": 43,
                "policy": "local_only",
                "mcc": 0.3,
                "f1": 0.4,
                "roc_auc": 0.8,
                "average_precision": 0.5,
                "mean_latency_ms": 3.0,
                "mean_total_gco2e": 0.2,
                "route_rate": None,
            },
            {
                "seed": 42,
                "policy": "adaptive_split",
                "mcc": 0.2,
                "f1": 0.25,
                "roc_auc": 0.65,
                "average_precision": 0.35,
                "mean_latency_ms": 2.0,
                "mean_total_gco2e": 0.15,
                "route_rate": 0.2,
            },
            {
                "seed": 43,
                "policy": "adaptive_split",
                "mcc": 0.4,
                "f1": 0.45,
                "roc_auc": 0.85,
                "average_precision": 0.55,
                "mean_latency_ms": 4.0,
                "mean_total_gco2e": 0.25,
                "route_rate": 0.3,
            },
        ]

    def test_aggregate_policy_rows_include_confidence_intervals(self) -> None:
        aggregate_rows = _aggregate_policy_rows(
            self.rows,
            bootstrap_repeats=200,
            confidence_level=0.95,
            bootstrap_seed=123,
        )
        by_policy = {row["policy"]: row for row in aggregate_rows}

        local_only = by_policy["local_only"]
        self.assertEqual(local_only["mcc_mean"], 0.2)
        self.assertEqual(local_only["mcc_std"], 0.1)
        self.assertLessEqual(local_only["mcc_ci_low"], local_only["mcc_mean"])
        self.assertGreaterEqual(local_only["mcc_ci_high"], local_only["mcc_mean"])
        self.assertIsNone(local_only["route_rate_mean"])
        self.assertIsNone(local_only["route_rate_ci_low"])
        self.assertIsNone(local_only["route_rate_ci_high"])

        adaptive_split = by_policy["adaptive_split"]
        self.assertEqual(adaptive_split["route_rate_mean"], 0.25)
        self.assertLessEqual(adaptive_split["route_rate_ci_low"], adaptive_split["route_rate_mean"])
        self.assertGreaterEqual(adaptive_split["route_rate_ci_high"], adaptive_split["route_rate_mean"])

    def test_aggregate_comparison_rows_summarize_paired_deltas(self) -> None:
        delta_rows = _paired_delta_rows(self.rows, "adaptive_split", "local_only")
        comparison_rows = _aggregate_comparison_rows(
            delta_rows,
            bootstrap_repeats=200,
            confidence_level=0.95,
            bootstrap_seed=456,
        )

        self.assertEqual(len(comparison_rows), 1)
        comparison = comparison_rows[0]
        self.assertEqual(comparison["comparison"], "adaptive_split_minus_local_only")
        self.assertEqual(comparison["num_runs"], 2)
        self.assertEqual(comparison["mcc_mean"], 0.1)
        self.assertEqual(comparison["mcc_std"], 0.0)
        self.assertEqual(comparison["mcc_ci_low"], 0.1)
        self.assertEqual(comparison["mcc_ci_high"], 0.1)

    def test_comparison_significance_tracks_lower_is_better_metrics(self) -> None:
        delta_rows = _paired_delta_rows(self.rows, "adaptive_split", "local_only")
        significance_rows = _comparison_significance_rows(delta_rows)
        by_metric = {row["metric"]: row for row in significance_rows}

        latency = by_metric["mean_latency_ms"]
        self.assertEqual(latency["wins"], 0)
        self.assertEqual(latency["losses"], 2)
        self.assertEqual(latency["effect_direction"], "favors_denominator")

        mcc = by_metric["mcc"]
        self.assertEqual(mcc["wins"], 2)
        self.assertEqual(mcc["losses"], 0)
        self.assertEqual(mcc["effect_direction"], "favors_numerator")

    def test_two_sided_sign_test_matches_small_exact_cases(self) -> None:
        self.assertEqual(_two_sided_sign_test_p_value(0, 0), 1.0)
        self.assertEqual(_two_sided_sign_test_p_value(2, 0), 0.5)
        self.assertEqual(_two_sided_sign_test_p_value(3, 0), 0.25)
