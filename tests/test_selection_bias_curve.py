"""Tests for the selection-bias scaling analysis and the carbon-scale sensitivity."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from edgeconbce.pipeline.carbon_scale_sensitivity import (
    break_even,
    build_report,
    scaled_edge_cost,
)
from edgeconbce.pipeline.nested_fusion import FusionInputs
from edgeconbce.pipeline.selection_bias_curve import (
    predictor_agreement,
    subset_inputs,
    weight_grid,
)

REPO = Path(__file__).resolve().parents[1]
DECOMPOSITION = REPO / "outputs" / "nested_fusion_sci" / "carbon_decomposition.json"


def _toy_inputs(n_sequences: int = 4, per_sequence: int = 10) -> FusionInputs:
    rng = np.random.default_rng(0)
    n = n_sequences * per_sequence
    groups = np.repeat(np.arange(n_sequences), per_sequence)
    labels = (rng.random(n) < 0.3).astype(np.int64)
    return FusionInputs(
        sequence_ids=tuple(f"seq{i}" for i in range(n_sequences)),
        labels=labels,
        reference_scores=rng.random(n),
        candidate_scores=rng.random(n),
        groups=groups,
    )


class WeightGridTests(unittest.TestCase):
    def test_spans_the_unit_interval(self) -> None:
        grid = weight_grid(5)
        self.assertEqual(len(grid), 5)
        self.assertAlmostEqual(grid[0], 0.0)
        self.assertAlmostEqual(grid[-1], 1.0)

    def test_rejects_a_degenerate_grid(self) -> None:
        with self.assertRaises(ValueError):
            weight_grid(1)


class SubsetTests(unittest.TestCase):
    def test_keeps_only_the_named_antigens(self) -> None:
        inputs = _toy_inputs()
        sub = subset_inputs(inputs, [1, 3])
        self.assertEqual(sub.sequence_ids, ("seq1", "seq3"))
        self.assertEqual(len(sub.labels), 20)

    def test_renumbers_groups_contiguously(self) -> None:
        # The leave-one-out loop indexes antigens by group number, so a subset
        # that kept the original numbering would silently skip folds.
        sub = subset_inputs(_toy_inputs(), [1, 3])
        self.assertEqual(sorted(set(sub.groups.tolist())), [0, 1])

    def test_a_single_antigen_cannot_support_leave_one_out(self) -> None:
        with self.assertRaises(ValueError):
            subset_inputs(_toy_inputs(), [2])

    def test_scores_match_the_source_rows(self) -> None:
        inputs = _toy_inputs()
        sub = subset_inputs(inputs, [0, 2])
        expected = np.concatenate(
            [inputs.reference_scores[inputs.groups == 0], inputs.reference_scores[inputs.groups == 2]]
        )
        np.testing.assert_allclose(sub.reference_scores, expected)


class AgreementTests(unittest.TestCase):
    def test_identical_scores_agree_perfectly(self) -> None:
        inputs = _toy_inputs()
        same = FusionInputs(
            sequence_ids=inputs.sequence_ids,
            labels=inputs.labels,
            reference_scores=inputs.reference_scores,
            candidate_scores=inputs.reference_scores.copy(),
            groups=inputs.groups,
        )
        self.assertAlmostEqual(predictor_agreement(same), 1.0, places=9)

    def test_reversed_scores_disagree_perfectly(self) -> None:
        inputs = _toy_inputs()
        flipped = FusionInputs(
            sequence_ids=inputs.sequence_ids,
            labels=inputs.labels,
            reference_scores=inputs.reference_scores,
            candidate_scores=-inputs.reference_scores,
            groups=inputs.groups,
        )
        self.assertAlmostEqual(predictor_agreement(flipped), -1.0, places=9)


class BreakEvenTests(unittest.TestCase):
    def test_identity(self) -> None:
        self.assertAlmostEqual(break_even(1.0, 2.0), 0.5)

    def test_equal_costs_give_a_zero_threshold(self) -> None:
        # If a remote call costs what a local one does, any routing at all adds
        # the local pass on top, so routing is worse from the first antigen.
        self.assertAlmostEqual(break_even(0.5, 0.5), 0.0)

    def test_rejects_a_non_positive_remote_cost(self) -> None:
        with self.assertRaises(ValueError):
            break_even(1.0, 0.0)

    def test_scaling_is_applied_to_the_operational_term_only(self) -> None:
        self.assertAlmostEqual(scaled_edge_cost(0.1, 0.01, 1.0), 0.11)
        self.assertAlmostEqual(scaled_edge_cost(0.1, 0.01, 10.0), 0.20)

    def test_rejects_a_negative_scaling_factor(self) -> None:
        with self.assertRaises(ValueError):
            scaled_edge_cost(0.1, 0.01, -1.0)

    def test_break_even_rises_with_remote_cost(self) -> None:
        rising = [break_even(0.12, scaled_edge_cost(0.137, 0.0024, k)) for k in (1, 10, 100)]
        self.assertEqual(rising, sorted(rising))


@unittest.skipUnless(DECOMPOSITION.exists(), "carbon decomposition artifact not present")
class CarbonReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decomposition = json.loads(DECOMPOSITION.read_text(encoding="utf-8"))
        self.report = build_report(self.decomposition)

    def test_unscaled_case_reproduces_the_recorded_break_even(self) -> None:
        # Anchoring to the recorded remote total must leave the published
        # number untouched; otherwise the curve starts somewhere else.
        self.assertAlmostEqual(
            self.report["recorded"]["break_even_at_measured_scale"],
            self.decomposition["break_even_route_rate"],
            places=9,
        )

    def test_the_measured_pair_still_penalises_routing(self) -> None:
        measured = self.report["checkpoints"][0]
        self.assertEqual(measured["remote_checkpoint"], "esm2_t6_8M_UR50D")
        self.assertTrue(measured["routing_exceeds_edge_only"])

    def test_verdict_matches_the_checkpoint_rows(self) -> None:
        inverts = any(not c["routing_exceeds_edge_only"] for c in self.report["checkpoints"])
        expected = (
            "ordering_inverts_at_realistic_remote_scale"
            if inverts
            else "ordering_holds_across_released_checkpoints"
        )
        self.assertEqual(self.report["verdict"], expected)

    def test_break_even_is_monotone_in_remote_scale(self) -> None:
        values = [row["break_even_route_rate"] for row in self.report["sensitivity"]]
        self.assertEqual(values, sorted(values))


if __name__ == "__main__":
    unittest.main()
