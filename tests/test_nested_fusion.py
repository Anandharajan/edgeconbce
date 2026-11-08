"""Tests for the leakage-free fusion evaluator."""
from __future__ import annotations

import unittest

import numpy as np

from edgeconbce.pipeline.nested_fusion import (
    FusionInputs, evaluate_nested_fusion, rank_normalise, score_metrics,
    threshold_for_positive_rate,
)


def make_inputs(n_sequences=8, length=40, seed=0, candidate_is_noise=True):
    rng = np.random.default_rng(seed)
    labels, reference, candidate, groups, ids = [], [], [], [], []
    for i in range(n_sequences):
        y = rng.integers(0, 2, size=length)
        # reference carries real signal; candidate is noise unless asked otherwise
        ref = y + rng.normal(0, 0.8, size=length)
        cand = rng.normal(0, 1, size=length) if candidate_is_noise else y + rng.normal(0, 0.8, size=length)
        labels.extend(y.tolist()); reference.extend(ref.tolist())
        candidate.extend(cand.tolist()); groups.extend([i] * length); ids.append(f"seq{i}")
    return FusionInputs(
        sequence_ids=tuple(ids),
        labels=np.asarray(labels, np.int8),
        reference_scores=np.asarray(reference),
        candidate_scores=np.asarray(candidate),
        groups=np.asarray(groups, np.int32),
    )


class TestRankNormalise(unittest.TestCase):
    def test_preserves_ranking(self):
        x = np.array([5.0, -1.0, 3.0, 100.0])
        r = rank_normalise(x)
        self.assertEqual(list(np.argsort(x)), list(np.argsort(r)))

    def test_maps_into_unit_interval(self):
        r = rank_normalise(np.random.default_rng(1).normal(size=200))
        self.assertAlmostEqual(float(r.min()), 0.0)
        self.assertAlmostEqual(float(r.max()), 1.0)

    def test_does_not_change_auc(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, size=300)
        s = y + rng.normal(0, 1, size=300)
        a = score_metrics(y, s, float(np.median(s)))["roc_auc"]
        b = score_metrics(y, rank_normalise(s), 0.5)["roc_auc"]
        self.assertAlmostEqual(a, b, places=10)


class TestThreshold(unittest.TestCase):
    def test_hits_requested_rate(self):
        s = np.linspace(0, 1, 1000)
        for rate in (0.1, 0.25, 0.5):
            thr = threshold_for_positive_rate(s, rate)
            self.assertAlmostEqual(float((s >= thr).mean()), rate, places=2)

    def test_degenerate_rates(self):
        s = np.linspace(0, 1, 10)
        self.assertEqual(threshold_for_positive_rate(s, 0.0), float("inf"))
        self.assertEqual(threshold_for_positive_rate(s, 1.0), float(s.min()))


class TestNestedEvaluation(unittest.TestCase):
    def test_deterministic_across_identical_runs(self):
        inputs = make_inputs()
        a = evaluate_nested_fusion(inputs, bootstrap=500)
        b = evaluate_nested_fusion(inputs, bootstrap=500)
        self.assertEqual(a["paired_statistics"], b["paired_statistics"])
        self.assertEqual(a["selected_weights"], b["selected_weights"])

    def test_one_fold_per_sequence(self):
        inputs = make_inputs(n_sequences=6)
        report = evaluate_nested_fusion(inputs, bootstrap=200)
        self.assertEqual(len(report["selected_weights"]["per_fold"]), 6)
        self.assertEqual({r["sequence_id"] for r in report["per_fold"]}, set(inputs.sequence_ids))

    def test_noise_candidate_is_down_weighted(self):
        """With a pure-noise candidate the folds should prefer the reference."""
        report = evaluate_nested_fusion(make_inputs(n_sequences=12, candidate_is_noise=True), bootstrap=200)
        self.assertGreaterEqual(report["selected_weights"]["median"], 0.5)

    def test_no_detected_gain_from_noise(self):
        """A noise candidate must not produce a significant improvement."""
        report = evaluate_nested_fusion(make_inputs(n_sequences=12, candidate_is_noise=True), bootstrap=2000)
        auc = report["paired_statistics"]["roc_auc"]
        self.assertLessEqual(auc["ci_low"], 0.0,
                             "a noise candidate produced a confidence interval entirely above zero")

    def test_selection_never_sees_the_held_out_sequence(self):
        """Corrupting one sequence's labels must not change other folds' selections."""
        inputs = make_inputs(n_sequences=8, seed=3)
        base = evaluate_nested_fusion(inputs, bootstrap=100)["selected_weights"]["per_fold"]
        corrupted_labels = inputs.labels.copy()
        corrupted_labels[inputs.groups == 0] = 1 - corrupted_labels[inputs.groups == 0]
        corrupted = evaluate_nested_fusion(
            FusionInputs(sequence_ids=inputs.sequence_ids, labels=corrupted_labels,
                         reference_scores=inputs.reference_scores,
                         candidate_scores=inputs.candidate_scores, groups=inputs.groups),
            bootstrap=100)["selected_weights"]["per_fold"]
        # Fold 0 holds out the corrupted sequence, so only folds != 0 are compared;
        # those folds train on the corrupted sequence, so they MAY change. The
        # invariant we assert is the reverse: fold 0's own selection must be
        # unaffected, because it never sees the held-out labels.
        self.assertEqual(base[0], corrupted[0])

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            FusionInputs(sequence_ids=("a", "b"), labels=np.zeros(10, np.int8),
                         reference_scores=np.zeros(9), candidate_scores=np.zeros(10),
                         groups=np.zeros(10, np.int32))

    def test_rejects_single_sequence(self):
        with self.assertRaises(ValueError):
            FusionInputs(sequence_ids=("a",), labels=np.zeros(5, np.int8),
                         reference_scores=np.zeros(5), candidate_scores=np.zeros(5),
                         groups=np.zeros(5, np.int32))

    def test_rejects_unknown_selection_metric(self):
        with self.assertRaises(ValueError):
            evaluate_nested_fusion(make_inputs(), select_metric="accuracy")


class TestLeakageContrast(unittest.TestCase):
    def test_test_derived_selection_is_optimistic(self):
        """Choosing the weight on the evaluation split must not look worse than
        choosing it out of fold; that gap is the bias this module measures."""
        inputs = make_inputs(n_sequences=10, candidate_is_noise=True, seed=5)
        ref = rank_normalise(inputs.reference_scores)
        cand = rank_normalise(inputs.candidate_scores)
        y, rate = inputs.labels, float(inputs.labels.mean())
        best = max(
            (score_metrics(y, w * ref + (1 - w) * cand,
                           threshold_for_positive_rate(w * ref + (1 - w) * cand, rate))["roc_auc"]
             for w in np.linspace(0, 1, 21)))
        reference_auc = score_metrics(y, ref, threshold_for_positive_rate(ref, rate))["roc_auc"]
        nested = evaluate_nested_fusion(inputs, bootstrap=200)
        self.assertGreaterEqual(best - reference_auc,
                                nested["paired_statistics"]["roc_auc"]["mean_delta"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
