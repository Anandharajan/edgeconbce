"""Region routing must stay leakage-free and must bound its own disclosure."""

from __future__ import annotations

import numpy as np
import pytest

from edgeconbce.pipeline import region_routing as rr
from edgeconbce.pipeline.nested_fusion import FusionInputs


def _toy(n_antigens: int = 6, length: int = 40, seed: int = 3) -> FusionInputs:
    rng = np.random.default_rng(seed)
    n = n_antigens * length
    labels = (rng.random(n) < 0.2).astype(np.int8)
    # Remote is informative, local is a noisier view of the same signal, so a
    # sensible router has something to gain by escalating.
    remote = labels * 0.6 + rng.random(n) * 0.4
    local = labels * 0.3 + rng.random(n) * 0.7
    groups = np.repeat(np.arange(n_antigens), length)
    return FusionInputs(
        sequence_ids=tuple(f"a{i}" for i in range(n_antigens)),
        labels=labels,
        reference_scores=remote,
        candidate_scores=local,
        groups=groups,
    )


def test_window_bounds_tile_the_sequence_exactly() -> None:
    for length, w in ((40, 8), (41, 8), (7, 16), (100, 12)):
        bounds = rr._window_bounds(length, w)
        covered = sum(stop - start for start, stop in bounds)
        assert covered == length
        assert bounds[0][0] == 0 and bounds[-1][1] == length
        for (_, a), (b, _) in zip(bounds, bounds[1:]):
            assert a == b, "windows must be contiguous and non-overlapping"


def test_zero_budget_discloses_nothing_and_one_discloses_everything() -> None:
    scores = np.linspace(0.0, 1.0, 50)
    none = rr.plan_region_routing(scores, 0.5, window=10, budget=0.0)
    assert none.routed_mask.sum() == 0
    allp = rr.plan_region_routing(scores, 0.5, window=10, budget=1.0)
    assert allp.routed_mask.all()


def test_budget_is_never_exceeded() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(97)
    for budget in (0.05, 0.1, 0.33, 0.5, 0.9):
        for w in (8, 12, 16):
            plan = rr.plan_region_routing(scores, 0.5, window=w, budget=budget)
            assert plan.routed_mask.sum() <= np.floor(budget * len(scores))


def test_routing_prefers_the_least_confident_windows() -> None:
    """The first window is right at the threshold, the rest are far from it."""
    scores = np.concatenate([np.full(10, 0.50), np.full(40, 0.99)])
    plan = rr.plan_region_routing(scores, 0.5, window=10, budget=0.2)
    assert plan.routed_mask[:10].all()
    assert not plan.routed_mask[10:].any()


def test_splice_takes_remote_only_where_routed() -> None:
    local = np.zeros(10)
    remote = np.ones(10)
    mask = np.zeros(10, dtype=bool)
    mask[3:6] = True
    out = rr.splice(local, remote, mask)
    assert out[3:6].tolist() == [1.0, 1.0, 1.0]
    assert out[:3].tolist() == [0.0, 0.0, 0.0]
    assert out[6:].tolist() == [0.0] * 4


def test_out_of_fold_transform_is_fitted_on_the_other_antigens() -> None:
    """Each antigen's ECDF must come from the antigens that are not it.

    Checked against an independently constructed transform rather than against
    a perturbation, because perturbing an antigen's raw scores legitimately
    changes its normalised scores; what must not happen is that the antigen
    contributes to the transform applied to it.
    """
    from edgeconbce.pipeline.nested_fusion import apply_ecdf, fit_ecdf

    inputs = _toy()
    oof = rr.out_of_fold_scores(inputs, reference_threshold=None)
    groups = np.asarray(inputs.groups)

    for g in np.unique(groups):
        held = groups == g
        rest = ~held
        expected_local = apply_ecdf(
            fit_ecdf(inputs.candidate_scores[rest]), inputs.candidate_scores[held]
        )
        expected_remote = apply_ecdf(
            fit_ecdf(inputs.reference_scores[rest]), inputs.reference_scores[held]
        )
        assert np.allclose(oof["local"][held], expected_local)
        assert np.allclose(oof["remote"][held], expected_remote)


def test_region_and_antigen_coincide_at_the_endpoints() -> None:
    inputs = _toy()
    oof = rr.out_of_fold_scores(inputs, reference_threshold=None)
    for budget in (0.0, 1.0):
        r = rr.evaluate_pooled(
            inputs, oof, policy="region", window=8, budget=budget
        )
        a = rr.evaluate_pooled(
            inputs, oof, policy="antigen", window=0, budget=budget
        )
        assert r["mcc"] == pytest.approx(a["mcc"], abs=1e-12)


def test_antigen_routing_discloses_whole_antigens_only() -> None:
    inputs = _toy()
    oof = rr.out_of_fold_scores(inputs, reference_threshold=None)
    groups = np.asarray(inputs.groups)
    mask = rr._route_pooled(
        oof, groups, policy="antigen", window=0, budget=0.5
    )
    for g in np.unique(groups):
        sel = groups == g
        touched = mask[sel]
        assert touched.all() or not touched.any(), (
            "whole-antigen routing must not disclose part of an antigen"
        )


def test_region_routing_can_reach_disclosure_levels_antigen_routing_cannot() -> None:
    """The point of the architecture: the interior of the disclosure axis."""
    inputs = _toy()
    oof = rr.out_of_fold_scores(inputs, reference_threshold=None)
    groups = np.asarray(inputs.groups)
    region = rr._route_pooled(
        oof, groups, policy="region", window=8, budget=0.1
    )
    antigen = rr._route_pooled(
        oof, groups, policy="antigen", window=0, budget=0.1
    )
    # With 6 equal antigens, a 10% residue budget cannot fit a whole antigen.
    assert antigen.sum() == 0
    assert region.sum() > 0
