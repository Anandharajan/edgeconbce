"""The predicted inflation must behave the way the inequality says it should."""

from __future__ import annotations

import numpy as np
import pytest

from edgeconbce.pipeline import selection_bias_correction as sbc
from edgeconbce.pipeline.nested_fusion import FusionInputs


def _toy(n_antigens: int = 8, length: int = 30, seed: int = 5) -> FusionInputs:
    rng = np.random.default_rng(seed)
    n = n_antigens * length
    labels = (rng.random(n) < 0.25).astype(np.int8)
    remote = labels * 0.5 + rng.random(n) * 0.5
    local = labels * 0.4 + rng.random(n) * 0.6
    groups = np.repeat(np.arange(n_antigens), length)
    return FusionInputs(
        sequence_ids=tuple(f"a{i}" for i in range(n_antigens)),
        labels=labels,
        reference_scores=remote,
        candidate_scores=local,
        groups=groups,
    )


def test_expected_max_is_zero_for_a_single_candidate() -> None:
    """One grid point is no choice, so it carries no selection bias."""
    assert sbc.expected_max_normal(1) == 0.0
    assert sbc.expected_max_normal(0.5) == 0.0


def test_expected_max_grows_with_the_number_of_candidates() -> None:
    values = [sbc.expected_max_normal(k) for k in (2, 5, 10, 40, 100)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_expected_max_matches_known_values() -> None:
    """E[max] of 2 standard normals is 1/sqrt(pi); of 3 it is 1.5/sqrt(pi)."""
    assert sbc.expected_max_normal(2) == pytest.approx(1 / np.sqrt(np.pi), abs=0.01)
    assert sbc.expected_max_normal(3) == pytest.approx(
        1.5 / np.sqrt(np.pi), abs=0.01
    )


def test_expected_max_interpolates_non_integer_grid_sizes() -> None:
    lo, hi = sbc.expected_max_normal(4), sbc.expected_max_normal(5)
    mid = sbc.expected_max_normal(4.5)
    assert lo < mid < hi


def test_predicted_bias_is_non_negative() -> None:
    """The inequality is one-sided; a scan cannot be pessimistic in expectation."""
    inputs = _toy()
    est = sbc.estimate_bias(inputs, grid_points=11, resamples=25, seed=1)
    assert est["predicted_bias"] >= 0.0


def test_predicted_bias_grows_with_grid_density() -> None:
    """More candidates scanned on the same data means more room to be flattered."""
    inputs = _toy()
    coarse = sbc.estimate_bias(inputs, grid_points=3, resamples=40, seed=2)
    fine = sbc.estimate_bias(inputs, grid_points=41, resamples=40, seed=2)
    assert fine["effective_grid_points"] >= coarse["effective_grid_points"]
    assert fine["predicted_bias"] >= coarse["predicted_bias"]


def test_effective_grid_points_never_exceed_the_grid() -> None:
    inputs = _toy()
    for g in (3, 11, 41):
        est = sbc.estimate_bias(inputs, grid_points=g, resamples=25, seed=4)
        assert 1.0 <= est["effective_grid_points"] <= g + 1e-9


def test_perfectly_correlated_grid_collapses_to_one_candidate() -> None:
    """If every weight gives the same curve there is no selection, so no bias."""
    g = 21.0
    rho = 1.0 - 1e-12
    g_eff = g / (1.0 + (g - 1.0) * rho)
    assert g_eff == pytest.approx(1.0, abs=1e-6)
    assert sbc.expected_max_normal(g_eff) == pytest.approx(0.0, abs=1e-6)


def test_delta_curve_has_one_entry_per_weight() -> None:
    inputs = _toy()
    from edgeconbce.pipeline.selection_bias_curve import weight_grid

    grid = weight_grid(11)
    curve = sbc._delta_curve(
        inputs, grid=grid, metric="mcc", reference_threshold=None
    )
    assert curve.shape == (11,)
    assert np.isfinite(curve).all()
