"""The carbon break-even must move with remote model scale, and say where."""
from __future__ import annotations

import pytest

from edgeconbce.pipeline.carbon_scale_sensitivity import (
    break_even,
    exact_crossover,
    scaled_edge_cost,
)

# The recorded constants of the BP3C50ID run, to the precision stored in
# outputs/nested_fusion_sci/carbon_decomposition.json.
C_LOCAL = 0.120003
EMBODIED_EDGE = 0.13692803819053065
OPERATIONAL_EDGE = 0.00240696180946934
ROUTE_RATE = 0.286667


def test_break_even_matches_the_identity() -> None:
    assert break_even(1.0, 4.0) == pytest.approx(0.75)


def test_scaling_the_remote_tier_raises_the_break_even() -> None:
    previous = -1.0
    for k in (1.0, 2.0, 10.0, 100.0):
        r = break_even(C_LOCAL, scaled_edge_cost(EMBODIED_EDGE, OPERATIONAL_EDGE, k))
        assert r > previous, "r* must increase with remote cost"
        previous = r


def test_exact_crossover_is_the_root_not_a_grid_point() -> None:
    """r*(k*) must equal the routed fraction, to floating-point precision."""
    out = exact_crossover(C_LOCAL, EMBODIED_EDGE, OPERATIONAL_EDGE, ROUTE_RATE)
    assert out is not None
    k = out["operational_scale_factor"]
    r_at_k = break_even(C_LOCAL, scaled_edge_cost(EMBODIED_EDGE, OPERATIONAL_EDGE, k))
    assert r_at_k == pytest.approx(ROUTE_RATE, abs=1e-12)


def test_crossover_brackets_the_ordering() -> None:
    """Just below k* routing is costlier; just above it is not."""
    out = exact_crossover(C_LOCAL, EMBODIED_EDGE, OPERATIONAL_EDGE, ROUTE_RATE)
    k = out["operational_scale_factor"]
    below = break_even(C_LOCAL, scaled_edge_cost(EMBODIED_EDGE, OPERATIONAL_EDGE, k * 0.99))
    above = break_even(C_LOCAL, scaled_edge_cost(EMBODIED_EDGE, OPERATIONAL_EDGE, k * 1.01))
    assert ROUTE_RATE > below, "below the crossing, routing should cost more"
    assert ROUTE_RATE < above, "above the crossing, it should not"


def test_no_crossover_when_the_policy_routes_everything() -> None:
    assert exact_crossover(C_LOCAL, EMBODIED_EDGE, OPERATIONAL_EDGE, 1.0) is None


def test_no_crossover_when_the_ordering_never_held() -> None:
    """A routed fraction already below the measured r* has no crossing above k=1."""
    assert exact_crossover(C_LOCAL, EMBODIED_EDGE, OPERATIONAL_EDGE, 0.01) is None
