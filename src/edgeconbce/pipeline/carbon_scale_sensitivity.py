"""Does the carbon conclusion survive a realistic remote model?

The headline carbon result in this project is that uncertainty-triggered
routing is the most carbon-intensive policy of those compared. That follows
from an identity rather than from a measurement: a routed antigen is charged
for a local pass and a remote pass, so

    C_adaptive = C_local + r * C_edge

and routing costs more than always-remote execution whenever the routed
fraction exceeds

    r* = 1 - C_local / C_edge .

The measured pair here is an 8M-parameter local model against a heavier remote
tier, and for that pair r* is small, so the policy's routed fraction clears it.
The objection is that this says more about the checkpoint pair than about the
task: a laboratory that routes to a genuinely large remote model (BepiPred-3.0
is built on ESM-2 650M) pays far more for each remote call, which pushes r* up
and may put the policy below it.

This module tests that objection. Two things matter for why the answer moves
so sharply:

1. In the recorded cost model the embodied term dominates. Operational cost is
   about 0.006% of the local total and 1.7% of the remote total, so the
   remote total is nearly all embodied charge.
2. Embodied charge is levied per request and does not scale with model size,
   while operational cost does. Scaling the remote model therefore grows a
   term that starts small, and r* moves only once that term becomes comparable
   to the embodied charge.

The function below is exact given the recorded constants: it takes the
operational scaling factor k (how much more operational cost a remote call
incurs than the measured one) and returns r*(k). Only the mapping from k to a
named checkpoint is an assumption, and it is stated rather than buried:
operational cost is taken to scale with parameter count at fixed sequence
length, which is the usual first-order approximation for a transformer forward
pass. It is a modelled result, not a metered one, exactly like the carbon
figures it extends.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Released ESM-2 checkpoints, by parameter count. The local tier is t6_8M.
ESM2_CHECKPOINTS: tuple[tuple[str, float], ...] = (
    ("esm2_t6_8M_UR50D", 8e6),
    ("esm2_t12_35M_UR50D", 35e6),
    ("esm2_t30_150M_UR50D", 150e6),
    ("esm2_t33_650M_UR50D", 650e6),
    ("esm2_t36_3B_UR50D", 3e9),
)
LOCAL_CHECKPOINT_PARAMS = 8e6


def break_even(c_local: float, c_edge: float) -> float:
    """Routed fraction above which routing costs more than always-remote.

    Exact given the two per-request costs: r* = 1 - C_local / C_edge.
    """
    if c_edge <= 0:
        raise ValueError(f"remote cost must be positive; got {c_edge}")
    return 1.0 - c_local / c_edge


def scaled_edge_cost(embodied_edge: float, operational_edge: float, k: float) -> float:
    """Remote per-request cost when operational cost is scaled by ``k``.

    The embodied charge is levied per request in the recorded model and is
    therefore left untouched. That is the model's own assumption, and it is
    the reason the break-even moves as slowly as it does at small ``k``.
    """
    if k < 0:
        raise ValueError(f"scaling factor must be non-negative; got {k}")
    return embodied_edge + operational_edge * k


def sensitivity(
    *,
    c_local: float,
    embodied_edge: float,
    operational_edge: float,
    route_rate: float,
    factors: Sequence[float],
) -> list[dict[str, Any]]:
    """r* over a range of operational scaling factors, with the verdict at each."""
    rows: list[dict[str, Any]] = []
    for k in factors:
        c_edge = scaled_edge_cost(embodied_edge, operational_edge, k)
        r_star = break_even(c_local, c_edge)
        rows.append(
            {
                "operational_scale_factor": float(k),
                "c_edge_gco2e": c_edge,
                "break_even_route_rate": r_star,
                "route_rate": route_rate,
                "routing_exceeds_edge_only": bool(route_rate > r_star),
            }
        )
    return rows


def checkpoint_rows(
    *,
    c_local: float,
    embodied_edge: float,
    operational_edge: float,
    route_rate: float,
) -> list[dict[str, Any]]:
    """The same curve, read at the parameter ratios of released checkpoints."""
    rows: list[dict[str, Any]] = []
    for name, params in ESM2_CHECKPOINTS:
        k = params / LOCAL_CHECKPOINT_PARAMS
        c_edge = scaled_edge_cost(embodied_edge, operational_edge, k)
        r_star = break_even(c_local, c_edge)
        rows.append(
            {
                "remote_checkpoint": name,
                "parameters": float(params),
                "operational_scale_factor": float(k),
                "c_edge_gco2e": c_edge,
                "break_even_route_rate": r_star,
                "route_rate": route_rate,
                "routing_exceeds_edge_only": bool(route_rate > r_star),
            }
        )
    return rows


def exact_crossover(
    c_local: float,
    embodied_edge: float,
    operational_edge: float,
    route_rate: float,
) -> dict[str, Any] | None:
    """The remote scale at which routing stops being the costlier policy.

    The grid above reports the first sampled factor past the crossing, which
    overstates it by however coarse the grid is. Solving r*(k) = route_rate
    for k gives the crossing itself:

        1 - c_local / (embodied + k * operational) = route_rate
        k = ((c_local / (1 - route_rate)) - embodied) / operational

    Returns None when the policy never catches up (route_rate >= 1) or when
    the crossing lies at or below the measured scale, k <= 1, in which case
    the ordering never held in the first place.
    """
    if operational_edge <= 0.0 or route_rate >= 1.0:
        return None
    k = ((c_local / (1.0 - route_rate)) - embodied_edge) / operational_edge
    if k <= 1.0:
        return None
    return {
        "operational_scale_factor": k,
        "implied_remote_parameters": LOCAL_CHECKPOINT_PARAMS * k,
        "c_edge_gco2e": scaled_edge_cost(embodied_edge, operational_edge, k),
        "note": (
            "Routing is the most carbon-intensive policy only while the remote "
            "tier costs less than this multiple of the measured remote call."
        ),
    }


def build_report(decomposition: dict[str, Any]) -> dict[str, Any]:
    c_local = float(decomposition["c_local_gco2e"])
    route_rate = float(decomposition["route_rate"])
    c_edge_recorded = float(decomposition["c_edge_gco2e"])
    operational_edge = float(decomposition["operational_term"]["mean_edge_gco2e_per_request"])
    # Anchor the curve to the recorded remote total rather than to the separately
    # stored embodied figure. The two differ by about 0.9% -- adding the stored
    # embodied and operational terms gives 0.1405 against a recorded 0.1393 --
    # and anchoring here keeps k = 1 reproducing the published break-even
    # exactly instead of a second, slightly different value.
    embodied_edge = c_edge_recorded - operational_edge

    factors = [1, 2, 5, 10, 20, 40, 81, 160, 375, 750]
    rows = sensitivity(
        c_local=c_local,
        embodied_edge=embodied_edge,
        operational_edge=operational_edge,
        route_rate=route_rate,
        factors=factors,
    )
    checkpoints = checkpoint_rows(
        c_local=c_local,
        embodied_edge=embodied_edge,
        operational_edge=operational_edge,
        route_rate=route_rate,
    )

    # The smallest scaling at which the recorded policy stops being the most
    # carbon-intensive option: the point the objection is really about.
    crossing = next((r for r in rows if not r["routing_exceeds_edge_only"]), None)
    inverts = next(
        (c for c in checkpoints if not c["routing_exceeds_edge_only"]), None
    )

    return {
        "project": "EdgeConBCE",
        "analysis": "carbon_scale_sensitivity",
        "question": (
            "Does the ordering that makes routing the most carbon-intensive policy "
            "survive a remote tier larger than the 8M-parameter model measured here?"
        ),
        "identity": "r* = 1 - C_local / C_edge",
        "assumption": (
            "Operational cost is taken to scale with remote parameter count at fixed "
            "sequence length; the embodied charge is per request in the recorded model "
            "and is held fixed. Modelled, not metered."
        ),
        "recorded": {
            "c_local_gco2e": c_local,
            "embodied_edge_gco2e_per_request": embodied_edge,
            "c_edge_recorded_gco2e": c_edge_recorded,
            "embodied_reconstruction_gap_gco2e": float(
                decomposition["embodied_term"]["implied_edge_gco2e_per_request"]
                + operational_edge
                - c_edge_recorded
            ),
            "operational_edge_gco2e_per_request": operational_edge,
            "operational_share_of_edge_total": operational_edge
            / (embodied_edge + operational_edge),
            "route_rate": route_rate,
            "break_even_at_measured_scale": break_even(
                c_local, embodied_edge + operational_edge
            ),
        },
        "sensitivity": rows,
        "checkpoints": checkpoints,
        "exact_crossover": exact_crossover(
            c_local, embodied_edge, operational_edge, route_rate
        ),
        "first_factor_without_penalty": crossing,
        "first_checkpoint_without_penalty": inverts,
        "verdict": (
            "ordering_inverts_at_realistic_remote_scale"
            if inverts is not None
            else "ordering_holds_across_released_checkpoints"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--carbon-decomposition", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    decomposition = json.loads(Path(args.carbon_decomposition).read_text(encoding="utf-8"))
    report = build_report(decomposition)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}: {report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
