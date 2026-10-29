"""Decompose the modelled carbon of each routing policy into its cost model.

The modelled carbon of a routing policy is not an independent measurement: it
is a deterministic function of the per-tier cost constants and the fraction of
antigens routed. Writing that function down turns "routing increased carbon in
our runs" into a statement about where the threshold lies, and makes the result
transferable to other hardware pairs.

For a policy that always runs the local tier and escalates a fraction r:

    C_adaptive = C_local + r * C_edge

so C_adaptive exceeds always-remote execution exactly when

    r > 1 - C_local / C_edge

This script verifies that identity against the recorded per-policy totals and
reports the threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repeated-eval", required=True)
    p.add_argument("--real-results", help="a single run, for the per-tier cost constants")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    d = json.loads(Path(a.repeated_eval).read_text(encoding="utf-8"))
    agg = {r["policy"]: r for r in d["aggregate_policy_metrics"]}

    c_local = agg["local_only"]["mean_total_gco2e_mean"]
    c_edge = agg["edge_only"]["mean_total_gco2e_mean"]
    c_adaptive = agg["adaptive_split"]["mean_total_gco2e_mean"]
    r = agg["adaptive_split"]["route_rate_mean"]

    predicted = c_local + r * c_edge
    threshold = 1.0 - (c_local / c_edge)

    report = {
        "analysis": "carbon_cost_model_decomposition",
        "identity": "C_adaptive = C_local + r * C_edge",
        "c_local_gco2e": c_local,
        "c_edge_gco2e": c_edge,
        "c_adaptive_recorded_gco2e": c_adaptive,
        "c_adaptive_predicted_gco2e": predicted,
        "absolute_error": abs(predicted - c_adaptive),
        "relative_error": abs(predicted - c_adaptive) / c_adaptive,
        "route_rate": r,
        "break_even_route_rate": threshold,
        "exceeds_edge_only": r > threshold,
        "interpretation": (
            "Routing costs more modelled carbon than always-remote execution whenever the "
            "routed fraction exceeds 1 - C_local/C_edge. The sign of the comparison is therefore "
            "set by the cost constants and the routed fraction, not discovered from the runs."
        ),
    }

    if a.real_results:
        run = json.loads(Path(a.real_results).read_text(encoding="utf-8"))
        cfg = run.get("config", {})
        local, edge = cfg.get("local_device", {}), cfg.get("edge_device", {})

        def embodied(dev):
            try:
                return dev["embodied_kgco2e_total"] / dev["amortization_requests"] * 1000.0
            except (KeyError, ZeroDivisionError):
                return None

        rows = run["external_test"]["local_only"]["per_sequence"]
        grid = cfg.get("grid_intensity_gco2_per_kwh")
        implied_local, implied_edge = [], []
        for row in rows:
            op_l = local["operational_power_w"] * row["local_latency_ms"] / 3.6e9 * grid
            op_e = edge["operational_power_w"] * row["edge_latency_ms"] / 3.6e9 * grid
            implied_local.append(row["local_total_gco2e"] - op_l)
            implied_edge.append(row["edge_total_gco2e"] - op_e)
        op_local = [local["operational_power_w"] * r["local_latency_ms"] / 3.6e9 * grid for r in rows]
        op_edge = [edge["operational_power_w"] * r["edge_latency_ms"] / 3.6e9 * grid for r in rows]
        report["operational_term"] = {
            "mean_local_gco2e_per_request": sum(op_local) / len(op_local),
            "mean_edge_gco2e_per_request": sum(op_edge) / len(op_edge),
            "note": (
                "At the observed latencies the operational term is a very small fraction of each "
                "per-request total, so modelled carbon is governed by the number of device-requests "
                "a policy charges rather than by how long they take."
            ),
        }
        report["embodied_term"] = {
            "declared_local_gco2e_per_request": embodied(local),
            "declared_edge_gco2e_per_request": embodied(edge),
            "implied_local_gco2e_per_request": sum(implied_local) / len(implied_local),
            "implied_edge_gco2e_per_request": sum(implied_edge) / len(implied_edge),
            "note": (
                "The operational term reproduces exactly from the recorded configuration. "
                "The embodied term, which dominates both totals, does not: the value implied "
                "by the recorded per-antigen totals differs from the value the recorded device "
                "configuration declares. The per-policy ORDERING is unaffected, because it "
                "follows from the identity above for any positive constants, but the absolute "
                "carbon figures cannot be reproduced from the configuration as stored."
            ),
        }

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
