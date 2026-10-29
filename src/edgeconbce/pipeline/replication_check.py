"""Reproduce the earlier fusion result under its own selection protocol.

The point of this script is attribution: if the published protocol, re-run
through the loaders and metrics used in the rest of the analysis, lands on the
published configuration and the published metrics, then any difference against
the nested estimate is caused by the protocol and not by this reimplementation.

Published values are not transcribed here. When ``--published-report`` is given,
the published artifact is read and every grid row is compared against the
recomputed one; the comparison, including the largest absolute disagreement, is
written into the output so it can be checked rather than asserted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nested_fusion import (DEFAULT_WEIGHT_GRID, build_bp3_inputs, build_iedb_inputs,  # noqa: E402
                           score_metrics)

TOLERANCE = 1e-6


def threshold_preserving_count(scores: np.ndarray, positive_count: int) -> float:
    """The published threshold rule: emit exactly ``positive_count`` positives."""
    if positive_count <= 0:
        return float(np.inf)
    if positive_count >= len(scores):
        return float(np.min(scores))
    return float(np.sort(scores)[-int(positive_count)])


def published_protocol(inputs, *, select_metric: str = "mcc",
                       reference_threshold: float | None = None) -> dict[str, Any]:
    """Blend raw scores, scan the weight grid on the evaluation split, keep the best.

    The published threshold rule gives the blended predictor the same number of
    positive calls that the reference predictor makes at its own operating
    threshold on this split.
    """
    y = inputs.labels
    ref, cand = inputs.reference_scores, inputs.candidate_scores
    if reference_threshold is None:
        reference_threshold = float(np.median(ref))
    reference_positive_count = int((ref >= reference_threshold).sum())
    reference_metrics = score_metrics(y, ref, reference_threshold)

    grid: list[dict[str, Any]] = []
    for weight in DEFAULT_WEIGHT_GRID:
        blended = weight * ref + (1.0 - weight) * cand
        metrics = score_metrics(y, blended, threshold_preserving_count(blended, reference_positive_count))
        grid.append({"weight": float(weight), **metrics})

    best = max(grid, key=lambda row: row[select_metric])
    return {
        "protocol": "weight and threshold selected on the evaluation split (as previously published)",
        "select_metric": select_metric,
        "positive_call_count": reference_positive_count,
        "selected": best,
        "reference": reference_metrics,
        "delta_vs_reference": {k: best[k] - reference_metrics[k] for k in reference_metrics},
        "grid": grid,
    }


def compare_with_published(grid: list[dict[str, Any]], published_path: Path) -> dict[str, Any]:
    payload = json.loads(published_path.read_text(encoding="utf-8"))
    rows = payload.get("weight_sensitivity") or payload.get("grid") or []
    if not rows:
        return {"status": "no_weight_grid_in_published_artifact", "path": str(published_path)}
    recomputed = {round(r["weight"], 2): r for r in grid}
    worst, worst_at = 0.0, None
    compared = 0
    for row in rows:
        weight = round(float(row.get("bepipred_weight", row.get("weight"))), 2)
        mine = recomputed.get(weight)
        if mine is None:
            continue
        for metric in ("roc_auc", "average_precision", "mcc", "f1"):
            if row.get(metric) is None:
                continue
            diff = abs(float(row[metric]) - float(mine[metric]))
            compared += 1
            if diff > worst:
                worst, worst_at = diff, f"w={weight} {metric}"
    return {
        "status": "reproduced" if worst <= TOLERANCE else "mismatch",
        "path": str(published_path),
        "rows_compared": compared,
        "max_absolute_difference": worst,
        "largest_difference_at": worst_at,
        "tolerance": TOLERANCE,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--carbon-predictions-bp3", required=True)
    p.add_argument("--bepipred-raw-scores", required=True)
    p.add_argument("--carbon-predictions-iedb")
    p.add_argument("--iedb-real-results")
    p.add_argument("--published-report-bp3")
    p.add_argument("--bepipred-threshold", type=float, default=0.1512,
                   help="BepiPred-3.0 operating threshold recorded in its rerun summary")
    p.add_argument("--iedb-reference-threshold", type=float, default=0.51,
                   help="local_only decision threshold recorded in the IEDB run")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    out: dict[str, Any] = {"analysis": "published_protocol_replication", "tolerance": TOLERANCE}

    bp3 = build_bp3_inputs(a.carbon_predictions_bp3, a.bepipred_raw_scores)
    out["bp3"] = published_protocol(bp3, reference_threshold=a.bepipred_threshold)
    out["bp3"]["reference_threshold"] = a.bepipred_threshold
    if a.published_report_bp3 and Path(a.published_report_bp3).exists():
        out["bp3"]["published_comparison"] = compare_with_published(
            out["bp3"]["grid"], Path(a.published_report_bp3))
    else:
        out["bp3"]["published_comparison"] = {
            "status": "published_artifact_not_available",
            "note": "rerun with --published-report-bp3 pointing at outputs/fusion_baseline_sci/fusion_report.json",
        }

    if a.carbon_predictions_iedb and a.iedb_real_results:
        iedb = build_iedb_inputs(a.carbon_predictions_iedb, a.iedb_real_results)
        out["iedb"] = published_protocol(iedb, reference_threshold=a.iedb_reference_threshold)
        out["iedb"]["reference_threshold"] = a.iedb_reference_threshold

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: {"selected_weight": v["selected"]["weight"],
                          "delta_vs_reference": v["delta_vs_reference"],
                          "published_comparison": v.get("published_comparison", {}).get("status")}
                      for k, v in out.items() if isinstance(v, dict) and "selected" in v}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
