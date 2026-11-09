"""Split the headline test-derived-minus-nested gap into its two components.

The manuscript's headline gap compares a test-derived figure computed by pooling
residues against a nested estimate that averages per-antigen differences. Those
two numbers differ in two ways at once:

  selection    where the weight and the decision threshold were fitted, which
               is the effect the paper is about;
  aggregation  whether residues are pooled into one metric or per-antigen
               differences are averaged, which is an accounting choice and not
               a selection effect at all.

Attributing the whole gap to selection overstates it. ``selection_bias_curve``
already defines a test-derived variant scored exactly the way the nested
estimator scores, which holds aggregation fixed and so isolates selection; this
module calls it and reports the split. At the full antigen split there is one
possible subset, so every quantity here is a single deterministic evaluation
rather than a mean over draws.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from edgeconbce.pipeline.selection_bias_curve import (
    METRICS,
    _nested_mean_delta,
    _test_derived_delta,
    _test_derived_per_antigen_delta,
    build_bp3_inputs,
    weight_grid,
)


def decompose(
    inputs: Any,
    *,
    metric: str = "mcc",
    grid_points: int = 21,
    reference_threshold: float | None = None,
) -> dict[str, Any]:
    """Return the headline gap and its selection and aggregation components."""
    grid = weight_grid(grid_points)
    kw = dict(metric=metric, grid=grid, reference_threshold=reference_threshold)

    pooled = _test_derived_delta(inputs, **kw)
    matched = _test_derived_per_antigen_delta(inputs, **kw)
    nested = _nested_mean_delta(inputs, **kw)

    headline = pooled - nested
    selection = matched - nested
    aggregation = pooled - matched
    return {
        "project": "EdgeConBCE",
        "analysis": "selection_bias_decomposition",
        "question": (
            "How much of the headline test-derived-minus-nested gap is selection, "
            "and how much is the difference between pooling residues and averaging "
            "over antigens?"
        ),
        "metric": metric,
        "grid_points": int(grid_points),
        "antigens": int(len(inputs.sequence_ids)),
        "replicates": 1,
        "deterministic": True,
        "test_derived_pooled": float(pooled),
        "test_derived_per_antigen": float(matched),
        "nested_per_antigen": float(nested),
        "headline_gap": float(headline),
        "selection_component": float(selection),
        "aggregation_component": float(aggregation),
        "selection_share_of_headline": (
            float(selection / headline) if headline else None
        ),
        "note": (
            "selection_component holds aggregation fixed by scoring the "
            "test-derived protocol per antigen, exactly as the nested estimator "
            "scores. aggregation_component is the residual and is not a "
            "selection effect."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--carbon-predictions", required=True)
    p.add_argument("--bepipred-raw-scores", required=True)
    p.add_argument("--reference-threshold", type=float, default=None)
    p.add_argument("--select-metric", default="mcc", choices=METRICS)
    p.add_argument("--grid-points", type=int, default=21)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)

    inputs = build_bp3_inputs(a.carbon_predictions, a.bepipred_raw_scores)
    report = decompose(
        inputs,
        metric=a.select_metric,
        grid_points=a.grid_points,
        reference_threshold=a.reference_threshold,
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
