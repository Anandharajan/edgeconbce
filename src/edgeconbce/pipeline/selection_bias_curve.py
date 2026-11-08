"""How far does selecting a fusion weight on the evaluation split move the estimate?

The nested-selection analysis elsewhere in this project reports one instance of
selection bias: on BP3C50ID the mean per-antigen MCC difference between a fused
predictor and its reference moves from a positive value under the published,
test-derived protocol to a negative one under leave-one-sequence-out nested
selection. One instance says that the bias exists. It does not say how large it
should be expected to be, or what governs it.

This module answers the second question by resampling the real data. For a
given number of antigens and a given weight grid it repeatedly draws a subset,
estimates the fused-minus-reference difference twice on that same subset, and
records the gap:

    test-derived   scan the weight grid on the subset, keep the best row
                   (the protocol a paper follows when it reports a tuned
                   combination parameter without holding anything out)
    nested         leave-one-sequence-out within the subset, so the weight and
                   threshold applied to a held-out antigen were chosen without
                   it

    bias = test-derived - nested

Both estimators see exactly the same antigens, so the gap is attributable to
the selection rule and not to which data were drawn. Averaged over draws it
estimates the optimism a reader should expect from a paper of that size using a
grid of that density.

Nothing here is specific to epitope prediction: the same construction applies
to any two-predictor combination whose parameter is chosen on the data the
result is reported on. The numbers are, of course, specific to these
predictors and this split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from edgeconbce.pipeline.nested_fusion import (
    METRICS,
    FusionInputs,
    apply_ecdf,
    build_bp3_inputs,
    evaluate_nested_fusion,
    fit_ecdf,
    score_metrics,
    threshold_for_positive_rate,
)
from edgeconbce.pipeline.replication_check import published_protocol

DEFAULT_SEED = 20261108
DEFAULT_REPLICATES = 100
DEFAULT_SIZES: tuple[int, ...] = (4, 6, 8, 10, 12, 15)
#: Grid densities to compare, as the number of equally spaced weights in [0, 1].
DEFAULT_GRID_SIZES: tuple[int, ...] = (3, 5, 11, 21, 41)


def weight_grid(n_points: int) -> tuple[float, ...]:
    """``n_points`` equally spaced weights spanning [0, 1]."""
    if n_points < 2:
        raise ValueError(f"a weight grid needs at least two points; got {n_points}")
    return tuple(round(float(w), 6) for w in np.linspace(0.0, 1.0, n_points))


def subset_inputs(inputs: FusionInputs, keep: Sequence[int]) -> FusionInputs:
    """Restrict ``inputs`` to the antigens at the given positions.

    Groups are renumbered to stay contiguous, because the leave-one-out loop
    indexes antigens by group number rather than by identity.
    """
    keep = list(keep)
    if len(keep) < 2:
        raise ValueError("leave-one-sequence-out needs at least two antigens")
    mask = np.isin(inputs.groups, keep)
    old = np.asarray(keep)
    remap = {int(g): i for i, g in enumerate(old)}
    groups = np.array([remap[int(g)] for g in inputs.groups[mask]], dtype=inputs.groups.dtype)
    return replace(
        inputs,
        sequence_ids=tuple(inputs.sequence_ids[i] for i in keep),
        labels=inputs.labels[mask],
        reference_scores=inputs.reference_scores[mask],
        candidate_scores=inputs.candidate_scores[mask],
        groups=groups,
    )


def predictor_agreement(inputs: FusionInputs) -> float:
    """Spearman correlation between the two predictors' residue scores.

    Reported as a covariate: two predictors that rank residues almost
    identically leave the weight little to do, so the grid scan has less room
    to overfit.
    """
    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average ties so the coefficient is the usual Spearman statistic
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts), dtype=np.float64)
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    a, b = ranks(inputs.reference_scores), ranks(inputs.candidate_scores)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def _nested_mean_delta(
    inputs: FusionInputs, *, metric: str, grid: Sequence[float], reference_threshold: float | None
) -> float:
    """Mean per-antigen fused-minus-reference difference under nested selection."""
    report = evaluate_nested_fusion(
        inputs,
        select_metric=metric,
        weight_grid=grid,
        bootstrap=2,  # intervals are not used here; the mean is what we need
        reference_threshold=reference_threshold,
    )
    deltas = [row[f"delta_{metric}"] for row in report["per_fold"]]
    return float(np.mean(deltas))


def _test_derived_delta(
    inputs: FusionInputs, *, metric: str, grid: Sequence[float], reference_threshold: float | None
) -> float:
    """Fused-minus-reference difference under the published, test-derived protocol."""
    report = published_protocol(
        inputs,
        select_metric=metric,
        reference_threshold=reference_threshold,
        weight_grid=grid,
    )
    return float(report["delta_vs_reference"][metric])



def _test_derived_per_antigen_delta(
    inputs: FusionInputs, *, metric: str, grid: Sequence[float], reference_threshold: float | None
) -> float:
    """Test-derived selection, scored the way the nested estimator scores.

    This is the nested loop with one thing changed: the ECDF, the weight and
    the threshold are all fitted on every antigen, including the one being
    scored, instead of on the training folds. Everything else -- the per-antigen
    scoring, and averaging deltas over antigens rather than pooling residues --
    matches ``_nested_mean_delta`` exactly.

    Comparing this against the nested estimate isolates the selection effect.
    Comparing the *pooled* test-derived figure against the nested estimate, as
    the two headline numbers in the manuscript do, mixes the selection effect
    with the difference between pooling residues and averaging over antigens.
    Both are reported so the two can be told apart.
    """
    labels, groups = inputs.labels, inputs.groups
    ref_ecdf = fit_ecdf(inputs.reference_scores)
    cand_ecdf = fit_ecdf(inputs.candidate_scores)
    ref = apply_ecdf(ref_ecdf, inputs.reference_scores)
    cand = apply_ecdf(cand_ecdf, inputs.candidate_scores)

    if reference_threshold is not None:
        rate = float((inputs.reference_scores >= reference_threshold).mean())
    else:
        rate = float(labels.mean())

    best_weight, best_value = float(grid[0]), -np.inf
    for weight in grid:
        blended = weight * ref + (1.0 - weight) * cand
        value = score_metrics(
            labels, blended, threshold_for_positive_rate(blended, rate)
        )[metric]
        if np.isfinite(value) and value > best_value:
            best_weight, best_value = float(weight), float(value)

    blended = best_weight * ref + (1.0 - best_weight) * cand
    threshold = threshold_for_positive_rate(blended, rate)
    ref_threshold = threshold_for_positive_rate(ref, rate)

    deltas: list[float] = []
    for held_out in range(len(inputs.sequence_ids)):
        mask = groups == held_out
        if not mask.any() or len(np.unique(labels[mask])) < 2:
            continue
        fused = score_metrics(labels[mask], blended[mask], threshold)[metric]
        baseline = score_metrics(labels[mask], ref[mask], ref_threshold)[metric]
        deltas.append(fused - baseline)
    if not deltas:
        return float("nan")
    return float(np.mean(deltas))


def bias_curve(
    inputs: FusionInputs,
    *,
    metric: str = "mcc",
    sizes: Sequence[int] = DEFAULT_SIZES,
    grid_sizes: Sequence[int] = DEFAULT_GRID_SIZES,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    reference_threshold: float | None = None,
) -> dict[str, Any]:
    """Estimate selection bias over antigen counts and weight-grid densities.

    Antigens are drawn without replacement, so every replicate is a design that
    could genuinely have been run, rather than one with duplicated antigens.
    """
    n_total = len(inputs.sequence_ids)
    sizes = [int(s) for s in sizes if 2 <= int(s) <= n_total]
    if not sizes:
        raise ValueError(f"no requested subset size fits a split of {n_total} antigens")
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    total_cells = len(grid_sizes) * len(sizes)
    cell = 0
    for n_points in grid_sizes:
        grid = weight_grid(n_points)
        for size in sizes:
            # This loop runs for tens of minutes at the default settings, so
            # it reports where it is. Progress goes to stderr, leaving stdout
            # for the one line that names the artifact.
            cell += 1
            print(
                f"  [{cell}/{total_cells}] grid={n_points:>3} antigens={size:>3}",
                file=sys.stderr,
                flush=True,
            )
            # At full size there is only one subset, so extra draws would be
            # identical work: the cell reduces to the headline analysis.
            draws = 1 if size == n_total else replicates
            biases: list[float] = []
            optimistic: list[float] = []
            honest: list[float] = []
            agreement: list[float] = []
            for _ in range(draws):
                keep = rng.choice(n_total, size=size, replace=False)
                sub = subset_inputs(inputs, sorted(int(k) for k in keep))
                try:
                    td = _test_derived_delta(
                        sub, metric=metric, grid=grid, reference_threshold=reference_threshold
                    )
                    ns = _nested_mean_delta(
                        sub, metric=metric, grid=grid, reference_threshold=reference_threshold
                    )
                except (ValueError, ZeroDivisionError):
                    # a draw with no positive residues in some fold is skipped
                    # rather than silently counted as zero bias
                    continue
                if not (np.isfinite(td) and np.isfinite(ns)):
                    continue
                optimistic.append(td)
                honest.append(ns)
                biases.append(td - ns)
                agreement.append(predictor_agreement(sub))
            if not biases:
                continue
            arr = np.asarray(biases, dtype=np.float64)
            rows.append(
                {
                    "metric": metric,
                    "grid_points": int(n_points),
                    "antigens": int(size),
                    "replicates_used": int(len(arr)),
                    "replicates_requested": int(draws),
                    "mean_bias": float(arr.mean()),
                    "median_bias": float(np.median(arr)),
                    "bias_p05": float(np.quantile(arr, 0.05)),
                    "bias_p95": float(np.quantile(arr, 0.95)),
                    "share_bias_positive": float((arr > 0).mean()),
                    "mean_test_derived_delta": float(np.mean(optimistic)),
                    "mean_nested_delta": float(np.mean(honest)),
                    "mean_predictor_agreement": float(np.mean(agreement)),
                }
            )

    return {
        "project": "EdgeConBCE",
        "analysis": "selection_bias_curve",
        "question": (
            "How much does choosing the fusion weight on the evaluation split inflate "
            "the reported fused-minus-reference difference, as a function of the number "
            "of antigens and the density of the weight grid?"
        ),
        "estimators": {
            "test_derived": "scan the weight grid on the evaluation split, keep the best row",
            "nested": "leave-one-sequence-out selection within the same subset",
            "bias": "test_derived - nested, on identical antigens",
        },
        "reference_model": inputs.reference_name,
        "candidate_model": inputs.candidate_name,
        "split": inputs.metadata.get("split"),
        "antigens_available": n_total,
        "select_metric": metric,
        "reference_threshold": reference_threshold,
        "sampling": "antigens drawn without replacement",
        "seed": seed,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--carbon-predictions", required=True)
    parser.add_argument("--bepipred-raw-scores", required=True)
    parser.add_argument("--reference-threshold", type=float, default=None)
    parser.add_argument("--select-metric", default="mcc", choices=METRICS)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    inputs = build_bp3_inputs(args.carbon_predictions, args.bepipred_raw_scores)
    report = bias_curve(
        inputs,
        metric=args.select_metric,
        replicates=args.replicates,
        seed=args.seed,
        reference_threshold=args.reference_threshold,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(report['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
