"""Estimate how much a grid-scanned result is inflated, without rerunning it.

The survey in this project found published combined predictors whose
combination parameter was chosen on the data the result is reported on. A
reader who wants to know what those numbers would have been under a
leakage-free protocol cannot simply rerun nested cross-validation: that needs
the authors' pipeline, their splits and their compute, none of which a reader
has.

What a reader does have is the benchmark, the size of the grid that was
scanned, and the number of evaluation units. This module turns those into an
estimate of the inflation.

The mechanism is the maximal-selection inequality that Section 2.4 of the
manuscript states. A grid scan reports the maximum of ``G`` noisy estimates of
the same underlying quantity. If those estimates have sampling standard
deviation ``sigma`` and behave like correlated normals, the expected maximum
sits above any pre-committed grid point by approximately

    bias ~= sigma * a(G_eff)

where ``a(k)`` is the expected maximum of ``k`` independent standard normals
and ``G_eff`` is the effective number of independent grid points, which falls
below ``G`` as neighbouring weights produce more similar scores. Both ``sigma``
and ``G_eff`` are estimated by resampling evaluation units at a *fixed* weight,
so nothing is refitted and the estimate costs one pass over the grid rather
than one nested cross-validation per candidate.

This is an approximation with an assumption that is visibly not exact here: a
metric computed on fifteen antigens is not normally distributed. The reason it
is worth doing anyway is that this project has something rare, a directly
measured bias surface over sample size and grid density with a hundred
replicates per cell, so the approximation can be checked rather than asserted.
``validate`` does that check and reports the error, including where it is bad.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from edgeconbce.pipeline.nested_fusion import (
    FusionInputs,
    apply_ecdf,
    build_bp3_inputs,
    fit_ecdf,
    score_metrics,
    threshold_for_positive_rate,
)
from edgeconbce.pipeline.selection_bias_curve import subset_inputs, weight_grid

METRICS = ("mcc", "roc_auc", "average_precision", "f1")


@lru_cache(maxsize=512)
def _expected_max_integer(n: int, draws: int, seed: int) -> float:
    """``E[max]`` of ``n`` independent standard normals. Cached: the Monte Carlo
    is the expensive part and the answer depends only on these three numbers."""
    if n <= 1:
        return 0.0
    rng = np.random.default_rng(seed)
    total = 0.0
    block = 20_000
    done = 0
    while done < draws:
        take = min(block, draws - done)
        total += float(rng.standard_normal((take, n)).max(axis=1).sum())
        done += take
    return total / draws


def expected_max_normal(k: float, *, draws: int = 200_000, seed: int = 11) -> float:
    """``E[max]`` of ``k`` independent standard normals, by Monte Carlo.

    Accepts a non-integer ``k`` because the effective number of independent
    grid points is not an integer; the two bracketing integers are interpolated.
    """
    if k <= 1.0:
        return 0.0
    lo, hi = int(np.floor(k)), int(np.ceil(k))
    if lo == hi:
        return _expected_max_integer(lo, draws, seed)
    frac = k - lo
    return (1.0 - frac) * _expected_max_integer(lo, draws, seed) + frac * (
        _expected_max_integer(hi, draws, seed)
    )


def _fast_metric(labels: np.ndarray, scores: np.ndarray, threshold: float, metric: str) -> float:
    """One metric, computed directly.

    ``score_metrics`` computes all four, and ROC-AUC and average precision are
    the expensive two. The resampling loop here evaluates a whole grid on
    hundreds of resamples, so paying for metrics nobody reads dominates the
    runtime. Values agree with ``score_metrics`` for the metric requested.
    """
    if metric == "mcc":
        pred = scores >= threshold
        pos = labels.astype(bool)
        tp = float(np.count_nonzero(pred & pos))
        fp = float(np.count_nonzero(pred & ~pos))
        fn = float(np.count_nonzero(~pred & pos))
        tn = float(np.count_nonzero(~pred & ~pos))
        if len(np.unique(pred)) < 2:
            return 0.0
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return float((tp * tn - fp * fn) / den) if den > 0 else 0.0
    if metric == "f1":
        pred = scores >= threshold
        pos = labels.astype(bool)
        tp = float(np.count_nonzero(pred & pos))
        fp = float(np.count_nonzero(pred & ~pos))
        fn = float(np.count_nonzero(~pred & pos))
        den = 2 * tp + fp + fn
        return float(2 * tp / den) if den > 0 else 0.0
    return float(score_metrics(labels, scores, threshold)[metric])


def _delta_curve(
    inputs: FusionInputs,
    *,
    grid: Sequence[float],
    metric: str,
    reference_threshold: float | None,
    keep: np.ndarray | None = None,
) -> np.ndarray:
    """The fused-minus-reference difference at every weight on the grid.

    Scored the way the nested estimator scores: per antigen, then averaged, so
    the curve carries no aggregation difference.
    """
    sub = inputs if keep is None else subset_inputs(inputs, keep)
    labels, groups = sub.labels, sub.groups
    ref = apply_ecdf(fit_ecdf(sub.reference_scores), sub.reference_scores)
    cand = apply_ecdf(fit_ecdf(sub.candidate_scores), sub.candidate_scores)
    rate = (
        float((sub.reference_scores >= reference_threshold).mean())
        if reference_threshold is not None
        else float(labels.mean())
    )
    ref_threshold = threshold_for_positive_rate(ref, rate)

    uniq_groups = np.unique(groups)
    two_class = {g: len(np.unique(labels[groups == g])) >= 2 for g in uniq_groups}

    out = np.empty(len(grid), dtype=float)
    for i, w in enumerate(grid):
        blended = w * ref + (1.0 - w) * cand
        t = threshold_for_positive_rate(blended, rate)
        deltas = []
        for g in uniq_groups:
            m = groups == g
            if not two_class[g]:
                continue
            deltas.append(
                _fast_metric(labels[m], blended[m], t, metric)
                - _fast_metric(labels[m], ref[m], ref_threshold, metric)
            )
        out[i] = float(np.mean(deltas)) if deltas else np.nan
    return out


def estimate_bias(
    inputs: FusionInputs,
    *,
    grid_points: int,
    metric: str = "mcc",
    reference_threshold: float | None = None,
    resamples: int = 400,
    seed: int = 20260112,
) -> dict[str, Any]:
    """Predicted inflation of a grid scan, from resampling at fixed weights.

    No weight is ever re-selected inside the resampling: the same grid is
    scored on each resampled set of antigens, which gives the sampling spread
    of the metric at each fixed weight and the correlation between weights.
    Re-selecting inside the loop would measure the thing being predicted and
    make the estimate circular.
    """
    grid = weight_grid(grid_points)
    rng = np.random.default_rng(seed)
    groups = np.asarray(inputs.groups)
    unique = np.unique(groups)
    n = len(unique)

    curves = np.empty((resamples, len(grid)), dtype=float)
    for b in range(resamples):
        draw = rng.choice(unique, size=n, replace=True)
        curves[b] = _delta_curve(
            inputs,
            grid=grid,
            metric=metric,
            reference_threshold=reference_threshold,
            keep=np.asarray(draw),
        )

    ok = ~np.isnan(curves).any(axis=1)
    curves = curves[ok]
    if len(curves) < 10:
        raise RuntimeError("too few usable resamples to estimate the bias")

    sigma = float(np.mean(np.std(curves, axis=0, ddof=1)))

    # A weight whose curve is constant across resamples has no variance and no
    # correlation with anything; it also contributes no selection pressure, so
    # it is dropped rather than allowed to poison the mean correlation.
    varying = np.std(curves, axis=0, ddof=1) > 0.0
    if varying.sum() >= 2:
        corr = np.corrcoef(curves[:, varying], rowvar=False)
        iu = np.triu_indices_from(corr, k=1)
        rho = float(np.nanmean(corr[iu])) if len(iu[0]) else 0.0
    else:
        rho = 1.0
    rho = min(max(rho, 0.0), 0.999999)

    g = float(len(grid))
    g_eff = g / (1.0 + (g - 1.0) * rho)
    a = expected_max_normal(g_eff)
    return {
        "grid_points": int(grid_points),
        "antigens": int(n),
        "metric": metric,
        "sigma": sigma,
        "mean_grid_correlation": rho,
        "effective_grid_points": g_eff,
        "expected_max_standard_normal": a,
        "predicted_bias": sigma * a,
        "resamples": int(len(curves)),
    }


def validate(
    inputs: FusionInputs,
    measured_curve_path: str | Path,
    *,
    metric: str = "mcc",
    reference_threshold: float | None = None,
    resamples: int = 400,
    seed: int = 20260112,
) -> dict[str, Any]:
    """Check the prediction against the directly measured bias surface.

    The measured surface comes from ``selection_bias_curve``, which resamples
    antigen subsets and runs both protocols on each. Every cell with more than
    one replicate is a usable test point; the full-split cell is a single
    deterministic run and is excluded, as it is everywhere else in this project.
    """
    measured = json.loads(Path(measured_curve_path).read_text(encoding="utf-8"))
    rows = [r for r in measured["rows"] if r["replicates_used"] > 1]

    rng = np.random.default_rng(seed)
    groups = np.asarray(inputs.groups)
    unique = np.unique(groups)

    out: list[dict[str, Any]] = []
    for row in rows:
        m, g = int(row["antigens"]), int(row["grid_points"])
        draw = rng.choice(unique, size=m, replace=False)
        sub = subset_inputs(inputs, np.asarray(draw))
        try:
            est = estimate_bias(
                sub,
                grid_points=g,
                metric=metric,
                reference_threshold=reference_threshold,
                resamples=resamples,
                seed=seed + m * 100 + g,
            )
        except RuntimeError:
            continue
        measured_bias = float(row["mean_bias"])
        out.append(
            {
                "antigens": m,
                "grid_points": g,
                "measured_bias": measured_bias,
                "predicted_bias": est["predicted_bias"],
                "error": est["predicted_bias"] - measured_bias,
                "ratio": (
                    est["predicted_bias"] / measured_bias
                    if measured_bias
                    else float("nan")
                ),
                "sigma": est["sigma"],
                "effective_grid_points": est["effective_grid_points"],
            }
        )

    if not out:
        raise RuntimeError("no cells could be validated")

    err = np.array([c["error"] for c in out])
    meas = np.array([c["measured_bias"] for c in out])
    pred = np.array([c["predicted_bias"] for c in out])
    return {
        "cells": out,
        "n_cells": len(out),
        "mean_absolute_error": float(np.mean(np.abs(err))),
        "median_absolute_error": float(np.median(np.abs(err))),
        "mean_signed_error": float(np.mean(err)),
        "mean_measured_bias": float(np.mean(meas)),
        "mean_predicted_bias": float(np.mean(pred)),
        "correlation": float(np.corrcoef(meas, pred)[0, 1]) if len(out) > 2 else float("nan"),
        "share_within_50pc": float(
            np.mean(np.abs(pred - meas) <= 0.5 * np.abs(meas))
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--carbon-predictions", required=True)
    p.add_argument("--bepipred-raw-scores", required=True)
    p.add_argument("--selection-bias-curve", required=True)
    p.add_argument("--reference-threshold", type=float, default=None)
    p.add_argument("--select-metric", default="mcc", choices=METRICS)
    p.add_argument("--resamples", type=int, default=400)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)

    inputs = build_bp3_inputs(a.carbon_predictions, a.bepipred_raw_scores)
    full = estimate_bias(
        inputs,
        grid_points=21,
        metric=a.select_metric,
        reference_threshold=a.reference_threshold,
        resamples=a.resamples,
    )
    check = validate(
        inputs,
        a.selection_bias_curve,
        metric=a.select_metric,
        reference_threshold=a.reference_threshold,
        resamples=a.resamples,
    )
    sources = {
        str(g): decompose_bias_sources(
            inputs,
            grid_points=g,
            metric=a.select_metric,
            reference_threshold=a.reference_threshold,
        )
        for g in (3, 5, 11, 21, 41)
    }
    report = {
        "project": "EdgeConBCE",
        "analysis": "selection_bias_correction",
        "question": (
            "Can the inflation of a grid-scanned result be predicted from the "
            "sampling spread of the metric and the size of the grid, without "
            "rerunning nested cross-validation?"
        ),
        "method": "bias ~= sigma * E[max of G_eff standard normals]",
        "full_split_estimate": full,
        "validation": check,
        "bias_sources": sources,
        "note": (
            "Weights are never re-selected inside the resampling loop; the "
            "same grid is scored on each resampled set of antigens. The "
            "validation compares the prediction against the directly measured "
            "bias surface, excluding the full-split cell, which is a single "
            "deterministic run."
        ),
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0




def decompose_bias_sources(
    inputs: FusionInputs,
    *,
    grid_points: int = 21,
    metric: str = "mcc",
    reference_threshold: float | None = None,
) -> dict[str, Any]:
    """Split the measured inflation by which quantity was fitted on test data.

    The grid-scan protocol fits three things on the evaluation split: the
    mixing weight, the decision threshold, and the ECDF that puts the two
    predictors on a common scale. The literature's advice, and the analytic
    correction above, address the first of these. This measures all three by
    moving them one at a time.

    Three estimators, differing only in what the evaluation split is allowed to
    determine:

    ``nested``      weight, threshold and ECDF all from the training folds.
    ``weight_only`` weight chosen on the evaluation split; threshold and ECDF
                    still from the training folds.
    ``full``        all three from the evaluation split.

    The gap between the first two is what scanning the grid costs. The gap
    between the last two is what refitting the threshold and the normalisation
    costs, which is not a hyperparameter search and is rarely reported at all.
    """
    grid = weight_grid(grid_points)
    labels, groups = inputs.labels, inputs.groups
    unique = np.unique(groups)

    def _rate(ref_raw: np.ndarray) -> float:
        if reference_threshold is None:
            return float(labels.mean())
        return float((ref_raw >= reference_threshold).mean())

    def _per_antigen_delta(
        ref_n: np.ndarray, cand_n: np.ndarray, w: float, rate: float
    ) -> float:
        blended = w * ref_n + (1.0 - w) * cand_n
        t = threshold_for_positive_rate(blended, rate)
        rt = threshold_for_positive_rate(ref_n, rate)
        out = []
        for g in unique:
            m = groups == g
            if len(np.unique(labels[m])) < 2:
                continue
            out.append(
                _fast_metric(labels[m], blended[m], t, metric)
                - _fast_metric(labels[m], ref_n[m], rt, metric)
            )
        return float(np.mean(out)) if out else float("nan")

    # --- full: everything fitted on the evaluation split -------------------
    ref_full = apply_ecdf(fit_ecdf(inputs.reference_scores), inputs.reference_scores)
    cand_full = apply_ecdf(fit_ecdf(inputs.candidate_scores), inputs.candidate_scores)
    rate_full = _rate(inputs.reference_scores)
    full = max(
        _per_antigen_delta(ref_full, cand_full, w, rate_full) for w in grid
    )

    # --- out-of-fold transforms: ECDF and rate from the other antigens -----
    ref_oof = np.empty_like(inputs.reference_scores, dtype=float)
    cand_oof = np.empty_like(inputs.candidate_scores, dtype=float)
    rates = []
    for g in unique:
        held = groups == g
        rest = ~held
        ref_oof[held] = apply_ecdf(
            fit_ecdf(inputs.reference_scores[rest]), inputs.reference_scores[held]
        )
        cand_oof[held] = apply_ecdf(
            fit_ecdf(inputs.candidate_scores[rest]), inputs.candidate_scores[held]
        )
        rates.append(_rate(inputs.reference_scores[rest]))
    rate_oof = float(np.mean(rates))

    # weight still chosen on the evaluation split, transforms are not
    weight_only = max(
        _per_antigen_delta(ref_oof, cand_oof, w, rate_oof) for w in grid
    )

    # nested: weight chosen per fold on the other antigens
    nested_deltas = []
    for g in unique:
        held = groups == g
        rest = ~held
        if len(np.unique(labels[held])) < 2:
            continue
        best_w, best_v = float(grid[0]), -np.inf
        for w in grid:
            blended = w * ref_oof[rest] + (1.0 - w) * cand_oof[rest]
            t = threshold_for_positive_rate(blended, rate_oof)
            v = _fast_metric(labels[rest], blended, t, metric)
            if v > best_v:
                best_w, best_v = float(w), v
        blended_all = best_w * ref_oof + (1.0 - best_w) * cand_oof
        t = threshold_for_positive_rate(blended_all, rate_oof)
        rt = threshold_for_positive_rate(ref_oof, rate_oof)
        nested_deltas.append(
            _fast_metric(labels[held], blended_all[held], t, metric)
            - _fast_metric(labels[held], ref_oof[held], rt, metric)
        )
    nested = float(np.mean(nested_deltas))

    total = full - nested
    grid_component = weight_only - nested
    transform_component = full - weight_only
    return {
        "grid_points": int(grid_points),
        "metric": metric,
        "nested": nested,
        "weight_only": weight_only,
        "full": full,
        "total_inflation": total,
        "grid_component": grid_component,
        "transform_component": transform_component,
        "grid_share": grid_component / total if total else float("nan"),
        "transform_share": transform_component / total if total else float("nan"),
        "note": (
            "grid_component is what scanning the weight grid on the evaluation "
            "split costs. transform_component is what additionally fitting the "
            "decision threshold and the ECDF on it costs. Neither is a "
            "hyperparameter count, and only the first is what limiting a grid "
            "would address."
        ),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
