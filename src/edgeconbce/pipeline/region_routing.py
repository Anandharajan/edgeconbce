"""Route contiguous regions of an antigen instead of the whole antigen.

Whole-antigen routing makes disclosure binary: either the entire construct
leaves the laboratory or none of it does. That is the wrong granularity for the
problem it claims to solve. A laboratory that will not disclose an unpublished
antigen will not disclose it because the remote model is uncertain about one
loop of it.

Region routing splits each antigen into contiguous windows, ranks the windows
by how uncertain the local predictor is on them, and sends only the least
confident windows to the remote predictor, up to a budget expressed as the
fraction of residues disclosed. Remote scores are spliced into the routed
windows and local scores are kept everywhere else. Disclosure becomes a
continuous quantity with a budget, rather than a property of an antigen.

Two choices here are deliberate and matter for whether the comparison is fair:

*Splicing needs a common scale.* The two predictors' scores are not
commensurable, so concatenating raw scores from different predictors into one
vector produces a ranking that is an artefact of their calibration rather than
of either model. Both score vectors are mapped through an ECDF fitted on the
training folds before splicing, which is the same transform the fusion analysis
uses, and the ECDF never sees the held-out antigen.

*The parameters are selected leave-one-antigen-out.* Window length and
disclosure budget are chosen on the training folds and applied unchanged to the
held-out antigen. Scanning them on the evaluation split is exactly the failure
this project is about, and a new architecture tuned that way would reproduce it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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

DEFAULT_WINDOWS: tuple[int, ...] = (8, 12, 16, 24, 32, 48)
DEFAULT_BUDGETS: tuple[float, ...] = (
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0
)
METRICS = ("mcc", "roc_auc", "average_precision", "f1")


@dataclass(frozen=True)
class RoutingPlan:
    """Which residues of one antigen are sent to the remote predictor."""

    routed_mask: np.ndarray
    disclosed_fraction: float
    n_windows_routed: int
    n_windows_total: int


def _window_bounds(length: int, window: int) -> list[tuple[int, int]]:
    """Contiguous non-overlapping windows covering ``length`` residues."""
    if window <= 0:
        raise ValueError("window must be positive")
    bounds = [(s, min(s + window, length)) for s in range(0, length, window)]
    return bounds


def local_margin(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Distance of each residue's score from the decision threshold.

    Small values mean the local predictor is close to flipping its call, which
    is the sense in which it is uncertain about that residue.
    """
    return np.abs(np.asarray(scores, dtype=float) - float(threshold))


def plan_region_routing(
    local_scores: np.ndarray,
    threshold: float,
    *,
    window: int,
    budget: float,
) -> RoutingPlan:
    """Select the least confident windows of one antigen, within ``budget``.

    ``budget`` is the maximum fraction of the antigen's residues that may be
    disclosed. Windows are taken in increasing order of mean local margin and
    admitted while they fit; a window that would overshoot the budget is
    skipped rather than truncated, because half a window is not a region a
    remote model can score in context.
    """
    length = len(local_scores)
    mask = np.zeros(length, dtype=bool)
    if budget <= 0.0 or length == 0:
        return RoutingPlan(mask, 0.0, 0, len(_window_bounds(length, window)))

    bounds = _window_bounds(length, window)
    margins = local_margin(local_scores, threshold)
    order = sorted(bounds, key=lambda b: float(np.mean(margins[b[0]:b[1]])))

    allowance = int(np.floor(budget * length))
    disclosed = 0
    routed = 0
    for start, stop in order:
        size = stop - start
        if disclosed + size > allowance:
            continue
        mask[start:stop] = True
        disclosed += size
        routed += 1
    return RoutingPlan(mask, disclosed / length if length else 0.0, routed, len(bounds))


def plan_antigen_routing(
    local_scores: np.ndarray,
    threshold: float,
    *,
    budget: float,
) -> RoutingPlan:
    """Whole-antigen routing: the antigen is disclosed entirely or not at all.

    The comparison baseline. Expressed against the same residue budget so the
    two policies can be read on one axis.
    """
    length = len(local_scores)
    route = budget >= 1.0 - 1e-12
    mask = np.ones(length, dtype=bool) if route else np.zeros(length, dtype=bool)
    return RoutingPlan(mask, 1.0 if route else 0.0, 1 if route else 0, 1)


def splice(
    local_norm: np.ndarray, remote_norm: np.ndarray, routed_mask: np.ndarray
) -> np.ndarray:
    """Remote scores where routed, local scores elsewhere, on a common scale."""
    out = np.array(local_norm, dtype=float, copy=True)
    out[routed_mask] = remote_norm[routed_mask]
    return out


def out_of_fold_scores(
    inputs: FusionInputs, *, reference_threshold: float | None
) -> dict[str, Any]:
    """Normalise every antigen using an ECDF fitted on the other antigens.

    Returns pooled out-of-fold score vectors. Each antigen's scores are mapped
    through transforms that never saw that antigen, so the pooled vectors can
    be used for a split-wide budget without leaking a held-out antigen into its
    own normalisation.
    """
    groups = np.asarray(inputs.groups)
    unique = np.unique(groups)
    n = len(inputs.labels)
    local_norm = np.empty(n, dtype=float)
    remote_norm = np.empty(n, dtype=float)
    gate = np.empty(n, dtype=float)
    rates: list[float] = []

    for held in unique:
        test_idx = np.where(groups == held)[0]
        train_idx = np.where(groups != held)[0]

        local_ecdf = fit_ecdf(inputs.candidate_scores[train_idx])
        remote_ecdf = fit_ecdf(inputs.reference_scores[train_idx])

        if reference_threshold is None:
            rate = 0.5
        else:
            rate = float(
                np.mean(inputs.reference_scores[train_idx] >= reference_threshold)
            )
        rates.append(rate)

        local_train_norm = apply_ecdf(local_ecdf, inputs.candidate_scores[train_idx])
        gate[test_idx] = threshold_for_positive_rate(local_train_norm, rate)

        local_norm[test_idx] = apply_ecdf(
            local_ecdf, inputs.candidate_scores[test_idx]
        )
        remote_norm[test_idx] = apply_ecdf(
            remote_ecdf, inputs.reference_scores[test_idx]
        )

    return {
        "local": local_norm,
        "remote": remote_norm,
        "gate": gate,
        "rate": float(np.mean(rates)),
    }


def _route_pooled(
    oof: dict[str, Any],
    groups: np.ndarray,
    *,
    policy: str,
    window: int,
    budget: float,
) -> np.ndarray:
    """Choose which residues of the whole split are disclosed, within ``budget``.

    Both policies spend one split-wide residue budget, which is what makes them
    comparable on a single disclosure axis. The difference is the unit they can
    spend it on: region routing admits windows, antigen routing admits entire
    antigens and cannot do anything finer.
    """
    n = len(groups)
    mask = np.zeros(n, dtype=bool)
    allowance = int(np.floor(float(budget) * n))
    if allowance <= 0:
        return mask

    margins = np.abs(oof["local"] - oof["gate"])
    units: list[tuple[float, np.ndarray]] = []

    if policy == "region":
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            for start, stop in _window_bounds(len(idx), window):
                sel = idx[start:stop]
                units.append((float(np.mean(margins[sel])), sel))
    elif policy == "antigen":
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            units.append((float(np.mean(margins[idx])), idx))
    else:  # pragma: no cover - guarded by the CLI
        raise ValueError(f"unknown policy {policy!r}")

    units.sort(key=lambda u: u[0])
    disclosed = 0
    for _, sel in units:
        if disclosed + len(sel) > allowance:
            continue
        mask[sel] = True
        disclosed += len(sel)
    return mask


def evaluate_pooled(
    inputs: FusionInputs,
    oof: dict[str, Any],
    *,
    policy: str,
    window: int,
    budget: float,
) -> dict[str, Any]:
    """Score one (policy, window, budget) over the whole out-of-fold split."""
    groups = np.asarray(inputs.groups)
    mask = _route_pooled(oof, groups, policy=policy, window=window, budget=budget)
    spliced = splice(oof["local"], oof["remote"], mask)
    decision = threshold_for_positive_rate(spliced, oof["rate"])
    scores = score_metrics(inputs.labels, spliced, decision)
    return {
        "policy": policy,
        "window": int(window),
        "budget": float(budget),
        "disclosed_fraction": float(mask.mean()),
        "antigens_touched": int(len(np.unique(groups[mask]))) if mask.any() else 0,
        **{k: float(scores[k]) for k in METRICS},
    }


def _evaluate_split(
    inputs: FusionInputs,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    window: int,
    budget: float,
    metric: str,
    reference_threshold: float | None,
    policy: str,
) -> dict[str, Any]:
    """Fit the transforms on the training folds, apply them to the held-out set.

    Nothing derived from ``test_idx`` enters the ECDF, the operating rate or the
    threshold.
    """
    local_train = inputs.candidate_scores[train_idx]
    remote_train = inputs.reference_scores[train_idx]

    local_ecdf = fit_ecdf(local_train)
    remote_ecdf = fit_ecdf(remote_train)

    # Operating rate: the reference predictor's own positive-call rate on the
    # training residues, as everywhere else in this project.
    if reference_threshold is None:
        rate = 0.5
    else:
        rate = float(np.mean(remote_train >= reference_threshold))

    local_train_norm = apply_ecdf(local_ecdf, local_train)
    gate_threshold = threshold_for_positive_rate(local_train_norm, rate)

    test_groups = inputs.groups[test_idx]
    local_test_norm = apply_ecdf(local_ecdf, inputs.candidate_scores[test_idx])
    remote_test_norm = apply_ecdf(remote_ecdf, inputs.reference_scores[test_idx])

    spliced = np.empty_like(local_test_norm)
    disclosed_residues = 0
    routed_windows = 0
    total_windows = 0
    for g in np.unique(test_groups):
        sel = test_groups == g
        loc = local_test_norm[sel]
        rem = remote_test_norm[sel]
        if policy == "region":
            plan = plan_region_routing(
                loc, gate_threshold, window=window, budget=budget
            )
        elif policy == "antigen":
            plan = plan_antigen_routing(loc, gate_threshold, budget=budget)
        else:  # pragma: no cover - guarded by the CLI
            raise ValueError(f"unknown policy {policy!r}")
        spliced[sel] = splice(loc, rem, plan.routed_mask)
        disclosed_residues += int(plan.routed_mask.sum())
        routed_windows += plan.n_windows_routed
        total_windows += plan.n_windows_total

    decision = threshold_for_positive_rate(local_train_norm, rate)
    scores = score_metrics(inputs.labels[test_idx], spliced, decision)
    n_test = len(test_idx)
    return {
        "metric_values": scores,
        "selected_metric": scores[metric],
        "disclosed_fraction": disclosed_residues / n_test if n_test else 0.0,
        "routed_windows": routed_windows,
        "total_windows": total_windows,
    }


def nested_region_routing(
    inputs: FusionInputs,
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    metric: str = "mcc",
    reference_threshold: float | None = None,
    policy: str = "region",
) -> dict[str, Any]:
    """Leave-one-antigen-out evaluation with the parameters chosen per fold.

    For each held-out antigen the window length and the budget are selected on
    the training folds alone and applied unchanged to the held-out antigen, so
    the reported figure is not the maximum of a scan over the evaluation data.
    """
    groups = np.asarray(inputs.groups)
    unique = np.unique(groups)
    per_fold: list[dict[str, Any]] = []

    for held in unique:
        test_idx = np.where(groups == held)[0]
        train_idx = np.where(groups != held)[0]

        best = None
        for w in windows:
            for b in budgets:
                inner = _evaluate_split(
                    inputs,
                    train_idx=train_idx,
                    test_idx=train_idx,  # selection sees training folds only
                    window=w,
                    budget=b,
                    metric=metric,
                    reference_threshold=reference_threshold,
                    policy=policy,
                )
                key = inner["selected_metric"]
                if best is None or key > best[0]:
                    best = (key, w, b)
        assert best is not None
        _, w_star, b_star = best

        outer = _evaluate_split(
            inputs,
            train_idx=train_idx,
            test_idx=test_idx,
            window=w_star,
            budget=b_star,
            metric=metric,
            reference_threshold=reference_threshold,
            policy=policy,
        )
        per_fold.append(
            {
                "held_out": str(held),
                "window": int(w_star),
                "budget": float(b_star),
                "disclosed_fraction": outer["disclosed_fraction"],
                **{k: outer["metric_values"][k] for k in METRICS},
            }
        )

    return {
        "policy": policy,
        "select_metric": metric,
        "n_folds": len(per_fold),
        "per_fold": per_fold,
        "mean": {
            k: float(np.nanmean([f[k] for f in per_fold])) for k in METRICS
        },
        "mean_disclosed_fraction": float(
            np.mean([f["disclosed_fraction"] for f in per_fold])
        ),
        "selected_windows": [f["window"] for f in per_fold],
        "selected_budgets": [f["budget"] for f in per_fold],
    }


def disclosure_frontier(
    inputs: FusionInputs,
    oof: dict[str, Any],
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    metric: str = "mcc",
) -> list[dict[str, Any]]:
    """Quality against disclosure for both policies on one budget axis.

    Each row answers: if a laboratory will disclose this fraction of its
    residues, what does the policy achieve? Whole-antigen routing spends the
    same budget but can only admit entire antigens, so it moves along the axis
    in coarse steps; region routing can place the budget where the local
    predictor is least confident. Both are scored on out-of-fold normalised
    scores over the whole split.

    The window length reported for region routing at each budget is the one
    that maximises ``metric``. That makes this frontier an upper envelope and
    not an out-of-sample claim, which is why the headline figure comes from
    ``nested_pooled_routing`` instead, where the window is chosen without
    seeing the fold it is scored on.
    """
    rows: list[dict[str, Any]] = []
    for b in budgets:
        best = None
        for w in windows:
            r = evaluate_pooled(inputs, oof, policy="region", window=w, budget=b)
            if best is None or r[metric] > best[metric]:
                best = r
        assert best is not None
        rows.append({**best, "envelope_over_windows": True})
        rows.append(
            evaluate_pooled(inputs, oof, policy="antigen", window=0, budget=b)
        )
    return rows


def nested_pooled_routing(
    inputs: FusionInputs,
    oof: dict[str, Any],
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    metric: str = "mcc",
    policy: str = "region",
) -> dict[str, Any]:
    """Choose the window length leave-one-antigen-out, at each fixed budget.

    For each held-out antigen the window is the one that scores best over the
    remaining antigens; the held-out antigen is then scored with it. The budget
    is a deployment constraint rather than a fitted parameter, so it is held
    fixed and reported across its range.
    """
    groups = np.asarray(inputs.groups)
    unique = np.unique(groups)
    rows: list[dict[str, Any]] = []

    margins = np.abs(oof["local"] - oof["gate"])
    n = len(inputs.labels)

    for b in budgets:
        # Per antigen, pick the window length on the OTHER antigens only.
        chosen: dict[Any, int] = {}
        for held in unique:
            train_sel = groups != held
            sub_groups = groups[train_sel]
            sub_oof = {
                "local": oof["local"][train_sel],
                "remote": oof["remote"][train_sel],
                "gate": oof["gate"][train_sel],
                "rate": oof["rate"],
            }
            best_w, best_v = windows[0], -np.inf
            for w in windows:
                m = _route_pooled(
                    sub_oof, sub_groups, policy=policy, window=w, budget=b
                )
                s = splice(sub_oof["local"], sub_oof["remote"], m)
                t = threshold_for_positive_rate(s, oof["rate"])
                v = score_metrics(inputs.labels[train_sel], s, t)[metric]
                if v > best_v:
                    best_w, best_v = w, v
            chosen[held] = best_w

        # One split-wide budget. Each antigen contributes units cut at the
        # window length its own fold never saw, so the unit boundaries carry no
        # information from the antigen they belong to.
        units: list[tuple[float, np.ndarray]] = []
        for g in unique:
            idx = np.where(groups == g)[0]
            if policy == "antigen":
                units.append((float(np.mean(margins[idx])), idx))
            else:
                for start, stop in _window_bounds(len(idx), chosen[g]):
                    sel = idx[start:stop]
                    units.append((float(np.mean(margins[sel])), sel))

        units.sort(key=lambda u: u[0])
        allowance = int(np.floor(float(b) * n))
        mask_all = np.zeros(n, dtype=bool)
        disclosed = 0
        for _, sel in units:
            if disclosed + len(sel) > allowance:
                continue
            mask_all[sel] = True
            disclosed += len(sel)

        spliced = splice(oof["local"], oof["remote"], mask_all)
        decision = threshold_for_positive_rate(spliced, oof["rate"])
        scores = score_metrics(inputs.labels, spliced, decision)
        rows.append(
            {
                "policy": policy,
                "budget": float(b),
                "disclosed_fraction": float(mask_all.mean()),
                "antigens_touched": (
                    int(len(np.unique(groups[mask_all]))) if mask_all.any() else 0
                ),
                "windows_selected": [int(chosen[g]) for g in unique],
                **{k: float(scores[k]) for k in METRICS},
            }
        )
    return {"select_metric": metric, "rows": rows}


def paired_bootstrap(
    inputs: FusionInputs,
    oof: dict[str, Any],
    *,
    region_rows: Sequence[dict[str, Any]],
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    metric: str = "mcc",
    resamples: int = 2000,
    seed: int = 20260112,
) -> list[dict[str, Any]]:
    """Interval on the region-minus-antigen difference, resampling antigens.

    The two policies are scored on the same resampled antigens in each draw, so
    the difference is paired and the interval is about the difference rather
    than about either policy separately. Antigens are the resampling unit
    because the claim is a claim about antigens, which is the same reason the
    rest of this project resamples them.

    Window lengths are taken from the leave-one-antigen-out selection and held
    fixed across draws: re-selecting them inside the bootstrap would make each
    draw a different estimator.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(inputs.groups)
    unique = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in unique}
    margins = np.abs(oof["local"] - oof["gate"])
    windows_by_budget = {
        float(r["budget"]): dict(zip(unique, r["windows_selected"]))
        for r in region_rows
    }

    def _score(draw: Sequence[Any], budget: float, policy: str) -> float:
        sel = np.concatenate([idx_by_group[g] for g in draw])
        n = len(sel)
        allowance = int(np.floor(budget * n))
        units: list[tuple[float, np.ndarray]] = []
        offset = 0
        for g in draw:
            idx = idx_by_group[g]
            local_pos = np.arange(offset, offset + len(idx))
            offset += len(idx)
            if policy == "antigen":
                units.append((float(np.mean(margins[idx])), local_pos))
            else:
                w = windows_by_budget[budget][g]
                for start, stop in _window_bounds(len(idx), w):
                    units.append(
                        (float(np.mean(margins[idx[start:stop]])),
                         local_pos[start:stop])
                    )
        units.sort(key=lambda u: u[0])
        mask = np.zeros(n, dtype=bool)
        disclosed = 0
        for _, pos in units:
            if disclosed + len(pos) > allowance:
                continue
            mask[pos] = True
            disclosed += len(pos)
        spliced = splice(oof["local"][sel], oof["remote"][sel], mask)
        t = threshold_for_positive_rate(spliced, oof["rate"])
        return float(score_metrics(inputs.labels[sel], spliced, t)[metric])

    out: list[dict[str, Any]] = []
    for b in budgets:
        diffs = np.empty(resamples, dtype=float)
        for i in range(resamples):
            draw = rng.choice(unique, size=len(unique), replace=True)
            diffs[i] = _score(draw, float(b), "region") - _score(
                draw, float(b), "antigen"
            )
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out.append(
            {
                "budget": float(b),
                "mean_difference": float(np.mean(diffs)),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "share_positive": float(np.mean(diffs > 0)),
                "excludes_zero": bool(lo > 0 or hi < 0),
                "resamples": int(resamples),
            }
        )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--carbon-predictions", required=True)
    p.add_argument("--bepipred-raw-scores", required=True)
    p.add_argument("--reference-threshold", type=float, default=None)
    p.add_argument("--select-metric", default="mcc", choices=METRICS)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260112)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)

    inputs = build_bp3_inputs(a.carbon_predictions, a.bepipred_raw_scores)
    oof = out_of_fold_scores(inputs, reference_threshold=a.reference_threshold)

    frontier = disclosure_frontier(inputs, oof, metric=a.select_metric)
    region = nested_pooled_routing(
        inputs, oof, metric=a.select_metric, policy="region"
    )
    antigen = nested_pooled_routing(
        inputs, oof, metric=a.select_metric, policy="antigen"
    )
    paired = paired_bootstrap(
        inputs,
        oof,
        region_rows=region["rows"],
        metric=a.select_metric,
        resamples=a.bootstrap,
        seed=a.seed,
    )

    report = {
        "project": "EdgeConBCE",
        "analysis": "region_routing",
        "question": (
            "Does routing contiguous regions rather than whole antigens buy "
            "remote-tier quality at a smaller disclosure cost?"
        ),
        "select_metric": a.select_metric,
        "antigens": len(inputs.sequence_ids),
        "residues": int(len(inputs.labels)),
        "window_grid": list(DEFAULT_WINDOWS),
        "budget_grid": list(DEFAULT_BUDGETS),
        "nested_region": region,
        "nested_antigen": antigen,
        "frontier": frontier,
        "paired_bootstrap": paired,
        "note": (
            "Every antigen is normalised by an ECDF fitted on the other "
            "antigens, so no antigen enters its own transform. The budget is a "
            "deployment constraint and is held fixed, not fitted. In "
            "nested_region the window length is chosen on the remaining "
            "antigens and applied to the held-out one; the frontier instead "
            "reports the best window per budget and is therefore an upper "
            "envelope, not an out-of-sample claim."
        ),
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
