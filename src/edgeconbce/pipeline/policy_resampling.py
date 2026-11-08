"""Antigen-level resampling for routing-policy comparisons.

The repeated-evaluation artifact reports confidence intervals over seeds on a
fixed test split. Those intervals describe how much a quantity moves when the
harness is re-run; they do not describe how much it would move on a different
set of antigens, because the same antigens appear in every seed. For a claim
about antigens the resampling unit must be the antigen.

This module recomputes every policy metric and every paired policy comparison
with a two-way bootstrap: antigens are resampled with replacement, and seeds
are resampled with replacement, so the reported interval carries both the
antigen-sampling and the re-run components. Seed-only intervals are retained
alongside, labelled as run-to-run dispersion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

QUALITY_METRICS = ("mcc", "f1", "roc_auc", "average_precision")
COST_METRICS = ("latency_ms", "total_gco2e", "raw_sequence_shared")
PROBABILITY_KEY = {
    "local_only": "local_probabilities",
    "edge_only": "edge_probabilities",
}
DEFAULT_BOOTSTRAP = 4000
DEFAULT_SEED = 20261025


def load_runs(run_paths: Sequence[Path]) -> dict[str, Any]:
    """Per-seed, per-policy, per-antigen labels, scores, decisions and costs."""
    seeds: list[int] = []
    policies: list[str] = []
    data: dict[int, dict[str, dict[str, Any]]] = {}
    antigen_ids: tuple[str, ...] | None = None

    for path in sorted(run_paths):
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        seed = int(run["config"]["seed"])
        seeds.append(seed)
        data[seed] = {}
        for policy, block in run["external_test"].items():
            if policy not in policies:
                policies.append(policy)
            rows = block["per_sequence"]
            ids = tuple(str(r["sequence_id"]) for r in rows)
            if antigen_ids is None:
                antigen_ids = ids
            elif ids != antigen_ids:
                raise ValueError(f"{path}: antigen order differs between runs/policies")
            key = PROBABILITY_KEY.get(policy, "adaptive_probabilities")
            per_antigen = []
            for r in rows:
                scores = r.get(key)
                if scores is None:
                    scores = r.get("adaptive_probabilities")
                per_antigen.append({
                    "labels": np.asarray(r["labels"], dtype=np.int8),
                    "scores": np.asarray(scores, dtype=np.float64),
                    "latency_ms": float(r.get("active_latency_ms", r.get("local_latency_ms", 0.0))),
                    "total_gco2e": float(
                        r.get("adaptive_total_gco2e", r.get("active_total_gco2e", 0.0))
                        if policy not in ("local_only", "edge_only")
                        else r.get("local_total_gco2e" if policy == "local_only" else "edge_total_gco2e", 0.0)
                    ),
                    "raw_sequence_shared": float(bool(r.get("raw_sequence_shared", policy == "edge_only"))),
                })
            data[seed][policy] = {
                "per_antigen": per_antigen,
                "threshold": float(block.get("decision_threshold", 0.5)),
            }
    return {"seeds": sorted(seeds), "policies": policies, "antigen_ids": antigen_ids, "data": data}


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney U statistic, with average ranks inside tied score groups."""
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # average the ranks of tied scores, as the standard definition requires
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, scores.size]):
        if stop - start > 1:
            ranks[order[start:stop]] = (start + stop + 1) / 2.0
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Step-wise average precision, matching sklearn's definition."""
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order].astype(np.float64)
    sorted_scores = scores[order]
    tp = np.cumsum(y)
    k = np.arange(1, y.size + 1, dtype=np.float64)
    # at tied scores every tied residue shares the group's cumulative counts
    last = np.r_[np.flatnonzero(np.diff(sorted_scores)), y.size - 1]
    tp_g, k_g = tp[last], k[last]
    recall = tp_g / n_pos
    precision = tp_g / k_g
    return float((np.diff(np.r_[0.0, recall]) * precision).sum())


def pooled_quality(per_antigen: list[dict], threshold: float, index: np.ndarray) -> dict[str, float]:
    """Metrics pooled over the selected antigens (sampled with replacement)."""
    labels = np.concatenate([per_antigen[i]["labels"] for i in index])
    scores = np.concatenate([per_antigen[i]["scores"] for i in index])
    predicted = scores >= threshold
    positive = labels == 1
    tp = float(np.count_nonzero(predicted & positive))
    fp = float(np.count_nonzero(predicted & ~positive))
    fn = float(np.count_nonzero(~predicted & positive))
    tn = float(np.count_nonzero(~predicted & ~positive))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "mcc": float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        "roc_auc": _roc_auc(labels, scores),
        "average_precision": _average_precision(labels, scores),
    }


def pooled_cost(per_antigen: list[dict], index: np.ndarray) -> dict[str, float]:
    return {m: float(np.mean([per_antigen[i][m] for i in index])) for m in COST_METRICS}


def _interval(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "mean": float(finite.mean()),
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
        "width": float(np.quantile(finite, 0.975) - np.quantile(finite, 0.025)),
    }


def two_way_bootstrap(
    loaded: dict[str, Any], *, bootstrap: int = DEFAULT_BOOTSTRAP, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    """Resample antigens and seeds together; report per-policy and paired results."""
    rng = np.random.default_rng(seed)
    seeds, policies = loaded["seeds"], loaded["policies"]
    data = loaded["data"]
    n_antigens = len(loaded["antigen_ids"])

    draws: dict[str, dict[str, list[float]]] = {
        p: {m: [] for m in QUALITY_METRICS + COST_METRICS} for p in policies
    }
    pair_draws: dict[tuple[str, str], dict[str, list[float]]] = {}

    for _ in range(int(bootstrap)):
        antigen_index = rng.integers(0, n_antigens, size=n_antigens)
        seed_index = rng.integers(0, len(seeds), size=len(seeds))
        chosen_seeds = [seeds[i] for i in seed_index]

        per_policy: dict[str, dict[str, float]] = {}
        for policy in policies:
            values: dict[str, list[float]] = {m: [] for m in QUALITY_METRICS + COST_METRICS}
            for s in chosen_seeds:
                block = data[s][policy]
                q = pooled_quality(block["per_antigen"], block["threshold"], antigen_index)
                c = pooled_cost(block["per_antigen"], antigen_index)
                for m in QUALITY_METRICS:
                    values[m].append(q[m])
                for m in COST_METRICS:
                    values[m].append(c[m])
            per_policy[policy] = {m: float(np.nanmean(v)) for m, v in values.items()}
            for m, v in per_policy[policy].items():
                draws[policy][m].append(v)

        for a in policies:
            for b in policies:
                if a >= b:
                    continue
                key = (a, b)
                pair_draws.setdefault(key, {m: [] for m in QUALITY_METRICS + COST_METRICS})
                for m in QUALITY_METRICS + COST_METRICS:
                    pair_draws[key][m].append(per_policy[a][m] - per_policy[b][m])

    return {
        "per_policy": {
            p: {m: _interval(np.asarray(v, dtype=np.float64)) for m, v in metrics.items()}
            for p, metrics in draws.items()
        },
        "paired": {
            f"{a}_minus_{b}": {
                m: _interval(np.asarray(v, dtype=np.float64)) for m, v in metrics.items()
            }
            for (a, b), metrics in pair_draws.items()
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--glob", default="*seed_*.json")
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    run_paths = sorted(Path(a.runs_dir).glob(a.glob))
    if not run_paths:
        raise SystemExit(f"no run files matched {a.runs_dir}/{a.glob}")
    loaded = load_runs(run_paths)
    result = two_way_bootstrap(loaded, bootstrap=a.bootstrap, seed=a.seed)

    report = {
        "analysis": "policy_antigen_level_resampling",
        "resampling_unit": "antigen and seed (two-way bootstrap with replacement)",
        "rationale": (
            "The repeated-evaluation artifact reports intervals over seeds on a fixed test "
            "split, which measure run-to-run dispersion rather than sampling uncertainty over "
            "antigens. These intervals resample both."
        ),
        "n_antigens": len(loaded["antigen_ids"]),
        "n_seeds": len(loaded["seeds"]),
        "seeds": loaded["seeds"],
        "bootstrap_repeats": a.bootstrap,
        "seed": a.seed,
        **result,
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'policy':<32}{'MCC [95% CI]':<34}{'ROC-AUC [95% CI]'}")
    for pol, m in report["per_policy"].items():
        q, r = m["mcc"], m["roc_auc"]
        print(f"{pol:<32}{q['mean']:.3f} [{q['ci_low']:.3f}, {q['ci_high']:.3f}]".ljust(66)
              + f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
