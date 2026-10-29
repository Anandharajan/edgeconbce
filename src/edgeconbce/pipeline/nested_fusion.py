"""Leakage-free evaluation of score fusion for conformational B-cell epitope prediction.

Motivation
----------
``fusion_baseline`` and ``iedb_robustness`` both report a fusion result whose
mixing weight was chosen by scanning a grid on the same split the result is
reported on, and whose decision threshold was set to reproduce a positive-call
count measured on that split. Both quantities are therefore functions of the
test labels, so the reported margin over the stronger single predictor is
optimistically biased.

This module re-estimates the same quantity under leave-one-sequence-out nested
selection: for each held-out antigen the mixing weight and the operating
threshold are chosen using only the remaining antigens, and are then applied
once to the held-out antigen. The fold-level differences against the reference
predictor are aggregated with a paired bootstrap, a Wilcoxon signed-rank test
and an exact sign test.

The module is deliberately independent of the two legacy pipelines so that the
corrected estimate can be compared against the published one rather than
silently replacing it.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

METRICS: tuple[str, ...] = ("roc_auc", "average_precision", "mcc", "f1")
DEFAULT_WEIGHT_GRID = tuple(round(float(w), 3) for w in np.linspace(0.0, 1.0, 21))
DEFAULT_SEED = 20261025
DEFAULT_BOOTSTRAP = 10000


@dataclass(frozen=True)
class FusionInputs:
    """Residue-level scores for two predictors over a sequence-grouped split."""

    sequence_ids: tuple[str, ...]
    labels: np.ndarray
    reference_scores: np.ndarray
    candidate_scores: np.ndarray
    groups: np.ndarray
    reference_name: str = "reference"
    candidate_name: str = "candidate"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.labels)
        if not (len(self.reference_scores) == len(self.candidate_scores) == len(self.groups) == n):
            raise ValueError(
                "labels, reference_scores, candidate_scores and groups must be the same length; "
                f"got {n}, {len(self.reference_scores)}, {len(self.candidate_scores)}, {len(self.groups)}"
            )
        if n == 0:
            raise ValueError("no residues supplied")
        if len(self.sequence_ids) < 2:
            raise ValueError("leave-one-sequence-out needs at least two sequences")


def rank_normalise(scores: np.ndarray) -> np.ndarray:
    """Map scores onto [0, 1] by rank.

    The two predictors are calibrated differently, so a convex combination of
    their raw outputs would be dominated by whichever has the wider spread.
    Ranking removes the scale difference without changing either predictor's
    own ranking, hence without changing its ROC-AUC or average precision.
    """
    if len(scores) < 2:
        return np.zeros_like(scores, dtype=np.float64)
    return np.argsort(np.argsort(scores)).astype(np.float64) / (len(scores) - 1)


def fit_ecdf(train_scores: np.ndarray) -> np.ndarray:
    """Sorted training scores, used as the empirical distribution for mapping."""
    return np.sort(np.asarray(train_scores, dtype=np.float64))


def apply_ecdf(sorted_train: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Map scores onto [0, 1] through a distribution estimated on training data.

    Held-out residues must not influence their own normalisation, so the
    mapping is fitted on the training folds and applied to the held-out fold,
    exactly as any other fitted transform would be.
    """
    if sorted_train.size == 0:
        return np.zeros_like(scores, dtype=np.float64)
    positions = np.searchsorted(sorted_train, scores, side="right")
    return positions.astype(np.float64) / float(sorted_train.size)


def threshold_for_positive_rate(scores: np.ndarray, rate: float) -> float:
    """Smallest threshold that calls approximately ``rate`` of residues positive."""
    k = int(round(float(rate) * len(scores)))
    if k <= 0:
        return float(np.inf)
    if k >= len(scores):
        return float(np.min(scores))
    return float(np.sort(scores)[-k])


def score_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(np.int8)
    both_classes_present = len(np.unique(labels)) > 1
    return {
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes_present else float("nan"),
        "average_precision": float(average_precision_score(labels, scores)) if both_classes_present else float("nan"),
        "mcc": float(matthews_corrcoef(labels, predictions)) if len(np.unique(predictions)) > 1 else 0.0,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def _paired_statistics(deltas: np.ndarray, *, bootstrap: int, seed: int) -> dict[str, Any]:
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        return {"n_folds": 0}
    rng = np.random.default_rng(seed)
    means = rng.choice(finite, size=(int(bootstrap), finite.size), replace=True).mean(axis=1)
    wins = int((finite > 0).sum())
    losses = int((finite < 0).sum())
    try:
        p_wilcoxon = float(wilcoxon(finite).pvalue)
    except ValueError:  # all differences zero
        p_wilcoxon = 1.0
    return {
        "n_folds": int(finite.size),
        "mean_delta": float(finite.mean()),
        "median_delta": float(np.median(finite)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "wins": wins,
        "losses": losses,
        "ties": int(finite.size - wins - losses),
        "p_wilcoxon": p_wilcoxon,
        "p_sign": float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0,
        "bootstrap_repeats": int(bootstrap),
    }


def evaluate_nested_fusion(
    inputs: FusionInputs,
    *,
    select_metric: str = "mcc",
    weight_grid: Sequence[float] = DEFAULT_WEIGHT_GRID,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    reference_threshold: float | None = None,
) -> dict[str, Any]:
    """Leave-one-sequence-out nested selection of the fusion weight and threshold."""
    if select_metric not in METRICS:
        raise ValueError(f"select_metric must be one of {METRICS}")

    labels = inputs.labels
    reference_raw = inputs.reference_scores
    candidate_raw = inputs.candidate_scores
    groups = inputs.groups

    out_of_fold_scores = np.zeros(len(labels), dtype=np.float64)
    out_of_fold_predictions = np.zeros(len(labels), dtype=np.int8)
    out_of_fold_reference = np.zeros(len(labels), dtype=np.float64)
    per_fold: list[dict[str, Any]] = []
    selected_weights: list[float] = []

    for held_out, sequence_id in enumerate(inputs.sequence_ids):
        train_mask = groups != held_out
        test_mask = groups == held_out
        if not train_mask.any() or not test_mask.any():
            continue

        # Every quantity below is estimated on the training folds alone.
        ref_ecdf = fit_ecdf(reference_raw[train_mask])
        cand_ecdf = fit_ecdf(candidate_raw[train_mask])
        ref_train = apply_ecdf(ref_ecdf, reference_raw[train_mask])
        cand_train = apply_ecdf(cand_ecdf, candidate_raw[train_mask])
        ref_test = apply_ecdf(ref_ecdf, reference_raw[test_mask])
        cand_test = apply_ecdf(cand_ecdf, candidate_raw[test_mask])

        if reference_threshold is not None:
            train_rate = float((reference_raw[train_mask] >= reference_threshold).mean())
        else:
            train_rate = float(labels[train_mask].mean())

        best_weight, best_value = float(weight_grid[0]), -np.inf
        for weight in weight_grid:
            blended = weight * ref_train + (1.0 - weight) * cand_train
            value = score_metrics(
                labels[train_mask], blended, threshold_for_positive_rate(blended, train_rate)
            )[select_metric]
            if np.isfinite(value) and value > best_value:
                best_weight, best_value = float(weight), float(value)

        selected_weights.append(best_weight)
        blended_train = best_weight * ref_train + (1.0 - best_weight) * cand_train
        blended_test = best_weight * ref_test + (1.0 - best_weight) * cand_test
        # The threshold is a value fitted on the training folds and then applied,
        # so the held-out antigen's own score distribution never sets its own
        # operating point. Taking the quantile of the held-out scores instead
        # would be transductive and could not be deployed.
        fold_threshold = threshold_for_positive_rate(blended_train, train_rate)
        out_of_fold_scores[test_mask] = blended_test
        out_of_fold_reference[test_mask] = ref_test
        out_of_fold_predictions[test_mask] = (blended_test >= fold_threshold).astype(np.int8)

        if len(np.unique(labels[test_mask])) > 1:
            fusion_fold = score_metrics(labels[test_mask], blended_test, fold_threshold)
            reference_fold = score_metrics(
                labels[test_mask], ref_test, threshold_for_positive_rate(ref_train, train_rate)
            )
            per_fold.append(
                {
                    "sequence_id": sequence_id,
                    "n_residues": int(test_mask.sum()),
                    "n_positive_residues": int(labels[test_mask].sum()),
                    "selected_weight": best_weight,
                    "train_selection_value": best_value,
                    "train_positive_rate": train_rate,
                    **{f"fusion_{k}": v for k, v in fusion_fold.items()},
                    **{f"reference_{k}": v for k, v in reference_fold.items()},
                    **{f"delta_{k}": fusion_fold[k] - reference_fold[k] for k in METRICS},
                }
            )

    pooled_rate = float(out_of_fold_predictions.mean())
    pooled_fusion = score_metrics(labels, out_of_fold_scores,
                                  threshold_for_positive_rate(out_of_fold_scores, pooled_rate))
    pooled_reference = score_metrics(labels, out_of_fold_reference,
                                     threshold_for_positive_rate(out_of_fold_reference, pooled_rate))

    statistics = {
        metric: _paired_statistics(
            np.array([row[f"delta_{metric}"] for row in per_fold], dtype=np.float64),
            bootstrap=bootstrap,
            seed=seed + index,
        )
        for index, metric in enumerate(METRICS)
    }

    return {
        "project": "edgecon-bce",
        "analysis": "nested_fusion_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": (
            "Leave-one-sequence-out nested selection. For each held-out antigen the fusion "
            "weight and the decision threshold are chosen using only the remaining antigens "
            "and then applied once to the held-out antigen."
        ),
        "reference_model": inputs.reference_name,
        "candidate_model": inputs.candidate_name,
        "select_metric": select_metric,
        "weight_grid": [float(w) for w in weight_grid],
        "reference_threshold": (float(reference_threshold) if reference_threshold is not None else None),
        "operating_rate_source": ("reference predictor threshold, recomputed per training split"
                                  if reference_threshold is not None else
                                  "training-fold label positive rate"),
        "dataset": {
            "n_sequences": len(inputs.sequence_ids),
            "n_residues": int(len(labels)),
            "n_positive_residues": int(labels.sum()),
            **inputs.metadata,
        },
        "selected_weights": {
            "per_fold": selected_weights,
            "median": float(np.median(selected_weights)) if selected_weights else float("nan"),
            "min": float(min(selected_weights)) if selected_weights else float("nan"),
            "max": float(max(selected_weights)) if selected_weights else float("nan"),
        },
        "pooled_out_of_fold": {
            "fusion": pooled_fusion,
            "reference": pooled_reference,
            "delta": {k: pooled_fusion[k] - pooled_reference[k] for k in METRICS},
            "positive_call_rate": pooled_rate,
        },
        "per_fold": per_fold,
        "paired_statistics": statistics,
        "seed": seed,
    }


# --------------------------------------------------------------------------- loaders


def load_carbon_predictions(path: str | Path) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sequence_ids: list[str] = []
    labels: list[int] = []
    scores: list[float] = []
    groups: list[int] = []
    for index, row in enumerate(payload["rows"]):
        sequence_ids.append(str(row["sequence_id"]))
        labels.extend(int(v) for v in row["labels"])
        scores.extend(float(v) for v in row["probabilities"])
        groups.extend([index] * int(row["length"]))
    return (
        tuple(sequence_ids),
        np.asarray(labels, dtype=np.int8),
        np.asarray(scores, dtype=np.float64),
        np.asarray(groups, dtype=np.int32),
    )


def load_bepipred_scores(path: str | Path) -> np.ndarray:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return np.asarray(
            [float(row["BepiPred-3.0 score"]) for row in csv.DictReader(handle)], dtype=np.float64
        )


def load_policy_scores(path: str | Path, policy: str) -> dict[str, tuple[list[int], list[float]]]:
    """Per-sequence labels and probabilities for one routing policy of a real-benchmark run."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    block = payload["external_test"][policy]
    key = {
        "local_only": "local_probabilities",
        "edge_only": "edge_probabilities",
    }.get(policy, "adaptive_probabilities")
    return {
        str(row["sequence_id"]): ([int(v) for v in row["labels"]], [float(v) for v in row[key]])
        for row in block["per_sequence"]
    }


def build_bp3_inputs(carbon_predictions: str | Path, bepipred_raw_scores: str | Path) -> FusionInputs:
    ids, labels, carbon, groups = load_carbon_predictions(carbon_predictions)
    bepipred = load_bepipred_scores(bepipred_raw_scores)
    if len(bepipred) != len(labels):
        raise ValueError(
            f"BepiPred score count ({len(bepipred)}) does not match residue count ({len(labels)}); "
            "the two files describe different splits"
        )
    return FusionInputs(
        sequence_ids=ids,
        labels=labels,
        reference_scores=bepipred,
        candidate_scores=carbon,
        groups=groups,
        reference_name="BepiPred-3.0 (official local rerun)",
        candidate_name="CARBON-BCE-R",
        metadata={"split": "BP3C50ID external test"},
    )


def build_iedb_inputs(
    carbon_predictions: str | Path, real_results: str | Path, policy: str = "local_only"
) -> FusionInputs:
    ids, labels, carbon, groups = load_carbon_predictions(carbon_predictions)
    per_sequence = load_policy_scores(real_results, policy)
    missing = [i for i in ids if i not in per_sequence]
    if missing:
        raise ValueError(f"{len(missing)} sequences absent from the real-results file, e.g. {missing[:3]}")
    reference: list[float] = []
    for index, sequence_id in enumerate(ids):
        seq_labels, seq_scores = per_sequence[sequence_id]
        expected = int((groups == index).sum())
        if len(seq_scores) != expected:
            raise ValueError(f"{sequence_id}: length {len(seq_scores)} does not match {expected}")
        if seq_labels != [int(v) for v in labels[groups == index]]:
            raise ValueError(f"{sequence_id}: labels disagree between the two artifacts")
        reference.extend(seq_scores)
    return FusionInputs(
        sequence_ids=ids,
        labels=labels,
        reference_scores=np.asarray(reference, dtype=np.float64),
        candidate_scores=carbon,
        groups=groups,
        reference_name=f"EdgeConBCE {policy}",
        candidate_name="CARBON-BCE-R",
        metadata={"split": "IEDB reduced bounded subset", "reference_policy": policy},
    )


def write_report(report: dict[str, Any], output_dir: str | Path, stem: str) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{stem}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    stats = report["paired_statistics"]
    lines = [
        f"# {report['analysis']}: {report['candidate_model']} fused onto {report['reference_model']}",
        "",
        f"Generated at: {report['created_at_utc']}",
        "",
        report["protocol"],
        "",
        f"Split: `{report['dataset'].get('split', 'n/a')}` — "
        f"{report['dataset']['n_sequences']} sequences, {report['dataset']['n_residues']} residues, "
        f"{report['dataset']['n_positive_residues']} positive residues.",
        "",
        (f"Reference operating threshold: `{report['reference_threshold']}` "
         f"({report['operating_rate_source']}).") if report.get("reference_threshold") is not None
        else f"Operating rate source: {report['operating_rate_source']}.",
        "",
        f"Selection metric: `{report['select_metric']}`. Weights chosen per fold: "
        f"median {report['selected_weights']['median']:.2f}, "
        f"range [{report['selected_weights']['min']:.2f}, {report['selected_weights']['max']:.2f}].",
        "",
        "## Pooled out-of-fold performance",
        "",
        "| metric | fusion | reference | delta |",
        "|---|---:|---:|---:|",
    ]
    pooled = report["pooled_out_of_fold"]
    for metric in METRICS:
        lines.append(
            f"| {metric} | {pooled['fusion'][metric]:.6f} | {pooled['reference'][metric]:.6f} "
            f"| {pooled['delta'][metric]:+.6f} |"
        )
    lines += [
        "",
        "## Per-sequence paired differences (fusion minus reference)",
        "",
        "| metric | mean delta | 95% CI | wins/losses | Wilcoxon p | sign-test p |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        s = stats[metric]
        if not s.get("n_folds"):
            continue
        lines.append(
            f"| {metric} | {s['mean_delta']:+.6f} | [{s['ci_low']:+.6f}, {s['ci_high']:+.6f}] "
            f"| {s['wins']}/{s['losses']} | {s['p_wilcoxon']:.4f} | {s['p_sign']:.4f} |"
        )
    md_path = out / f"{stem}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = out / f"{stem}_per_fold.csv"
    if report["per_fold"]:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["per_fold"][0].keys()))
            writer.writeheader()
            writer.writerows(report["per_fold"])
    return {"json": json_path, "markdown": md_path, "per_fold_csv": csv_path}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("bp3", "iedb"), required=True)
    parser.add_argument("--carbon-predictions", required=True)
    parser.add_argument("--bepipred-raw-scores")
    parser.add_argument("--real-results")
    parser.add_argument("--reference-policy", default="local_only")
    parser.add_argument("--reference-threshold", type=float, default=None)
    parser.add_argument("--select-metric", default="mcc", choices=METRICS)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stem", default=None)
    args = parser.parse_args(argv)

    if args.dataset == "bp3":
        if not args.bepipred_raw_scores:
            parser.error("--bepipred-raw-scores is required for --dataset bp3")
        inputs = build_bp3_inputs(args.carbon_predictions, args.bepipred_raw_scores)
    else:
        if not args.real_results:
            parser.error("--real-results is required for --dataset iedb")
        inputs = build_iedb_inputs(args.carbon_predictions, args.real_results, args.reference_policy)

    report = evaluate_nested_fusion(
        inputs,
        select_metric=args.select_metric,
        bootstrap=args.bootstrap,
        seed=args.seed,
        reference_threshold=args.reference_threshold,
    )
    stem = args.stem or f"nested_fusion_{args.dataset}_{args.select_metric}"
    paths = write_report(report, args.output_dir, stem)
    print(json.dumps({"written": {k: str(v) for k, v in paths.items()},
                      "pooled_delta": report["pooled_out_of_fold"]["delta"],
                      "selected_weight_median": report["selected_weights"]["median"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
