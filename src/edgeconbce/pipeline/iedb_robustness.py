from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

from edgeconbce.artifacts import write_json_artifact
from edgeconbce.pipeline.report import write_csv, write_markdown
from edgeconbce.utils.env import system_metadata


def _load_carbon(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    labels: list[int] = []
    scores: list[float] = []
    rows = list(payload["rows"])
    for row in rows:
        labels.extend(int(value) for value in row["labels"])
        scores.extend(float(value) for value in row["probabilities"])
    return np.asarray(labels, dtype=np.int8), np.asarray(scores, dtype=np.float64), rows


def _load_policy(real_payload: dict[str, Any], policy: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rows = list(real_payload["external_test"][policy]["per_sequence"])
    labels: list[int] = []
    scores: list[float] = []
    predictions: list[int] = []
    score_key = "adaptive_probabilities"
    prediction_key = "adaptive_predictions"
    for row in rows:
        labels.extend(int(value) for value in row["labels"])
        scores.extend(float(value) for value in row[score_key])
        predictions.extend(int(value) for value in row[prediction_key])
    return np.asarray(labels, dtype=np.int8), np.asarray(scores, dtype=np.float64), np.asarray(predictions, dtype=np.int8), rows


def _threshold_for_positive_count(scores: np.ndarray, positive_count: int) -> float:
    if positive_count <= 0:
        return float(np.inf)
    if positive_count >= len(scores):
        return float(np.min(scores))
    return float(np.sort(scores)[-positive_count])


def _summary(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = (scores >= threshold).astype(np.int8)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0,
        "average_precision": float(average_precision_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0,
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "predicted_positive_residues": int(predictions.sum()),
    }


def _sequence_slices(rows: list[dict[str, Any]]) -> list[slice]:
    slices: list[slice] = []
    offset = 0
    for row in rows:
        length = int(row["length"])
        slices.append(slice(offset, offset + length))
        offset += length
    return slices


def _bootstrap_deltas(
    *,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    fusion_scores: np.ndarray,
    baseline_threshold: float,
    fusion_threshold: float,
    repeats: int,
) -> list[dict[str, Any]]:
    slices = _sequence_slices(rows)
    rng = np.random.default_rng(20260621)
    values: dict[str, list[float]] = {metric: [] for metric in ("roc_auc", "average_precision", "mcc", "f1")}
    for _ in range(repeats):
        sampled = rng.integers(0, len(slices), size=len(slices))
        indices = np.concatenate([np.arange(slices[int(index)].start, slices[int(index)].stop) for index in sampled])
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        baseline_summary = _summary(sampled_labels, baseline_scores[indices], baseline_threshold)
        fusion_summary = _summary(sampled_labels, fusion_scores[indices], fusion_threshold)
        for metric in values:
            values[metric].append(float(fusion_summary[metric] - baseline_summary[metric]))
    rows_out: list[dict[str, Any]] = []
    for metric, metric_values in values.items():
        array = np.asarray(metric_values, dtype=np.float64)
        rows_out.append(
            {
                "metric": metric,
                "mean_delta": round(float(np.mean(array)), 6),
                "ci_low": round(float(np.quantile(array, 0.025)), 6),
                "ci_high": round(float(np.quantile(array, 0.975)), 6),
                "positive_delta_rate": round(float(np.mean(array > 0.0)), 6),
                "bootstrap_repeats": int(len(array)),
            }
        )
    return rows_out


def generate_iedb_robustness(
    *,
    real_results_path: str | Path,
    carbon_predictions_path: str | Path,
    output_dir: str | Path,
    base_policy: str = "local_only",
    count_policy: str = "adaptive_split",
    base_weight: float = 0.80,
    bootstrap_repeats: int = 2000,
) -> dict[str, Any]:
    real_payload = json.loads(Path(real_results_path).read_text(encoding="utf-8"))
    labels, base_scores, _, base_rows = _load_policy(real_payload, base_policy)
    _, count_scores, count_predictions, _ = _load_policy(real_payload, count_policy)
    carbon_labels, carbon_scores, _ = _load_carbon(carbon_predictions_path)
    if not np.array_equal(labels, carbon_labels):
        raise ValueError("Real-result and CARBON labels do not align.")

    positive_count = int(np.sum(count_predictions))
    baseline_threshold = _threshold_for_positive_count(count_scores, positive_count)
    baseline_summary = _summary(labels, count_scores, baseline_threshold)

    fusion_scores = (float(base_weight) * base_scores) + ((1.0 - float(base_weight)) * carbon_scores)
    fusion_threshold = _threshold_for_positive_count(fusion_scores, positive_count)
    fusion_summary = _summary(labels, fusion_scores, fusion_threshold)

    sensitivity_rows: list[dict[str, Any]] = []
    for candidate_weight in np.linspace(0.0, 1.0, 11):
        candidate_scores = (float(candidate_weight) * base_scores) + ((1.0 - float(candidate_weight)) * carbon_scores)
        candidate_threshold = _threshold_for_positive_count(candidate_scores, positive_count)
        candidate_summary = _summary(labels, candidate_scores, candidate_threshold)
        sensitivity_rows.append(
            {
                "base_weight": round(float(candidate_weight), 2),
                "carbon_weight": round(float(1.0 - candidate_weight), 2),
                "threshold": round(float(candidate_threshold), 6),
                "roc_auc": round(float(candidate_summary["roc_auc"]), 6),
                "mcc": round(float(candidate_summary["mcc"]), 6),
                "f1": round(float(candidate_summary["f1"]), 6),
                "average_precision": round(float(candidate_summary["average_precision"]), 6),
                "predicted_positive_residues": int(candidate_summary["predicted_positive_residues"]),
            }
        )

    bootstrap_rows = _bootstrap_deltas(
        rows=base_rows,
        labels=labels,
        baseline_scores=count_scores,
        fusion_scores=fusion_scores,
        baseline_threshold=baseline_threshold,
        fusion_threshold=fusion_threshold,
        repeats=bootstrap_repeats,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sensitivity_csv = output_path / "iedb_fusion_weight_sensitivity.csv"
    bootstrap_csv = output_path / "iedb_fusion_vs_adaptive_bootstrap.csv"
    markdown_path = output_path / "iedb_robustness_report.md"
    json_path = output_path / "iedb_robustness_report.json"

    write_csv(
        sensitivity_csv,
        ["base_weight", "carbon_weight", "threshold", "roc_auc", "mcc", "f1", "average_precision", "predicted_positive_residues"],
        sensitivity_rows,
    )
    write_csv(
        bootstrap_csv,
        ["metric", "mean_delta", "ci_low", "ci_high", "positive_delta_rate", "bootstrap_repeats"],
        bootstrap_rows,
    )
    write_markdown(
        markdown_path,
        "\n".join(
            [
                "# IEDB Bounded Robustness Report",
                "",
                f"Dataset: `{real_payload['config']['test_fasta_path']}`",
                f"Base policy: `{base_policy}`",
                f"Positive-count rule: preserve `{count_policy}` count = `{positive_count}`",
                f"Selected fusion: `{base_weight:.2f}` base / `{1.0 - base_weight:.2f}` CARBON",
                "",
                "| model | ROC AUC | MCC | F1 | AP |",
                "|---|---:|---:|---:|---:|",
                f"| {count_policy} baseline | {baseline_summary['roc_auc']:.6f} | {baseline_summary['mcc']:.6f} | {baseline_summary['f1']:.6f} | {baseline_summary['average_precision']:.6f} |",
                f"| IEDB local/CARBON fusion | {fusion_summary['roc_auc']:.6f} | {fusion_summary['mcc']:.6f} | {fusion_summary['f1']:.6f} | {fusion_summary['average_precision']:.6f} |",
                "",
                "## Weight Sensitivity",
                "",
                "| base weight | CARBON weight | ROC AUC | MCC | F1 | AP |",
                "|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {row['base_weight']:.2f} | {row['carbon_weight']:.2f} | {row['roc_auc']} | {row['mcc']} | {row['f1']} | {row['average_precision']} |"
                    for row in sensitivity_rows
                ],
                "",
                "## Sequence Bootstrap Versus Adaptive Baseline",
                "",
                "| metric | mean delta | 95% CI | positive delta rate |",
                "|---|---:|---:|---:|",
                *[
                    f"| {row['metric']} | {row['mean_delta']} | [{row['ci_low']}, {row['ci_high']}] | {row['positive_delta_rate']} |"
                    for row in bootstrap_rows
                ],
            ]
        ),
    )

    payload = {
        "project": "edgecon-bce",
        "analysis": "iedb_bounded_robustness",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "inputs": {
            "real_results_path": str(real_results_path),
            "carbon_predictions_path": str(carbon_predictions_path),
        },
        "algorithm": {
            "base_policy": base_policy,
            "count_policy": count_policy,
            "base_weight": float(base_weight),
            "carbon_weight": float(1.0 - base_weight),
            "threshold_rule": "preserve_count_policy_positive_call_count",
            "positive_count": positive_count,
        },
        "baseline_summary": baseline_summary,
        "fusion_summary": fusion_summary,
        "weight_sensitivity": sensitivity_rows,
        "sequence_bootstrap_delta_vs_baseline": bootstrap_rows,
        "reports": {
            "markdown": str(markdown_path),
            "json": str(json_path),
            "sensitivity_csv": str(sensitivity_csv),
            "bootstrap_csv": str(bootstrap_csv),
        },
    }
    write_json_artifact(json_path, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate IEDB bounded robustness analysis.")
    parser.add_argument("--real-results", required=True)
    parser.add_argument("--carbon-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-policy", default="local_only")
    parser.add_argument("--count-policy", default="adaptive_split")
    parser.add_argument("--base-weight", type=float, default=0.80)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = generate_iedb_robustness(
        real_results_path=args.real_results,
        carbon_predictions_path=args.carbon_predictions,
        output_dir=args.output_dir,
        base_policy=args.base_policy,
        count_policy=args.count_policy,
        base_weight=args.base_weight,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(Path(report["reports"]["markdown"]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
