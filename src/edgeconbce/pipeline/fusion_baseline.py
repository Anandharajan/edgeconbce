from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

from edgeconbce.artifacts import write_json_artifact
from edgeconbce.pipeline.report import write_csv, write_markdown
from edgeconbce.utils.env import system_metadata


def _load_carbon_predictions(path: str | Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = list(payload["rows"])
    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        labels.extend(int(value) for value in row["labels"])
        scores.extend(float(value) for value in row["probabilities"])
    return rows, np.asarray(labels, dtype=np.int8), np.asarray(scores, dtype=np.float64)


def _load_bepipred_scores(path: str | Path) -> np.ndarray:
    scores: list[float] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scores.append(float(row["BepiPred-3.0 score"]))
    return np.asarray(scores, dtype=np.float64)


def _metric_summary(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | None]:
    predictions = (scores >= threshold).astype(np.int8)
    summary: dict[str, float | None] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "roc_auc": None,
        "average_precision": None,
        "predicted_positive_residues": int(predictions.sum()),
    }
    if len(np.unique(labels)) > 1:
        summary["roc_auc"] = float(roc_auc_score(labels, scores))
        summary["average_precision"] = float(average_precision_score(labels, scores))
    return summary


def _metric_values(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    summary = _metric_summary(labels, scores, threshold)
    return {
        "roc_auc": float(summary["roc_auc"] or 0.0),
        "average_precision": float(summary["average_precision"] or 0.0),
        "mcc": float(summary["mcc"] or 0.0),
        "f1": float(summary["f1"] or 0.0),
    }


def _sequence_slices(rows: list[dict[str, Any]]) -> list[slice]:
    slices: list[slice] = []
    offset = 0
    for row in rows:
        length = int(row["length"])
        slices.append(slice(offset, offset + length))
        offset += length
    return slices


def _bootstrap_sequence_deltas(
    *,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    fusion_scores: np.ndarray,
    bepipred_scores: np.ndarray,
    fusion_threshold: float,
    bepipred_threshold: float,
    repeats: int,
    seed: int = 20260621,
) -> list[dict[str, Any]]:
    sequence_slices = _sequence_slices(rows)
    if len(sequence_slices) < 2 or repeats <= 0:
        return []
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {metric: [] for metric in ("roc_auc", "average_precision", "mcc", "f1")}
    for _ in range(int(repeats)):
        sampled = rng.integers(0, len(sequence_slices), size=len(sequence_slices))
        indices = np.concatenate([np.arange(sequence_slices[int(index)].start, sequence_slices[int(index)].stop) for index in sampled])
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        fusion_values = _metric_values(sampled_labels, fusion_scores[indices], fusion_threshold)
        bepipred_values = _metric_values(sampled_labels, bepipred_scores[indices], bepipred_threshold)
        for metric in deltas:
            deltas[metric].append(float(fusion_values[metric] - bepipred_values[metric]))
    rows_out: list[dict[str, Any]] = []
    for metric, values in deltas.items():
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
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


def _threshold_for_positive_count(scores: np.ndarray, positive_count: int) -> float:
    if positive_count <= 0:
        return float(np.inf)
    if positive_count >= len(scores):
        return float(np.min(scores))
    return float(np.sort(scores)[-positive_count])


def generate_fusion_baseline(
    *,
    carbon_predictions_path: str | Path,
    bepipred_raw_scores_path: str | Path,
    bepipred_summary_path: str | Path,
    output_dir: str | Path,
    bepipred_weight: float = 0.90,
    model_name: str = "BepiPred-3.0 + CARBON-BCE-R fixed fusion",
    bootstrap_repeats: int = 2000,
) -> dict[str, Any]:
    if not 0.0 <= bepipred_weight <= 1.0:
        raise ValueError("bepipred_weight must be in [0, 1].")

    carbon_rows, labels, carbon_scores = _load_carbon_predictions(carbon_predictions_path)
    bepipred_scores = _load_bepipred_scores(bepipred_raw_scores_path)
    if len(bepipred_scores) != len(carbon_scores):
        raise ValueError("BepiPred and CARBON score lengths differ.")

    bepipred_summary = json.loads(Path(bepipred_summary_path).read_text(encoding="utf-8"))
    bepipred_threshold = float(bepipred_summary["threshold"])
    bepipred_positive_count = int(np.sum(bepipred_scores >= bepipred_threshold))

    fusion_scores = (float(bepipred_weight) * bepipred_scores) + ((1.0 - float(bepipred_weight)) * carbon_scores)
    fusion_threshold = _threshold_for_positive_count(fusion_scores, bepipred_positive_count)
    fusion_summary = _metric_summary(labels, fusion_scores, fusion_threshold)
    bootstrap_delta_rows = _bootstrap_sequence_deltas(
        rows=carbon_rows,
        labels=labels,
        fusion_scores=fusion_scores,
        bepipred_scores=bepipred_scores,
        fusion_threshold=fusion_threshold,
        bepipred_threshold=bepipred_threshold,
        repeats=bootstrap_repeats,
    )
    sensitivity_rows: list[dict[str, Any]] = []
    for candidate_weight in np.linspace(0.0, 1.0, 11):
        candidate_scores = (float(candidate_weight) * bepipred_scores) + ((1.0 - float(candidate_weight)) * carbon_scores)
        candidate_threshold = _threshold_for_positive_count(candidate_scores, bepipred_positive_count)
        candidate_summary = _metric_summary(labels, candidate_scores, candidate_threshold)
        sensitivity_rows.append(
            {
                "bepipred_weight": round(float(candidate_weight), 2),
                "carbon_weight": round(float(1.0 - candidate_weight), 2),
                "threshold": round(float(candidate_threshold), 6),
                "roc_auc": None if candidate_summary["roc_auc"] is None else round(float(candidate_summary["roc_auc"]), 6),
                "mcc": round(float(candidate_summary["mcc"]), 6),
                "f1": round(float(candidate_summary["f1"]), 6),
                "average_precision": None
                if candidate_summary["average_precision"] is None
                else round(float(candidate_summary["average_precision"]), 6),
                "predicted_positive_residues": int(candidate_summary["predicted_positive_residues"]),
            }
        )

    baseline_row = {
        "model": model_name,
        "source_type": "local_rerun_fusion",
        "dataset": "BP3C50ID external test (15 antigens)",
        "roc_auc": None if fusion_summary["roc_auc"] is None else round(float(fusion_summary["roc_auc"]), 6),
        "mcc": round(float(fusion_summary["mcc"]), 6),
        "f1": round(float(fusion_summary["f1"]), 6),
        "average_precision": None if fusion_summary["average_precision"] is None else round(float(fusion_summary["average_precision"]), 6),
        "latency_ms": None,
        "total_gco2e": None,
        "raw_sequence_share_rate": None,
        "notes": (
            "Fixed-score fusion using official BepiPred-3.0 raw scores and CARBON-BCE-R probabilities. "
            "Threshold preserves the official BepiPred positive-call count instead of optimizing test MCC."
        ),
        "source_url": "https://services.healthtech.dtu.dk/services/BepiPred-3.0/",
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    baseline_json = output_path / "fusion_baseline_row.json"
    report_json = output_path / "fusion_report.json"
    report_md = output_path / "fusion_report.md"
    per_sequence_csv = output_path / "fusion_per_sequence.csv"
    sensitivity_csv = output_path / "fusion_weight_sensitivity.csv"
    bootstrap_csv = output_path / "fusion_vs_bepipred_sequence_bootstrap.csv"

    rows: list[dict[str, Any]] = []
    offset = 0
    predictions = (fusion_scores >= fusion_threshold).astype(np.int8)
    for row in carbon_rows:
        length = int(row["length"])
        row_scores = fusion_scores[offset : offset + length]
        row_predictions = predictions[offset : offset + length]
        rows.append(
            {
                "sequence_id": row["sequence_id"],
                "length": length,
                "predicted_positive_residues": int(row_predictions.sum()),
                "mean_fusion_score": round(float(np.mean(row_scores)), 6),
            }
        )
        offset += length

    write_json_artifact(baseline_json, {"rows": [baseline_row]})
    write_csv(per_sequence_csv, ["sequence_id", "length", "predicted_positive_residues", "mean_fusion_score"], rows)
    write_csv(
        sensitivity_csv,
        [
            "bepipred_weight",
            "carbon_weight",
            "threshold",
            "roc_auc",
            "mcc",
            "f1",
            "average_precision",
            "predicted_positive_residues",
        ],
        sensitivity_rows,
    )
    write_csv(
        bootstrap_csv,
        ["metric", "mean_delta", "ci_low", "ci_high", "positive_delta_rate", "bootstrap_repeats"],
        bootstrap_delta_rows,
    )
    write_markdown(
        report_md,
        "\n".join(
            [
                "# BepiPred/CARBON Fixed Fusion Report",
                "",
                f"BepiPred weight: `{bepipred_weight:.2f}`",
                f"Threshold rule: `preserve_bepipred_positive_count={bepipred_positive_count}`",
                f"Fusion threshold: `{fusion_threshold:.6f}`",
                "",
                "| model | ROC AUC | MCC | F1 | AP |",
                "|---|---:|---:|---:|---:|",
                f"| {baseline_row['model']} | {baseline_row['roc_auc']} | {baseline_row['mcc']} | {baseline_row['f1']} | {baseline_row['average_precision']} |",
                "",
                "## Weight Sensitivity",
                "",
                "| BepiPred weight | CARBON weight | ROC AUC | MCC | F1 | AP |",
                "|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {row['bepipred_weight']:.2f} | {row['carbon_weight']:.2f} | {row['roc_auc']} | {row['mcc']} | {row['f1']} | {row['average_precision']} |"
                    for row in sensitivity_rows
                ],
                "",
                "## Sequence Bootstrap Versus BepiPred",
                "",
                "| metric | mean delta | 95% CI | positive delta rate |",
                "|---|---:|---:|---:|",
                *[
                    f"| {row['metric']} | {row['mean_delta']} | [{row['ci_low']}, {row['ci_high']}] | {row['positive_delta_rate']} |"
                    for row in bootstrap_delta_rows
                ],
                "",
                "Use `fusion_baseline_row.json` as an extra baseline when regenerating baseline comparison.",
            ]
        ),
    )

    payload = {
        "project": "edgecon-bce",
        "analysis": "bepipred_carbon_fixed_fusion",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "inputs": {
            "carbon_predictions_path": str(carbon_predictions_path),
            "bepipred_raw_scores_path": str(bepipred_raw_scores_path),
            "bepipred_summary_path": str(bepipred_summary_path),
        },
        "algorithm": {
            "name": model_name,
            "bepipred_weight": float(bepipred_weight),
            "carbon_weight": float(1.0 - float(bepipred_weight)),
            "threshold_rule": "preserve_official_bepipred_positive_call_count",
            "bepipred_threshold": bepipred_threshold,
            "bepipred_positive_count": bepipred_positive_count,
            "fusion_threshold": fusion_threshold,
        },
        "summary": fusion_summary,
        "weight_sensitivity": sensitivity_rows,
        "sequence_bootstrap_delta_vs_bepipred": bootstrap_delta_rows,
        "baseline_row": baseline_row,
        "reports": {
            "baseline_json": str(baseline_json),
            "per_sequence_csv": str(per_sequence_csv),
            "sensitivity_csv": str(sensitivity_csv),
            "bootstrap_csv": str(bootstrap_csv),
            "markdown": str(report_md),
            "json": str(report_json),
        },
    }
    write_json_artifact(report_json, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a fixed BepiPred/CARBON fusion baseline.")
    parser.add_argument("--carbon-predictions", required=True)
    parser.add_argument("--bepipred-raw-scores", required=True)
    parser.add_argument("--bepipred-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bepipred-weight", type=float, default=0.90)
    parser.add_argument("--model-name", default="BepiPred-3.0 + CARBON-BCE-R fixed fusion")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = generate_fusion_baseline(
        carbon_predictions_path=args.carbon_predictions,
        bepipred_raw_scores_path=args.bepipred_raw_scores,
        bepipred_summary_path=args.bepipred_summary,
        output_dir=args.output_dir,
        bepipred_weight=args.bepipred_weight,
        model_name=args.model_name,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(Path(report["reports"]["markdown"]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
