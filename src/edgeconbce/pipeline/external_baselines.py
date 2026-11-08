from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edgeconbce.artifacts import write_json_artifact
from edgeconbce.metrics.ranking import auc_score, average_precision, summarize_binary_metrics
from edgeconbce.pipeline.report import write_csv, write_markdown
from edgeconbce.utils.env import system_metadata


DEFAULT_TRUTH_FASTA = Path("BP3C50ID_external_test_set.fasta")
DEFAULT_OUTPUT_DIR = Path("outputs/external_baselines")
DEFAULT_THRESHOLD = 0.1512


def _parse_labeled_fasta(path: str | Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    header: str | None = None
    parts: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                sequence = "".join(parts)
                records[header] = {
                    "sequence": sequence.upper(),
                    "labels": [char.isupper() for char in sequence],
                }
            header = line[1:].split()[0]
            parts = []
        else:
            parts.append(line)
    if header is not None:
        sequence = "".join(parts)
        records[header] = {
            "sequence": sequence.upper(),
            "labels": [char.isupper() for char in sequence],
        }
    return records


def _parse_prediction_fasta(path: str | Path) -> dict[str, list[bool]]:
    records: dict[str, list[bool]] = {}
    header: str | None = None
    parts: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records[header] = [char.isupper() for char in "".join(parts)]
            header = line[1:].split()[0]
            parts = []
        else:
            parts.append(line)
    if header is not None:
        records[header] = [char.isupper() for char in "".join(parts)]
    return records


def _normalise_key(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.strip().lower())).strip("_")


def _find_column(fieldnames: list[str], candidates: set[str]) -> str | None:
    by_key = {_normalise_key(name): name for name in fieldnames}
    for candidate in candidates:
        if candidate in by_key:
            return by_key[candidate]
    return None


def _parse_score_csv(path: str | Path) -> dict[str, list[float]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        reader = csv.DictReader(handle, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"Raw score CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        sequence_column = _find_column(fieldnames, {"id", "seqid", "sequence_id", "protein", "protein_id", "header", "name", "accession"})
        position_column = _find_column(fieldnames, {"position", "pos", "residue_number", "index", "aa_index"})
        score_column = _find_column(
            fieldnames,
            {
                "score",
                "probability",
                "bepipred_3_0_score",
                "bepipred_3_0_score",
                "bepipred3_score",
                "epitope_score",
                "b_cell_epitope_score",
                "bcell_epitope_score",
            },
        )
        if sequence_column is None or score_column is None:
            raise ValueError(
                "Could not infer sequence and score columns from raw score CSV. "
                f"Columns: {fieldnames}"
            )

        indexed_scores: dict[str, dict[int, float]] = {}
        next_position_by_sequence: dict[str, int] = {}
        for row in reader:
            sequence_id = str(row[sequence_column]).strip().split()[0]
            if not sequence_id:
                continue
            if position_column is None:
                position = next_position_by_sequence.get(sequence_id, 0)
                next_position_by_sequence[sequence_id] = position + 1
            else:
                position = int(float(str(row[position_column]).strip()))
            score = float(str(row[score_column]).strip())
            indexed_scores.setdefault(sequence_id, {})[position] = score

    scores: dict[str, list[float]] = {}
    for sequence_id, by_position in indexed_scores.items():
        offset = 0 if 0 in by_position else 1
        scores[sequence_id] = [by_position[index + offset] for index in range(len(by_position))]
    return scores


def write_unlabeled_submission_fasta(
    *,
    truth_fasta_path: str | Path = DEFAULT_TRUTH_FASTA,
    output_fasta_path: str | Path = DEFAULT_OUTPUT_DIR / "bp3c50id_external_unlabeled.fasta",
) -> Path:
    truth_records = _parse_labeled_fasta(truth_fasta_path)
    output_path = Path(output_fasta_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for sequence_id, record in truth_records.items():
        lines.append(f">{sequence_id}")
        sequence = str(record["sequence"])
        for start in range(0, len(sequence), 80):
            lines.append(sequence[start : start + 80])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def evaluate_bepipred_output(
    *,
    truth_fasta_path: str | Path,
    output_dir: str | Path,
    prediction_fasta_path: str | Path | None = None,
    raw_score_csv_path: str | Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = "BepiPred-3.0 official local rerun",
    source_url: str = "https://services.healthtech.dtu.dk/services/BepiPred-3.0/",
) -> dict[str, Any]:
    if prediction_fasta_path is None and raw_score_csv_path is None:
        raise ValueError("Provide prediction_fasta_path and/or raw_score_csv_path.")

    truth_records = _parse_labeled_fasta(truth_fasta_path)
    score_rows = _parse_score_csv(raw_score_csv_path) if raw_score_csv_path is not None else {}
    prediction_rows = _parse_prediction_fasta(prediction_fasta_path) if prediction_fasta_path is not None else {}

    labels: list[bool] = []
    predictions: list[bool] = []
    scores: list[float] = []
    per_sequence_rows: list[dict[str, Any]] = []

    for sequence_id, truth_record in truth_records.items():
        truth_labels = list(truth_record["labels"])
        sequence_scores = score_rows.get(sequence_id)
        sequence_predictions = prediction_rows.get(sequence_id)
        if sequence_predictions is None and sequence_scores is not None:
            sequence_predictions = [score >= threshold for score in sequence_scores]
        if sequence_predictions is None:
            raise ValueError(f"No predictions found for sequence {sequence_id}")
        if len(sequence_predictions) != len(truth_labels):
            raise ValueError(f"Length mismatch for {sequence_id}: truth={len(truth_labels)} prediction={len(sequence_predictions)}")
        if sequence_scores is not None and len(sequence_scores) != len(truth_labels):
            raise ValueError(f"Length mismatch for {sequence_id}: truth={len(truth_labels)} scores={len(sequence_scores)}")

        labels.extend(truth_labels)
        predictions.extend(sequence_predictions)
        if sequence_scores is not None:
            scores.extend(sequence_scores)
        per_sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "length": len(truth_labels),
                "true_positive_residues": sum(1 for label in truth_labels if label),
                "predicted_positive_residues": sum(1 for prediction in sequence_predictions if prediction),
                "has_raw_scores": sequence_scores is not None,
            }
        )

    binary_metrics = summarize_binary_metrics(labels, predictions)
    row = {
        "model": model_name,
        "source_type": "local_rerun",
        "dataset": "BP3C50ID external test (15 antigens)",
        "roc_auc": round(auc_score(labels, scores), 6) if scores else None,
        "mcc": round(binary_metrics["mcc"], 6),
        "f1": round(binary_metrics["f1"], 6),
        "average_precision": round(average_precision(labels, scores), 6) if scores else None,
        "latency_ms": None,
        "total_gco2e": None,
        "raw_sequence_share_rate": None,
        "notes": "Imported from official BepiPred-3.0 downloaded output on the BP3C50ID external FASTA.",
        "source_url": source_url,
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    extra_baselines_path = output_path / "bepipred3_local_rerun_baseline.json"
    per_sequence_csv = output_path / "bepipred3_per_sequence_summary.csv"
    markdown_path = output_path / "bepipred3_local_rerun_summary.md"
    payload_path = output_path / "bepipred3_local_rerun_summary.json"

    write_json_artifact(extra_baselines_path, {"rows": [row]})
    write_csv(per_sequence_csv, list(per_sequence_rows[0].keys()) if per_sequence_rows else ["sequence_id"], per_sequence_rows)
    write_markdown(
        markdown_path,
        "\n".join(
            [
                "# BepiPred-3.0 Local Rerun Import",
                "",
                "| model | ROC AUC | MCC | F1 | AP |",
                "|---|---:|---:|---:|---:|",
                f"| {row['model']} | {row['roc_auc'] if row['roc_auc'] is not None else 'n/a'} | {row['mcc']} | {row['f1']} | {row['average_precision'] if row['average_precision'] is not None else 'n/a'} |",
                "",
                "Use the generated JSON as `--extra-baselines` for `edgeconbce.pipeline.baseline_comparison`.",
            ]
        ),
    )
    payload = {
        "project": "edgecon-bce",
        "analysis": "bepipred3_local_rerun_import",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "threshold": threshold,
        "truth_fasta_path": str(Path(truth_fasta_path)),
        "prediction_fasta_path": str(Path(prediction_fasta_path)) if prediction_fasta_path is not None else None,
        "raw_score_csv_path": str(Path(raw_score_csv_path)) if raw_score_csv_path is not None else None,
        "baseline_row": row,
        "per_sequence_rows": per_sequence_rows,
        "reports": {
            "extra_baselines_json": str(extra_baselines_path),
            "per_sequence_csv": str(per_sequence_csv),
            "markdown": str(markdown_path),
            "json": str(payload_path),
        },
    }
    write_json_artifact(payload_path, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and import official external BCE baseline outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-bp3-fasta", help="Write an unlabeled BP3C50ID external FASTA for web-server submission.")
    prepare.add_argument("--truth-fasta", default=str(DEFAULT_TRUTH_FASTA))
    prepare.add_argument("--output-fasta", default=str(DEFAULT_OUTPUT_DIR / "bp3c50id_external_unlabeled.fasta"))

    evaluate = subparsers.add_parser("evaluate-bepipred", help="Evaluate downloaded BepiPred-3.0 output against BP3C50ID labels.")
    evaluate.add_argument("--truth-fasta", default=str(DEFAULT_TRUTH_FASTA))
    evaluate.add_argument("--prediction-fasta", default=None)
    evaluate.add_argument("--raw-score-csv", default=None)
    evaluate.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    evaluate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-bp3-fasta":
        path = write_unlabeled_submission_fasta(truth_fasta_path=args.truth_fasta, output_fasta_path=args.output_fasta)
        print(Path(path).resolve())
        return 0
    if args.command == "evaluate-bepipred":
        report = evaluate_bepipred_output(
            truth_fasta_path=args.truth_fasta,
            prediction_fasta_path=args.prediction_fasta,
            raw_score_csv_path=args.raw_score_csv,
            threshold=args.threshold,
            output_dir=args.output_dir,
        )
        print(Path(report["reports"]["markdown"]).resolve())
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
