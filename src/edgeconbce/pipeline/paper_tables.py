from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from edgeconbce.pipeline.real_benchmark import _policy_data_movement_summary
from edgeconbce.pipeline.report import write_csv, write_markdown
from edgeconbce.pipeline.repeated_eval import (
    SUMMARY_BOOTSTRAP_REPEATS,
    SUMMARY_CONFIDENCE_LEVEL,
    _aggregate_comparison_rows,
    _comparison_significance_rows,
)


DEFAULT_REAL_RESULTS = Path("outputs/edgeconbce_bp3_real_results_v3.json")
DEFAULT_REPEATED_RESULTS = Path("outputs/bp3_repeated_eval/bp3_real_multiseed_aggregate_results.json")
DEFAULT_OUTPUT_DIR = Path("outputs/paper_tables")
POLICY_METRICS = [
    "mcc",
    "f1",
    "roc_auc",
    "average_precision",
    "mean_latency_ms",
    "mean_total_gco2e",
    "route_rate",
    "raw_sequence_share_rate",
    "mean_bytes_transferred",
]
COMPARISON_METRICS = [
    "mcc",
    "f1",
    "roc_auc",
    "average_precision",
    "mean_latency_ms",
    "mean_total_gco2e",
    "raw_sequence_share_rate",
    "mean_bytes_transferred",
]


def _policy_sort_key(policy_name: str) -> tuple[int, str]:
    order = {
        "local_only": 0,
        "edge_only": 1,
        "adaptive_split": 2,
        "adaptive_carbon_guard": 3,
        "plm_local_only": 4,
        "plm_edge_only": 5,
        "plm_region_constrained_edge": 6,
        "plm_adaptive_split": 7,
        "plm_adaptive_conformal_guard": 8,
    }
    return (order.get(policy_name, 99), policy_name)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _format_numeric(value: float | int | None, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{decimals}f}"


def _format_interval(row: dict[str, Any], metric: str, decimals: int = 4) -> str:
    mean_value = row.get(f"{metric}_mean")
    if mean_value is None:
        return "n/a"
    ci_low = row.get(f"{metric}_ci_low")
    ci_high = row.get(f"{metric}_ci_high")
    if ci_low is None or ci_high is None:
        return _format_numeric(float(mean_value), decimals=decimals)
    return f"{_format_numeric(float(mean_value), decimals=decimals)} [{_format_numeric(float(ci_low), decimals=decimals)}, {_format_numeric(float(ci_high), decimals=decimals)}]"


def _comparison_summary_rows(repeated_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    delta_rows: list[dict[str, Any]] = []
    for rows in repeated_payload.get("paired_deltas", {}).values():
        delta_rows.extend(rows)

    summary_rows = repeated_payload.get("paired_delta_summary")
    if not summary_rows:
        summary_rows = _aggregate_comparison_rows(
            delta_rows,
            bootstrap_repeats=SUMMARY_BOOTSTRAP_REPEATS,
            confidence_level=SUMMARY_CONFIDENCE_LEVEL,
            bootstrap_seed=404,
        )

    significance_rows = repeated_payload.get("paired_significance")
    if not significance_rows:
        significance_rows = _comparison_significance_rows(delta_rows)
    return summary_rows, significance_rows


def _policy_table_rows(repeated_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sorted(repeated_payload.get("aggregate_policy_metrics", []), key=lambda item: _policy_sort_key(str(item["policy"]))):
        rows.append(
            {
                "policy": row["policy"],
                "mcc": _format_interval(row, "mcc"),
                "f1": _format_interval(row, "f1"),
                "roc_auc": _format_interval(row, "roc_auc"),
                "average_precision": _format_interval(row, "average_precision"),
                "mean_latency_ms": _format_interval(row, "mean_latency_ms"),
                "mean_total_gco2e": _format_interval(row, "mean_total_gco2e"),
                "route_rate": _format_interval(row, "route_rate"),
                "raw_sequence_share_rate": _format_interval(row, "raw_sequence_share_rate"),
                "mean_bytes_transferred": _format_interval(row, "mean_bytes_transferred"),
            }
        )
    return rows


def _comparison_table_rows(repeated_payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary_rows, significance_rows = _comparison_summary_rows(repeated_payload)
    significance_by_key = {(row["comparison"], row["metric"]): row for row in significance_rows}

    table_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        display_row: dict[str, Any] = {"comparison": row["comparison"]}
        for metric in COMPARISON_METRICS:
            significance = significance_by_key.get((row["comparison"], metric), {})
            display_row[f"{metric}_delta"] = _format_interval(row, metric)
            display_row[f"{metric}_p_value"] = _format_numeric(significance.get("p_value_two_sided_sign_test"), decimals=4)
            display_row[f"{metric}_direction"] = significance.get("effect_direction", "n/a")
        table_rows.append(display_row)
    return table_rows


def _data_movement_rows(real_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name, payload in sorted(real_payload.get("external_test", {}).items(), key=lambda item: _policy_sort_key(str(item[0]))):
        data_movement = payload.get("data_movement")
        if data_movement is None:
            data_movement = _policy_data_movement_summary(payload.get("per_sequence", []))
        rows.append(
            {
                "policy": policy_name,
                "transfer_mode": data_movement["transfer_mode"],
                "payload_basis": data_movement["payload_basis"],
                "sequence_count": data_movement["sequence_count"],
                "shared_sequence_count": data_movement["shared_sequence_count"],
                "raw_sequence_share_rate": _format_numeric(data_movement["raw_sequence_share_rate"], decimals=4),
                "mean_bytes_transferred": _format_numeric(data_movement["mean_bytes_transferred"], decimals=2),
                "total_bytes_transferred": data_movement["total_bytes_transferred"],
                "max_bytes_transferred": data_movement["max_bytes_transferred"],
            }
        )
    return rows


def _region_metric_rows(real_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name, payload in sorted(real_payload.get("external_test", {}).items(), key=lambda item: _policy_sort_key(str(item[0]))):
        region_metrics = payload.get("summary", {}).get("region_metrics")
        if not region_metrics:
            continue
        rows.append(
            {
                "policy": policy_name,
                "predicted_region_count": region_metrics["predicted_region_count"],
                "true_region_count": region_metrics["true_region_count"],
                "predicted_positive_count": region_metrics["predicted_positive_count"],
                "true_positive_count": region_metrics["true_positive_count"],
                "region_overlap_count": region_metrics["region_overlap_count"],
                "region_precision": _format_numeric(region_metrics["region_precision"], decimals=4),
                "region_recall": _format_numeric(region_metrics["region_recall"], decimals=4),
                "largest_predicted_region_fraction": _format_numeric(region_metrics["largest_predicted_region_fraction"], decimals=4),
            }
        )
    return rows


def _provenance_rows(repeated_payload: dict[str, Any], real_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact_name, payload in (("repeated_eval", repeated_payload), ("single_run_real", real_payload)):
        provenance = payload.get("measurement_provenance", {})
        for field_name, source in sorted(provenance.items()):
            rows.append(
                {
                    "artifact": artifact_name,
                    "field": field_name,
                    "provenance": source,
                }
            )
    return rows


def _paper_markdown(
    repeated_payload: dict[str, Any],
    policy_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    data_movement_rows: list[dict[str, Any]],
    region_metric_rows: list[dict[str, Any]],
    provenance_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# EdgeConBCE Manuscript Tables",
        "",
        f"Repeated-eval artifact: {repeated_payload.get('run_name', 'n/a')}",
        "",
        "## Policy Summary Table",
        "",
        "| policy | MCC | F1 | ROC AUC | AP | latency ms | total gCO2e | route rate | raw share | mean bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy_rows:
        lines.append(
            f"| {row['policy']} | {row['mcc']} | {row['f1']} | {row['roc_auc']} | {row['average_precision']} | "
            f"{row['mean_latency_ms']} | {row['mean_total_gco2e']} | {row['route_rate']} | "
            f"{row['raw_sequence_share_rate']} | {row['mean_bytes_transferred']} |"
        )

    lines.extend(
        [
            "",
            "## Paired Adaptive Comparisons",
            "",
            "| comparison | MCC delta | p | F1 delta | p | latency delta | p | carbon delta | p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| {row['comparison']} | {row['mcc_delta']} | {row['mcc_p_value']} | {row['f1_delta']} | {row['f1_p_value']} | "
            f"{row['mean_latency_ms_delta']} | {row['mean_latency_ms_p_value']} | {row['mean_total_gco2e_delta']} | {row['mean_total_gco2e_p_value']} |"
        )

    lines.extend(
        [
            "",
            "## Data Movement Table",
            "",
            "| policy | raw share | mean bytes | total bytes | shared seqs | total seqs |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in data_movement_rows:
        lines.append(
            f"| {row['policy']} | {row['raw_sequence_share_rate']} | {row['mean_bytes_transferred']} | {row['total_bytes_transferred']} | "
            f"{row['shared_sequence_count']} | {row['sequence_count']} |"
        )

    if region_metric_rows:
        lines.extend(
            [
                "",
                "## Region Metrics Table",
                "",
                "| policy | predicted regions | predicted positives | overlap residues | region precision | region recall | largest region fraction |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in region_metric_rows:
            lines.append(
                f"| {row['policy']} | {row['predicted_region_count']} | {row['predicted_positive_count']} | "
                f"{row['region_overlap_count']} | {row['region_precision']} | {row['region_recall']} | "
                f"{row['largest_predicted_region_fraction']} |"
            )

    if provenance_rows:
        lines.extend(
            [
                "",
                "## Measurement Provenance Table",
                "",
                "| artifact | field | provenance |",
                "|---|---|---|",
            ]
        )
        for row in provenance_rows:
            lines.append(f"| {row['artifact']} | {row['field']} | {row['provenance']} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Policy summary cells use repeated-run means with bootstrap confidence intervals when available in the repeated-eval artifact.",
            "- Paired p-values are exact two-sided sign tests over per-seed policy deltas.",
            "- Data-movement bytes in the BP3 prediction artifact are estimated as one ASCII byte per shared amino-acid residue; measured serialized payload sizes are reported separately in the transport benchmark artifact.",
            "- Region metrics are single-run external-test summaries until the repeated BP3 artifact is regenerated with the region-constrained policy.",
        ]
    )
    return "\n".join(lines)


def generate_paper_tables(
    *,
    repeated_results_path: str | Path,
    real_results_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    repeated_payload = _load_json(repeated_results_path)
    real_payload = _load_json(real_results_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    policy_rows = _policy_table_rows(repeated_payload)
    comparison_rows = _comparison_table_rows(repeated_payload)
    data_movement_rows = _data_movement_rows(real_payload)
    region_metric_rows = _region_metric_rows(real_payload)
    provenance_rows = _provenance_rows(repeated_payload, real_payload)

    policy_csv = output_path / "policy_summary_table.csv"
    comparison_csv = output_path / "paired_comparison_table.csv"
    data_movement_csv = output_path / "data_movement_table.csv"
    region_metrics_csv = output_path / "region_metrics_table.csv"
    provenance_csv = output_path / "measurement_provenance_table.csv"
    markdown_path = output_path / "paper_tables.md"

    write_csv(
        policy_csv,
        [
            "policy",
            "mcc",
            "f1",
            "roc_auc",
            "average_precision",
            "mean_latency_ms",
            "mean_total_gco2e",
            "route_rate",
            "raw_sequence_share_rate",
            "mean_bytes_transferred",
        ],
        policy_rows,
    )
    write_csv(comparison_csv, list(comparison_rows[0].keys()) if comparison_rows else ["comparison"], comparison_rows)
    write_csv(
        data_movement_csv,
        [
            "policy",
            "transfer_mode",
            "payload_basis",
            "sequence_count",
            "shared_sequence_count",
            "raw_sequence_share_rate",
            "mean_bytes_transferred",
            "total_bytes_transferred",
            "max_bytes_transferred",
        ],
        data_movement_rows,
    )
    write_csv(
        region_metrics_csv,
        [
            "policy",
            "predicted_region_count",
            "true_region_count",
            "predicted_positive_count",
            "true_positive_count",
            "region_overlap_count",
            "region_precision",
            "region_recall",
            "largest_predicted_region_fraction",
        ],
        region_metric_rows,
    )
    write_csv(provenance_csv, ["artifact", "field", "provenance"], provenance_rows)
    write_markdown(
        markdown_path,
        _paper_markdown(
            repeated_payload,
            policy_rows,
            comparison_rows,
            data_movement_rows,
            region_metric_rows,
            provenance_rows,
        ),
    )

    return {
        "policy_csv": str(policy_csv),
        "comparison_csv": str(comparison_csv),
        "data_movement_csv": str(data_movement_csv),
        "region_metrics_csv": str(region_metrics_csv),
        "provenance_csv": str(provenance_csv),
        "markdown": str(markdown_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate EdgeConBCE manuscript-ready tables from benchmark artifacts.")
    parser.add_argument("--repeated-results", default=str(DEFAULT_REPEATED_RESULTS), help="Path to repeated-eval aggregate JSON.")
    parser.add_argument("--real-results", default=str(DEFAULT_REAL_RESULTS), help="Path to single-run BP3 real benchmark JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated manuscript tables.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    reports = generate_paper_tables(
        repeated_results_path=args.repeated_results,
        real_results_path=args.real_results,
        output_dir=args.output_dir,
    )
    print(Path(reports["markdown"]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
