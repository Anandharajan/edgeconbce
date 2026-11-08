from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edgeconbce.artifacts import write_json_artifact
from edgeconbce.pipeline.report import write_csv, write_markdown
from edgeconbce.utils.env import system_metadata


DEFAULT_REPEATED_RESULTS = Path("outputs/bp3_repeated_eval/bp3_real_multiseed_aggregate_results.json")
DEFAULT_REAL_RESULTS = Path("outputs/edgeconbce_bp3_real_results_v3.json")
DEFAULT_BASELINE_RESULTS = Path("outputs/baseline_comparison/baseline_comparison.json")
DEFAULT_TRANSPORT_RESULTS = Path("outputs/transport_benchmark/transport_benchmark.json")
DEFAULT_OUTPUT_DIR = Path("outputs/figure_inputs")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate_figure_inputs(
    *,
    repeated_results_path: str | Path,
    real_results_path: str | Path,
    baseline_results_path: str | Path,
    transport_results_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    repeated_payload = _load_json(repeated_results_path)
    real_payload = _load_json(real_results_path)
    baseline_payload = _load_json(baseline_results_path)
    transport_payload = _load_json(transport_results_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pareto_rows = [
        {
            "policy": row["policy"],
            "mcc_mean": row.get("mcc_mean"),
            "f1_mean": row.get("f1_mean"),
            "roc_auc_mean": row.get("roc_auc_mean"),
            "mean_latency_ms_mean": row.get("mean_latency_ms_mean"),
            "mean_total_gco2e_mean": row.get("mean_total_gco2e_mean"),
            "raw_sequence_share_rate_mean": row.get("raw_sequence_share_rate_mean"),
            "mean_bytes_transferred_mean": row.get("mean_bytes_transferred_mean"),
        }
        for row in repeated_payload.get("aggregate_policy_metrics", [])
    ]

    route_rows: list[dict[str, Any]] = []
    for policy_name, payload in real_payload.get("external_test", {}).items():
        for sequence_row in payload.get("per_sequence", []):
            if sequence_row.get("route") is None:
                continue
            route_rows.append(
                {
                    "policy": policy_name,
                    "sequence_id": sequence_row["sequence_id"],
                    "length": sequence_row["length"],
                    "route": bool(sequence_row["route"]),
                    "route_probability": sequence_row["route_probability"],
                    "raw_sequence_shared": bool(sequence_row["raw_sequence_shared"]),
                }
            )

    baseline_rows = baseline_payload.get("rows", [])
    transport_rows = transport_payload.get("summary_rows", [])

    pareto_csv = output_path / "pareto_policy_points.csv"
    routes_csv = output_path / "route_distribution.csv"
    baseline_csv = output_path / "baseline_points.csv"
    transport_csv = output_path / "transport_points.csv"
    markdown_path = output_path / "figure_inputs.md"
    json_path = output_path / "figure_inputs.json"

    write_csv(pareto_csv, list(pareto_rows[0].keys()) if pareto_rows else ["policy"], pareto_rows)
    write_csv(routes_csv, list(route_rows[0].keys()) if route_rows else ["policy"], route_rows)
    write_csv(baseline_csv, list(baseline_rows[0].keys()) if baseline_rows else ["model"], baseline_rows)
    write_csv(transport_csv, list(transport_rows[0].keys()) if transport_rows else ["policy"], transport_rows)
    write_markdown(
        markdown_path,
        "\n".join(
            [
                "# Figure Inputs",
                "",
                "- `pareto_policy_points.csv`: quality/latency/carbon/privacy trade-off scatter input.",
                "- `route_distribution.csv`: per-sequence route decisions for hybrid policies.",
                "- `baseline_points.csv`: local repeated policies plus published BepiPred-3.0 comparison rows.",
                "- `transport_points.csv`: measured loopback transport payloads and round-trip latencies.",
            ]
        ),
    )

    payload = {
        "project": "edgecon-bce",
        "analysis": "figure_inputs",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "reports": {
            "pareto_csv": str(pareto_csv),
            "route_distribution_csv": str(routes_csv),
            "baseline_csv": str(baseline_csv),
            "transport_csv": str(transport_csv),
            "markdown": str(markdown_path),
            "json": str(json_path),
        },
    }
    write_json_artifact(json_path, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate figure-ready CSV inputs from EdgeConBCE outputs.")
    parser.add_argument("--repeated-results", default=str(DEFAULT_REPEATED_RESULTS))
    parser.add_argument("--real-results", default=str(DEFAULT_REAL_RESULTS))
    parser.add_argument("--baseline-results", default=str(DEFAULT_BASELINE_RESULTS))
    parser.add_argument("--transport-results", default=str(DEFAULT_TRANSPORT_RESULTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    reports = generate_figure_inputs(
        repeated_results_path=args.repeated_results,
        real_results_path=args.real_results,
        baseline_results_path=args.baseline_results,
        transport_results_path=args.transport_results,
        output_dir=args.output_dir,
    )
    print(Path(reports["reports"]["markdown"]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
