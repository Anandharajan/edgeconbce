from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from edgeconbce.artifacts import write_json_artifact
from edgeconbce.deployment.transport import post_json_payload
from edgeconbce.pipeline.report import write_csv, write_markdown
from edgeconbce.pipeline.transport_benchmark import _load_json, edge_transport_payloads
from edgeconbce.utils.env import system_metadata


DEFAULT_REAL_RESULTS = Path("outputs/edgeconbce_bp3_real_results_v3.json")
DEFAULT_OUTPUT_DIR = Path("outputs/two_node_transport_sci")


def generate_two_node_transport_benchmark(
    *,
    real_results_path: str | Path,
    edge_url: str,
    output_dir: str | Path,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    real_payload = _load_json(real_results_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    request_rows: list[dict[str, Any]] = []
    for transport_payload in edge_transport_payloads(real_payload):
        response = post_json_payload(transport_payload["payload"], url=edge_url, timeout_seconds=timeout_seconds)
        request_rows.append(
            {
                "policy": transport_payload["policy"],
                "payload_mode": transport_payload["payload_mode"],
                "sequence_id": transport_payload["sequence_id"],
                "request_bytes_measured": response.request_bytes,
                "response_bytes_measured": response.response_bytes,
                "roundtrip_ms_measured": response.roundtrip_ms,
                "status_code": response.status_code,
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in request_rows:
        grouped[(str(row["policy"]), str(row["payload_mode"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (policy, payload_mode), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "policy": policy,
                "payload_mode": payload_mode,
                "num_requests": len(rows),
                "mean_request_bytes_measured": round(mean(float(row["request_bytes_measured"]) for row in rows), 6),
                "mean_response_bytes_measured": round(mean(float(row["response_bytes_measured"]) for row in rows), 6),
                "mean_roundtrip_ms_measured": round(mean(float(row["roundtrip_ms_measured"]) for row in rows), 6),
                "max_roundtrip_ms_measured": round(max(float(row["roundtrip_ms_measured"]) for row in rows), 6),
            }
        )

    request_csv = output_path / "two_node_transport_request_rows.csv"
    summary_csv = output_path / "two_node_transport_summary.csv"
    markdown_path = output_path / "two_node_transport_benchmark.md"
    json_path = output_path / "two_node_transport_benchmark.json"

    write_csv(request_csv, list(request_rows[0].keys()) if request_rows else ["policy"], request_rows)
    write_csv(summary_csv, list(summary_rows[0].keys()) if summary_rows else ["policy"], summary_rows)
    write_markdown(
        markdown_path,
        "\n".join(
            [
                "# Two-Node Transport Benchmark",
                "",
                f"Edge URL: `{edge_url}`",
                "",
                "Measured request/response payload sizes and round-trip latencies against an explicitly configured edge endpoint.",
                "",
                "| policy | payload mode | num requests | mean request bytes | mean response bytes | mean roundtrip ms |",
                "|---|---|---:|---:|---:|---:|",
                *[
                    f"| {row['policy']} | {row['payload_mode']} | {row['num_requests']} | {row['mean_request_bytes_measured']} | {row['mean_response_bytes_measured']} | {row['mean_roundtrip_ms_measured']} |"
                    for row in summary_rows
                ],
            ]
        ),
    )
    payload = {
        "project": "edgecon-bce",
        "analysis": "measured_two_node_transport",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "edge_url": edge_url,
        "source_real_results_path": str(Path(real_results_path)),
        "request_rows": request_rows,
        "summary_rows": summary_rows,
        "reports": {
            "request_csv": str(request_csv),
            "summary_csv": str(summary_csv),
            "summary_markdown": str(markdown_path),
            "summary_json": str(json_path),
        },
    }
    write_json_artifact(json_path, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a two-node transport benchmark against an EdgeConBCE edge endpoint.")
    parser.add_argument("--edge-url", required=True, help="URL of the edge endpoint, e.g. http://192.168.1.12:8088/edge")
    parser.add_argument("--real-results", default=str(DEFAULT_REAL_RESULTS), help="Path to a single-run real benchmark artifact.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated transport artifacts.")
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    reports = generate_two_node_transport_benchmark(
        real_results_path=args.real_results,
        edge_url=args.edge_url,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
    )
    print(Path(reports["reports"]["summary_markdown"]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
