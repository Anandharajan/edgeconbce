from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from edgeconbce.artifacts import write_json_artifact
from edgeconbce.deployment.transport import benchmark_loopback_payload
from edgeconbce.pipeline.report import write_csv, write_markdown
from edgeconbce.utils.env import system_metadata


DEFAULT_REAL_RESULTS = Path("outputs/edgeconbce_bp3_real_results_v3.json")
DEFAULT_OUTPUT_DIR = Path("outputs/transport_benchmark")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_fasta(path: str | Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    header: str | None = None
    parts: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                sequences[header] = "".join(parts).upper()
            header = line[1:]
            parts = []
            continue
        parts.append(line)
    if header is not None:
        sequences[header] = "".join(parts).upper()
    return sequences


def _feature_payload(sequence_row: dict[str, Any]) -> dict[str, Any]:
    local_probabilities = [float(value) for value in sequence_row["local_probabilities"]]
    return {
        "sequence_id": str(sequence_row["sequence_id"]),
        "length": int(sequence_row["length"]),
        "local_probability_mean": round(mean(local_probabilities), 6),
        "local_probability_std": round(float((sum((value - mean(local_probabilities)) ** 2 for value in local_probabilities) / max(1, len(local_probabilities))) ** 0.5), 6),
        "mean_local_uncertainty": float(sequence_row["mean_local_uncertainty"]),
        "positive_prediction_count": int(sequence_row["positive_prediction_count"]),
    }


def edge_transport_payloads(real_payload: dict[str, Any]) -> list[dict[str, Any]]:
    test_sequences = _parse_fasta(real_payload["config"]["test_fasta_path"])
    payloads: list[dict[str, Any]] = []
    for policy_name, policy_payload in real_payload.get("external_test", {}).items():
        for sequence_row in policy_payload.get("per_sequence", []):
            if not bool(sequence_row.get("raw_sequence_shared")):
                continue
            sequence_id = str(sequence_row["sequence_id"])
            raw_payload = {
                "sequence_id": sequence_id,
                "sequence": test_sequences[sequence_id],
                "policy": policy_name,
            }
            feature_payload = {
                "sequence_id": sequence_id,
                "feature_summary": _feature_payload(sequence_row),
                "policy": policy_name,
            }
            payloads.append(
                {
                    "policy": policy_name,
                    "payload_mode": "raw_sequence",
                    "sequence_id": sequence_id,
                    "payload": raw_payload,
                }
            )
            payloads.append(
                {
                    "policy": policy_name,
                    "payload_mode": "feature_summary",
                    "sequence_id": sequence_id,
                    "payload": feature_payload,
                }
            )
    return payloads


def generate_transport_benchmark(
    *,
    real_results_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    real_payload = _load_json(real_results_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    request_rows: list[dict[str, Any]] = []
    for transport_payload in edge_transport_payloads(real_payload):
        response = benchmark_loopback_payload(transport_payload["payload"])
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

    request_csv = output_path / "transport_request_rows.csv"
    summary_csv = output_path / "transport_summary.csv"
    markdown_path = output_path / "transport_benchmark.md"
    json_path = output_path / "transport_benchmark.json"

    write_csv(request_csv, list(request_rows[0].keys()) if request_rows else ["policy"], request_rows)
    write_csv(summary_csv, list(summary_rows[0].keys()) if summary_rows else ["policy"], summary_rows)
    write_markdown(
        markdown_path,
        "\n".join(
            [
                "# Loopback Transport Benchmark",
                "",
                "Measured localhost request/response payload sizes and round-trip latencies for policies that offload sequences to the edge.",
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
        "analysis": "transport_benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
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
    parser = argparse.ArgumentParser(description="Run a localhost transport benchmark over edge-routed EdgeConBCE requests.")
    parser.add_argument("--real-results", default=str(DEFAULT_REAL_RESULTS), help="Path to a single-run real benchmark artifact.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for transport benchmark outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    reports = generate_transport_benchmark(real_results_path=args.real_results, output_dir=args.output_dir)
    print(Path(reports["reports"]["summary_markdown"]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
