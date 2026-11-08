from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edgeconbce.artifacts import write_json_artifact
from edgeconbce.pipeline.report import write_csv, write_markdown
from edgeconbce.utils.env import system_metadata


DEFAULT_REPEATED_RESULTS = Path("outputs/bp3_repeated_eval_sci/bp3_real_multiseed_sci_aggregate_results.json")
DEFAULT_BASELINE_RESULTS = Path("outputs/baseline_comparison_sci/baseline_comparison.json")
DEFAULT_TRANSPORT_RESULTS = Path("outputs/transport_benchmark_sci/transport_benchmark.json")
DEFAULT_TWO_NODE_TRANSPORT_RESULTS = Path("outputs/two_node_transport_sci/two_node_transport_benchmark.json")
DEFAULT_REAL_RESULTS = Path("outputs/edgeconbce_bp3_real_results_v3.json")
DEFAULT_OUTPUT_DIR = Path("outputs/sci_readiness_sci")

REQUIRED_POLICIES = {
    "local_only",
    "edge_only",
    "adaptive_split",
    "adaptive_carbon_guard",
    "plm_local_only",
    "plm_edge_only",
    "plm_adaptive_split",
    "plm_adaptive_conformal_guard",
}
MODERN_BASELINE_PREFIXES = ("SEMA", "GraphBepi", "DynaBCE", "RoBep", "CALIBER", "CBTOPE2", "BConformeR", "CARBON-BCE-R")
PROVENANCE_FIELDS = {
    "latency",
    "energy",
    "operational_carbon",
    "embodied_carbon",
    "bytes_transferred",
}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _status(passed: bool, *, blocked: bool = False) -> str:
    if passed:
        return "pass"
    if blocked:
        return "blocked_external"
    return "needs_work"


def _check_row(item: str, passed: bool, evidence: str, next_step: str, *, blocked: bool = False) -> dict[str, Any]:
    return {
        "item": item,
        "status": _status(passed, blocked=blocked),
        "evidence": evidence,
        "next_step": next_step,
    }


def _seed_count(repeated_payload: dict[str, Any]) -> int:
    seeds = repeated_payload.get("seeds")
    if isinstance(seeds, list):
        return len(seeds)
    per_seed_rows = repeated_payload.get("per_seed_policy_metrics", [])
    return len({row.get("seed") for row in per_seed_rows if "seed" in row})


def _policy_set(repeated_payload: dict[str, Any]) -> set[str]:
    return {str(row["policy"]) for row in repeated_payload.get("aggregate_policy_metrics", []) if "policy" in row}


def _dataset_labels(*payloads: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    for payload in payloads:
        for row in payload.get("rows", []):
            dataset = row.get("dataset")
            if dataset:
                labels.add(str(dataset))
        config = payload.get("config", {})
        test_path = config.get("test_fasta_path")
        if test_path:
            labels.add(Path(str(test_path)).name)
    return labels


def _dataset_families(labels: set[str]) -> set[str]:
    families: set[str] = set()
    for label in labels:
        normalized = label.lower()
        if "bp3c50id" in normalized or "bp3" in normalized:
            families.add("BP3C50ID")
        else:
            families.add(label)
    return families


def generate_sci_readiness_report(
    *,
    repeated_results_path: str | Path,
    baseline_results_path: str | Path,
    transport_results_path: str | Path,
    two_node_transport_results_path: str | Path | None = None,
    real_results_path: str | Path,
    additional_real_results_paths: list[str | Path] | None = None,
    output_dir: str | Path,
) -> dict[str, Any]:
    repeated_payload = _load_json(repeated_results_path)
    baseline_payload = _load_json(baseline_results_path)
    transport_payload = _load_json(transport_results_path)
    two_node_transport_payload = _load_json(two_node_transport_results_path) if two_node_transport_results_path is not None and Path(two_node_transport_results_path).exists() else {}
    real_payload = _load_json(real_results_path)
    additional_real_payloads = [_load_json(path) for path in additional_real_results_paths or []]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    policies = _policy_set(repeated_payload)
    missing_policies = sorted(REQUIRED_POLICIES - policies)
    seed_count = _seed_count(repeated_payload)
    baseline_source_types = {str(row.get("source_type")) for row in baseline_payload.get("rows", [])}
    baseline_models = {str(row.get("model")) for row in baseline_payload.get("rows", [])}
    modern_baseline_models = sorted(model for model in baseline_models if model.startswith(MODERN_BASELINE_PREFIXES))
    transport_rows = transport_payload.get("summary_rows", [])
    two_node_transport_rows = two_node_transport_payload.get("summary_rows", [])
    has_two_node_transport = two_node_transport_payload.get("analysis") == "measured_two_node_transport" and bool(two_node_transport_rows)
    repeated_provenance = set(repeated_payload.get("measurement_provenance", {}).keys())
    real_provenance = set(real_payload.get("measurement_provenance", {}).keys())
    datasets = _dataset_labels(baseline_payload, real_payload, *additional_real_payloads)
    dataset_families = _dataset_families(datasets)

    rows = [
        _check_row(
            "10-seed repeated evaluation",
            seed_count >= 10,
            f"{seed_count} repeated seeds detected.",
            "Run the repeated BP3 configuration until at least 10 seeds are aggregated.",
        ),
        _check_row(
            "Required policy coverage",
            not missing_policies,
            "All required policies are present." if not missing_policies else f"Missing policies: {', '.join(missing_policies)}.",
            "Regenerate repeated evaluation with the missing policies enabled.",
        ),
        _check_row(
            "Published BepiPred-3.0 comparator",
            any("BepiPred-3.0" in model for model in baseline_models),
            "BepiPred-3.0 appears in the baseline comparison artifact.",
            "Add the published same-split BepiPred-3.0 comparator row.",
        ),
        _check_row(
            "Official or local BepiPred-3.0 rerun",
            "local_rerun" in baseline_source_types,
            "No local_rerun baseline row detected." if "local_rerun" not in baseline_source_types else "At least one local_rerun comparator is present.",
            "Run the official executable/service on the same split and import the outputs as source_type=local_rerun.",
            blocked="local_rerun" not in baseline_source_types,
        ),
        _check_row(
            "Second modern BCE baseline",
            bool(modern_baseline_models),
            (
                f"Modern comparator rows detected: {', '.join(modern_baseline_models)}."
                if modern_baseline_models
                else "No second modern BCE comparator detected."
            ),
            "Add a same-split local rerun where possible; otherwise add a clearly labeled published_different_split row outside headline claims.",
            blocked=not modern_baseline_models,
        ),
        _check_row(
            "Measured loopback transport",
            bool(transport_rows),
            f"{len(transport_rows)} transport summary rows detected.",
            "Run edgeconbce.pipeline.transport_benchmark against the real BP3 artifact.",
        ),
        _check_row(
            "Real two-node deployment transport",
            has_two_node_transport,
            (
                f"{len(two_node_transport_rows)} measured two-node transport summary rows detected."
                if has_two_node_transport
                else "Only localhost loopback transport is represented in current artifacts."
            ),
            "Run the transport benchmark across separate local and edge hosts and store the artifact as measured_two_node_transport.",
            blocked=not has_two_node_transport,
        ),
        _check_row(
            "Measurement provenance coverage",
            PROVENANCE_FIELDS <= (repeated_provenance | real_provenance),
            f"Repeated provenance fields: {sorted(repeated_provenance)}; real-run provenance fields: {sorted(real_provenance)}.",
            "Add missing provenance labels to generated artifacts before using the fields in claims.",
        ),
        _check_row(
            "Dataset breadth beyond BP3C50ID",
            len(dataset_families) > 1,
            f"Dataset families detected: {', '.join(sorted(dataset_families)) or 'none'}; labels: {', '.join(sorted(datasets)) or 'none'}.",
            "Add a second benchmark family or external validation artifact.",
            blocked=len(dataset_families) <= 1,
        ),
    ]

    pass_count = sum(1 for row in rows if row["status"] == "pass")
    blocked_count = sum(1 for row in rows if row["status"] == "blocked_external")
    needs_work_count = sum(1 for row in rows if row["status"] == "needs_work")
    overall_status = "sci_draftable_not_final_strong"
    if needs_work_count == 0 and blocked_count == 0:
        overall_status = "sci_ready"

    csv_path = output_path / "sci_readiness_checklist.csv"
    markdown_path = output_path / "sci_readiness_report.md"
    json_path = output_path / "sci_readiness_report.json"

    write_csv(csv_path, ["item", "status", "evidence", "next_step"], rows)
    write_markdown(
        markdown_path,
        "\n".join(
            [
                "# EdgeConBCE SCI Readiness Report",
                "",
                f"Generated at: {datetime.now(timezone.utc).isoformat()}",
                "",
                f"Overall status: `{overall_status}`",
                "",
                f"Checklist: {pass_count} pass, {needs_work_count} needs work, {blocked_count} blocked on external data/service work.",
                "",
                "| item | status | evidence | next step |",
                "|---|---|---|---|",
                *[f"| {row['item']} | {row['status']} | {row['evidence']} | {row['next_step']} |" for row in rows],
                "",
                "## Claim Boundary",
                "",
                "- Current artifacts support a reproducible BP3C50ID local-edge BCE benchmark and trade-off analysis.",
                "- Strong state-of-the-art, real-world deployment, and measured-carbon claims remain blocked until the external checks pass.",
            ]
        ),
    )

    payload = {
        "project": "edgecon-bce",
        "analysis": "sci_readiness",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "overall_status": overall_status,
        "summary": {
            "pass": pass_count,
            "needs_work": needs_work_count,
            "blocked_external": blocked_count,
        },
        "rows": rows,
        "source_artifacts": {
            "repeated_results": str(Path(repeated_results_path)),
            "baseline_results": str(Path(baseline_results_path)),
            "transport_results": str(Path(transport_results_path)),
            "two_node_transport_results": str(Path(two_node_transport_results_path)) if two_node_transport_results_path is not None else None,
            "real_results": str(Path(real_results_path)),
            "additional_real_results": [str(Path(path)) for path in additional_real_results_paths or []],
        },
        "reports": {
            "csv": str(csv_path),
            "markdown": str(markdown_path),
            "json": str(json_path),
        },
    }
    write_json_artifact(json_path, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an SCI-readiness checklist from EdgeConBCE artifacts.")
    parser.add_argument("--repeated-results", default=str(DEFAULT_REPEATED_RESULTS))
    parser.add_argument("--baseline-results", default=str(DEFAULT_BASELINE_RESULTS))
    parser.add_argument("--transport-results", default=str(DEFAULT_TRANSPORT_RESULTS))
    parser.add_argument("--two-node-transport-results", default=str(DEFAULT_TWO_NODE_TRANSPORT_RESULTS))
    parser.add_argument("--real-results", default=str(DEFAULT_REAL_RESULTS))
    parser.add_argument("--additional-real-results", action="append", default=[], help="Optional extra real benchmark artifact for dataset-breadth checks.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = generate_sci_readiness_report(
        repeated_results_path=args.repeated_results,
        baseline_results_path=args.baseline_results,
        transport_results_path=args.transport_results,
        two_node_transport_results_path=args.two_node_transport_results,
        real_results_path=args.real_results,
        additional_real_results_paths=args.additional_real_results,
        output_dir=args.output_dir,
    )
    print(Path(report["reports"]["markdown"]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
