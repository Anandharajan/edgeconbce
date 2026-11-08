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
DEFAULT_OUTPUT_DIR = Path("outputs/baseline_comparison")

PUBLISHED_BASELINES = [
    {
        "model": "BepiPred-3.0 FFNN (ESM-1b)",
        "source_type": "published_same_split",
        "dataset": "BP3C50ID external test (15 antigens)",
        "roc_auc": 0.697,
        "mcc": 0.220,
        "f1": 0.401,
        "average_precision": None,
        "latency_ms": None,
        "total_gco2e": None,
        "raw_sequence_share_rate": None,
        "notes": "Published Table 3 comparator on the same BP3C50ID external test.",
        "source_url": "https://storage.prod.researchhub.com/uploads/papers/2023/10/26/2022.07.11.499418.full.pdf",
    },
    {
        "model": "BepiPred-3.0 final FFNN (+ SeqLen)",
        "source_type": "published_same_split",
        "dataset": "BP3C50ID external test (15 antigens)",
        "roc_auc": 0.714,
        "mcc": 0.241,
        "f1": None,
        "average_precision": None,
        "latency_ms": None,
        "total_gco2e": None,
        "raw_sequence_share_rate": None,
        "notes": "Published Table 4 final web-server model on the same BP3C50ID external test.",
        "source_url": "https://storage.prod.researchhub.com/uploads/papers/2023/10/26/2022.07.11.499418.full.pdf",
    },
]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _policy_sort_key(policy_name: str) -> tuple[int, str]:
    order = {
        "local_only": 0,
        "edge_only": 1,
        "adaptive_split": 2,
        "adaptive_carbon_guard": 3,
        "plm_local_only": 4,
        "plm_edge_only": 5,
        "plm_adaptive_split": 6,
        "plm_adaptive_conformal_guard": 7,
    }
    return (order.get(policy_name, 99), policy_name)


def _format_value(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{decimals}f}"


def generate_baseline_comparison(
    *,
    repeated_results_path: str | Path,
    output_dir: str | Path,
    extra_baselines_path: str | Path | list[str | Path] | None = None,
) -> dict[str, Any]:
    repeated_payload = _load_json(repeated_results_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for policy_row in sorted(repeated_payload.get("aggregate_policy_metrics", []), key=lambda row: _policy_sort_key(str(row["policy"]))):
        rows.append(
            {
                "model": str(policy_row["policy"]),
                "source_type": "local_repeated_eval",
                "dataset": "BP3C50ID external test (15 antigens)",
                "roc_auc": policy_row.get("roc_auc_mean"),
                "mcc": policy_row.get("mcc_mean"),
                "f1": policy_row.get("f1_mean"),
                "average_precision": policy_row.get("average_precision_mean"),
                "latency_ms": policy_row.get("mean_latency_ms_mean"),
                "total_gco2e": policy_row.get("mean_total_gco2e_mean"),
                "raw_sequence_share_rate": policy_row.get("raw_sequence_share_rate_mean"),
                "notes": "Repeated-run mean over seeds from this repository.",
                "source_url": None,
            }
        )
    rows.extend(PUBLISHED_BASELINES)
    extra_baseline_paths: list[str | Path] = []
    if extra_baselines_path is not None:
        extra_baseline_paths = extra_baselines_path if isinstance(extra_baselines_path, list) else [extra_baselines_path]
    for extra_path in extra_baseline_paths:
        extra_payload = _load_json(extra_path)
        extra_rows = extra_payload.get("rows", extra_payload if isinstance(extra_payload, list) else [])
        for row in extra_rows:
            rows.append(
                {
                    "model": row["model"],
                    "source_type": row["source_type"],
                    "dataset": row["dataset"],
                    "roc_auc": row.get("roc_auc"),
                    "mcc": row.get("mcc"),
                    "f1": row.get("f1"),
                    "average_precision": row.get("average_precision"),
                    "latency_ms": row.get("latency_ms"),
                    "total_gco2e": row.get("total_gco2e"),
                    "raw_sequence_share_rate": row.get("raw_sequence_share_rate"),
                    "notes": row.get("notes", "Imported external comparator row."),
                    "source_url": row.get("source_url"),
                }
            )

    has_local_rerun = any(row["source_type"] == "local_rerun" for row in rows)
    has_fusion = any(row["source_type"] == "local_rerun_fusion" for row in rows)
    csv_path = output_path / "baseline_comparison.csv"
    markdown_path = output_path / "baseline_comparison.md"
    json_path = output_path / "baseline_comparison.json"

    write_csv(csv_path, list(rows[0].keys()) if rows else ["model"], rows)
    write_markdown(
        markdown_path,
        "\n".join(
            [
                "# BP3 Baseline Comparison",
                "",
                "Published BepiPred-3.0 comparators are reported on the same BP3C50ID external test split, but they are literature values rather than local reruns.",
                "",
                "| model | source | ROC AUC | MCC | F1 | AP | latency ms | total gCO2e | raw share |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
                *[
                    f"| {row['model']} | {row['source_type']} | {_format_value(row['roc_auc'])} | {_format_value(row['mcc'])} | {_format_value(row['f1'])} | {_format_value(row['average_precision'])} | {_format_value(row['latency_ms'])} | {_format_value(row['total_gco2e'])} | {_format_value(row['raw_sequence_share_rate'])} |"
                    for row in rows
                ],
                "",
                "## Notes",
                "",
                "- Published BepiPred-3.0 values come from the primary BP3C50ID external-test tables in the BepiPred-3.0 paper.",
                (
                    "- Rows marked `local_rerun` were imported from downloaded official/service outputs on the same BP3C50ID external-test FASTA."
                    if has_local_rerun
                    else "- Local rows are repeated-run means from this repository and therefore directly comparable for workflow trade-offs, but not a substitute for rerunning the official executable."
                ),
                (
                    "- Rows marked `local_rerun_fusion` combine same-split local rerun outputs with repository predictions and should be framed as fusion/stacking evidence, not as a standalone predictor."
                    if has_fusion
                    else ""
                ),
            ]
        ),
    )
    payload = {
        "project": "edgecon-bce",
        "analysis": "baseline_comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "source_repeated_results_path": str(Path(repeated_results_path)),
        "source_extra_baselines_path": [str(Path(path)) for path in extra_baseline_paths],
        "rows": rows,
        "reports": {
            "csv": str(csv_path),
            "markdown": str(markdown_path),
            "json": str(json_path),
        },
    }
    write_json_artifact(json_path, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate manuscript baseline comparisons against published BepiPred-3.0 BP3C50ID results.")
    parser.add_argument("--repeated-results", default=str(DEFAULT_REPEATED_RESULTS), help="Path to the repeated-eval aggregate JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated comparison artifacts.")
    parser.add_argument("--extra-baselines", action="append", default=[], help="Optional JSON list or {'rows': [...]} with additional comparator rows. May be repeated.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    reports = generate_baseline_comparison(
        repeated_results_path=args.repeated_results,
        output_dir=args.output_dir,
        extra_baselines_path=args.extra_baselines,
    )
    print(Path(reports["reports"]["markdown"]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
