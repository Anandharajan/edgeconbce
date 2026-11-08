from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from edgeconbce.artifacts import write_json_artifact
from edgeconbce.pipeline.real_benchmark import build_legacy_bp3_config, run_legacy_bp3_real_v3
from edgeconbce.pipeline.report import write_csv, write_markdown
from edgeconbce.schemas import RepeatedBP3EvalConfig
from edgeconbce.utils.env import system_metadata


SUMMARY_METRICS = [
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
SUMMARY_STAT_SUFFIXES = ("mean", "std", "ci_low", "ci_high")
SUMMARY_CONFIDENCE_LEVEL = 0.95
SUMMARY_BOOTSTRAP_REPEATS = 2000
HIGHER_IS_BETTER_METRICS = {"mcc", "f1", "roc_auc", "average_precision", "route_rate"}
COMPARISON_SIGNIFICANCE_METRICS = [
    "mcc",
    "f1",
    "roc_auc",
    "average_precision",
    "mean_latency_ms",
    "mean_total_gco2e",
    "raw_sequence_share_rate",
    "mean_bytes_transferred",
]


def _policy_summary_rows(seed: int, results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_name, payload in results["external_test"].items():
        summary = payload["summary"]
        row = {"seed": seed, "policy": policy_name}
        for metric in SUMMARY_METRICS:
            row[metric] = summary.get(metric)
        rows.append(row)
    return rows


def _bootstrap_mean_interval(
    values: list[float],
    *,
    repeats: int,
    confidence_level: float,
    seed: int,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if repeats <= 0:
        raise ValueError("Bootstrap repeats must be positive.")
    if len(values) == 1:
        value = round(float(values[0]), 6)
        return value, value

    samples = np.asarray(values, dtype=np.float64)
    sample_size = len(samples)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        bootstrap_sample = samples[rng.integers(0, sample_size, size=sample_size)]
        bootstrap_means[index] = float(np.mean(bootstrap_sample))

    alpha = 1.0 - confidence_level
    lower = round(float(np.quantile(bootstrap_means, alpha / 2.0)), 6)
    upper = round(float(np.quantile(bootstrap_means, 1.0 - (alpha / 2.0))), 6)
    return lower, upper


def _metric_summary(
    values: list[float],
    *,
    repeats: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "ci_low": None, "ci_high": None}

    ci_low, ci_high = _bootstrap_mean_interval(
        values,
        repeats=repeats,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {
        "mean": round(mean(values), 6),
        "std": round(pstdev(values), 6) if len(values) > 1 else 0.0,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _aggregate_policy_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_repeats: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["policy"]), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for policy_index, (policy, policy_rows) in enumerate(grouped.items()):
        aggregate_row: dict[str, Any] = {"policy": policy, "num_runs": len(policy_rows)}
        for metric_index, metric in enumerate(SUMMARY_METRICS):
            values = [float(row[metric]) for row in policy_rows if row.get(metric) is not None]
            summary = _metric_summary(
                values,
                repeats=bootstrap_repeats,
                confidence_level=confidence_level,
                seed=bootstrap_seed + (policy_index * 1000) + metric_index,
            )
            for suffix, value in summary.items():
                aggregate_row[f"{metric}_{suffix}"] = value
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _paired_delta_rows(rows: list[dict[str, Any]], numerator_policy: str, denominator_policy: str) -> list[dict[str, Any]]:
    by_seed_policy = {(int(row["seed"]), str(row["policy"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    delta_rows: list[dict[str, Any]] = []
    for seed in seeds:
        left = by_seed_policy[(seed, numerator_policy)]
        right = by_seed_policy[(seed, denominator_policy)]
        delta_row: dict[str, Any] = {"seed": seed, "comparison": f"{numerator_policy}_minus_{denominator_policy}"}
        for metric in SUMMARY_METRICS:
            left_value = left.get(metric)
            right_value = right.get(metric)
            delta_row[metric] = None if left_value is None or right_value is None else round(float(left_value) - float(right_value), 6)
        delta_rows.append(delta_row)
    return delta_rows


def _aggregate_comparison_rows(
    delta_rows: list[dict[str, Any]],
    *,
    bootstrap_repeats: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in delta_rows:
        grouped.setdefault(str(row["comparison"]), []).append(row)

    comparison_rows: list[dict[str, Any]] = []
    for comparison_index, (comparison, rows_for_comparison) in enumerate(grouped.items()):
        comparison_row: dict[str, Any] = {"comparison": comparison, "num_runs": len(rows_for_comparison)}
        for metric_index, metric in enumerate(SUMMARY_METRICS):
            values = [float(row[metric]) for row in rows_for_comparison if row.get(metric) is not None]
            summary = _metric_summary(
                values,
                repeats=bootstrap_repeats,
                confidence_level=confidence_level,
                seed=bootstrap_seed + (comparison_index * 1000) + metric_index,
            )
            for suffix, value in summary.items():
                comparison_row[f"{metric}_{suffix}"] = value
        comparison_rows.append(comparison_row)
    return comparison_rows


def _comparison_pairs(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    policies = {str(row["policy"]) for row in rows}
    pairs: list[tuple[str, str]] = []
    if {"adaptive_split", "local_only"} <= policies:
        pairs.append(("adaptive_split", "local_only"))
    if {"adaptive_split", "edge_only"} <= policies:
        pairs.append(("adaptive_split", "edge_only"))
    if {"adaptive_carbon_guard", "local_only"} <= policies:
        pairs.append(("adaptive_carbon_guard", "local_only"))
    if {"adaptive_carbon_guard", "edge_only"} <= policies:
        pairs.append(("adaptive_carbon_guard", "edge_only"))
    if {"adaptive_carbon_guard", "adaptive_split"} <= policies:
        pairs.append(("adaptive_carbon_guard", "adaptive_split"))
    if {"plm_adaptive_split", "plm_local_only"} <= policies:
        pairs.append(("plm_adaptive_split", "plm_local_only"))
    if {"plm_adaptive_split", "plm_edge_only"} <= policies:
        pairs.append(("plm_adaptive_split", "plm_edge_only"))
    if {"plm_region_constrained_edge", "plm_edge_only"} <= policies:
        pairs.append(("plm_region_constrained_edge", "plm_edge_only"))
    if {"plm_region_constrained_edge", "plm_local_only"} <= policies:
        pairs.append(("plm_region_constrained_edge", "plm_local_only"))
    if {"plm_adaptive_conformal_guard", "plm_local_only"} <= policies:
        pairs.append(("plm_adaptive_conformal_guard", "plm_local_only"))
    if {"plm_adaptive_conformal_guard", "plm_edge_only"} <= policies:
        pairs.append(("plm_adaptive_conformal_guard", "plm_edge_only"))
    if {"plm_adaptive_conformal_guard", "plm_adaptive_split"} <= policies:
        pairs.append(("plm_adaptive_conformal_guard", "plm_adaptive_split"))
    if {"plm_local_only", "local_only"} <= policies:
        pairs.append(("plm_local_only", "local_only"))
    if {"plm_edge_only", "edge_only"} <= policies:
        pairs.append(("plm_edge_only", "edge_only"))
    if {"plm_adaptive_conformal_guard", "adaptive_carbon_guard"} <= policies:
        pairs.append(("plm_adaptive_conformal_guard", "adaptive_carbon_guard"))
    if {"plm_adaptive_conformal_guard", "adaptive_split"} <= policies:
        pairs.append(("plm_adaptive_conformal_guard", "adaptive_split"))
    return pairs


def _two_sided_sign_test_p_value(wins: int, losses: int) -> float:
    decisive_pairs = wins + losses
    if decisive_pairs == 0:
        return 1.0
    threshold = max(wins, losses)
    tail_probability = sum(comb(decisive_pairs, index) for index in range(threshold, decisive_pairs + 1)) / (2 ** decisive_pairs)
    return round(min(1.0, 2.0 * tail_probability), 6)


def _comparison_significance_rows(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in delta_rows:
        grouped.setdefault(str(row["comparison"]), []).append(row)

    significance_rows: list[dict[str, Any]] = []
    for comparison, rows_for_comparison in grouped.items():
        for metric in COMPARISON_SIGNIFICANCE_METRICS:
            direction_multiplier = 1.0 if metric in HIGHER_IS_BETTER_METRICS else -1.0
            transformed_values: list[float] = []
            for row in rows_for_comparison:
                value = row.get(metric)
                if value is None:
                    continue
                transformed_values.append(direction_multiplier * float(value))

            wins = sum(1 for value in transformed_values if value > 0.0)
            losses = sum(1 for value in transformed_values if value < 0.0)
            ties = sum(1 for value in transformed_values if value == 0.0)
            decisive_pairs = wins + losses
            win_rate = None if decisive_pairs == 0 else round(wins / decisive_pairs, 6)
            effect_direction = "tie"
            if wins > losses:
                effect_direction = "favors_numerator"
            elif losses > wins:
                effect_direction = "favors_denominator"

            raw_values = [float(row[metric]) for row in rows_for_comparison if row.get(metric) is not None]
            significance_rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "higher_is_better": metric in HIGHER_IS_BETTER_METRICS,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "decisive_pairs": decisive_pairs,
                    "win_rate_decisive": win_rate,
                    "mean_delta": None if not raw_values else round(mean(raw_values), 6),
                    "median_delta": None if not raw_values else round(float(np.median(np.asarray(raw_values, dtype=np.float64))), 6),
                    "p_value_two_sided_sign_test": _two_sided_sign_test_p_value(wins, losses),
                    "effect_direction": effect_direction,
                }
            )
    return significance_rows


def _format_interval(row: dict[str, Any], metric: str) -> str:
    mean_value = row.get(f"{metric}_mean")
    if mean_value is None:
        return "n/a"
    ci_low = row.get(f"{metric}_ci_low")
    ci_high = row.get(f"{metric}_ci_high")
    if ci_low is None or ci_high is None:
        return str(mean_value)
    return f"{mean_value} [{ci_low}, {ci_high}]"


def _markdown_summary(
    config: RepeatedBP3EvalConfig,
    aggregate_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    significance_rows: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {config.run_name}",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Seeds: {', '.join(str(seed) for seed in config.seeds)}",
        "",
        f"Aggregate statistics: mean with {int(SUMMARY_CONFIDENCE_LEVEL * 100)}% bootstrap confidence intervals over seeds ({SUMMARY_BOOTSTRAP_REPEATS} repeats)",
        "",
        "## Aggregate Policy Metrics",
        "",
        "| policy | mcc | f1 | roc_auc | avg_precision | latency_ms | total_gco2e | route_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['policy']} | {_format_interval(row, 'mcc')} | {_format_interval(row, 'f1')} | {_format_interval(row, 'roc_auc')} | "
            f"{_format_interval(row, 'average_precision')} | {_format_interval(row, 'mean_latency_ms')} | {_format_interval(row, 'mean_total_gco2e')} | "
            f"{_format_interval(row, 'route_rate')} |"
        )
    lines.extend(
        [
            "",
            "## Data Movement",
            "",
            "| policy | raw_sequence_share_rate | mean_bytes_transferred |",
            "|---|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            f"| {row['policy']} | {_format_interval(row, 'raw_sequence_share_rate')} | {_format_interval(row, 'mean_bytes_transferred')} |"
        )
    lines.extend(["", "## Paired Delta Summaries", ""])
    for row in comparison_rows:
        lines.append(
            f"- {row['comparison']}: "
            f"mcc={_format_interval(row, 'mcc')}, f1={_format_interval(row, 'f1')}, "
            f"roc_auc={_format_interval(row, 'roc_auc')}, avg_precision={_format_interval(row, 'average_precision')}, "
            f"latency_ms={_format_interval(row, 'mean_latency_ms')}, total_gco2e={_format_interval(row, 'mean_total_gco2e')}, "
            f"raw_share={_format_interval(row, 'raw_sequence_share_rate')}, bytes={_format_interval(row, 'mean_bytes_transferred')}"
        )
    lines.extend(["", "## Paired Sign Tests", ""])
    for row in significance_rows:
        lines.append(
            f"- {row['comparison']} {row['metric']}: wins={row['wins']}, losses={row['losses']}, ties={row['ties']}, "
            f"win_rate={row['win_rate_decisive']}, p={row['p_value_two_sided_sign_test']}, direction={row['effect_direction']}"
        )
    lines.extend(
        [
            "",
            "## Measurement Provenance",
            "",
            "- latency: measured from benchmark execution timings inside the legacy runner",
            "- energy: estimated from fixed operational power assumptions and latency",
            "- carbon: estimated from energy and configured grid intensity",
            "- embodied carbon: amortized estimate from configured device metadata",
        ]
    )
    return "\n".join(lines)


def run_repeated_bp3_eval(config: RepeatedBP3EvalConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []
    run_artifacts: list[dict[str, Any]] = []
    for seed in config.seeds:
        per_seed_output = output_dir / "runs" / f"{config.run_name}_seed_{seed}.json"
        single_run_config = build_legacy_bp3_config(config, seed=seed, output_json_path=str(per_seed_output))
        results = run_legacy_bp3_real_v3(single_run_config)
        per_seed_rows.extend(_policy_summary_rows(seed, results))
        run_artifacts.append(
            {
                "seed": seed,
                "output_json_path": str(per_seed_output),
                "run_id": results.get("run_id"),
            }
        )

    aggregate_rows = _aggregate_policy_rows(
        per_seed_rows,
        bootstrap_repeats=SUMMARY_BOOTSTRAP_REPEATS,
        confidence_level=SUMMARY_CONFIDENCE_LEVEL,
        bootstrap_seed=(sum(config.seeds) * 17) + 101,
    )
    comparison_pairs = _comparison_pairs(per_seed_rows)
    paired_delta_map: dict[str, list[dict[str, Any]]] = {}
    all_delta_rows: list[dict[str, Any]] = []
    for numerator_policy, denominator_policy in comparison_pairs:
        delta_rows = _paired_delta_rows(per_seed_rows, numerator_policy, denominator_policy)
        paired_delta_map[f"{numerator_policy}_minus_{denominator_policy}"] = delta_rows
        all_delta_rows.extend(delta_rows)
    comparison_rows = _aggregate_comparison_rows(
        all_delta_rows,
        bootstrap_repeats=SUMMARY_BOOTSTRAP_REPEATS,
        confidence_level=SUMMARY_CONFIDENCE_LEVEL,
        bootstrap_seed=(sum(config.seeds) * 17) + 404,
    )
    significance_rows = _comparison_significance_rows(all_delta_rows)

    per_seed_csv = output_dir / f"{config.run_name}_per_seed_policy_metrics.csv"
    aggregate_csv = output_dir / f"{config.run_name}_aggregate_policy_metrics.csv"
    comparison_csv = output_dir / f"{config.run_name}_paired_delta_summary.csv"
    significance_csv = output_dir / f"{config.run_name}_paired_significance.csv"
    markdown_path = output_dir / f"{config.run_name}_summary.md"
    json_path = output_dir / f"{config.run_name}_aggregate_results.json"

    write_csv(per_seed_csv, ["seed", "policy", *SUMMARY_METRICS], per_seed_rows)
    write_csv(
        aggregate_csv,
        [
            "policy",
            "num_runs",
            *[f"{metric}_{suffix}" for metric in SUMMARY_METRICS for suffix in SUMMARY_STAT_SUFFIXES],
        ],
        aggregate_rows,
    )
    write_csv(
        comparison_csv,
        [
            "comparison",
            "num_runs",
            *[f"{metric}_{suffix}" for metric in SUMMARY_METRICS for suffix in SUMMARY_STAT_SUFFIXES],
        ],
        comparison_rows,
    )
    write_csv(
        significance_csv,
        [
            "comparison",
            "metric",
            "higher_is_better",
            "wins",
            "losses",
            "ties",
            "decisive_pairs",
            "win_rate_decisive",
            "mean_delta",
            "median_delta",
            "p_value_two_sided_sign_test",
            "effect_direction",
        ],
        significance_rows,
    )
    write_markdown(markdown_path, _markdown_summary(config, aggregate_rows, comparison_rows, significance_rows))

    aggregate_payload = {
        "project": "edgecon-bce",
        "mode": config.mode,
        "run_name": config.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "config": asdict(config),
        "measurement_provenance": {
            "latency": "timed benchmark execution",
            "energy": "estimated_from_fixed_power_and_latency",
            "operational_carbon": "estimated_from_energy_and_grid_intensity",
            "embodied_carbon": "amortized_estimate_from_device_metadata",
        },
        "summary_statistics": {
            "unit_of_analysis": "per-seed repeated benchmark run",
            "estimator": "mean",
            "confidence_level": SUMMARY_CONFIDENCE_LEVEL,
            "confidence_interval_method": "percentile_bootstrap",
            "bootstrap_repeats": SUMMARY_BOOTSTRAP_REPEATS,
        },
        "comparison_statistics": {
            "test": "two_sided_sign_test",
            "unit_of_analysis": "paired per-seed policy delta",
            "metric_directionality": {
                "higher_is_better": sorted(HIGHER_IS_BETTER_METRICS),
                "lower_is_better": sorted(metric for metric in COMPARISON_SIGNIFICANCE_METRICS if metric not in HIGHER_IS_BETTER_METRICS),
            },
        },
        "run_artifacts": run_artifacts,
        "per_seed_policy_metrics": per_seed_rows,
        "aggregate_policy_metrics": aggregate_rows,
        "paired_delta_summary": comparison_rows,
        "paired_significance": significance_rows,
        "paired_deltas": paired_delta_map,
        "reports": {
            "per_seed_csv": str(per_seed_csv),
            "aggregate_csv": str(aggregate_csv),
            "paired_delta_summary_csv": str(comparison_csv),
            "paired_significance_csv": str(significance_csv),
            "summary_markdown": str(markdown_path),
        },
    }
    write_json_artifact(json_path, aggregate_payload)
    return aggregate_payload
