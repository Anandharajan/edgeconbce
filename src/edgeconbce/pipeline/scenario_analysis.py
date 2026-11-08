from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from edgeconbce.artifacts import write_json_artifact
from edgeconbce.metrics.ranking import auc_score, average_precision, summarize_binary_metrics
from edgeconbce.pipeline.real_benchmark import _policy_data_movement_summary
from edgeconbce.pipeline.repeated_eval import (
    HIGHER_IS_BETTER_METRICS,
    SUMMARY_BOOTSTRAP_REPEATS,
    SUMMARY_CONFIDENCE_LEVEL,
    SUMMARY_METRICS,
    _comparison_significance_rows,
    _metric_summary,
)
from edgeconbce.pipeline.report import write_csv, write_markdown
from edgeconbce.utils.env import system_metadata


SCENARIO_METRICS = SUMMARY_METRICS
# The scenario layer is intended to be runnable over archived BP3 artifacts without
# another multi-minute model pass, so keep the nested sequence bootstrap modest by default.
PAIRWISE_BOOTSTRAP_REPEATS = 100


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    grid_intensity_multiplier: float = 1.0
    additional_edge_latency_ms: float = 0.0
    edge_available: bool = True


def _default_scenarios() -> tuple[ScenarioSpec, ...]:
    return (
        ScenarioSpec(name="baseline", description="Original benchmark configuration."),
        ScenarioSpec(
            name="low_grid_0p5x",
            description="Operational carbon recomputed at 0.5x the configured grid intensity.",
            grid_intensity_multiplier=0.5,
        ),
        ScenarioSpec(
            name="high_grid_1p5x",
            description="Operational carbon recomputed at 1.5x the configured grid intensity.",
            grid_intensity_multiplier=1.5,
        ),
        ScenarioSpec(
            name="low_network_plus_5ms",
            description="Adds 5 ms overhead to each edge-routed sequence.",
            additional_edge_latency_ms=5.0,
        ),
        ScenarioSpec(
            name="high_network_plus_50ms",
            description="Adds 50 ms overhead to each edge-routed sequence.",
            additional_edge_latency_ms=50.0,
        ),
        ScenarioSpec(
            name="edge_unavailable",
            description="Forces local fallback when edge execution is unavailable.",
            edge_available=False,
        ),
    )


def _available_policy_names(run_payload: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(policy_name) for policy_name in run_payload["external_test"].keys())


def _analysis_comparison_pairs(policy_names: tuple[str, ...]) -> list[tuple[str, str]]:
    policies = set(policy_names)
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
    return pairs


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


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _local_embodied_gco2e(config: dict[str, Any]) -> float:
    local = config["local_device"]
    return float((float(local["embodied_kgco2e_total"]) / int(local["amortization_requests"])) * 0.20 * 1000.0)


def _edge_embodied_gco2e(config: dict[str, Any], *, edge_uses_local_score: bool) -> float:
    edge = config["edge_device"]
    edge_embodied = float((float(edge["embodied_kgco2e_total"]) / int(edge["amortization_requests"])) * 0.65 * 1000.0)
    if not edge_uses_local_score:
        return edge_embodied
    return edge_embodied + _local_embodied_gco2e(config)


def _operational_gco2e(latency_ms: float, power_w: float, grid_intensity_gco2_per_kwh: float) -> float:
    energy_kwh = float(power_w) * (float(latency_ms) / 3_600_000.0) / 1000.0
    return energy_kwh * float(grid_intensity_gco2_per_kwh)


def _local_total_gco2e(sequence_summary: dict[str, Any], config: dict[str, Any], grid_intensity_gco2_per_kwh: float) -> float:
    return _operational_gco2e(
        float(sequence_summary["local_latency_ms"]),
        float(config["local_device"]["operational_power_w"]),
        grid_intensity_gco2_per_kwh,
    ) + _local_embodied_gco2e(config)


def _edge_total_gco2e(
    sequence_summary: dict[str, Any],
    run_payload: dict[str, Any],
    grid_intensity_gco2_per_kwh: float,
) -> float:
    config = run_payload["config"]
    edge_uses_local_score = _edge_uses_local_score(run_payload)
    local_component_ms = float(sequence_summary["local_latency_ms"]) if edge_uses_local_score else 0.0
    raw_edge_ms = float(sequence_summary["edge_latency_ms"]) - local_component_ms
    local_operational = 0.0
    if edge_uses_local_score:
        local_operational = _operational_gco2e(
            local_component_ms,
            float(config["local_device"]["operational_power_w"]),
            grid_intensity_gco2_per_kwh,
        )
    edge_operational = _operational_gco2e(
        raw_edge_ms,
        float(config["edge_device"]["operational_power_w"]),
        grid_intensity_gco2_per_kwh,
    )
    return local_operational + edge_operational + _edge_embodied_gco2e(config, edge_uses_local_score=edge_uses_local_score)


def _edge_uses_local_score(run_payload: dict[str, Any]) -> bool:
    edge_weight = float(run_payload["model_selection"]["edge_optimisation"]["selected_edge_weight"])
    return edge_weight < 0.999


def _policy_thresholds(external_test: dict[str, Any], policy_name: str) -> tuple[float, float]:
    payload = external_test[policy_name]
    if "decision_thresholds" in payload:
        thresholds = payload["decision_thresholds"]
        return float(thresholds["local"]), float(thresholds["edge"])

    family_prefix = "plm_" if policy_name.startswith("plm_") else ""
    local_policy_name = f"{family_prefix}local_only"
    edge_policy_name = f"{family_prefix}edge_only"
    local_payload = external_test.get(local_policy_name, {})
    edge_payload = external_test.get(edge_policy_name, {})
    fallback_threshold = float(payload.get("decision_threshold", 0.5))

    local_threshold = float(local_payload.get("decision_threshold", fallback_threshold))
    edge_threshold = float(edge_payload.get("decision_threshold", fallback_threshold))

    if policy_name == edge_policy_name:
        edge_threshold = float(payload.get("decision_threshold", edge_threshold))
    if policy_name == local_policy_name:
        local_threshold = float(payload.get("decision_threshold", local_threshold))
    return local_threshold, edge_threshold


def _policy_uses_edge(policy_name: str, sequence_summary: dict[str, Any]) -> bool:
    if policy_name in {"edge_only", "plm_edge_only"}:
        return True
    if policy_name in {"local_only", "plm_local_only"}:
        return False
    return bool(sequence_summary.get("route"))


def _policy_sequence_rows(run_payload: dict[str, Any], policy_name: str, scenario: ScenarioSpec) -> list[dict[str, Any]]:
    config = run_payload["config"]
    base_grid = float(config["grid_intensity_gco2_per_kwh"])
    scenario_grid = base_grid * float(scenario.grid_intensity_multiplier)
    external_test = run_payload["external_test"]
    local_threshold, edge_threshold = _policy_thresholds(external_test, policy_name)
    payload = external_test[policy_name]
    edge_uses_local_score = _edge_uses_local_score(run_payload)
    rows: list[dict[str, Any]] = []

    for sequence_summary in payload["per_sequence"]:
        use_edge = _policy_uses_edge(policy_name, sequence_summary)

        if not scenario.edge_available:
            use_edge = False

        if use_edge:
            probabilities = [float(value) for value in sequence_summary["edge_probabilities"]]
            predictions = [bool(float(value) >= edge_threshold) for value in probabilities]
            if policy_name not in {"edge_only", "plm_edge_only"} and not edge_uses_local_score:
                active_latency_ms = (
                    float(sequence_summary["local_latency_ms"])
                    + float(sequence_summary["edge_latency_ms"])
                    + float(scenario.additional_edge_latency_ms)
                )
                active_total_gco2e = (
                    _local_total_gco2e(sequence_summary, config, scenario_grid)
                    + _edge_total_gco2e(sequence_summary, run_payload, scenario_grid)
                )
            else:
                active_latency_ms = float(sequence_summary["edge_latency_ms"]) + float(scenario.additional_edge_latency_ms)
                active_total_gco2e = _edge_total_gco2e(sequence_summary, run_payload, scenario_grid)
            raw_sequence_shared = True
            bytes_transferred = int(sequence_summary["length"])
        else:
            probabilities = [float(value) for value in sequence_summary["local_probabilities"]]
            predictions = [bool(float(value) >= local_threshold) for value in probabilities]
            active_latency_ms = float(sequence_summary["local_latency_ms"])
            active_total_gco2e = _local_total_gco2e(sequence_summary, config, scenario_grid)
            raw_sequence_shared = False
            bytes_transferred = 0

        rows.append(
            {
                "sequence_id": str(sequence_summary["sequence_id"]),
                "length": int(sequence_summary["length"]),
                "labels": [bool(value) for value in sequence_summary["labels"]],
                "probabilities": probabilities,
                "predictions": predictions,
                "active_latency_ms": active_latency_ms,
                "active_total_gco2e": active_total_gco2e,
                "edge_used": use_edge,
                "raw_sequence_shared": raw_sequence_shared,
                "bytes_transferred": bytes_transferred,
                "route_probability": sequence_summary.get("route_probability"),
            }
        )
    return rows


def _summarize_policy_rows(policy_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [label for row in rows for label in row["labels"]]
    predictions = [prediction for row in rows for prediction in row["predictions"]]
    scores = [score for row in rows for score in row["probabilities"]]
    metrics = summarize_binary_metrics(labels, predictions)
    data_movement = _policy_data_movement_summary(
        [
            {
                "length": row["length"],
                "raw_sequence_shared": row["raw_sequence_shared"],
            }
            for row in rows
        ]
    )
    return {
        "policy": policy_name,
        "accuracy": round(metrics["accuracy"], 6),
        "precision": round(metrics["precision"], 6),
        "recall": round(metrics["recall"], 6),
        "f1": round(metrics["f1"], 6),
        "mcc": round(metrics["mcc"], 6),
        "roc_auc": round(auc_score(labels, scores), 6),
        "average_precision": round(average_precision(labels, scores), 6),
        "mean_latency_ms": round(mean(float(row["active_latency_ms"]) for row in rows), 6),
        "mean_total_gco2e": round(mean(float(row["active_total_gco2e"]) for row in rows), 6),
        "route_rate": round(sum(1 for row in rows if row["edge_used"]) / len(rows), 6),
        "raw_sequence_share_rate": data_movement["raw_sequence_share_rate"],
        "mean_bytes_transferred": data_movement["mean_bytes_transferred"],
    }


def _aggregate_scenario_rows(
    rows: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scenario"]), str(row["policy"])), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for group_index, ((scenario, policy), group_rows) in enumerate(grouped.items()):
        aggregate_row: dict[str, Any] = {"scenario": scenario, "policy": policy, "num_runs": len(group_rows)}
        for metric_index, metric in enumerate(SCENARIO_METRICS):
            values = [float(row[metric]) for row in group_rows if row.get(metric) is not None]
            summary = _metric_summary(
                values,
                repeats=SUMMARY_BOOTSTRAP_REPEATS,
                confidence_level=SUMMARY_CONFIDENCE_LEVEL,
                seed=bootstrap_seed + (group_index * 1000) + metric_index,
            )
            for suffix, value in summary.items():
                aggregate_row[f"{metric}_{suffix}"] = value
        aggregate_rows.append(aggregate_row)
    return aggregate_rows


def _scenario_delta_rows(rows: list[dict[str, Any]], numerator_policy: str, denominator_policy: str) -> list[dict[str, Any]]:
    by_key = {(str(row["scenario"]), int(row["seed"]), str(row["policy"])): row for row in rows}
    scenarios = sorted({str(row["scenario"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    delta_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for seed in seeds:
            left = by_key[(scenario, seed, numerator_policy)]
            right = by_key[(scenario, seed, denominator_policy)]
            delta_row: dict[str, Any] = {
                "scenario": scenario,
                "seed": seed,
                "comparison": f"{numerator_policy}_minus_{denominator_policy}",
            }
            for metric in SCENARIO_METRICS:
                left_value = left.get(metric)
                right_value = right.get(metric)
                delta_row[metric] = None if left_value is None or right_value is None else round(float(left_value) - float(right_value), 6)
            delta_rows.append(delta_row)
    return delta_rows


def _aggregate_scenario_comparisons(delta_rows: list[dict[str, Any]], *, bootstrap_seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in delta_rows:
        grouped.setdefault((str(row["scenario"]), str(row["comparison"])), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for group_index, ((scenario, comparison), group_rows) in enumerate(grouped.items()):
        summary_row: dict[str, Any] = {"scenario": scenario, "comparison": comparison, "num_runs": len(group_rows)}
        for metric_index, metric in enumerate(SCENARIO_METRICS):
            values = [float(row[metric]) for row in group_rows if row.get(metric) is not None]
            summary = _metric_summary(
                values,
                repeats=SUMMARY_BOOTSTRAP_REPEATS,
                confidence_level=SUMMARY_CONFIDENCE_LEVEL,
                seed=bootstrap_seed + (group_index * 1000) + metric_index,
            )
            for suffix, value in summary.items():
                summary_row[f"{metric}_{suffix}"] = value
        summary_rows.append(summary_row)
    return summary_rows


def _scenario_significance_rows(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenario_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in delta_rows:
        grouped.setdefault(str(row["scenario"]), []).append(row)

    for scenario, rows_for_scenario in grouped.items():
        stripped_rows = [{key: value for key, value in row.items() if key != "scenario"} for row in rows_for_scenario]
        for significance in _comparison_significance_rows(stripped_rows):
            scenario_rows.append({"scenario": scenario, **significance})
    return scenario_rows


def _paired_sequence_bootstrap_rows(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    seed: int,
    scenario: str,
    comparison: str,
) -> list[dict[str, Any]]:
    if len(left_rows) != len(right_rows):
        raise ValueError("Paired bootstrap requires equal sequence counts.")
    left_by_sequence = {row["sequence_id"]: row for row in left_rows}
    right_by_sequence = {row["sequence_id"]: row for row in right_rows}
    sequence_ids = sorted(left_by_sequence)
    if sequence_ids != sorted(right_by_sequence):
        raise ValueError("Policy sequence ids do not match for paired bootstrap.")

    rng = np.random.default_rng(seed + 7000)
    delta_samples: dict[str, list[float]] = {metric: [] for metric in SCENARIO_METRICS}
    for _ in range(PAIRWISE_BOOTSTRAP_REPEATS):
        sampled_sequence_ids = [sequence_ids[int(index)] for index in rng.integers(0, len(sequence_ids), size=len(sequence_ids))]
        sampled_left = [left_by_sequence[sequence_id] for sequence_id in sampled_sequence_ids]
        sampled_right = [right_by_sequence[sequence_id] for sequence_id in sampled_sequence_ids]
        left_summary = _summarize_policy_rows("left", sampled_left)
        right_summary = _summarize_policy_rows("right", sampled_right)
        for metric in SCENARIO_METRICS:
            delta_samples[metric].append(float(left_summary[metric]) - float(right_summary[metric]))

    rows: list[dict[str, Any]] = []
    for metric in SCENARIO_METRICS:
        samples = np.asarray(delta_samples[metric], dtype=np.float64)
        ci_low = round(float(np.quantile(samples, 0.025)), 6)
        ci_high = round(float(np.quantile(samples, 0.975)), 6)
        mean_delta = round(float(np.mean(samples)), 6)
        median_delta = round(float(np.median(samples)), 6)
        if metric in HIGHER_IS_BETTER_METRICS:
            probability_numerator_better = round(float(np.mean(samples > 0.0)), 6)
        else:
            probability_numerator_better = round(float(np.mean(samples < 0.0)), 6)
        rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "comparison": comparison,
                "metric": metric,
                "mean_delta": mean_delta,
                "median_delta": median_delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "probability_numerator_better": probability_numerator_better,
            }
        )
    return rows


def _pairwise_bootstrap_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scenario"]), str(row["comparison"]), str(row["metric"])), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (scenario, comparison, metric), group_rows in grouped.items():
        summary_rows.append(
            {
                "scenario": scenario,
                "comparison": comparison,
                "metric": metric,
                "num_runs": len(group_rows),
                "mean_delta_mean": round(mean(float(row["mean_delta"]) for row in group_rows), 6),
                "mean_delta_ci_low_mean": round(mean(float(row["ci_low"]) for row in group_rows), 6),
                "mean_delta_ci_high_mean": round(mean(float(row["ci_high"]) for row in group_rows), 6),
                "probability_numerator_better_mean": round(
                    mean(float(row["probability_numerator_better"]) for row in group_rows),
                    6,
                ),
            }
        )
    return summary_rows


def _scenario_markdown(
    scenario_specs: tuple[ScenarioSpec, ...],
    aggregate_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    significance_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
) -> str:
    aggregate_index = {(row["scenario"], row["policy"]): row for row in aggregate_rows}
    bootstrap_index = {(row["scenario"], row["comparison"], row["metric"]): row for row in bootstrap_rows}
    policy_names = sorted({str(row["policy"]) for row in aggregate_rows}, key=_policy_sort_key)
    comparison_names = sorted({str(row["comparison"]) for row in comparison_rows})
    lines = [
        "# EdgeConBCE Scenario Analysis",
        "",
        "## Scenario Definitions",
        "",
    ]
    for scenario in scenario_specs:
        lines.append(f"- {scenario.name}: {scenario.description}")

    lines.extend(
        [
            "",
            "## Aggregate Scenario Summary",
            "",
            "| scenario | policy | MCC | F1 | latency ms | total gCO2e | route rate | raw share | mean bytes |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in scenario_specs:
        for policy in policy_names:
            row = aggregate_index[(scenario.name, policy)]
            lines.append(
                f"| {scenario.name} | {policy} | {row['mcc_mean']} | {row['f1_mean']} | {row['mean_latency_ms_mean']} | "
                f"{row['mean_total_gco2e_mean']} | {row['route_rate_mean']} | {row['raw_sequence_share_rate_mean']} | "
                f"{row['mean_bytes_transferred_mean']} |"
            )

    lines.extend(["", "## Paired Scenario Comparisons", ""])
    for row in comparison_rows:
        lines.append(
            f"- {row['scenario']} {row['comparison']}: "
            f"mcc={row['mcc_mean']} [{row['mcc_ci_low']}, {row['mcc_ci_high']}], "
            f"latency={row['mean_latency_ms_mean']} [{row['mean_latency_ms_ci_low']}, {row['mean_latency_ms_ci_high']}], "
            f"carbon={row['mean_total_gco2e_mean']} [{row['mean_total_gco2e_ci_low']}, {row['mean_total_gco2e_ci_high']}], "
            f"raw_share={row['raw_sequence_share_rate_mean']} [{row['raw_sequence_share_rate_ci_low']}, {row['raw_sequence_share_rate_ci_high']}]"
        )

    lines.extend(["", "## Paired Sign Tests", ""])
    for row in significance_rows:
        if row["metric"] not in {"mcc", "f1", "mean_latency_ms", "mean_total_gco2e", "raw_sequence_share_rate"}:
            continue
        lines.append(
            f"- {row['scenario']} {row['comparison']} {row['metric']}: "
            f"wins={row['wins']}, losses={row['losses']}, ties={row['ties']}, p={row['p_value_two_sided_sign_test']}, "
            f"direction={row['effect_direction']}"
        )

    if bootstrap_rows:
        lines.extend(["", "## Sequence-Bootstrap Comparison Means", ""])
        for comparison in comparison_names:
            for scenario in scenario_specs:
                mcc = bootstrap_index.get((scenario.name, comparison, "mcc"))
                latency = bootstrap_index.get((scenario.name, comparison, "mean_latency_ms"))
                carbon = bootstrap_index.get((scenario.name, comparison, "mean_total_gco2e"))
                if mcc is None or latency is None or carbon is None:
                    continue
                lines.append(
                    f"- {scenario.name} {comparison}: "
                    f"mcc_delta={mcc['mean_delta_mean']} [{mcc['mean_delta_ci_low_mean']}, {mcc['mean_delta_ci_high_mean']}], "
                    f"latency_delta={latency['mean_delta_mean']} [{latency['mean_delta_ci_low_mean']}, {latency['mean_delta_ci_high_mean']}], "
                    f"carbon_delta={carbon['mean_delta_mean']} [{carbon['mean_delta_ci_low_mean']}, {carbon['mean_delta_ci_high_mean']}]"
                )
    return "\n".join(lines)


def generate_scenario_analysis(
    *,
    repeated_results_path: str | Path,
    output_dir: str | Path,
    include_sequence_bootstrap: bool = False,
) -> dict[str, Any]:
    repeated_payload = _load_json(repeated_results_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scenario_specs = _default_scenarios()
    policy_names = _available_policy_names(_load_json(repeated_payload["run_artifacts"][0]["output_json_path"]))
    comparison_pairs = _analysis_comparison_pairs(policy_names)

    per_seed_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for artifact in repeated_payload["run_artifacts"]:
        run_payload = _load_json(artifact["output_json_path"])
        seed = int(artifact["seed"])
        for scenario in scenario_specs:
            policy_rows = {
                policy_name: _policy_sequence_rows(run_payload, policy_name, scenario)
                for policy_name in policy_names
            }
            for policy_name, rows in policy_rows.items():
                per_seed_rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario.name,
                        **_summarize_policy_rows(policy_name, rows),
                    }
                )
            if include_sequence_bootstrap:
                for numerator_policy, denominator_policy in comparison_pairs:
                    bootstrap_rows.extend(
                        _paired_sequence_bootstrap_rows(
                            policy_rows[numerator_policy],
                            policy_rows[denominator_policy],
                            seed=seed,
                            scenario=scenario.name,
                            comparison=f"{numerator_policy}_minus_{denominator_policy}",
                        )
                    )

    aggregate_rows = _aggregate_scenario_rows(per_seed_rows, bootstrap_seed=1101)
    all_delta_rows: list[dict[str, Any]] = []
    for numerator_policy, denominator_policy in comparison_pairs:
        all_delta_rows.extend(_scenario_delta_rows(per_seed_rows, numerator_policy, denominator_policy))
    comparison_rows = _aggregate_scenario_comparisons(all_delta_rows, bootstrap_seed=2202)
    significance_rows = _scenario_significance_rows(all_delta_rows)
    bootstrap_summary_rows = _pairwise_bootstrap_summary(bootstrap_rows) if bootstrap_rows else []

    per_seed_csv = output_path / "scenario_per_seed_policy_metrics.csv"
    aggregate_csv = output_path / "scenario_aggregate_policy_metrics.csv"
    comparison_csv = output_path / "scenario_comparison_summary.csv"
    significance_csv = output_path / "scenario_significance.csv"
    markdown_path = output_path / "scenario_analysis.md"
    json_path = output_path / "scenario_analysis.json"

    write_csv(per_seed_csv, list(per_seed_rows[0].keys()), per_seed_rows)
    write_csv(aggregate_csv, list(aggregate_rows[0].keys()), aggregate_rows)
    write_csv(comparison_csv, list(comparison_rows[0].keys()), comparison_rows)
    write_csv(significance_csv, list(significance_rows[0].keys()), significance_rows)
    write_markdown(
        markdown_path,
        _scenario_markdown(scenario_specs, aggregate_rows, comparison_rows, significance_rows, bootstrap_summary_rows),
    )
    bootstrap_csv = None
    bootstrap_summary_csv = None
    if bootstrap_rows:
        bootstrap_csv = output_path / "scenario_sequence_bootstrap.csv"
        bootstrap_summary_csv = output_path / "scenario_sequence_bootstrap_summary.csv"
        write_csv(bootstrap_csv, list(bootstrap_rows[0].keys()), bootstrap_rows)
        write_csv(bootstrap_summary_csv, list(bootstrap_summary_rows[0].keys()), bootstrap_summary_rows)

    payload = {
        "project": "edgecon-bce",
        "analysis": "scenario_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "source_repeated_results_path": str(Path(repeated_results_path)),
        "scenario_definitions": [
            {
                "name": scenario.name,
                "description": scenario.description,
                "grid_intensity_multiplier": scenario.grid_intensity_multiplier,
                "additional_edge_latency_ms": scenario.additional_edge_latency_ms,
                "edge_available": scenario.edge_available,
            }
            for scenario in scenario_specs
        ],
        "summary_statistics": {
            "confidence_level": SUMMARY_CONFIDENCE_LEVEL,
            "bootstrap_repeats": SUMMARY_BOOTSTRAP_REPEATS,
            "sequence_bootstrap_repeats": PAIRWISE_BOOTSTRAP_REPEATS if include_sequence_bootstrap else 0,
            "sequence_bootstrap_enabled": include_sequence_bootstrap,
        },
        "per_seed_policy_metrics": per_seed_rows,
        "aggregate_policy_metrics": aggregate_rows,
        "paired_delta_summary": comparison_rows,
        "paired_significance": significance_rows,
        "sequence_bootstrap": bootstrap_rows,
        "sequence_bootstrap_summary": bootstrap_summary_rows,
        "reports": {
            "per_seed_csv": str(per_seed_csv),
            "aggregate_csv": str(aggregate_csv),
            "comparison_csv": str(comparison_csv),
            "significance_csv": str(significance_csv),
            "sequence_bootstrap_csv": None if bootstrap_csv is None else str(bootstrap_csv),
            "sequence_bootstrap_summary_csv": None if bootstrap_summary_csv is None else str(bootstrap_summary_csv),
            "summary_markdown": str(markdown_path),
            "summary_json": str(json_path),
        },
    }
    write_json_artifact(json_path, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate scenario and robustness analysis from repeated EdgeConBCE BP3 artifacts.")
    parser.add_argument(
        "--repeated-results",
        default="outputs/bp3_repeated_eval/bp3_real_multiseed_aggregate_results.json",
        help="Path to repeated-eval aggregate JSON with run_artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/scenario_analysis",
        help="Directory for generated scenario analysis outputs.",
    )
    parser.add_argument(
        "--with-sequence-bootstrap",
        action="store_true",
        help="Enable slower per-sequence paired bootstrap comparisons in addition to repeated-run summaries.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    results = generate_scenario_analysis(
        repeated_results_path=args.repeated_results,
        output_dir=args.output_dir,
        include_sequence_bootstrap=bool(args.with_sequence_bootstrap),
    )
    print(Path(results["reports"]["summary_markdown"]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
