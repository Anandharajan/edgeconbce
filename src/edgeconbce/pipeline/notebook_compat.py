from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

from edgeconbce.artifacts import write_json_artifact
from edgeconbce.backends.edge_proxy import edge_model_probability
from edgeconbce.backends.local_heuristic import local_model_probability
from edgeconbce.data.synthetic import build_dataset
from edgeconbce.features.pacvf import pacvf_like_features
from edgeconbce.features.structure_proxy import structure_proxy_features
from edgeconbce.metrics.carbon import (
    edge_latency_ms,
    embodied_gco2e_per_request,
    energy_kwh,
    local_latency_ms,
    operational_gco2e,
    split_latency_ms,
)
from edgeconbce.metrics.summary import quality_score, summarize_policy
from edgeconbce.policies.router import choose_route
from edgeconbce.schemas import NotebookCompatConfig, SampleRecord, validate_notebook_bundle
from edgeconbce.utils.env import system_metadata
from edgeconbce.utils.hashing import sample_fingerprint


def evaluate_sample(sample, policy: str, request_index: int, config: NotebookCompatConfig) -> SampleRecord:
    features = pacvf_like_features(sample.sequence, config.max_sequence_length)
    proxies = structure_proxy_features(sample.sequence, config.max_sequence_length)
    local_probability = local_model_probability(features)
    edge_probability = edge_model_probability(features, proxies, local_probability)
    route = choose_route(policy, sample, local_probability, edge_probability, features, proxies, config)
    sequence_length = len(sample.sequence)
    if route == "local":
        latency_ms = local_latency_ms(sequence_length)
        energy = energy_kwh(config.local_device.operational_power_w, latency_ms)
        embodied = embodied_gco2e_per_request(config.local_device)
        final_probability = local_probability
        raw_sequence_shared = False
    elif route == "edge_full":
        latency_ms = edge_latency_ms(sequence_length)
        energy = energy_kwh(config.edge_device.operational_power_w, latency_ms)
        embodied = embodied_gco2e_per_request(config.edge_device)
        final_probability = edge_probability
        raw_sequence_shared = True
    elif route == "split_to_edge":
        local_stage_ms = 16.0 + (0.15 * sequence_length)
        edge_stage_ms = split_latency_ms(sequence_length) - local_stage_ms
        latency_ms = split_latency_ms(sequence_length)
        energy = (
            energy_kwh(config.local_device.operational_power_w, local_stage_ms)
            + energy_kwh(config.edge_device.operational_power_w, edge_stage_ms)
        )
        embodied = (
            0.20 * embodied_gco2e_per_request(config.local_device)
            + 0.65 * embodied_gco2e_per_request(config.edge_device)
        )
        final_probability = edge_probability
        raw_sequence_shared = not config.privacy_mode
    else:
        raise ValueError(f"Unsupported route: {route}")
    operational = operational_gco2e(energy, config.grid_intensity_gco2_per_kwh)
    total_gco2e = operational + embodied
    prediction = final_probability >= config.decision_threshold
    return SampleRecord(
        request_id=f"req_{request_index:04d}",
        sample_id=sample.sample_id,
        mode=sample.mode,
        sequence_length=sequence_length,
        policy=policy,
        route=route,
        local_probability=round(local_probability, 6),
        final_probability=round(final_probability, 6),
        prediction=prediction,
        label=sample.label,
        correct=prediction == sample.label,
        quality_score=round(quality_score(final_probability, sample.label), 6),
        latency_ms=round(latency_ms, 3),
        energy_kwh=round(energy, 9),
        operational_gco2e=round(operational, 6),
        embodied_gco2e=round(embodied, 6),
        total_gco2e=round(total_gco2e, 6),
        raw_sequence_shared=raw_sequence_shared,
    )


def run_benchmark(config: NotebookCompatConfig) -> dict:
    dataset = build_dataset(config)
    positives = sum(sample.label for sample in dataset)
    policy_results: list[dict] = []
    for policy in config.policies:
        records = [evaluate_sample(sample, policy, index, config) for index, sample in enumerate(dataset, start=1)]
        summary = summarize_policy(policy, records)
        policy_results.append(
            {
                "policy": policy,
                "summary": asdict(summary),
                "records": [asdict(record) for record in records],
            }
        )
    result_bundle = {
        "artifact_version": config.artifact_version,
        "benchmark_mode": config.mode,
        "run_id": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ_edgeconbce_notebook"),
        "project": "edgecon-bce",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": system_metadata(),
        "measurement_provenance": {
            "latency": "estimated_from_analytic_latency_models",
            "energy": "estimated_from_fixed_power_and_latency",
            "operational_carbon": "estimated_from_energy_and_grid_intensity",
            "embodied_carbon": "amortized_estimate_from_device_metadata",
        },
        "config": asdict(config),
        "dataset": {
            "num_samples": len(dataset),
            "num_positive": positives,
            "num_negative": len(dataset) - positives,
            "sample_fingerprint": sample_fingerprint([sample.sample_id for sample in dataset]),
        },
        "policy_results": policy_results,
    }
    validate_notebook_bundle(result_bundle, config)
    write_json_artifact(config.output_json_path, result_bundle)
    return result_bundle
