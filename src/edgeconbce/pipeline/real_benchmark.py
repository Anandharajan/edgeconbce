from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any
import sys

from edgeconbce import __version__
from edgeconbce.artifacts import write_json_artifact
from edgeconbce.pipeline.bp3_modern_extension import extend_with_plm_policies
from edgeconbce.schemas import LegacyBP3RealV3Config, RepeatedBP3EvalConfig


def _load_legacy_module() -> Any:
    project_root = Path(__file__).resolve().parents[3]
    legacy_path = project_root / "edgeconbce_real_benchmark_v2.py"
    spec = spec_from_file_location("edgeconbce_real_benchmark_v2_legacy", legacy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy benchmark script at {legacy_path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_legacy_bp3_real_v3(config: LegacyBP3RealV3Config) -> dict[str, Any]:
    legacy = _load_legacy_module()
    legacy_config = _legacy_config_from_package_config(legacy, config)
    results = legacy.run_benchmark(legacy_config)
    results = extend_with_plm_policies(results, config)
    augmented_results = _augment_legacy_results(results, config)
    write_json_artifact(config.output_json_path, augmented_results)
    return augmented_results


def _estimate_sequence_transfer_bytes(sequence_summary: dict[str, Any]) -> int:
    if not bool(sequence_summary.get("raw_sequence_shared")):
        return 0
    return int(sequence_summary["length"])


def _policy_data_movement_summary(per_sequence: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_sequence:
        return {
            "transfer_mode": "raw_sequence_only",
            "payload_basis": "1_byte_per_amino_acid_ascii_estimate",
            "sequence_count": 0,
            "shared_sequence_count": 0,
            "raw_sequence_share_rate": 0.0,
            "mean_bytes_transferred": 0.0,
            "total_bytes_transferred": 0,
            "max_bytes_transferred": 0,
        }

    bytes_transferred = [_estimate_sequence_transfer_bytes(sequence_summary) for sequence_summary in per_sequence]
    shared_sequence_count = sum(1 for sequence_summary in per_sequence if bool(sequence_summary.get("raw_sequence_shared")))
    sequence_count = len(per_sequence)
    raw_sequence_share_rate = round(shared_sequence_count / sequence_count, 6)
    mean_bytes_transferred = round(sum(bytes_transferred) / sequence_count, 6)
    return {
        "transfer_mode": "raw_sequence_only",
        "payload_basis": "1_byte_per_amino_acid_ascii_estimate",
        "sequence_count": sequence_count,
        "shared_sequence_count": shared_sequence_count,
        "raw_sequence_share_rate": raw_sequence_share_rate,
        "mean_bytes_transferred": mean_bytes_transferred,
        "total_bytes_transferred": int(sum(bytes_transferred)),
        "max_bytes_transferred": int(max(bytes_transferred)),
    }


def _augment_legacy_results(results: dict[str, Any], config: LegacyBP3RealV3Config) -> dict[str, Any]:
    results["artifact_version"] = 2
    results["benchmark_mode"] = config.mode
    results["generator"] = {
        "package": "edgecon-bce",
        "version": __version__,
        "entrypoint": "edgeconbce.pipeline.real_benchmark.run_legacy_bp3_real_v3",
        "legacy_backend": "edgeconbce_real_benchmark_v2.py",
    }
    results["measurement_provenance"] = {
        "latency": "measured_from_model_prediction_timing",
        "energy": "estimated_from_fixed_power_and_latency",
        "operational_carbon": "estimated_from_energy_and_grid_intensity",
        "embodied_carbon": "amortized_estimate_from_device_metadata",
        "bytes_transferred": "estimated_from_raw_sequence_length_ascii_bytes",
    }
    # The embodied term is charged at a device-utilisation fraction applied in the
    # legacy backend. It dominates every per-request carbon total, so it is
    # recorded here rather than left implicit in the code; see
    # edgeconbce_real_benchmark_v2.py where these values are applied.
    results["embodied_utilisation_fraction"] = {
        "local_device": 0.20,
        "edge_device": 0.65,
        "applies_to": "embodied_gco2e_per_request",
        "source": "literal in edgeconbce_real_benchmark_v2.py; not configurable",
    }

    external_test = results.get("external_test", {})
    for policy_name, payload in external_test.items():
        per_sequence = payload.get("per_sequence", [])
        data_movement = _policy_data_movement_summary(per_sequence)
        payload["data_movement"] = data_movement
        summary = payload.setdefault("summary", {})
        summary["raw_sequence_share_rate"] = data_movement["raw_sequence_share_rate"]
        summary["mean_bytes_transferred"] = data_movement["mean_bytes_transferred"]
        payload["policy"] = policy_name
    return results


def _legacy_config_from_package_config(legacy: Any, config: LegacyBP3RealV3Config) -> Any:
    return legacy.BenchmarkConfig(
        seed=config.seed,
        train_fasta_path=config.train_fasta_path,
        test_fasta_path=config.test_fasta_path,
        output_json_path=config.output_json_path,
        validation_fraction=config.validation_fraction,
        route_budget_max=config.route_budget_max,
        threshold_grid=config.threshold_grid,
        route_probability_grid=config.route_probability_grid,
        edge_blend_weight_grid=config.edge_blend_weight_grid,
        smoothing_radius_grid=config.smoothing_radius_grid,
        smoothing_alpha_grid=config.smoothing_alpha_grid,
        local_device=legacy.DeviceConfig(
            name=config.local_device.name,
            operational_power_w=config.local_device.operational_power_w,
            embodied_kgco2e_total=config.local_device.embodied_kgco2e_total,
            amortization_requests=config.local_device.amortization_requests,
        ),
        edge_device=legacy.DeviceConfig(
            name=config.edge_device.name,
            operational_power_w=config.edge_device.operational_power_w,
            embodied_kgco2e_total=config.edge_device.embodied_kgco2e_total,
            amortization_requests=config.edge_device.amortization_requests,
        ),
        grid_intensity_gco2_per_kwh=config.grid_intensity_gco2_per_kwh,
    )


def build_legacy_bp3_config(
    config: RepeatedBP3EvalConfig,
    *,
    seed: int,
    output_json_path: str,
) -> LegacyBP3RealV3Config:
    return LegacyBP3RealV3Config(
        mode="legacy_bp3_real_v3",
        output_json_path=output_json_path,
        seed=seed,
        train_fasta_path=config.train_fasta_path,
        test_fasta_path=config.test_fasta_path,
        validation_fraction=config.validation_fraction,
        route_budget_max=config.route_budget_max,
        threshold_grid=config.threshold_grid,
        route_probability_grid=config.route_probability_grid,
        edge_blend_weight_grid=config.edge_blend_weight_grid,
        smoothing_radius_grid=config.smoothing_radius_grid,
        smoothing_alpha_grid=config.smoothing_alpha_grid,
        local_device=config.local_device,
        edge_device=config.edge_device,
        grid_intensity_gco2_per_kwh=config.grid_intensity_gco2_per_kwh,
    )
