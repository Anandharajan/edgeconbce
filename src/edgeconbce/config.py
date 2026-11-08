from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from edgeconbce.schemas import (
    DeviceCarbonConfig,
    LegacyBP3RealV3Config,
    NotebookCompatConfig,
    PolicyWeights,
    RepeatedBP3EvalConfig,
    RunConfigLike,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _device_config(raw: dict[str, Any]) -> DeviceCarbonConfig:
    return DeviceCarbonConfig(
        name=str(raw["name"]),
        operational_power_w=float(raw["operational_power_w"]),
        embodied_kgco2e_total=float(raw["embodied_kgco2e_total"]),
        amortization_requests=int(raw["amortization_requests"]),
    )


def _weights_config(raw: dict[str, Any]) -> PolicyWeights:
    return PolicyWeights(
        uncertainty=float(raw["uncertainty"]),
        sequence_length=float(raw["sequence_length"]),
        contact_complexity=float(raw["contact_complexity"]),
        quality_risk=float(raw["quality_risk"]),
        operational_carbon=float(raw["operational_carbon"]),
        local_confidence=float(raw["local_confidence"]),
    )


def _grid(raw: Any) -> tuple[float, ...]:
    if isinstance(raw, list):
        return tuple(float(value) for value in raw)
    if isinstance(raw, dict):
        start = float(raw["start"])
        stop = float(raw["stop"])
        count = int(raw["count"])
        return tuple(round(float(value), 2) for value in np.linspace(start, stop, count))
    raise TypeError(f"Unsupported grid spec: {type(raw)!r}")


def _int_grid(raw: Any) -> tuple[int, ...]:
    if isinstance(raw, list):
        return tuple(int(value) for value in raw)
    raise TypeError(f"Unsupported integer grid spec: {type(raw)!r}")


def load_run_config(path: str | Path) -> RunConfigLike:
    raw = _load_yaml(path)
    mode = str(raw["mode"])
    if mode == "notebook_compat":
        return NotebookCompatConfig(
            mode=mode,
            artifact_version=int(raw.get("artifact_version", 1)),
            output_json_path=str(raw["output_json_path"]),
            seed=int(raw["seed"]),
            num_samples=int(raw["num_samples"]),
            min_sequence_length=int(raw["min_sequence_length"]),
            max_sequence_length=int(raw["max_sequence_length"]),
            decision_threshold=float(raw["decision_threshold"]),
            grid_intensity_gco2_per_kwh=float(raw["grid_intensity_gco2_per_kwh"]),
            privacy_mode=bool(raw["privacy_mode"]),
            policies=tuple(str(policy) for policy in raw["policies"]),
            local_device=_device_config(raw["local_device"]),
            edge_device=_device_config(raw["edge_device"]),
            weights=_weights_config(raw["weights"]),
        )
    if mode == "legacy_bp3_real_v3":
        return LegacyBP3RealV3Config(
            mode=mode,
            output_json_path=str(raw["output_json_path"]),
            seed=int(raw["seed"]),
            train_fasta_path=str(raw["train_fasta_path"]),
            test_fasta_path=str(raw["test_fasta_path"]),
            validation_fraction=float(raw["validation_fraction"]),
            route_budget_max=float(raw["route_budget_max"]),
            threshold_grid=_grid(raw["threshold_grid"]),
            route_probability_grid=_grid(raw["route_probability_grid"]),
            edge_blend_weight_grid=_grid(raw["edge_blend_weight_grid"]),
            smoothing_radius_grid=_int_grid(raw["smoothing_radius_grid"]),
            smoothing_alpha_grid=_grid(raw["smoothing_alpha_grid"]),
            local_device=_device_config(raw["local_device"]),
            edge_device=_device_config(raw["edge_device"]),
            grid_intensity_gco2_per_kwh=float(raw["grid_intensity_gco2_per_kwh"]),
        )
    if mode == "bp3_repeated_eval":
        return RepeatedBP3EvalConfig(
            mode=mode,
            output_dir=str(raw["output_dir"]),
            run_name=str(raw["run_name"]),
            seeds=tuple(int(seed) for seed in raw["seeds"]),
            train_fasta_path=str(raw["train_fasta_path"]),
            test_fasta_path=str(raw["test_fasta_path"]),
            validation_fraction=float(raw["validation_fraction"]),
            route_budget_max=float(raw["route_budget_max"]),
            threshold_grid=_grid(raw["threshold_grid"]),
            route_probability_grid=_grid(raw["route_probability_grid"]),
            edge_blend_weight_grid=_grid(raw["edge_blend_weight_grid"]),
            smoothing_radius_grid=_int_grid(raw["smoothing_radius_grid"]),
            smoothing_alpha_grid=_grid(raw["smoothing_alpha_grid"]),
            local_device=_device_config(raw["local_device"]),
            edge_device=_device_config(raw["edge_device"]),
            grid_intensity_gco2_per_kwh=float(raw["grid_intensity_gco2_per_kwh"]),
        )
    raise ValueError(f"Unsupported config mode: {mode}")
