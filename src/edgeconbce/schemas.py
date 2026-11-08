from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias


PolicyName: TypeAlias = Literal["local_only", "edge_only", "carbon_first", "adaptive_split"]
RunMode: TypeAlias = Literal["notebook_compat", "legacy_bp3_real_v3", "bp3_repeated_eval"]


@dataclass(frozen=True)
class DeviceCarbonConfig:
    name: str
    operational_power_w: float
    embodied_kgco2e_total: float
    amortization_requests: int


@dataclass(frozen=True)
class PolicyWeights:
    uncertainty: float
    sequence_length: float
    contact_complexity: float
    quality_risk: float
    operational_carbon: float
    local_confidence: float


@dataclass(frozen=True)
class NotebookCompatConfig:
    mode: RunMode
    artifact_version: int
    output_json_path: str
    seed: int
    num_samples: int
    min_sequence_length: int
    max_sequence_length: int
    decision_threshold: float
    grid_intensity_gco2_per_kwh: float
    privacy_mode: bool
    policies: tuple[str, ...]
    local_device: DeviceCarbonConfig
    edge_device: DeviceCarbonConfig
    weights: PolicyWeights


@dataclass(frozen=True)
class LegacyBP3RealV3Config:
    mode: RunMode
    output_json_path: str
    seed: int
    train_fasta_path: str
    test_fasta_path: str
    validation_fraction: float
    route_budget_max: float
    threshold_grid: tuple[float, ...]
    route_probability_grid: tuple[float, ...]
    edge_blend_weight_grid: tuple[float, ...]
    smoothing_radius_grid: tuple[int, ...]
    smoothing_alpha_grid: tuple[float, ...]
    local_device: DeviceCarbonConfig
    edge_device: DeviceCarbonConfig
    grid_intensity_gco2_per_kwh: float


@dataclass(frozen=True)
class RepeatedBP3EvalConfig:
    mode: RunMode
    output_dir: str
    run_name: str
    seeds: tuple[int, ...]
    train_fasta_path: str
    test_fasta_path: str
    validation_fraction: float
    route_budget_max: float
    threshold_grid: tuple[float, ...]
    route_probability_grid: tuple[float, ...]
    edge_blend_weight_grid: tuple[float, ...]
    smoothing_radius_grid: tuple[int, ...]
    smoothing_alpha_grid: tuple[float, ...]
    local_device: DeviceCarbonConfig
    edge_device: DeviceCarbonConfig
    grid_intensity_gco2_per_kwh: float


@dataclass(frozen=True)
class SequenceSample:
    sample_id: str
    sequence: str
    label: bool
    hidden_signal: float
    mode: str


@dataclass(frozen=True)
class SampleRecord:
    request_id: str
    sample_id: str
    mode: str
    sequence_length: int
    policy: str
    route: str
    local_probability: float
    final_probability: float
    prediction: bool
    label: bool
    correct: bool
    quality_score: float
    latency_ms: float
    energy_kwh: float
    operational_gco2e: float
    embodied_gco2e: float
    total_gco2e: float
    raw_sequence_shared: bool


@dataclass(frozen=True)
class PolicySummary:
    policy: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float
    auc: float
    map: float
    mean_latency_ms: float
    p95_latency_ms: float
    mean_total_gco2e: float
    edge_route_rate: float
    raw_sequence_share_rate: float


RunConfigLike: TypeAlias = NotebookCompatConfig | LegacyBP3RealV3Config | RepeatedBP3EvalConfig


def validate_notebook_bundle(payload: dict[str, Any], config: NotebookCompatConfig) -> None:
    required_keys = {
        "artifact_version",
        "run_id",
        "project",
        "created_at_utc",
        "system",
        "config",
        "dataset",
        "policy_results",
    }
    missing = required_keys - payload.keys()
    if missing:
        raise ValueError(f"Notebook bundle missing keys: {sorted(missing)}")

    dataset = payload["dataset"]
    if int(dataset["num_samples"]) != config.num_samples:
        raise ValueError("Dataset sample count does not match config.")
    if len(payload["policy_results"]) != len(config.policies):
        raise ValueError("Policy result count does not match config.")

    for policy_result in payload["policy_results"]:
        summary = policy_result["summary"]
        records = policy_result["records"]
        if len(records) != config.num_samples:
            raise ValueError("Record count does not match config sample count.")
        if not (0.0 <= float(summary["auc"]) <= 1.0):
            raise ValueError("AUC is out of range.")
        if not (0.0 <= float(summary["map"]) <= 1.0):
            raise ValueError("Average precision is out of range.")
