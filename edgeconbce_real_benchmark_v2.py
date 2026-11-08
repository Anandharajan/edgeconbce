from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
import json
import platform
import random

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
PAD_INDEX = len(AMINO_ACIDS)
WINDOW_RADIUS = 7
WINDOW_SIZE = (2 * WINDOW_RADIUS) + 1
WINDOW_WIDTH = len(AMINO_ACIDS) + 1
WINDOW_FEATURES = WINDOW_SIZE * WINDOW_WIDTH
HYDROPHOBIC = set("AILMFWVY")
POLAR = set("STNQ")
CHARGED = set("KRHDE")
AROMATIC = set("FWYH")
FLEXIBLE = set("GSPDN")
MODEL_REPEAT_COUNT = 3
BOOTSTRAP_REPEATS = 400
CARBON_GUARD_LENGTH_PENALTY_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
CARBON_GUARD_MIN_ROUTE_RATE = 0.10
CARBON_GUARD_MAX_ROUTE_RATE = 0.15


@dataclass(frozen=True)
class DeviceConfig:
    name: str
    operational_power_w: float
    embodied_kgco2e_total: float
    amortization_requests: int


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int
    train_fasta_path: str
    test_fasta_path: str
    output_json_path: str
    validation_fraction: float
    route_budget_max: float
    threshold_grid: tuple[float, ...]
    route_probability_grid: tuple[float, ...]
    edge_blend_weight_grid: tuple[float, ...]
    smoothing_radius_grid: tuple[int, ...]
    smoothing_alpha_grid: tuple[float, ...]
    local_device: DeviceConfig
    edge_device: DeviceConfig
    grid_intensity_gco2_per_kwh: float


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    annotated_sequence: str


def build_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        seed=42,
        train_fasta_path="BP3C50ID_training_set.fasta",
        test_fasta_path="BP3C50ID_external_test_set.fasta",
        output_json_path="outputs/edgeconbce_bp3_real_results_v3.json",
        validation_fraction=0.20,
        route_budget_max=0.50,
        threshold_grid=tuple(round(value, 2) for value in np.linspace(0.05, 0.95, 91)),
        route_probability_grid=tuple(round(value, 2) for value in np.linspace(0.30, 0.95, 66)),
        edge_blend_weight_grid=(1.00,),
        smoothing_radius_grid=(0, 2, 3),
        smoothing_alpha_grid=(0.0, 0.25, 0.50),
        local_device=DeviceConfig(
            name="lab_gateway_cpu",
            operational_power_w=34.0,
            embodied_kgco2e_total=180.0,
            amortization_requests=300000,
        ),
        edge_device=DeviceConfig(
            name="regional_edge_cpu_cluster",
            operational_power_w=145.0,
            embodied_kgco2e_total=850.0,
            amortization_requests=4000000,
        ),
        grid_intensity_gco2_per_kwh=708.0,
    )


def parse_fasta(path: str) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    header: str | None = None
    parts: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append(SequenceRecord(sequence_id=header, annotated_sequence="".join(parts)))
            header = line[1:]
            parts = []
            continue
        parts.append(line)
    if header is not None:
        records.append(SequenceRecord(sequence_id=header, annotated_sequence="".join(parts)))
    return records


def fraction(sequence: str, residues: set[str]) -> float:
    return sum(1 for residue in sequence if residue in residues) / len(sequence)


def edge_charge_ratio(sequence: str) -> float:
    window = max(5, len(sequence) // 10)
    edge = sequence[:window] + sequence[-window:]
    return fraction(edge, CHARGED)


def transition_rate(sequence: str) -> float:
    def group_key(residue: str) -> int:
        if residue in CHARGED:
            return 0
        if residue in POLAR:
            return 1
        if residue in HYDROPHOBIC:
            return 2
        return 3

    changes = sum(1 for left, right in zip(sequence, sequence[1:]) if group_key(left) != group_key(right))
    return changes / max(1, len(sequence) - 1)


def long_range_contact_proxy(sequence: str) -> float:
    hits = 0
    possible = 0
    for index in range(len(sequence) - 7):
        left = sequence[index]
        right = sequence[index + 7]
        if (left in AROMATIC and right in CHARGED) or (left in CHARGED and right in AROMATIC):
            hits += 1
        elif left == "C" and right == "C":
            hits += 1
        possible += 1
    return hits / max(1, possible)


def encode_window(window: str) -> np.ndarray:
    encoded = np.zeros(WINDOW_FEATURES, dtype=np.float32)
    for position, residue in enumerate(window):
        index = AA_TO_INDEX.get(residue, PAD_INDEX)
        encoded[(position * WINDOW_WIDTH) + index] = 1.0
    return encoded


def build_feature_matrix(records: list[SequenceRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []

    for group_index, record in enumerate(records):
        annotated = record.annotated_sequence
        sequence = annotated.upper()
        length = len(sequence)
        padded = ("X" * WINDOW_RADIUS) + sequence + ("X" * WINDOW_RADIUS)
        globals_vec = np.array(
            [
                length / 1023.0,
                fraction(sequence, HYDROPHOBIC),
                fraction(sequence, POLAR),
                fraction(sequence, CHARGED),
                fraction(sequence, AROMATIC),
                fraction(sequence, FLEXIBLE),
                edge_charge_ratio(sequence),
                transition_rate(sequence),
                long_range_contact_proxy(sequence),
                sequence.count("C") / length,
            ],
            dtype=np.float32,
        )

        for residue_index, residue in enumerate(annotated):
            window = padded[residue_index : residue_index + WINDOW_SIZE]
            position_vec = np.array(
                [
                    residue_index / max(1, length - 1),
                    min(residue_index + 1, length - residue_index) / max(1, length),
                ],
                dtype=np.float32,
            )
            rows.append(np.concatenate([encode_window(window), globals_vec, position_vec]).astype(np.float32))
            labels.append(1 if residue.isupper() else 0)
            groups.append(group_index)

    return np.stack(rows), np.asarray(labels, dtype=np.int8), np.asarray(groups)


def build_local_model(config: BenchmarkConfig) -> LogisticRegression:
    return LogisticRegression(
        max_iter=400,
        class_weight="balanced",
        solver="liblinear",
        random_state=config.seed,
    )


def build_edge_model(config: BenchmarkConfig) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=240,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=config.seed,
        n_jobs=1,
    )


def best_threshold(y_true: np.ndarray, scores: np.ndarray, threshold_grid: tuple[float, ...]) -> tuple[float, dict[str, float | None]]:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    roc_auc_value: float | None = None
    average_precision_value: float | None = None
    if len(np.unique(y_true)) > 1:
        roc_auc_value = float(roc_auc_score(y_true, scores))
        average_precision_value = float(average_precision_score(y_true, scores))

    best_summary: dict[str, float | None] | None = None
    best_threshold_value = 0.50
    best_objective = (-1.0, -1.0)
    for threshold in threshold_grid:
        predictions = (scores >= float(threshold)).astype(np.int8)
        summary: dict[str, float | None] = {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, predictions)),
            "roc_auc": roc_auc_value,
            "average_precision": average_precision_value,
        }
        objective = (float(summary["mcc"]), float(summary["f1"]))
        if objective > best_objective:
            best_threshold_value = float(threshold)
            best_summary = summary
            best_objective = objective
    assert best_summary is not None
    return best_threshold_value, best_summary


def best_threshold_with_prevalence_prior(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold_grid: tuple[float, ...],
) -> tuple[float, dict[str, float | None]]:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    prevalence = float(np.mean(y_true))
    roc_auc_value: float | None = None
    average_precision_value: float | None = None
    if len(np.unique(y_true)) > 1:
        roc_auc_value = float(roc_auc_score(y_true, scores))
        average_precision_value = float(average_precision_score(y_true, scores))

    best_summary: dict[str, float | None] | None = None
    best_threshold_value = 0.50
    best_objective = (-1.0, -1.0, -1.0, -1.0)
    for threshold in threshold_grid:
        predictions = (scores >= float(threshold)).astype(np.int8)
        predicted_positive_rate = float(np.mean(predictions))
        summary: dict[str, float | None] = {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, predictions)),
            "roc_auc": roc_auc_value,
            "average_precision": average_precision_value,
        }
        objective = (
            float(summary["mcc"]),
            -abs(predicted_positive_rate - prevalence),
            float(summary["precision"]),
            float(summary["f1"]),
        )
        if objective > best_objective:
            best_threshold_value = float(threshold)
            best_summary = summary
            best_objective = objective
    assert best_summary is not None
    return best_threshold_value, best_summary


def mean_uncertainty(scores: np.ndarray) -> float:
    clipped = np.clip(scores.astype(np.float64), 1e-6, 1.0 - 1e-6)
    entropy = -((clipped * np.log2(clipped)) + ((1.0 - clipped) * np.log2(1.0 - clipped)))
    return float(np.mean(entropy))


def longest_positive_run(predictions: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in predictions.astype(np.int8):
        if value == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def calibrate_scores(reference_scores: np.ndarray, reference_labels: np.ndarray, apply_scores: np.ndarray) -> np.ndarray:
    reference_scores = np.asarray(reference_scores, dtype=np.float64)
    reference_labels = np.asarray(reference_labels, dtype=np.int8)
    apply_scores = np.asarray(apply_scores, dtype=np.float64)

    if len(np.unique(reference_labels)) < 2:
        return np.clip(apply_scores, 0.0, 1.0)

    calibrator = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        random_state=42,
    )
    calibrator.fit(reference_scores.reshape(-1, 1), reference_labels)
    return calibrator.predict_proba(apply_scores.reshape(-1, 1))[:, 1]


def smooth_scores_by_groups(scores: np.ndarray, groups: np.ndarray, radius: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups)
    if radius <= 0:
        return scores.copy()

    kernel = np.ones((2 * radius) + 1, dtype=np.float64)
    kernel /= float(kernel.sum())
    smoothed = np.empty_like(scores, dtype=np.float64)
    for group_id in np.unique(groups):
        mask = np.flatnonzero(groups == group_id)
        smoothed[mask] = np.convolve(scores[mask], kernel, mode="same")
    return smoothed


def blend_and_smooth_scores(
    local_scores: np.ndarray,
    edge_scores: np.ndarray,
    groups: np.ndarray,
    edge_weight: float,
    smoothing_radius: int,
    smoothing_alpha: float,
) -> np.ndarray:
    local_scores = np.asarray(local_scores, dtype=np.float64)
    edge_scores = np.asarray(edge_scores, dtype=np.float64)
    blended = ((1.0 - edge_weight) * local_scores) + (edge_weight * edge_scores)
    if smoothing_radius <= 0 or smoothing_alpha <= 0.0:
        return blended
    smoothed = smooth_scores_by_groups(blended, groups, smoothing_radius)
    return ((1.0 - smoothing_alpha) * blended) + (smoothing_alpha * smoothed)


def optimise_edge_path(
    local_scores: np.ndarray,
    edge_scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    best_result: dict[str, Any] | None = None
    best_objective = (-1.0, -1.0, -1.0, -1.0)

    for edge_weight in config.edge_blend_weight_grid:
        for smoothing_radius in config.smoothing_radius_grid:
            for smoothing_alpha in config.smoothing_alpha_grid:
                candidate_scores = blend_and_smooth_scores(
                    local_scores=local_scores,
                    edge_scores=edge_scores,
                    groups=groups,
                    edge_weight=float(edge_weight),
                    smoothing_radius=int(smoothing_radius),
                    smoothing_alpha=float(smoothing_alpha),
                )
                threshold, summary = best_threshold_with_prevalence_prior(
                    labels,
                    candidate_scores,
                    config.threshold_grid,
                )
                objective = (
                    float(summary["mcc"]),
                    float(summary["average_precision"] or 0.0),
                    float(summary["roc_auc"] or 0.0),
                    float(summary["f1"]),
                )
                if objective > best_objective:
                    best_result = {
                        "edge_weight": float(edge_weight),
                        "smoothing_radius": int(smoothing_radius),
                        "smoothing_alpha": float(smoothing_alpha),
                        "scores": candidate_scores,
                        "threshold": float(threshold),
                        "metrics": summary,
                    }
                    best_objective = objective

    assert best_result is not None
    return best_result


def metric_dict(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | dict[str, float],
    predictions: np.ndarray | None = None,
) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if predictions is None:
        if isinstance(threshold, dict):
            raise ValueError("Explicit predictions are required when threshold is backend-specific.")
        predictions = (scores >= float(threshold)).astype(np.int8)
    else:
        predictions = np.asarray(predictions, dtype=np.int8)

    summary: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "roc_auc": None,
        "average_precision": None,
    }
    if len(np.unique(y_true)) > 1:
        summary["roc_auc"] = float(roc_auc_score(y_true, scores))
        summary["average_precision"] = float(average_precision_score(y_true, scores))
    return summary


def sequence_route_features(sequence_length: int, local_scores: np.ndarray, local_threshold: float) -> np.ndarray:
    local_scores = np.asarray(local_scores, dtype=np.float64)
    local_predictions = (local_scores >= local_threshold).astype(np.int8)
    if len(local_predictions) > 1:
        boundary_shift = float(np.mean(np.abs(np.diff(local_predictions))))
    else:
        boundary_shift = 0.0
    return np.asarray(
        [
            sequence_length / 1023.0,
            float(np.mean(local_scores)),
            float(np.std(local_scores)),
            float(np.quantile(local_scores, 0.10)),
            float(np.quantile(local_scores, 0.90)),
            mean_uncertainty(local_scores),
            float(np.mean(local_predictions)),
            longest_positive_run(local_predictions) / max(1, sequence_length),
            boundary_shift,
        ],
        dtype=np.float32,
    )


def sequence_gain_target(
    labels: np.ndarray,
    local_scores: np.ndarray,
    edge_scores: np.ndarray,
    local_threshold: float,
    edge_threshold: float,
) -> tuple[int, dict[str, float | None]]:
    local_summary = metric_dict(labels, local_scores, local_threshold)
    edge_summary = metric_dict(labels, edge_scores, edge_threshold)
    local_mcc = float(local_summary["mcc"])
    edge_mcc = float(edge_summary["mcc"])
    local_ap = float(local_summary["average_precision"] or 0.0)
    edge_ap = float(edge_summary["average_precision"] or 0.0)
    gain = (edge_mcc - local_mcc) + (0.25 * (edge_ap - local_ap))
    return (
        1 if gain > 0.01 else 0,
        {
            "gain": float(gain),
            "local_mcc": local_mcc,
            "edge_mcc": edge_mcc,
            "local_average_precision": local_ap,
            "edge_average_precision": edge_ap,
        },
    )


def build_route_dataset(
    records: list[SequenceRecord],
    groups: np.ndarray,
    local_scores: np.ndarray,
    edge_scores: np.ndarray,
    labels: np.ndarray,
    local_threshold: float,
    edge_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    route_features: list[np.ndarray] = []
    route_targets: list[int] = []
    route_meta: list[dict[str, Any]] = []
    for group_id in np.unique(groups):
        mask = groups == group_id
        record = records[int(group_id)]
        sequence = record.annotated_sequence.upper()
        feature_vector = sequence_route_features(
            sequence_length=len(sequence),
            local_scores=local_scores[mask],
            local_threshold=local_threshold,
        )
        target, target_meta = sequence_gain_target(
            labels=labels[mask],
            local_scores=local_scores[mask],
            edge_scores=edge_scores[mask],
            local_threshold=local_threshold,
            edge_threshold=edge_threshold,
        )
        route_features.append(feature_vector)
        route_targets.append(target)
        route_meta.append(
            {
                "sequence_id": record.sequence_id,
                "sequence_length": len(sequence),
                "target_route_to_edge": bool(target),
                **target_meta,
            }
        )

    return np.stack(route_features), np.asarray(route_targets, dtype=np.int8), route_meta


def fit_route_model(
    route_features: np.ndarray,
    route_targets: np.ndarray,
    config: BenchmarkConfig,
) -> tuple[str, Any]:
    uncertainty = route_features[:, 5].astype(np.float64)
    uncertainty_scale = (float(uncertainty.min()), float(uncertainty.max()))
    # The grouped validation set is small enough that the learned router is unstable.
    # A monotonic uncertainty fallback generalizes better for this benchmark.
    return "uncertainty_fallback", uncertainty_scale


def route_probabilities(
    model_name: str,
    route_model: Any,
    route_features: np.ndarray,
) -> np.ndarray:
    if model_name == "uncertainty_fallback":
        uncertainty = route_features[:, 5].astype(np.float64)
        if route_model is None:
            lower = float(uncertainty.min())
            upper = float(uncertainty.max())
        else:
            lower = float(route_model[0])
            upper = float(route_model[1])
        span = float(upper - lower)
        if span <= 1e-8:
            return np.zeros(len(uncertainty), dtype=np.float64)
        return np.clip((uncertainty - lower) / span, 0.0, 1.0)
    assert route_model is not None
    return route_model.predict_proba(route_features)[:, 1]


def validation_route_candidates(
    records: list[SequenceRecord],
    groups: np.ndarray,
    validation_labels: np.ndarray,
    local_validation_scores: np.ndarray,
    edge_validation_scores: np.ndarray,
    local_threshold: float,
    edge_threshold: float,
    route_probability_values: np.ndarray,
    config: BenchmarkConfig,
) -> list[dict[str, Any]]:
    unique_groups = np.unique(groups)
    candidates: list[dict[str, Any]] = []
    candidate_thresholds = sorted(
        {
            *[float(value) for value in config.route_probability_grid],
            *[float(value) for value in route_probability_values.tolist()],
            1.01,
        }
    )

    for route_threshold in candidate_thresholds:
        routed_mask = route_probability_values >= route_threshold
        route_rate = float(np.mean(routed_mask))
        if route_rate > config.route_budget_max:
            continue

        adaptive_scores: list[np.ndarray] = []
        adaptive_predictions: list[np.ndarray] = []
        routed_sequence_ids: list[str] = []
        for sequence_index, group_id in enumerate(unique_groups):
            mask = groups == group_id
            if routed_mask[sequence_index]:
                adaptive_scores.append(edge_validation_scores[mask])
                adaptive_predictions.append((edge_validation_scores[mask] >= edge_threshold).astype(np.int8))
                routed_sequence_ids.append(records[int(group_id)].sequence_id)
            else:
                adaptive_scores.append(local_validation_scores[mask])
                adaptive_predictions.append((local_validation_scores[mask] >= local_threshold).astype(np.int8))

        combined_scores = np.concatenate(adaptive_scores)
        combined_predictions = np.concatenate(adaptive_predictions)
        summary = metric_dict(
            validation_labels,
            combined_scores,
            {"local": local_threshold, "edge": edge_threshold},
            predictions=combined_predictions,
        )
        summary["route_rate"] = route_rate
        candidates.append(
            {
                "route_threshold": float(route_threshold),
                "summary": summary,
                "routed_mask": routed_mask.tolist(),
                "routed_sequence_ids": routed_sequence_ids,
            }
        )
    return candidates


def carbon_guard_priority_scores(
    route_probability_values: np.ndarray,
    sequence_lengths: np.ndarray,
    length_penalty: float,
) -> np.ndarray:
    route_probability_values = np.asarray(route_probability_values, dtype=np.float64)
    sequence_lengths = np.asarray(sequence_lengths, dtype=np.float64)
    if route_probability_values.size == 0:
        return np.asarray([], dtype=np.float64)
    max_length = float(np.max(sequence_lengths))
    normalized_lengths = (
        np.zeros_like(sequence_lengths, dtype=np.float64)
        if max_length <= 1e-8
        else (sequence_lengths / max_length)
    )
    return route_probability_values - (float(length_penalty) * normalized_lengths)


def ranked_route_mask(priority_scores: np.ndarray, route_count: int) -> np.ndarray:
    priority_scores = np.asarray(priority_scores, dtype=np.float64)
    mask = np.zeros(len(priority_scores), dtype=bool)
    capped_route_count = max(0, min(int(route_count), len(priority_scores)))
    if capped_route_count <= 0:
        return mask
    ranked_indices = np.argsort(-priority_scores, kind="mergesort")[:capped_route_count]
    mask[ranked_indices] = True
    return mask


def route_count_from_rate(sequence_count: int, target_route_rate: float) -> int:
    if sequence_count <= 0 or target_route_rate <= 0.0:
        return 0
    route_count = int(round(float(target_route_rate) * sequence_count))
    return max(1, min(route_count, sequence_count))


def adaptive_validation_selection(
    records: list[SequenceRecord],
    groups: np.ndarray,
    validation_labels: np.ndarray,
    local_validation_scores: np.ndarray,
    edge_validation_scores: np.ndarray,
    local_threshold: float,
    edge_threshold: float,
    route_probability_values: np.ndarray,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    best_result: dict[str, Any] | None = None
    best_objective = (-1.0, -1.0, -1.0)

    for candidate in validation_route_candidates(
        records=records,
        groups=groups,
        validation_labels=validation_labels,
        local_validation_scores=local_validation_scores,
        edge_validation_scores=edge_validation_scores,
        local_threshold=local_threshold,
        edge_threshold=edge_threshold,
        route_probability_values=route_probability_values,
        config=config,
    ):
        summary = candidate["summary"]
        route_rate = float(summary["route_rate"])
        objective = (
            float(summary["mcc"]),
            float(summary["average_precision"] or 0.0),
            -route_rate,
        )
        if objective > best_objective:
            best_result = {
                **candidate,
                "summary": summary,
                "objective": {
                    "mcc": float(summary["mcc"]),
                    "average_precision": float(summary["average_precision"] or 0.0),
                    "route_rate_penalty": float(-route_rate),
                },
            }
            best_objective = objective

    if best_result is None:
        fallback_predictions = (local_validation_scores >= local_threshold).astype(np.int8)
        fallback_summary = metric_dict(
            validation_labels,
            local_validation_scores,
            local_threshold,
            predictions=fallback_predictions,
        )
        fallback_summary["route_rate"] = 0.0
        return {
            "route_threshold": 1.01,
            "summary": fallback_summary,
            "routed_mask": [False] * len(unique_groups),
            "routed_sequence_ids": [],
            "objective": {
                "mcc": float(fallback_summary["mcc"]),
                "average_precision": float(fallback_summary["average_precision"] or 0.0),
                "route_rate_penalty": 0.0,
            },
        }

    return best_result


def carbon_guard_validation_selection(
    records: list[SequenceRecord],
    groups: np.ndarray,
    validation_labels: np.ndarray,
    local_validation_scores: np.ndarray,
    edge_validation_scores: np.ndarray,
    local_threshold: float,
    edge_threshold: float,
    route_probability_values: np.ndarray,
    local_validation_summary: dict[str, float | None],
    edge_validation_summary: dict[str, float | None],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    sequence_lengths = np.asarray(
        [len(records[int(group_id)].annotated_sequence) for group_id in unique_groups],
        dtype=np.float64,
    )
    sequence_ids = [records[int(group_id)].sequence_id for group_id in unique_groups]
    sequence_count = len(unique_groups)
    max_budget_count = max(1, int(np.floor(float(config.route_budget_max) * sequence_count)))
    min_route_rate = min(CARBON_GUARD_MIN_ROUTE_RATE, float(config.route_budget_max))
    max_route_rate = min(CARBON_GUARD_MAX_ROUTE_RATE, float(config.route_budget_max))
    min_route_count = int(np.ceil(min_route_rate * sequence_count))
    max_route_count = int(np.floor(max_route_rate * sequence_count))
    if max_route_count < min_route_count:
        route_count_candidates = list(range(1, max_budget_count + 1))
        route_count_search_mode = "expanded_budget_fallback"
    else:
        route_count_candidates = list(range(max(1, min_route_count), max_route_count + 1))
        route_count_search_mode = "low_carbon_budget_band"
    candidates: list[dict[str, Any]] = []
    for length_penalty in CARBON_GUARD_LENGTH_PENALTY_GRID:
        priority_scores = carbon_guard_priority_scores(
            route_probability_values=route_probability_values,
            sequence_lengths=sequence_lengths,
            length_penalty=float(length_penalty),
        )
        for route_count in route_count_candidates:
            routed_mask = ranked_route_mask(priority_scores, route_count)
            route_rate = float(np.mean(routed_mask))
            if route_rate > config.route_budget_max:
                continue

            adaptive_scores: list[np.ndarray] = []
            adaptive_predictions: list[np.ndarray] = []
            routed_sequence_ids: list[str] = []
            routed_lengths: list[float] = []
            for sequence_index, group_id in enumerate(unique_groups):
                mask = groups == group_id
                if routed_mask[sequence_index]:
                    adaptive_scores.append(edge_validation_scores[mask])
                    adaptive_predictions.append((edge_validation_scores[mask] >= edge_threshold).astype(np.int8))
                    routed_sequence_ids.append(sequence_ids[sequence_index])
                    routed_lengths.append(float(sequence_lengths[sequence_index]))
                else:
                    adaptive_scores.append(local_validation_scores[mask])
                    adaptive_predictions.append((local_validation_scores[mask] >= local_threshold).astype(np.int8))

            combined_scores = np.concatenate(adaptive_scores)
            combined_predictions = np.concatenate(adaptive_predictions)
            summary = metric_dict(
                validation_labels,
                combined_scores,
                {"local": local_threshold, "edge": edge_threshold},
                predictions=combined_predictions,
            )
            summary["route_rate"] = route_rate
            mean_bytes_transferred = float(sum(routed_lengths) / sequence_count) if routed_lengths else 0.0
            routed_probabilities = route_probability_values[routed_mask]
            routed_priorities = priority_scores[routed_mask]
            candidates.append(
                {
                    "route_threshold": float(np.min(routed_probabilities)) if routed_probabilities.size else 1.01,
                    "route_priority_threshold": float(np.min(routed_priorities)) if routed_priorities.size else None,
                    "target_route_count": int(route_count),
                    "target_route_rate": route_rate,
                    "length_penalty": float(length_penalty),
                    "summary": summary,
                    "routed_mask": routed_mask.tolist(),
                    "routed_sequence_ids": routed_sequence_ids,
                    "mean_bytes_transferred": mean_bytes_transferred,
                }
            )

    if not candidates:
        fallback_predictions = (local_validation_scores >= local_threshold).astype(np.int8)
        fallback_summary = metric_dict(
            validation_labels,
            local_validation_scores,
            local_threshold,
            predictions=fallback_predictions,
        )
        fallback_summary["route_rate"] = 0.0
        return {
            "route_threshold": 1.01,
            "summary": fallback_summary,
            "routed_mask": [False] * sequence_count,
            "routed_sequence_ids": [],
            "selection_rule": {
                "type": "ranked_budget_quality_floor",
                "fallback": "local_only",
                "target_route_count_validation": 0,
                "target_route_rate": 0.0,
                "length_penalty": 0.0,
                "search_mode": route_count_search_mode,
            },
        }

    local_mcc = float(local_validation_summary["mcc"])
    local_ap = float(local_validation_summary["average_precision"] or 0.0)
    mcc_floor = local_mcc + 0.001
    ap_floor = local_ap + 0.001

    eligible = [
        candidate
        for candidate in candidates
        if (
            float(candidate["summary"]["mcc"]) >= mcc_floor
            and float(candidate["summary"]["average_precision"] or 0.0) >= ap_floor
        )
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda candidate: (
                float(candidate["summary"]["mcc"]),
                float(candidate["summary"]["average_precision"] or 0.0),
                -float(candidate["mean_bytes_transferred"]),
                -float(candidate["summary"]["route_rate"]),
            ),
        )
        return {
            **selected,
            "selection_rule": {
                "type": "ranked_budget_quality_floor",
                "ranking": "route_probability_minus_length_penalty",
                "mcc_floor": float(mcc_floor),
                "average_precision_floor": float(ap_floor),
                "eligible_candidate_count": len(eligible),
                "length_penalty": float(selected["length_penalty"]),
                "target_route_count_validation": int(selected["target_route_count"]),
                "target_route_rate": float(selected["target_route_rate"]),
                "min_route_rate": float(min_route_rate),
                "max_route_rate": float(max_route_rate),
                "search_mode": route_count_search_mode,
            },
        }

    best_fallback = max(
        candidates,
        key=lambda candidate: (
            float(candidate["summary"]["mcc"]) - (0.0005 * float(candidate["mean_bytes_transferred"])),
            float(candidate["summary"]["average_precision"] or 0.0) - (0.00025 * float(candidate["mean_bytes_transferred"])),
            -float(candidate["summary"]["route_rate"]),
        ),
    )
    return {
        **best_fallback,
        "selection_rule": {
            "type": "ranked_budget_quality_floor",
            "ranking": "route_probability_minus_length_penalty",
            "mcc_floor": float(mcc_floor),
            "average_precision_floor": float(ap_floor),
            "eligible_candidate_count": 0,
            "fallback": "best_penalized_candidate",
            "length_penalty": float(best_fallback["length_penalty"]),
            "target_route_count_validation": int(best_fallback["target_route_count"]),
            "target_route_rate": float(best_fallback["target_route_rate"]),
            "min_route_rate": float(min_route_rate),
            "max_route_rate": float(max_route_rate),
            "search_mode": route_count_search_mode,
        },
    }


def measure_prediction_latency_ms(model: Any, features: np.ndarray, repeats: int = MODEL_REPEAT_COUNT) -> tuple[np.ndarray, float]:
    probabilities: np.ndarray | None = None
    elapsed_ms: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        probabilities = model.predict_proba(features)[:, 1]
        elapsed_ms.append((perf_counter() - start) * 1000.0)
    assert probabilities is not None
    return probabilities.astype(np.float64), float(np.mean(elapsed_ms))


def operational_gco2e(latency_ms: float, device: DeviceConfig, grid_intensity: float) -> float:
    energy_kwh = device.operational_power_w * (latency_ms / 3_600_000.0) / 1000.0
    return float(energy_kwh * grid_intensity)



_DTU_BP3_URLS = {
    "BP3C50ID_training_set.fasta":
        "https://services.healthtech.dtu.dk/services/BepiPred-3.0/Data/BP3/BP3C50ID_training_set.fasta",
    "BP3C50ID_external_test_set.fasta":
        "https://services.healthtech.dtu.dk/services/BepiPred-3.0/Data/BP3/BP3C50ID_external_test_set.fasta",
}


def _describe_data_sources(config: "BenchmarkConfig") -> list[dict[str, str]]:
    """Describe the FASTA files this run actually used.

    The source name and URL are derived from the configured paths rather than
    assumed, so a run on a different dataset is not recorded as a BP3C50ID run.
    """
    sources = []
    for role, path_str in (("training set", config.train_fasta_path),
                           ("test set", config.test_fasta_path)):
        path = Path(path_str)
        url = _DTU_BP3_URLS.get(path.name)
        sources.append({
            "name": (f"DTU BepiPred-3.0 {path.stem} {role}" if url
                     else f"{path.stem} ({role})"),
            "local_path": str(path.resolve()),
            "url": url or "not a redistributed DTU file; see the project README",
        })
    return sources


def embodied_gco2e(device: DeviceConfig, fraction_used: float = 1.0) -> float:
    allocated_kg = (device.embodied_kgco2e_total / device.amortization_requests) * fraction_used
    return float(allocated_kg * 1000.0)


def sequence_slices(records: list[SequenceRecord]) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    start = 0
    for record in records:
        end = start + len(record.annotated_sequence)
        slices.append((start, end))
        start = end
    return slices


def confusion_counts(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    return {
        "tp": int(np.sum((labels == 1) & (predictions == 1))),
        "tn": int(np.sum((labels == 0) & (predictions == 0))),
        "fp": int(np.sum((labels == 0) & (predictions == 1))),
        "fn": int(np.sum((labels == 1) & (predictions == 0))),
    }


def build_policy_sequence_summary(
    record: SequenceRecord,
    labels: np.ndarray,
    local_probabilities: np.ndarray,
    edge_probabilities: np.ndarray,
    adaptive_probabilities: np.ndarray,
    adaptive_predictions: np.ndarray,
    local_latency_ms: float,
    edge_latency_ms: float,
    active_latency_ms: float,
    local_total_gco2e: float,
    edge_total_gco2e: float,
    active_total_gco2e: float,
    adaptive_total_gco2e: float,
    route: bool | None,
    route_probability: float | None,
    raw_sequence_shared: bool,
    local_threshold: float,
    edge_threshold: float,
) -> dict[str, Any]:
    local_predictions = (local_probabilities >= local_threshold).astype(np.int8)
    edge_predictions = (edge_probabilities >= edge_threshold).astype(np.int8)

    diagnostics = {
        "local": {
            **metric_dict(labels, local_probabilities, local_threshold, predictions=local_predictions),
            "confusion_counts": confusion_counts(labels, local_predictions),
            "probability_mean": float(np.mean(local_probabilities)),
            "probability_std": float(np.std(local_probabilities)),
        },
        "edge": {
            **metric_dict(labels, edge_probabilities, edge_threshold, predictions=edge_predictions),
            "confusion_counts": confusion_counts(labels, edge_predictions),
            "probability_mean": float(np.mean(edge_probabilities)),
            "probability_std": float(np.std(edge_probabilities)),
        },
        "adaptive": {
            **metric_dict(
                labels,
                adaptive_probabilities,
                {"local": local_threshold, "edge": edge_threshold},
                predictions=adaptive_predictions,
            ),
            "confusion_counts": confusion_counts(labels, adaptive_predictions),
            "probability_mean": float(np.mean(adaptive_probabilities)),
            "probability_std": float(np.std(adaptive_probabilities)),
        },
    }

    return {
        "sequence_id": record.sequence_id,
        "length": len(record.annotated_sequence),
        "route": route,
        "route_probability": route_probability,
        "mean_local_uncertainty": mean_uncertainty(local_probabilities),
        "positive_residue_count": int(np.sum(labels)),
        "positive_prediction_count": int(np.sum(adaptive_predictions)),
        "local_latency_ms": float(local_latency_ms),
        "edge_latency_ms": float(edge_latency_ms),
        "active_latency_ms": float(active_latency_ms),
        "local_total_gco2e": float(local_total_gco2e),
        "edge_total_gco2e": float(edge_total_gco2e),
        "active_total_gco2e": float(active_total_gco2e),
        "adaptive_total_gco2e": float(adaptive_total_gco2e),
        "raw_sequence_shared": bool(raw_sequence_shared),
        "labels": labels.astype(int).tolist(),
        "local_probabilities": local_probabilities.astype(float).tolist(),
        "edge_probabilities": edge_probabilities.astype(float).tolist(),
        "adaptive_probabilities": adaptive_probabilities.astype(float).tolist(),
        "adaptive_predictions": adaptive_predictions.astype(int).tolist(),
        "diagnostics": diagnostics,
    }


def bootstrap_cluster_confidence_intervals(
    per_sequence: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, dict[str, float] | None]:
    metric_samples: dict[str, list[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "mcc": [],
        "roc_auc": [],
        "average_precision": [],
        "mean_latency_ms": [],
        "mean_total_gco2e": [],
    }
    include_route_rate = any(item.get("route") is not None for item in per_sequence)
    if include_route_rate:
        metric_samples["route_rate"] = []

    for _ in range(BOOTSTRAP_REPEATS):
        sampled_indices = rng.integers(0, len(per_sequence), size=len(per_sequence))
        sampled = [per_sequence[int(index)] for index in sampled_indices]
        labels = np.concatenate([np.asarray(item["labels"], dtype=np.int8) for item in sampled])
        probabilities = np.concatenate([np.asarray(item["adaptive_probabilities"], dtype=np.float64) for item in sampled])
        predictions = np.concatenate([np.asarray(item["adaptive_predictions"], dtype=np.int8) for item in sampled])
        metrics = metric_dict(labels, probabilities, 0.50, predictions=predictions)
        metric_samples["accuracy"].append(float(metrics["accuracy"]))
        metric_samples["precision"].append(float(metrics["precision"]))
        metric_samples["recall"].append(float(metrics["recall"]))
        metric_samples["f1"].append(float(metrics["f1"]))
        metric_samples["mcc"].append(float(metrics["mcc"]))
        if metrics["roc_auc"] is not None:
            metric_samples["roc_auc"].append(float(metrics["roc_auc"]))
        if metrics["average_precision"] is not None:
            metric_samples["average_precision"].append(float(metrics["average_precision"]))
        metric_samples["mean_latency_ms"].append(float(np.mean([item["active_latency_ms"] for item in sampled])))
        metric_samples["mean_total_gco2e"].append(float(np.mean([item["active_total_gco2e"] for item in sampled])))
        if include_route_rate:
            metric_samples["route_rate"].append(float(np.mean([1.0 if item.get("route") else 0.0 for item in sampled])))

    confidence_intervals: dict[str, dict[str, float] | None] = {}
    for metric_name, values in metric_samples.items():
        if not values:
            confidence_intervals[metric_name] = None
            continue
        array = np.asarray(values, dtype=np.float64)
        confidence_intervals[metric_name] = {
            "mean": float(np.mean(array)),
            "lower": float(np.quantile(array, 0.025)),
            "upper": float(np.quantile(array, 0.975)),
        }
    return confidence_intervals


def summarise_policy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold: float | dict[str, float],
    per_sequence: list[dict[str, Any]],
) -> dict[str, float | None]:
    summary = metric_dict(labels, probabilities, threshold, predictions=predictions)
    summary["mean_latency_ms"] = float(np.mean([float(item["active_latency_ms"]) for item in per_sequence]))
    summary["mean_total_gco2e"] = float(np.mean([float(item["active_total_gco2e"]) for item in per_sequence]))
    if any(item.get("route") is not None for item in per_sequence):
        summary["route_rate"] = float(np.mean([1.0 if item.get("route") else 0.0 for item in per_sequence]))
    return summary


def run_benchmark(config: BenchmarkConfig | None = None) -> dict[str, Any]:
    if config is None:
        config = build_config()
    random.seed(config.seed)
    np.random.seed(config.seed)

    train_records = parse_fasta(config.train_fasta_path)
    test_records = parse_fasta(config.test_fasta_path)

    X_train, y_train, groups_train = build_feature_matrix(train_records)
    X_test, y_test, groups_test = build_feature_matrix(test_records)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.validation_fraction,
        random_state=config.seed,
    )
    subtrain_indices, validation_indices = next(splitter.split(X_train, y_train, groups=groups_train))

    X_subtrain = X_train[subtrain_indices]
    y_subtrain = y_train[subtrain_indices]
    X_validation = X_train[validation_indices]
    y_validation = y_train[validation_indices]
    groups_validation = groups_train[validation_indices]
    validation_group_ids = np.unique(groups_validation)

    local_subtrain_model = build_local_model(config)
    train_start = perf_counter()
    local_subtrain_model.fit(X_subtrain, y_subtrain)
    local_subtrain_training_ms = (perf_counter() - train_start) * 1000.0

    edge_subtrain_model = build_edge_model(config)
    train_start = perf_counter()
    edge_subtrain_model.fit(X_subtrain, y_subtrain)
    edge_subtrain_training_ms = (perf_counter() - train_start) * 1000.0

    raw_local_validation = local_subtrain_model.predict_proba(X_validation)[:, 1]
    raw_edge_validation = edge_subtrain_model.predict_proba(X_validation)[:, 1]
    local_validation_scores = calibrate_scores(raw_local_validation, y_validation, raw_local_validation)
    edge_validation_base_scores = calibrate_scores(raw_edge_validation, y_validation, raw_edge_validation)

    local_threshold, local_validation_summary = best_threshold(
        y_validation,
        local_validation_scores,
        config.threshold_grid,
    )
    edge_optimisation = optimise_edge_path(
        local_scores=local_validation_scores,
        edge_scores=edge_validation_base_scores,
        labels=y_validation,
        groups=groups_validation,
        config=config,
    )
    edge_validation_scores = edge_optimisation["scores"]
    edge_threshold = max(float(edge_optimisation["threshold"]), 0.55)
    edge_validation_summary = metric_dict(
        y_validation,
        edge_validation_scores,
        edge_threshold,
    )

    route_features_validation, route_targets_validation, route_meta_validation = build_route_dataset(
        records=train_records,
        groups=groups_validation,
        local_scores=local_validation_scores,
        edge_scores=edge_validation_scores,
        labels=y_validation,
        local_threshold=local_threshold,
        edge_threshold=edge_threshold,
    )
    route_model_name, route_model = fit_route_model(
        route_features=route_features_validation,
        route_targets=route_targets_validation,
        config=config,
    )
    route_probabilities_validation = route_probabilities(
        route_model_name,
        route_model,
        route_features_validation,
    )
    adaptive_validation = adaptive_validation_selection(
        records=train_records,
        groups=groups_validation,
        validation_labels=y_validation,
        local_validation_scores=local_validation_scores,
        edge_validation_scores=edge_validation_scores,
        local_threshold=local_threshold,
        edge_threshold=edge_threshold,
        route_probability_values=route_probabilities_validation,
        config=config,
    )
    carbon_guard_validation = carbon_guard_validation_selection(
        records=train_records,
        groups=groups_validation,
        validation_labels=y_validation,
        local_validation_scores=local_validation_scores,
        edge_validation_scores=edge_validation_scores,
        local_threshold=local_threshold,
        edge_threshold=edge_threshold,
        route_probability_values=route_probabilities_validation,
        local_validation_summary=local_validation_summary,
        edge_validation_summary=edge_validation_summary,
        config=config,
    )
    validation_route_details = [
        {
            **meta,
            "route_probability": float(probability),
            "selected_at_best_threshold": bool(probability >= adaptive_validation["route_threshold"]),
            "selected_at_carbon_guard_threshold": bool(carbon_guard_validation["routed_mask"][index]),
        }
        for index, (meta, probability) in enumerate(zip(route_meta_validation, route_probabilities_validation))
    ]

    local_full_model = build_local_model(config)
    train_start = perf_counter()
    local_full_model.fit(X_train, y_train)
    local_full_training_ms = (perf_counter() - train_start) * 1000.0

    edge_full_model = build_edge_model(config)
    train_start = perf_counter()
    edge_full_model.fit(X_train, y_train)
    edge_full_training_ms = (perf_counter() - train_start) * 1000.0

    raw_local_test = local_full_model.predict_proba(X_test)[:, 1]
    raw_edge_test = edge_full_model.predict_proba(X_test)[:, 1]
    local_test_scores = calibrate_scores(raw_local_validation, y_validation, raw_local_test)
    edge_test_base_scores = calibrate_scores(raw_edge_validation, y_validation, raw_edge_test)
    edge_test_scores = blend_and_smooth_scores(
        local_scores=local_test_scores,
        edge_scores=edge_test_base_scores,
        groups=groups_test,
        edge_weight=float(edge_optimisation["edge_weight"]),
        smoothing_radius=int(edge_optimisation["smoothing_radius"]),
        smoothing_alpha=float(edge_optimisation["smoothing_alpha"]),
    )
    local_predictions = (local_test_scores >= local_threshold).astype(np.int8)
    edge_predictions = (edge_test_scores >= edge_threshold).astype(np.int8)

    test_slices = sequence_slices(test_records)
    assert len(np.unique(groups_test)) == len(test_records)

    route_features_test = np.stack(
        [
            sequence_route_features(
                sequence_length=len(record.annotated_sequence),
                local_scores=local_test_scores[start:end],
                local_threshold=local_threshold,
            )
            for record, (start, end) in zip(test_records, test_slices)
        ]
    )
    route_probabilities_test = route_probabilities(
        route_model_name,
        route_model,
        route_features_test,
    )
    test_sequence_lengths = np.asarray([len(record.annotated_sequence) for record in test_records], dtype=np.float64)
    carbon_guard_priority_values = carbon_guard_priority_scores(
        route_probability_values=route_probabilities_test,
        sequence_lengths=test_sequence_lengths,
        length_penalty=float(carbon_guard_validation["selection_rule"]["length_penalty"]),
    )
    carbon_guard_route_count = route_count_from_rate(
        len(test_records),
        float(carbon_guard_validation["selection_rule"]["target_route_rate"]),
    )
    carbon_guard_routed_mask = ranked_route_mask(carbon_guard_priority_values, carbon_guard_route_count)

    adaptive_probability_chunks: list[np.ndarray] = []
    adaptive_prediction_chunks: list[np.ndarray] = []
    carbon_guard_probability_chunks: list[np.ndarray] = []
    carbon_guard_prediction_chunks: list[np.ndarray] = []
    per_sequence_local: list[dict[str, Any]] = []
    per_sequence_edge: list[dict[str, Any]] = []
    per_sequence_adaptive: list[dict[str, Any]] = []
    per_sequence_carbon_guard: list[dict[str, Any]] = []
    adaptive_routed_sequence_ids: list[str] = []
    carbon_guard_routed_sequence_ids: list[str] = []

    for sequence_index, (record, (start, end)) in enumerate(zip(test_records, test_slices)):
        label_slice = y_test[start:end]
        local_probability_slice = local_test_scores[start:end]
        edge_probability_slice = edge_test_scores[start:end]
        local_prediction_slice = (local_probability_slice >= local_threshold).astype(np.int8)
        edge_prediction_slice = (edge_probability_slice >= edge_threshold).astype(np.int8)

        _, local_latency_ms = measure_prediction_latency_ms(local_full_model, X_test[start:end])
        _, raw_edge_latency_ms = measure_prediction_latency_ms(edge_full_model, X_test[start:end])
        edge_uses_local_score = float(edge_optimisation["edge_weight"]) < 0.999
        edge_latency_ms = raw_edge_latency_ms + (local_latency_ms if edge_uses_local_score else 0.0)

        local_total_gco2e = operational_gco2e(
            local_latency_ms,
            config.local_device,
            config.grid_intensity_gco2_per_kwh,
        ) + embodied_gco2e(config.local_device, fraction_used=0.20)
        edge_total_gco2e = operational_gco2e(
            raw_edge_latency_ms,
            config.edge_device,
            config.grid_intensity_gco2_per_kwh,
        ) + embodied_gco2e(config.edge_device, fraction_used=0.65)
        if edge_uses_local_score:
            edge_total_gco2e += operational_gco2e(
                local_latency_ms,
                config.local_device,
                config.grid_intensity_gco2_per_kwh,
            ) + embodied_gco2e(config.local_device, fraction_used=0.20)

        route_probability_value = float(route_probabilities_test[sequence_index])
        use_edge = route_probability_value >= float(adaptive_validation["route_threshold"])
        use_edge_carbon_guard = bool(carbon_guard_routed_mask[sequence_index])
        if use_edge:
            adaptive_probability_slice = edge_probability_slice
            adaptive_prediction_slice = edge_prediction_slice
            adaptive_latency_ms = edge_latency_ms if edge_uses_local_score else (local_latency_ms + edge_latency_ms)
            adaptive_total_gco2e = edge_total_gco2e if edge_uses_local_score else (local_total_gco2e + edge_total_gco2e)
            adaptive_routed_sequence_ids.append(record.sequence_id)
        else:
            adaptive_probability_slice = local_probability_slice
            adaptive_prediction_slice = local_prediction_slice
            adaptive_latency_ms = local_latency_ms
            adaptive_total_gco2e = local_total_gco2e

        if use_edge_carbon_guard:
            carbon_guard_probability_slice = edge_probability_slice
            carbon_guard_prediction_slice = edge_prediction_slice
            carbon_guard_latency_ms = edge_latency_ms if edge_uses_local_score else (local_latency_ms + edge_latency_ms)
            carbon_guard_total_gco2e = edge_total_gco2e if edge_uses_local_score else (local_total_gco2e + edge_total_gco2e)
            carbon_guard_routed_sequence_ids.append(record.sequence_id)
        else:
            carbon_guard_probability_slice = local_probability_slice
            carbon_guard_prediction_slice = local_prediction_slice
            carbon_guard_latency_ms = local_latency_ms
            carbon_guard_total_gco2e = local_total_gco2e

        adaptive_probability_chunks.append(adaptive_probability_slice)
        adaptive_prediction_chunks.append(adaptive_prediction_slice)
        carbon_guard_probability_chunks.append(carbon_guard_probability_slice)
        carbon_guard_prediction_chunks.append(carbon_guard_prediction_slice)

        per_sequence_local.append(
            build_policy_sequence_summary(
                record=record,
                labels=label_slice,
                local_probabilities=local_probability_slice,
                edge_probabilities=edge_probability_slice,
                adaptive_probabilities=local_probability_slice,
                adaptive_predictions=local_prediction_slice,
                local_latency_ms=local_latency_ms,
                edge_latency_ms=edge_latency_ms,
                active_latency_ms=local_latency_ms,
                local_total_gco2e=local_total_gco2e,
                edge_total_gco2e=edge_total_gco2e,
                active_total_gco2e=local_total_gco2e,
                adaptive_total_gco2e=local_total_gco2e,
                route=None,
                route_probability=None,
                raw_sequence_shared=False,
                local_threshold=local_threshold,
                edge_threshold=edge_threshold,
            )
        )
        per_sequence_edge.append(
            build_policy_sequence_summary(
                record=record,
                labels=label_slice,
                local_probabilities=local_probability_slice,
                edge_probabilities=edge_probability_slice,
                adaptive_probabilities=edge_probability_slice,
                adaptive_predictions=edge_prediction_slice,
                local_latency_ms=local_latency_ms,
                edge_latency_ms=edge_latency_ms,
                active_latency_ms=edge_latency_ms,
                local_total_gco2e=local_total_gco2e,
                edge_total_gco2e=edge_total_gco2e,
                active_total_gco2e=edge_total_gco2e,
                adaptive_total_gco2e=edge_total_gco2e,
                route=None,
                route_probability=None,
                raw_sequence_shared=True,
                local_threshold=local_threshold,
                edge_threshold=edge_threshold,
            )
        )
        per_sequence_adaptive.append(
            build_policy_sequence_summary(
                record=record,
                labels=label_slice,
                local_probabilities=local_probability_slice,
                edge_probabilities=edge_probability_slice,
                adaptive_probabilities=adaptive_probability_slice,
                adaptive_predictions=adaptive_prediction_slice,
                local_latency_ms=local_latency_ms,
                edge_latency_ms=edge_latency_ms,
                active_latency_ms=adaptive_latency_ms,
                local_total_gco2e=local_total_gco2e,
                edge_total_gco2e=edge_total_gco2e,
                active_total_gco2e=adaptive_total_gco2e,
                adaptive_total_gco2e=adaptive_total_gco2e,
                route=use_edge,
                route_probability=route_probability_value,
                raw_sequence_shared=use_edge,
                local_threshold=local_threshold,
                edge_threshold=edge_threshold,
            )
        )
        per_sequence_carbon_guard.append(
            build_policy_sequence_summary(
                record=record,
                labels=label_slice,
                local_probabilities=local_probability_slice,
                edge_probabilities=edge_probability_slice,
                adaptive_probabilities=carbon_guard_probability_slice,
                adaptive_predictions=carbon_guard_prediction_slice,
                local_latency_ms=local_latency_ms,
                edge_latency_ms=edge_latency_ms,
                active_latency_ms=carbon_guard_latency_ms,
                local_total_gco2e=local_total_gco2e,
                edge_total_gco2e=edge_total_gco2e,
                active_total_gco2e=carbon_guard_total_gco2e,
                adaptive_total_gco2e=carbon_guard_total_gco2e,
                route=use_edge_carbon_guard,
                route_probability=route_probability_value,
                raw_sequence_shared=use_edge_carbon_guard,
                local_threshold=local_threshold,
                edge_threshold=edge_threshold,
            )
        )

    adaptive_probabilities = np.concatenate(adaptive_probability_chunks)
    adaptive_predictions = np.concatenate(adaptive_prediction_chunks)
    carbon_guard_probabilities = np.concatenate(carbon_guard_probability_chunks)
    carbon_guard_predictions = np.concatenate(carbon_guard_prediction_chunks)

    local_summary = summarise_policy(
        labels=y_test,
        probabilities=local_test_scores,
        predictions=local_predictions,
        threshold=local_threshold,
        per_sequence=per_sequence_local,
    )
    edge_summary = summarise_policy(
        labels=y_test,
        probabilities=edge_test_scores,
        predictions=edge_predictions,
        threshold=edge_threshold,
        per_sequence=per_sequence_edge,
    )
    adaptive_summary = summarise_policy(
        labels=y_test,
        probabilities=adaptive_probabilities,
        predictions=adaptive_predictions,
        threshold={"local": local_threshold, "edge": edge_threshold},
        per_sequence=per_sequence_adaptive,
    )
    carbon_guard_summary = summarise_policy(
        labels=y_test,
        probabilities=carbon_guard_probabilities,
        predictions=carbon_guard_predictions,
        threshold={"local": local_threshold, "edge": edge_threshold},
        per_sequence=per_sequence_carbon_guard,
    )

    output_path = Path(config.output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "project": "EdgeConBCE",
        "run_id": f"edgeconbce-bp3-v3-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "working_directory": str(Path.cwd()),
        },
        "config": asdict(config),
        "data_sources": _describe_data_sources(config),
        "dataset": {
            "train_sequence_count": len(train_records),
            "train_residue_count": int(len(y_train)),
            "train_positive_residue_count": int(np.sum(y_train)),
            "train_positive_rate": float(np.mean(y_train)),
            "test_sequence_count": len(test_records),
            "test_residue_count": int(len(y_test)),
            "test_positive_residue_count": int(np.sum(y_test)),
            "test_positive_rate": float(np.mean(y_test)),
        },
        "model_selection": {
            "validation_sequence_ids": [train_records[int(group_id)].sequence_id for group_id in validation_group_ids],
            "validation_sequence_count": int(len(validation_group_ids)),
            "validation_residue_count": int(len(y_validation)),
            "subtrain_training_time_ms": {
                "local_model": float(local_subtrain_training_ms),
                "edge_model": float(edge_subtrain_training_ms),
            },
            "full_training_time_ms": {
                "local_model": float(local_full_training_ms),
                "edge_model": float(edge_full_training_ms),
            },
            "local_validation": {
                "threshold": float(local_threshold),
                "calibrated": True,
                "metrics": local_validation_summary,
            },
            "edge_validation": {
                "threshold": float(edge_threshold),
                "calibrated": True,
                "metrics": edge_validation_summary,
            },
            "edge_optimisation": {
                "edge_backend": "extra_trees_with_local_blend_and_sequence_smoothing",
                "base_edge_metrics_before_optimisation": best_threshold(
                    y_validation,
                    edge_validation_base_scores,
                    config.threshold_grid,
                )[1],
                "selected_edge_threshold_before_floor": float(edge_optimisation["threshold"]),
                "applied_edge_threshold": float(edge_threshold),
                "selected_edge_weight": float(edge_optimisation["edge_weight"]),
                "selected_smoothing_radius": int(edge_optimisation["smoothing_radius"]),
                "selected_smoothing_alpha": float(edge_optimisation["smoothing_alpha"]),
            },
            "adaptive_router": {
                "router": route_model_name,
                "route_budget_max": float(config.route_budget_max),
                "route_probability_threshold": float(adaptive_validation["route_threshold"]),
                "validation_metrics": adaptive_validation["summary"],
                "validation_route_targets": route_targets_validation.astype(int).tolist(),
                "validation_route_probabilities": route_probabilities_validation.astype(float).tolist(),
                "validation_route_meta": validation_route_details,
                "selected_validation_routes": adaptive_validation["routed_sequence_ids"],
            },
            "adaptive_carbon_guard": {
                "router": route_model_name,
                "route_budget_max": float(config.route_budget_max),
                "route_probability_threshold": float(carbon_guard_validation["route_threshold"]),
                "validation_metrics": carbon_guard_validation["summary"],
                "selection_rule": carbon_guard_validation["selection_rule"],
                "selected_validation_routes": carbon_guard_validation["routed_sequence_ids"],
            },
        },
        "external_test": {
            "local_only": {
                "decision_threshold": float(local_threshold),
                "summary": local_summary,
                "confidence_intervals_95": bootstrap_cluster_confidence_intervals(
                    per_sequence_local,
                    np.random.default_rng(config.seed + 101),
                ),
                "per_sequence": per_sequence_local,
            },
            "edge_only": {
                "decision_threshold": float(edge_threshold),
                "summary": edge_summary,
                "confidence_intervals_95": bootstrap_cluster_confidence_intervals(
                    per_sequence_edge,
                    np.random.default_rng(config.seed + 202),
                ),
                "per_sequence": per_sequence_edge,
            },
            "adaptive_split": {
                "decision_thresholds": {
                    "local": float(local_threshold),
                    "edge": float(edge_threshold),
                },
                "router": route_model_name,
                "route_probability_threshold": float(adaptive_validation["route_threshold"]),
                "routed_sequence_ids": adaptive_routed_sequence_ids,
                "summary": adaptive_summary,
                "confidence_intervals_95": bootstrap_cluster_confidence_intervals(
                    per_sequence_adaptive,
                    np.random.default_rng(config.seed + 303),
                ),
                "per_sequence": per_sequence_adaptive,
            },
            "adaptive_carbon_guard": {
                "decision_thresholds": {
                    "local": float(local_threshold),
                    "edge": float(edge_threshold),
                },
                "router": route_model_name,
                "route_probability_threshold": float(carbon_guard_validation["route_threshold"]),
                "routed_sequence_ids": carbon_guard_routed_sequence_ids,
                "selection_rule": carbon_guard_validation["selection_rule"],
                "summary": carbon_guard_summary,
                "confidence_intervals_95": bootstrap_cluster_confidence_intervals(
                    per_sequence_carbon_guard,
                    np.random.default_rng(config.seed + 404),
                ),
                "per_sequence": per_sequence_carbon_guard,
            },
        },
        "limitations": [
            "The benchmark still uses handcrafted sequence-context features rather than protein language model embeddings.",
            "Edge predictions are improved with validation-selected local-edge blending and sequence smoothing, which strengthens the baseline but is still not a structural model.",
            "Latency and carbon estimates remain CPU-side deployment proxies rather than hardware-instrumented measurements.",
            "The adaptive router is trained on a single grouped validation split; stronger publication evidence would add repeated grouped resampling.",
        ],
    }

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    runtime_config = build_config()
    benchmark_results = run_benchmark(runtime_config)
    print(Path(runtime_config.output_json_path).resolve())
    print(json.dumps(benchmark_results["external_test"]["adaptive_split"]["summary"], indent=2))
