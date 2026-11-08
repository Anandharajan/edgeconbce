from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from math import ceil
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any
import argparse
import os
import re
import warnings

import numpy as np
import torch
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.serialization import add_safe_globals

import esm

from edgeconbce.metrics.regions import summarize_region_predictions
from edgeconbce.schemas import LegacyBP3RealV3Config


DEFAULT_ESM_MODEL_NAME = "esm2_t6_8M_UR50D"


def _resolve_esm_model_name() -> str:
    """The ESM-2 checkpoint to use, overridable without editing this module.

    Set ``EDGECONBCE_ESM_MODEL`` to run the PLM family on a larger checkpoint,
    for example ``esm2_t33_650M_UR50D`` on a 16 GB accelerator or
    ``esm2_t12_35M_UR50D`` on a 4 GB one. The default keeps earlier runs
    reproducible.
    """
    return os.environ.get("EDGECONBCE_ESM_MODEL", DEFAULT_ESM_MODEL_NAME).strip() or DEFAULT_ESM_MODEL_NAME


def _resolve_esm_layer(model_name: str) -> int:
    """Final encoder layer for an ESM-2 checkpoint, taken from its name.

    ESM-2 checkpoints are named ``esm2_t<layers>_<size>_UR50D``; representations
    are read from the last layer, so the layer index follows the checkpoint and
    must not be left at a value belonging to a different one.
    """
    match = re.search(r"_t(\d+)_", model_name)
    if match is None:
        raise ValueError(
            f"Cannot infer the encoder depth from ESM checkpoint name {model_name!r}; "
            "expected a name of the form esm2_t<layers>_<size>_UR50D."
        )
    return int(match.group(1))


def _resolve_fit_jobs() -> int:
    """Worker count for the tree ensembles, overridable with EDGECONBCE_N_JOBS.

    The edge refiner dominates run time (roughly 330 s of a 332 s seed in the
    10-seed BP3 runs) and was previously fitted on a single core. Forest fitting
    in scikit-learn is deterministic given ``random_state`` regardless of the
    worker count, so this changes wall-clock time and nothing else; a run with
    ``EDGECONBCE_N_JOBS=1`` reproduces earlier artifacts exactly.
    """
    raw = os.environ.get("EDGECONBCE_N_JOBS", "-1").strip()
    try:
        value = int(raw)
    except ValueError:
        return -1
    return value if value != 0 else 1


FIT_N_JOBS = _resolve_fit_jobs()
ESM_MODEL_NAME = _resolve_esm_model_name()
ESM_LAYER = _resolve_esm_layer(ESM_MODEL_NAME)
CONTACT_BOOTSTRAP_REPEATS = 200
CONFORMAL_ALPHA = 0.20
CONFORMAL_LENGTH_PENALTIES = (0.0, 0.05, 0.10, 0.15, 0.20)
REGION_RADII = (3, 5, 8)
REGION_OUTSIDE_SCALES = (0.25, 0.40, 0.55)


@dataclass(frozen=True)
class AnnotatedSequence:
    sequence_id: str
    annotated_sequence: str

    @property
    def sequence(self) -> str:
        return self.annotated_sequence.upper()

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([1 if residue.isupper() else 0 for residue in self.annotated_sequence], dtype=np.int8)


_ESM_CACHE: dict[str, Any] = {}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _esm_checkpoint_path() -> Path:
    return _project_root() / ".torchcache" / "hub" / "checkpoints" / f"{ESM_MODEL_NAME}.pt"


def _esm_contact_regression_path() -> Path:
    return _project_root() / ".torchcache" / "hub" / "checkpoints" / f"{ESM_MODEL_NAME}-contact-regression.pt"


def _esm_cache_dir() -> Path:
    return _project_root() / "outputs" / "feature_cache" / ESM_MODEL_NAME


def _ensure_esm_loaded() -> tuple[Any, Any, Any]:
    cached = _ESM_CACHE.get("model_bundle")
    if cached is not None:
        return cached

    checkpoint_path = _esm_checkpoint_path()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing local ESM checkpoint at {checkpoint_path}. "
            "Download the checkpoint into the workspace before running the modern extension."
        )
    add_safe_globals([argparse.Namespace])
    contact_regression_path = _esm_contact_regression_path()
    contacts_available = contact_regression_path.exists()
    if contacts_available:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(checkpoint_path))
    else:
        warnings.warn(
            f"Missing local ESM contact-regression checkpoint at {contact_regression_path}; "
            "continuing with contact-derived features set to zero.",
            RuntimeWarning,
            stacklevel=2,
        )
        model_data = torch.load(str(checkpoint_path), map_location="cpu")
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(checkpoint_path.stem, model_data, regression_data=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    bundle = (model, alphabet, batch_converter)
    _ESM_CACHE["model_bundle"] = bundle
    _ESM_CACHE["contacts_available"] = contacts_available
    return bundle


def _esm_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _parse_fasta(path: str) -> list[AnnotatedSequence]:
    records: list[AnnotatedSequence] = []
    header: str | None = None
    parts: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append(AnnotatedSequence(sequence_id=header, annotated_sequence="".join(parts)))
            header = line[1:]
            parts = []
            continue
        parts.append(line)
    if header is not None:
        records.append(AnnotatedSequence(sequence_id=header, annotated_sequence="".join(parts)))
    return records


def _cache_path_for_sequence(sequence: str) -> Path:
    digest = sha1(sequence.encode("utf-8")).hexdigest()
    return _esm_cache_dir() / f"{digest}.npz"


def _smooth_vector(values: np.ndarray, radius: int = 2) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if radius <= 0 or len(values) == 0:
        return values.astype(np.float32)
    kernel = np.ones((2 * radius) + 1, dtype=np.float64)
    kernel /= float(kernel.sum())
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _compute_plm_features(sequence: str) -> dict[str, np.ndarray]:
    model, _, batch_converter = _ensure_esm_loaded()
    _, _, batch_tokens = batch_converter([("sequence", sequence)])
    batch_tokens = batch_tokens.to(_esm_device(model))
    contacts_available = bool(_ESM_CACHE.get("contacts_available", False))
    with torch.no_grad():
        outputs = model(batch_tokens, repr_layers=[ESM_LAYER], return_contacts=contacts_available)

    embeddings = outputs["representations"][ESM_LAYER][0, 1 : len(sequence) + 1].cpu().numpy().astype(np.float32)
    if contacts_available and "contacts" in outputs:
        contacts = outputs["contacts"][0].cpu().numpy().astype(np.float32)
    else:
        contacts = np.zeros((len(sequence), len(sequence)), dtype=np.float32)

    if len(sequence) == 1:
        contact_density = np.zeros(1, dtype=np.float32)
        long_range_density = np.zeros(1, dtype=np.float32)
    else:
        contact_density = ((contacts.sum(axis=1) - np.diag(contacts)) / max(1, len(sequence) - 1)).astype(np.float32)
        contact_mask = np.abs(np.subtract.outer(np.arange(len(sequence)), np.arange(len(sequence)))) >= 6
        long_range_counts = np.maximum(1, contact_mask.sum(axis=1))
        long_range_density = ((contacts * contact_mask).sum(axis=1) / long_range_counts).astype(np.float32)

    return {
        "embeddings": embeddings,
        "contact_density": contact_density,
        "contact_long_range": long_range_density,
        "contact_density_smooth": _smooth_vector(contact_density, radius=2),
    }


def _load_or_compute_features(sequence: str) -> dict[str, np.ndarray]:
    cache_path = _cache_path_for_sequence(sequence)
    if cache_path.exists():
        cached = np.load(cache_path)
        return {
            "embeddings": cached["embeddings"].astype(np.float32),
            "contact_density": cached["contact_density"].astype(np.float32),
            "contact_long_range": cached["contact_long_range"].astype(np.float32),
            "contact_density_smooth": cached["contact_density_smooth"].astype(np.float32),
        }

    features = _compute_plm_features(sequence)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=features["embeddings"].astype(np.float16),
        contact_density=features["contact_density"].astype(np.float16),
        contact_long_range=features["contact_long_range"].astype(np.float16),
        contact_density_smooth=features["contact_density_smooth"].astype(np.float16),
    )
    return features


def _local_feature_rows(feature_bundle: dict[str, np.ndarray]) -> np.ndarray:
    embeddings = feature_bundle["embeddings"]
    length = embeddings.shape[0]
    positions = np.arange(length, dtype=np.float32)
    length_norm = np.full((length, 1), length / 1023.0, dtype=np.float32)
    pos_norm = (positions / max(1.0, float(length - 1))).reshape(-1, 1).astype(np.float32)
    edge_distance = (np.minimum(positions + 1.0, float(length) - positions) / max(1.0, float(length))).reshape(-1, 1).astype(np.float32)
    return np.concatenate([embeddings, length_norm, pos_norm, edge_distance], axis=1).astype(np.float32)


def _edge_feature_rows(feature_bundle: dict[str, np.ndarray], local_probabilities: np.ndarray) -> np.ndarray:
    base_rows = _local_feature_rows(feature_bundle)
    local_probabilities = np.asarray(local_probabilities, dtype=np.float32)
    uncertainty = (1.0 - np.abs(local_probabilities - 0.5) * 2.0).astype(np.float32)
    local_positive_context = _smooth_vector(local_probabilities, radius=2)
    extras = np.column_stack(
        [
            local_probabilities,
            uncertainty,
            feature_bundle["contact_density"],
            feature_bundle["contact_long_range"],
            feature_bundle["contact_density_smooth"],
            local_positive_context,
        ]
    ).astype(np.float32)
    return np.concatenate([base_rows, extras], axis=1).astype(np.float32)


def _stack_rows(
    records: list[AnnotatedSequence],
    feature_map: dict[str, dict[str, np.ndarray]],
    *,
    edge_mode: bool,
    local_scores: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for record in records:
        bundle = feature_map[record.sequence_id]
        if edge_mode:
            if local_scores is None:
                raise ValueError("Edge mode requires local scores.")
            rows.append(_edge_feature_rows(bundle, local_scores[record.sequence_id]))
        else:
            rows.append(_local_feature_rows(bundle))
        labels.append(record.labels)
    return np.concatenate(rows, axis=0), np.concatenate(labels, axis=0)


def _predict_by_sequence(
    model: Any,
    records: list[AnnotatedSequence],
    feature_map: dict[str, dict[str, np.ndarray]],
    *,
    edge_mode: bool,
    local_scores: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for record in records:
        bundle = feature_map[record.sequence_id]
        if edge_mode:
            if local_scores is None:
                raise ValueError("Edge mode requires local scores.")
            rows = _edge_feature_rows(bundle, local_scores[record.sequence_id])
        else:
            rows = _local_feature_rows(bundle)
        predictions[record.sequence_id] = model.predict_proba(rows)[:, 1].astype(np.float64)
    return predictions


def _calibrate_scores(reference_scores: np.ndarray, reference_labels: np.ndarray, apply_scores: np.ndarray) -> np.ndarray:
    reference_scores = np.asarray(reference_scores, dtype=np.float64)
    reference_labels = np.asarray(reference_labels, dtype=np.int8)
    apply_scores = np.asarray(apply_scores, dtype=np.float64)
    if len(np.unique(reference_labels)) < 2:
        return np.clip(apply_scores, 0.0, 1.0)
    calibrator = LogisticRegression(max_iter=200, class_weight="balanced", random_state=42)
    calibrator.fit(reference_scores.reshape(-1, 1), reference_labels)
    return calibrator.predict_proba(apply_scores.reshape(-1, 1))[:, 1]


def _score_summary(y_true: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
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
        summary["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        summary["average_precision"] = float(average_precision_score(y_true, probabilities))
    return summary


def _best_threshold(y_true: np.ndarray, scores: np.ndarray, threshold_grid: tuple[float, ...]) -> tuple[float, dict[str, float | None]]:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    prevalence = float(np.mean(y_true))
    best_threshold = 0.5
    best_summary: dict[str, float | None] | None = None
    best_objective = (-1.0, -1.0, -1.0, -1.0)
    for threshold in threshold_grid:
        predictions = (scores >= float(threshold)).astype(np.int8)
        predicted_positive_rate = float(np.mean(predictions))
        summary = _score_summary(y_true, scores, predictions)
        objective = (
            float(summary["mcc"]),
            -abs(predicted_positive_rate - prevalence),
            float(summary["precision"]),
            float(summary["f1"]),
        )
        if objective > best_objective:
            best_threshold = float(threshold)
            best_summary = summary
            best_objective = objective
    assert best_summary is not None
    return best_threshold, best_summary


def _longest_positive_run(predictions: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in predictions.astype(np.int8):
        if value == 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _normalise_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value <= min_value:
        return np.zeros(values.shape, dtype=np.float32)
    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def _region_constrained_scores(
    probabilities: np.ndarray,
    feature_bundle: dict[str, np.ndarray],
    *,
    threshold: float,
    radius: int,
    outside_scale: float,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    length = probabilities.shape[0]
    if length == 0:
        return probabilities

    local_signal = _smooth_vector(probabilities, radius=2).astype(np.float64)
    contact_signal = _normalise_vector(feature_bundle["contact_density_smooth"]).astype(np.float64)
    region_signal = (0.85 * local_signal) + (0.15 * contact_signal)
    base_predictions = probabilities >= float(threshold)
    target_positive_count = int(np.sum(base_predictions))
    if target_positive_count == 0:
        target_positive_count = max(1, int(round(float(np.mean(probabilities >= np.quantile(probabilities, 0.90))) * length)))

    window_scores: list[tuple[float, int, int]] = []
    for center in range(length):
        start = max(0, center - int(radius))
        end = min(length, center + int(radius) + 1)
        if start >= end:
            continue
        window_scores.append((float(np.mean(region_signal[start:end])), start, end))

    selected = np.zeros(length, dtype=bool)
    for _, start, end in sorted(window_scores, key=lambda item: item[0], reverse=True):
        selected[start:end] = True
        if int(np.sum(selected)) >= target_positive_count:
            break

    constrained = probabilities.copy()
    constrained[~selected] *= float(outside_scale)
    constrained[selected] = np.maximum(constrained[selected], local_signal[selected])
    return np.clip(constrained, 0.0, 1.0)


def _region_summary_for_records(
    records: list[AnnotatedSequence],
    score_map: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    labels = np.concatenate([record.labels for record in records], axis=0)
    predictions = np.concatenate([(score_map[record.sequence_id] >= threshold).astype(np.int8) for record in records], axis=0)
    return summarize_region_predictions(labels, predictions)


def _select_region_constraint(
    records: list[AnnotatedSequence],
    edge_scores: dict[str, np.ndarray],
    feature_map: dict[str, dict[str, np.ndarray]],
    edge_threshold: float,
) -> dict[str, Any]:
    best_result: dict[str, Any] | None = None
    best_objective = (-1.0, -1.0, -1.0)
    labels = np.concatenate([record.labels for record in records], axis=0)
    for radius in REGION_RADII:
        for outside_scale in REGION_OUTSIDE_SCALES:
            constrained_scores = {
                record.sequence_id: _region_constrained_scores(
                    edge_scores[record.sequence_id],
                    feature_map[record.sequence_id],
                    threshold=edge_threshold,
                    radius=radius,
                    outside_scale=outside_scale,
                )
                for record in records
            }
            probabilities = np.concatenate([constrained_scores[record.sequence_id] for record in records], axis=0)
            predictions = (probabilities >= edge_threshold).astype(np.int8)
            score_summary = _score_summary(labels, probabilities, predictions)
            region_summary = summarize_region_predictions(labels, predictions)
            objective = (
                float(score_summary["mcc"]),
                float(score_summary["f1"]),
                float(region_summary["largest_predicted_region_fraction"]),
            )
            if best_result is None or objective > best_objective:
                best_result = {
                    "radius": int(radius),
                    "outside_scale": float(outside_scale),
                    "validation_metrics": score_summary,
                    "validation_region_metrics": region_summary,
                }
                best_objective = objective
    assert best_result is not None
    return best_result


def _route_features(length: int, local_scores: np.ndarray, threshold: float) -> np.ndarray:
    local_scores = np.asarray(local_scores, dtype=np.float64)
    predictions = (local_scores >= threshold).astype(np.int8)
    if len(predictions) > 1:
        boundary_shift = float(np.mean(np.abs(np.diff(predictions))))
    else:
        boundary_shift = 0.0
    uncertainty = 1.0 - np.abs(local_scores - 0.5) * 2.0
    return np.asarray(
        [
            length / 1023.0,
            float(np.mean(local_scores)),
            float(np.std(local_scores)),
            float(np.quantile(local_scores, 0.10)),
            float(np.quantile(local_scores, 0.90)),
            float(np.mean(uncertainty)),
            float(np.mean(predictions)),
            _longest_positive_run(predictions) / max(1, length),
            boundary_shift,
        ],
        dtype=np.float32,
    )


def _sequence_uncertainty_score(local_scores: np.ndarray) -> float:
    local_scores = np.asarray(local_scores, dtype=np.float64)
    return float(np.mean(1.0 - np.abs(local_scores - 0.5) * 2.0))


def _sequence_gain_target(
    labels: np.ndarray,
    local_scores: np.ndarray,
    edge_scores: np.ndarray,
    local_threshold: float,
    edge_threshold: float,
) -> tuple[int, dict[str, float]]:
    local_predictions = (local_scores >= local_threshold).astype(np.int8)
    edge_predictions = (edge_scores >= edge_threshold).astype(np.int8)
    local_summary = _score_summary(labels, local_scores, local_predictions)
    edge_summary = _score_summary(labels, edge_scores, edge_predictions)
    local_mcc = float(local_summary["mcc"])
    edge_mcc = float(edge_summary["mcc"])
    local_ap = float(local_summary["average_precision"] or 0.0)
    edge_ap = float(edge_summary["average_precision"] or 0.0)
    gain = (edge_mcc - local_mcc) + (0.25 * (edge_ap - local_ap))
    return (
        1 if gain > 0.01 else 0,
        {
            "gain": round(float(gain), 6),
            "local_mcc": round(local_mcc, 6),
            "edge_mcc": round(edge_mcc, 6),
            "local_average_precision": round(local_ap, 6),
            "edge_average_precision": round(edge_ap, 6),
        },
    )


def _build_route_dataset(
    records: list[AnnotatedSequence],
    local_scores: dict[str, np.ndarray],
    edge_scores: dict[str, np.ndarray],
    local_threshold: float,
    edge_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    route_features: list[np.ndarray] = []
    route_targets: list[int] = []
    route_meta: list[dict[str, Any]] = []
    for record in records:
        features = _route_features(len(record.sequence), local_scores[record.sequence_id], local_threshold)
        target, meta = _sequence_gain_target(
            record.labels,
            local_scores[record.sequence_id],
            edge_scores[record.sequence_id],
            local_threshold,
            edge_threshold,
        )
        route_features.append(features)
        route_targets.append(target)
        route_meta.append(
            {
                "sequence_id": record.sequence_id,
                "sequence_length": len(record.sequence),
                "target_route_to_edge": bool(target),
                **meta,
            }
        )
    return np.stack(route_features), np.asarray(route_targets, dtype=np.int8), route_meta


def _evaluate_route_threshold(
    records: list[AnnotatedSequence],
    route_probabilities: np.ndarray,
    local_scores: dict[str, np.ndarray],
    edge_scores: dict[str, np.ndarray],
    local_threshold: float,
    edge_threshold: float,
    threshold: float,
) -> dict[str, Any]:
    route_mask = route_probabilities >= float(threshold)
    all_labels: list[int] = []
    all_probabilities: list[float] = []
    all_predictions: list[int] = []
    for index, record in enumerate(records):
        use_edge = bool(route_mask[index])
        scores = edge_scores[record.sequence_id] if use_edge else local_scores[record.sequence_id]
        active_threshold = edge_threshold if use_edge else local_threshold
        predictions = (scores >= active_threshold).astype(np.int8)
        all_labels.extend(int(value) for value in record.labels)
        all_probabilities.extend(float(value) for value in scores)
        all_predictions.extend(int(value) for value in predictions)
    summary = _score_summary(
        np.asarray(all_labels, dtype=np.int8),
        np.asarray(all_probabilities, dtype=np.float64),
        np.asarray(all_predictions, dtype=np.int8),
    )
    return {
        "summary": summary,
        "route_rate": float(np.mean(route_mask)) if len(route_mask) else 0.0,
        "routed_sequence_ids": [record.sequence_id for index, record in enumerate(records) if route_mask[index]],
    }


def _select_adaptive_threshold(
    records: list[AnnotatedSequence],
    route_probabilities: np.ndarray,
    local_scores: dict[str, np.ndarray],
    edge_scores: dict[str, np.ndarray],
    local_threshold: float,
    edge_threshold: float,
    route_probability_grid: tuple[float, ...],
    route_budget_max: float,
) -> dict[str, Any]:
    best_result: dict[str, Any] | None = None
    best_objective = (-1.0, -1.0, -1.0)
    for threshold in route_probability_grid:
        candidate = _evaluate_route_threshold(
            records,
            route_probabilities,
            local_scores,
            edge_scores,
            local_threshold,
            edge_threshold,
            float(threshold),
        )
        if candidate["route_rate"] > route_budget_max:
            continue
        summary = candidate["summary"]
        objective = (
            float(summary["mcc"]),
            float(summary["f1"]),
            -float(candidate["route_rate"]),
        )
        if objective > best_objective:
            best_result = {
                "route_threshold": float(threshold),
                "summary": summary,
                "routed_sequence_ids": candidate["routed_sequence_ids"],
            }
            best_objective = objective
    if best_result is None:
        best_result = {
            "route_threshold": 1.01,
            "summary": _evaluate_route_threshold(
                records,
                route_probabilities,
                local_scores,
                edge_scores,
                local_threshold,
                edge_threshold,
                1.01,
            )["summary"],
            "routed_sequence_ids": [],
        }
    return best_result


def _route_count_from_rate(num_sequences: int, route_rate: float) -> int:
    if num_sequences <= 0 or route_rate <= 0.0:
        return 0
    return min(num_sequences, max(1, int(round(num_sequences * route_rate))))


def _ranked_route_mask(priority_values: np.ndarray, route_count: int) -> np.ndarray:
    priority_values = np.asarray(priority_values, dtype=np.float64)
    if route_count <= 0:
        return np.zeros(priority_values.shape[0], dtype=bool)
    order = np.argsort(-priority_values, kind="mergesort")
    mask = np.zeros(priority_values.shape[0], dtype=bool)
    mask[order[:route_count]] = True
    return mask


def _select_conformal_policy(
    records: list[AnnotatedSequence],
    route_probabilities: np.ndarray,
    route_targets: np.ndarray,
    local_scores: dict[str, np.ndarray],
    edge_scores: dict[str, np.ndarray],
    local_threshold: float,
    edge_threshold: float,
    route_budget_max: float,
    minimum_threshold: float = 0.0,
) -> dict[str, Any]:
    helpful = route_probabilities[route_targets == 1]
    if helpful.size == 0:
        return {
            "route_probability_threshold": 1.01,
            "selection_rule": {
                "method": "conformal_positive_quantile",
                "alpha": CONFORMAL_ALPHA,
                "length_penalty": 0.0,
                "target_route_rate": route_budget_max,
                "max_routed_sequences": 0,
            },
            "summary": _evaluate_route_threshold(
                records,
                route_probabilities,
                local_scores,
                edge_scores,
                local_threshold,
                edge_threshold,
                1.01,
            )["summary"],
            "routed_sequence_ids": [],
        }

    nonconformity = 1.0 - helpful
    quantile_level = min(1.0, ceil((helpful.size + 1) * (1.0 - CONFORMAL_ALPHA)) / helpful.size)
    base_quantile = float(np.quantile(nonconformity, quantile_level, method="higher"))
    base_threshold = max(float(minimum_threshold), max(0.0, min(1.0, 1.0 - base_quantile)))
    candidate_mask = route_probabilities >= base_threshold
    max_routes = _route_count_from_rate(len(records), route_budget_max)
    sequence_lengths = np.asarray([len(record.sequence) for record in records], dtype=np.float64)
    length_norm = sequence_lengths / max(1.0, float(sequence_lengths.max(initial=1.0)))

    best_result: dict[str, Any] | None = None
    best_objective = (-1.0, -1.0, 1.0)
    for length_penalty in CONFORMAL_LENGTH_PENALTIES:
        priority = route_probabilities - (float(length_penalty) * length_norm)
        if candidate_mask.sum() > max_routes > 0:
            restricted_priority = np.where(candidate_mask, priority, -1e9)
            route_mask = _ranked_route_mask(restricted_priority, max_routes)
        else:
            route_mask = candidate_mask.copy()

        routed_ids = [record.sequence_id for index, record in enumerate(records) if route_mask[index]]
        summary = _evaluate_route_threshold(
            records,
            np.where(route_mask, 1.0, 0.0),
            local_scores,
            edge_scores,
            local_threshold,
            edge_threshold,
            0.5,
        )["summary"]
        route_rate = float(np.mean(route_mask)) if len(route_mask) else 0.0
        objective = (
            float(summary["mcc"]),
            float(summary["f1"]),
            -route_rate,
        )
        if best_result is None or objective > best_objective:
            best_result = {
                "route_probability_threshold": round(base_threshold, 6),
                "selection_rule": {
                    "method": "conformal_positive_quantile",
                    "alpha": CONFORMAL_ALPHA,
                    "length_penalty": float(length_penalty),
                    "target_route_rate": float(route_budget_max),
                    "max_routed_sequences": int(max_routes),
                },
                "summary": summary,
                "routed_sequence_ids": routed_ids,
                "route_mask": route_mask,
            }
            best_objective = objective

    assert best_result is not None
    return best_result


def _operational_gco2e(latency_ms: float, power_w: float, grid_intensity_gco2_per_kwh: float) -> float:
    energy_kwh = float(power_w) * (float(latency_ms) / 3_600_000.0) / 1000.0
    return energy_kwh * float(grid_intensity_gco2_per_kwh)


def _embodied_gco2e_per_request(total_kgco2e: float, amortization_requests: int, fraction_used: float) -> float:
    return (float(total_kgco2e) / int(amortization_requests)) * float(fraction_used) * 1000.0


def _measure_forward_latency_ms(
    sequence: str,
    local_model: Any,
    edge_model: Any,
    feature_bundle: dict[str, np.ndarray],
) -> tuple[float, float]:
    model, _, batch_converter = _ensure_esm_loaded()
    _, _, batch_tokens = batch_converter([("sequence", sequence)])
    batch_tokens = batch_tokens.to(_esm_device(model))

    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = perf_counter()
        local_outputs = model(batch_tokens, repr_layers=[ESM_LAYER], return_contacts=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        local_embeddings = local_outputs["representations"][ESM_LAYER][0, 1 : len(sequence) + 1].cpu().numpy().astype(np.float32)
        local_rows = np.concatenate(
            [
                local_embeddings,
                np.full((len(sequence), 1), len(sequence) / 1023.0, dtype=np.float32),
                (np.arange(len(sequence), dtype=np.float32) / max(1, len(sequence) - 1)).reshape(-1, 1),
                (np.minimum(np.arange(len(sequence), dtype=np.float32) + 1.0, float(len(sequence)) - np.arange(len(sequence), dtype=np.float32)) / max(1.0, float(len(sequence)))).reshape(-1, 1),
            ],
            axis=1,
        ).astype(np.float32)
        local_probabilities = local_model.predict_proba(local_rows)[:, 1]
        local_latency_ms = (perf_counter() - start) * 1000.0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = perf_counter()
        edge_outputs = model(batch_tokens, repr_layers=[ESM_LAYER], return_contacts=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        edge_embeddings = edge_outputs["representations"][ESM_LAYER][0, 1 : len(sequence) + 1].cpu().numpy().astype(np.float32)
        contacts = edge_outputs["contacts"][0].cpu().numpy().astype(np.float32)
        contact_density = ((contacts.sum(axis=1) - np.diag(contacts)) / max(1, len(sequence) - 1)).astype(np.float32) if len(sequence) > 1 else np.zeros(1, dtype=np.float32)
        contact_mask = np.abs(np.subtract.outer(np.arange(len(sequence)), np.arange(len(sequence)))) >= 6
        long_range_counts = np.maximum(1, contact_mask.sum(axis=1))
        contact_long_range = ((contacts * contact_mask).sum(axis=1) / long_range_counts).astype(np.float32) if len(sequence) > 1 else np.zeros(1, dtype=np.float32)
        contact_density_smooth = _smooth_vector(contact_density, radius=2)
        uncertainty = (1.0 - np.abs(local_probabilities - 0.5) * 2.0).astype(np.float32)
        local_positive_context = _smooth_vector(local_probabilities, radius=2)
        edge_rows = np.concatenate(
            [
                np.concatenate(
                    [
                        edge_embeddings,
                        np.full((len(sequence), 1), len(sequence) / 1023.0, dtype=np.float32),
                        (np.arange(len(sequence), dtype=np.float32) / max(1, len(sequence) - 1)).reshape(-1, 1),
                        (np.minimum(np.arange(len(sequence), dtype=np.float32) + 1.0, float(len(sequence)) - np.arange(len(sequence), dtype=np.float32)) / max(1.0, float(len(sequence)))).reshape(-1, 1),
                    ],
                    axis=1,
                ),
                np.column_stack(
                    [
                        local_probabilities.astype(np.float32),
                        uncertainty,
                        contact_density,
                        contact_long_range,
                        contact_density_smooth,
                        local_positive_context,
                    ]
                ).astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        edge_model.predict_proba(edge_rows)
        edge_latency_ms = (perf_counter() - start) * 1000.0

    return float(local_latency_ms), float(edge_latency_ms)


def _sequence_summary_row(
    record: AnnotatedSequence,
    local_probabilities: np.ndarray,
    edge_probabilities: np.ndarray,
    active_probabilities: np.ndarray,
    active_predictions: np.ndarray,
    local_latency_ms: float,
    edge_latency_ms: float,
    active_latency_ms: float,
    local_total_gco2e: float,
    edge_total_gco2e: float,
    active_total_gco2e: float,
    *,
    route: bool | None,
    route_probability: float | None,
    raw_sequence_shared: bool,
) -> dict[str, Any]:
    return {
        "sequence_id": record.sequence_id,
        "length": len(record.sequence),
        "labels": record.labels.astype(int).tolist(),
        "local_probabilities": np.asarray(local_probabilities, dtype=np.float64).tolist(),
        "edge_probabilities": np.asarray(edge_probabilities, dtype=np.float64).tolist(),
        "adaptive_probabilities": np.asarray(active_probabilities, dtype=np.float64).tolist(),
        "adaptive_predictions": np.asarray(active_predictions, dtype=np.int8).astype(int).tolist(),
        "local_latency_ms": round(float(local_latency_ms), 6),
        "edge_latency_ms": round(float(edge_latency_ms), 6),
        "active_latency_ms": round(float(active_latency_ms), 6),
        "local_total_gco2e": round(float(local_total_gco2e), 6),
        "edge_total_gco2e": round(float(edge_total_gco2e), 6),
        "adaptive_total_gco2e": round(float(active_total_gco2e), 6),
        "active_total_gco2e": round(float(active_total_gco2e), 6),
        "route": route,
        "route_probability": None if route_probability is None else round(float(route_probability), 6),
        "raw_sequence_shared": bool(raw_sequence_shared),
        "positive_prediction_count": int(np.sum(active_predictions)),
        "positive_residue_count": int(np.sum(record.labels)),
        "mean_local_uncertainty": round(float(np.mean(1.0 - np.abs(np.asarray(local_probabilities, dtype=np.float64) - 0.5) * 2.0)), 6),
        "diagnostics": {
            "local_probability_mean": round(float(np.mean(local_probabilities)), 6),
            "edge_probability_mean": round(float(np.mean(edge_probabilities)), 6),
        },
    }


def _policy_summary(per_sequence_rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([value for row in per_sequence_rows for value in row["labels"]], dtype=np.int8)
    probabilities = np.asarray([value for row in per_sequence_rows for value in row["adaptive_probabilities"]], dtype=np.float64)
    predictions = np.asarray([value for row in per_sequence_rows for value in row["adaptive_predictions"]], dtype=np.int8)
    summary = _score_summary(labels, probabilities, predictions)
    region_summary = summarize_region_predictions(labels, predictions)
    route_values = [row["route"] for row in per_sequence_rows if row["route"] is not None]
    summary.update(
        {
            "mean_latency_ms": round(mean(float(row["active_latency_ms"]) for row in per_sequence_rows), 6),
            "mean_total_gco2e": round(mean(float(row["active_total_gco2e"]) for row in per_sequence_rows), 6),
            "route_rate": None if not route_values else round(sum(1 for value in route_values if value) / len(route_values), 6),
            "region_metrics": region_summary,
        }
    )
    return summary


def _policy_bootstrap_confidence_intervals(per_sequence_rows: list[dict[str, Any]], seed: int) -> dict[str, list[float]]:
    if not per_sequence_rows:
        return {}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "mcc": [],
        "f1": [],
        "roc_auc": [],
        "average_precision": [],
        "mean_latency_ms": [],
        "mean_total_gco2e": [],
        "route_rate": [],
    }
    for _ in range(CONTACT_BOOTSTRAP_REPEATS):
        sampled_rows = [per_sequence_rows[int(index)] for index in rng.integers(0, len(per_sequence_rows), size=len(per_sequence_rows))]
        summary = _policy_summary(sampled_rows)
        for key in samples:
            value = summary.get(key)
            if value is not None:
                samples[key].append(float(value))
    intervals: dict[str, list[float]] = {}
    for key, values in samples.items():
        if not values:
            continue
        intervals[key] = [
            round(float(np.quantile(values, 0.025)), 6),
            round(float(np.quantile(values, 0.975)), 6),
        ]
    return intervals


def _record_map(records: list[AnnotatedSequence]) -> dict[str, AnnotatedSequence]:
    return {record.sequence_id: record for record in records}


def extend_with_plm_policies(results: dict[str, Any], config: LegacyBP3RealV3Config) -> dict[str, Any]:
    train_records = _parse_fasta(config.train_fasta_path)
    test_records = _parse_fasta(config.test_fasta_path)
    validation_ids = set(str(sequence_id) for sequence_id in results["model_selection"]["validation_sequence_ids"])
    train_record_map = _record_map(train_records)

    validation_records = [train_record_map[sequence_id] for sequence_id in results["model_selection"]["validation_sequence_ids"]]
    subtrain_records = [record for record in train_records if record.sequence_id not in validation_ids]

    feature_map: dict[str, dict[str, np.ndarray]] = {}
    for record in [*subtrain_records, *validation_records, *test_records]:
        feature_map[record.sequence_id] = _load_or_compute_features(record.sequence)

    local_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    early_stopping=True,
                    max_iter=80,
                    random_state=config.seed,
                ),
            ),
        ]
    )
    X_local_subtrain, y_local_subtrain = _stack_rows(subtrain_records, feature_map, edge_mode=False)
    local_model.fit(X_local_subtrain, y_local_subtrain)

    raw_local_validation = _predict_by_sequence(local_model, validation_records, feature_map, edge_mode=False)
    y_validation = np.concatenate([record.labels for record in validation_records], axis=0)
    calibrated_local_validation = _calibrate_scores(
        np.concatenate([raw_local_validation[record.sequence_id] for record in validation_records], axis=0),
        y_validation,
        np.concatenate([raw_local_validation[record.sequence_id] for record in validation_records], axis=0),
    )
    local_validation_scores: dict[str, np.ndarray] = {}
    offset = 0
    for record in validation_records:
        length = len(record.sequence)
        local_validation_scores[record.sequence_id] = calibrated_local_validation[offset : offset + length]
        offset += length
    local_threshold, local_validation_summary = _best_threshold(y_validation, calibrated_local_validation, config.threshold_grid)

    raw_local_subtrain = _predict_by_sequence(local_model, subtrain_records, feature_map, edge_mode=False)
    calibrated_local_subtrain_flat = _calibrate_scores(
        np.concatenate([raw_local_validation[record.sequence_id] for record in validation_records], axis=0),
        y_validation,
        np.concatenate([raw_local_subtrain[record.sequence_id] for record in subtrain_records], axis=0),
    )
    local_subtrain_scores: dict[str, np.ndarray] = {}
    offset = 0
    for record in subtrain_records:
        length = len(record.sequence)
        local_subtrain_scores[record.sequence_id] = calibrated_local_subtrain_flat[offset : offset + length]
        offset += length

    edge_model = ExtraTreesClassifier(
        n_estimators=240,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=config.seed,
        n_jobs=FIT_N_JOBS,
    )
    X_edge_subtrain, y_edge_subtrain = _stack_rows(
        subtrain_records,
        feature_map,
        edge_mode=True,
        local_scores=local_subtrain_scores,
    )
    edge_model.fit(X_edge_subtrain, y_edge_subtrain)

    raw_edge_validation = _predict_by_sequence(
        edge_model,
        validation_records,
        feature_map,
        edge_mode=True,
        local_scores=local_validation_scores,
    )
    calibrated_edge_validation = _calibrate_scores(
        np.concatenate([raw_edge_validation[record.sequence_id] for record in validation_records], axis=0),
        y_validation,
        np.concatenate([raw_edge_validation[record.sequence_id] for record in validation_records], axis=0),
    )
    edge_validation_scores: dict[str, np.ndarray] = {}
    offset = 0
    for record in validation_records:
        length = len(record.sequence)
        edge_validation_scores[record.sequence_id] = calibrated_edge_validation[offset : offset + length]
        offset += length
    edge_threshold, edge_validation_summary = _best_threshold(y_validation, calibrated_edge_validation, config.threshold_grid)
    region_constraint = _select_region_constraint(
        validation_records,
        edge_validation_scores,
        feature_map,
        edge_threshold,
    )

    route_features_validation, route_targets_validation, validation_route_meta = _build_route_dataset(
        validation_records,
        local_validation_scores,
        edge_validation_scores,
        local_threshold,
        edge_threshold,
    )
    route_model_name = "mean_local_uncertainty"
    route_probabilities_validation = np.asarray(
        [_sequence_uncertainty_score(local_validation_scores[record.sequence_id]) for record in validation_records],
        dtype=np.float64,
    )
    adaptive_validation = _select_adaptive_threshold(
        validation_records,
        route_probabilities_validation,
        local_validation_scores,
        edge_validation_scores,
        local_threshold,
        edge_threshold,
        config.route_probability_grid,
        config.route_budget_max,
    )
    conformal_validation = _select_conformal_policy(
        validation_records,
        route_probabilities_validation,
        route_targets_validation,
        local_validation_scores,
        edge_validation_scores,
        local_threshold,
        edge_threshold,
        config.route_budget_max,
        minimum_threshold=float(adaptive_validation["route_threshold"]),
    )

    raw_local_test = _predict_by_sequence(local_model, test_records, feature_map, edge_mode=False)
    calibrated_local_test = _calibrate_scores(
        np.concatenate([raw_local_validation[record.sequence_id] for record in validation_records], axis=0),
        y_validation,
        np.concatenate([raw_local_test[record.sequence_id] for record in test_records], axis=0),
    )

    local_test_scores: dict[str, np.ndarray] = {}
    edge_test_scores: dict[str, np.ndarray] = {}
    offset = 0
    for record in test_records:
        length = len(record.sequence)
        local_test_scores[record.sequence_id] = calibrated_local_test[offset : offset + length]
        offset += length

    raw_edge_test = _predict_by_sequence(edge_model, test_records, feature_map, edge_mode=True, local_scores=local_test_scores)
    y_test = np.concatenate([record.labels for record in test_records], axis=0)
    calibrated_edge_test = _calibrate_scores(
        np.concatenate([raw_edge_validation[record.sequence_id] for record in validation_records], axis=0),
        y_validation,
        np.concatenate([raw_edge_test[record.sequence_id] for record in test_records], axis=0),
    )
    offset = 0
    for record in test_records:
        length = len(record.sequence)
        edge_test_scores[record.sequence_id] = calibrated_edge_test[offset : offset + length]
        offset += length
    region_edge_test_scores: dict[str, np.ndarray] = {
        record.sequence_id: _region_constrained_scores(
            edge_test_scores[record.sequence_id],
            feature_map[record.sequence_id],
            threshold=edge_threshold,
            radius=int(region_constraint["radius"]),
            outside_scale=float(region_constraint["outside_scale"]),
        )
        for record in test_records
    }

    route_features_test = np.stack(
        [_route_features(len(record.sequence), local_test_scores[record.sequence_id], local_threshold) for record in test_records]
    )
    route_probabilities_test = np.asarray(
        [_sequence_uncertainty_score(local_test_scores[record.sequence_id]) for record in test_records],
        dtype=np.float64,
    )
    conformal_route_mask = np.zeros(len(test_records), dtype=bool)
    conformal_route_threshold = float(conformal_validation["route_probability_threshold"])
    if len(test_records):
        candidate_mask = route_probabilities_test >= conformal_route_threshold
        max_routes = int(conformal_validation["selection_rule"]["max_routed_sequences"])
        sequence_lengths = np.asarray([len(record.sequence) for record in test_records], dtype=np.float64)
        length_norm = sequence_lengths / max(1.0, float(sequence_lengths.max(initial=1.0)))
        priority = route_probabilities_test - (float(conformal_validation["selection_rule"]["length_penalty"]) * length_norm)
        if candidate_mask.sum() > max_routes > 0:
            restricted_priority = np.where(candidate_mask, priority, -1e9)
            conformal_route_mask = _ranked_route_mask(restricted_priority, max_routes)
        else:
            conformal_route_mask = candidate_mask

    external_test = results.setdefault("external_test", {})
    modern_policy_rows: dict[str, list[dict[str, Any]]] = {
        "plm_local_only": [],
        "plm_edge_only": [],
        "plm_region_constrained_edge": [],
        "plm_adaptive_split": [],
        "plm_adaptive_conformal_guard": [],
    }
    adaptive_route_threshold = float(adaptive_validation["route_threshold"])

    for sequence_index, record in enumerate(test_records):
        local_probabilities = local_test_scores[record.sequence_id]
        edge_probabilities = edge_test_scores[record.sequence_id]
        region_edge_probabilities = region_edge_test_scores[record.sequence_id]
        local_predictions = (local_probabilities >= local_threshold).astype(np.int8)
        edge_predictions = (edge_probabilities >= edge_threshold).astype(np.int8)
        region_edge_predictions = (region_edge_probabilities >= edge_threshold).astype(np.int8)
        local_latency_ms, edge_latency_ms = _measure_forward_latency_ms(record.sequence, local_model, edge_model, feature_map[record.sequence_id])

        local_total_gco2e = _operational_gco2e(local_latency_ms, config.local_device.operational_power_w, config.grid_intensity_gco2_per_kwh) + _embodied_gco2e_per_request(
            config.local_device.embodied_kgco2e_total,
            config.local_device.amortization_requests,
            0.20,
        )
        edge_total_gco2e = _operational_gco2e(edge_latency_ms, config.edge_device.operational_power_w, config.grid_intensity_gco2_per_kwh) + _embodied_gco2e_per_request(
            config.edge_device.embodied_kgco2e_total,
            config.edge_device.amortization_requests,
            0.65,
        )

        route_probability = float(route_probabilities_test[sequence_index])
        use_edge_adaptive = route_probability >= adaptive_route_threshold
        use_edge_conformal = bool(conformal_route_mask[sequence_index])

        adaptive_probabilities = edge_probabilities if use_edge_adaptive else local_probabilities
        adaptive_predictions = edge_predictions if use_edge_adaptive else local_predictions
        adaptive_latency_ms = edge_latency_ms + local_latency_ms if use_edge_adaptive else local_latency_ms
        adaptive_total_gco2e = edge_total_gco2e + local_total_gco2e if use_edge_adaptive else local_total_gco2e

        conformal_probabilities = edge_probabilities if use_edge_conformal else local_probabilities
        conformal_predictions = edge_predictions if use_edge_conformal else local_predictions
        conformal_latency_ms = edge_latency_ms + local_latency_ms if use_edge_conformal else local_latency_ms
        conformal_total_gco2e = edge_total_gco2e + local_total_gco2e if use_edge_conformal else local_total_gco2e

        modern_policy_rows["plm_local_only"].append(
            _sequence_summary_row(
                record,
                local_probabilities,
                edge_probabilities,
                local_probabilities,
                local_predictions,
                local_latency_ms,
                edge_latency_ms,
                local_latency_ms,
                local_total_gco2e,
                edge_total_gco2e,
                local_total_gco2e,
                route=None,
                route_probability=None,
                raw_sequence_shared=False,
            )
        )
        modern_policy_rows["plm_edge_only"].append(
            _sequence_summary_row(
                record,
                local_probabilities,
                edge_probabilities,
                edge_probabilities,
                edge_predictions,
                local_latency_ms,
                edge_latency_ms,
                edge_latency_ms,
                local_total_gco2e,
                edge_total_gco2e,
                edge_total_gco2e,
                route=None,
                route_probability=None,
                raw_sequence_shared=True,
            )
        )
        modern_policy_rows["plm_region_constrained_edge"].append(
            _sequence_summary_row(
                record,
                local_probabilities,
                edge_probabilities,
                region_edge_probabilities,
                region_edge_predictions,
                local_latency_ms,
                edge_latency_ms,
                edge_latency_ms,
                local_total_gco2e,
                edge_total_gco2e,
                edge_total_gco2e,
                route=None,
                route_probability=None,
                raw_sequence_shared=True,
            )
        )
        modern_policy_rows["plm_adaptive_split"].append(
            _sequence_summary_row(
                record,
                local_probabilities,
                edge_probabilities,
                adaptive_probabilities,
                adaptive_predictions,
                local_latency_ms,
                edge_latency_ms,
                adaptive_latency_ms,
                local_total_gco2e,
                edge_total_gco2e,
                adaptive_total_gco2e,
                route=use_edge_adaptive,
                route_probability=route_probability,
                raw_sequence_shared=use_edge_adaptive,
            )
        )
        modern_policy_rows["plm_adaptive_conformal_guard"].append(
            _sequence_summary_row(
                record,
                local_probabilities,
                edge_probabilities,
                conformal_probabilities,
                conformal_predictions,
                local_latency_ms,
                edge_latency_ms,
                conformal_latency_ms,
                local_total_gco2e,
                edge_total_gco2e,
                conformal_total_gco2e,
                route=use_edge_conformal,
                route_probability=route_probability,
                raw_sequence_shared=use_edge_conformal,
            )
        )

    external_test["plm_local_only"] = {
        "decision_threshold": round(float(local_threshold), 6),
        "summary": _policy_summary(modern_policy_rows["plm_local_only"]),
        "confidence_intervals_95": _policy_bootstrap_confidence_intervals(modern_policy_rows["plm_local_only"], config.seed + 1101),
        "per_sequence": modern_policy_rows["plm_local_only"],
    }
    external_test["plm_edge_only"] = {
        "decision_threshold": round(float(edge_threshold), 6),
        "summary": _policy_summary(modern_policy_rows["plm_edge_only"]),
        "confidence_intervals_95": _policy_bootstrap_confidence_intervals(modern_policy_rows["plm_edge_only"], config.seed + 1202),
        "per_sequence": modern_policy_rows["plm_edge_only"],
    }
    external_test["plm_region_constrained_edge"] = {
        "decision_threshold": round(float(edge_threshold), 6),
        "region_constraint": region_constraint,
        "summary": _policy_summary(modern_policy_rows["plm_region_constrained_edge"]),
        "confidence_intervals_95": _policy_bootstrap_confidence_intervals(modern_policy_rows["plm_region_constrained_edge"], config.seed + 1252),
        "per_sequence": modern_policy_rows["plm_region_constrained_edge"],
    }
    external_test["plm_adaptive_split"] = {
        "decision_thresholds": {"local": round(float(local_threshold), 6), "edge": round(float(edge_threshold), 6)},
        "router": route_model_name,
        "route_probability_threshold": round(adaptive_route_threshold, 6),
        "routed_sequence_ids": [row["sequence_id"] for row in modern_policy_rows["plm_adaptive_split"] if row["route"]],
        "summary": _policy_summary(modern_policy_rows["plm_adaptive_split"]),
        "confidence_intervals_95": _policy_bootstrap_confidence_intervals(modern_policy_rows["plm_adaptive_split"], config.seed + 1303),
        "per_sequence": modern_policy_rows["plm_adaptive_split"],
    }
    external_test["plm_adaptive_conformal_guard"] = {
        "decision_thresholds": {"local": round(float(local_threshold), 6), "edge": round(float(edge_threshold), 6)},
        "router": route_model_name,
        "route_probability_threshold": round(conformal_route_threshold, 6),
        "selection_rule": conformal_validation["selection_rule"],
        "routed_sequence_ids": [row["sequence_id"] for row in modern_policy_rows["plm_adaptive_conformal_guard"] if row["route"]],
        "summary": _policy_summary(modern_policy_rows["plm_adaptive_conformal_guard"]),
        "confidence_intervals_95": _policy_bootstrap_confidence_intervals(modern_policy_rows["plm_adaptive_conformal_guard"], config.seed + 1404),
        "per_sequence": modern_policy_rows["plm_adaptive_conformal_guard"],
    }

    results.setdefault("measurement_provenance", {}).update(
        {
            "plm_local_latency": "measured_end_to_end_esm_forward_without_contacts_plus_residue_head",
            "plm_edge_latency": "measured_end_to_end_esm_forward_with_contact_head_plus_structure_aware_refiner",
            "plm_feature_cache": f"frozen_{ESM_MODEL_NAME}_residue_embeddings_and_contact_summaries_cached_on_disk",
            "region_metrics": "computed_from contiguous positive residue runs on the external test labels and predictions",
        }
    )
    results.setdefault("modern_extension", {}).update(
        {
            "enabled": True,
            "plm_model": ESM_MODEL_NAME,
            "local_model": "frozen_esm_embeddings_plus_logistic_residue_head",
            "edge_model": "frozen_esm_embeddings_plus_contact_features_plus_extra_trees_refiner",
            "region_constrained_edge": "validation_selected_contiguous_region_postprocessor_using_edge_probabilities_and_contact_density",
            "route_model": route_model_name,
            "validation_split_reused_from_legacy_runner": True,
            "cache_directory": str(_esm_cache_dir()),
        }
    )
    results["model_selection"]["plm_local_validation"] = {
        "threshold": round(float(local_threshold), 6),
        "metrics": local_validation_summary,
    }
    results["model_selection"]["plm_edge_validation"] = {
        "threshold": round(float(edge_threshold), 6),
        "metrics": edge_validation_summary,
    }
    results["model_selection"]["plm_region_constraint"] = region_constraint
    results["model_selection"]["plm_adaptive_router"] = {
        "router": route_model_name,
            "route_budget_max": float(config.route_budget_max),
            "route_probability_threshold": round(float(adaptive_validation["route_threshold"]), 6),
            "validation_metrics": adaptive_validation["summary"],
            "validation_route_targets": route_targets_validation.astype(int).tolist(),
            "validation_route_probabilities": route_probabilities_validation.astype(float).tolist(),
            "validation_route_scores": route_probabilities_validation.astype(float).tolist(),
            "validation_route_meta": validation_route_meta,
            "selected_validation_routes": adaptive_validation["routed_sequence_ids"],
        }
    results["model_selection"]["plm_adaptive_conformal_guard"] = {
        "router": route_model_name,
        "route_probability_threshold": round(float(conformal_validation["route_probability_threshold"]), 6),
        "validation_metrics": conformal_validation["summary"],
        "selection_rule": conformal_validation["selection_rule"],
        "selected_validation_routes": conformal_validation["routed_sequence_ids"],
    }
    results.setdefault("limitations", []).append(
        f"The PLM extension uses the frozen {ESM_MODEL_NAME} checkpoint; conclusions about the PLM family are bounded to that checkpoint."
    )
    return results
