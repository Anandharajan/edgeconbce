from __future__ import annotations

from typing import Any

import numpy as np


def positive_runs(values: np.ndarray) -> list[tuple[int, int]]:
    predictions = np.asarray(values, dtype=np.int8)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(predictions):
        if value == 1 and start is None:
            start = index
        elif value == 0 and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(predictions)))
    return runs


def summarize_region_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have the same shape.")

    pred_runs = positive_runs(predictions)
    label_runs = positive_runs(labels)
    predicted_positive_count = int(np.sum(predictions))
    true_positive_count = int(np.sum(labels))
    overlap_count = int(np.sum((labels == 1) & (predictions == 1)))
    longest_predicted_run = max((end - start for start, end in pred_runs), default=0)

    return {
        "predicted_region_count": len(pred_runs),
        "true_region_count": len(label_runs),
        "predicted_positive_count": predicted_positive_count,
        "true_positive_count": true_positive_count,
        "region_overlap_count": overlap_count,
        "largest_predicted_region_fraction": 0.0
        if predicted_positive_count == 0
        else round(longest_predicted_run / predicted_positive_count, 6),
        "region_recall": 0.0 if true_positive_count == 0 else round(overlap_count / true_positive_count, 6),
        "region_precision": 0.0 if predicted_positive_count == 0 else round(overlap_count / predicted_positive_count, 6),
    }
