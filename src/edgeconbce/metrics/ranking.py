from __future__ import annotations

from typing import Sequence
import math


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def auc_score(labels: Sequence[bool], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hit_count = 0
    precision_sum = 0.0
    for index, (_, label) in enumerate(ranked, start=1):
        if label:
            hit_count += 1
            precision_sum += hit_count / index
    return precision_sum / positives


def summarize_binary_metrics(labels: Sequence[bool], predictions: Sequence[bool]) -> dict[str, float]:
    tp = sum(1 for truth, pred in zip(labels, predictions) if truth and pred)
    tn = sum(1 for truth, pred in zip(labels, predictions) if (not truth) and (not pred))
    fp = sum(1 for truth, pred in zip(labels, predictions) if (not truth) and pred)
    fn = sum(1 for truth, pred in zip(labels, predictions) if truth and (not pred))
    accuracy = (tp + tn) / max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)
    denominator = math.sqrt(max(1.0, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }

