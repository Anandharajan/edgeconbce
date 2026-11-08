from __future__ import annotations

import statistics

from edgeconbce.metrics.ranking import auc_score, average_precision, percentile, summarize_binary_metrics
from edgeconbce.schemas import PolicySummary, SampleRecord


def quality_score(probability: float, label: bool) -> float:
    return 1.0 - abs(probability - float(label))


def summarize_policy(policy: str, records: list[SampleRecord]) -> PolicySummary:
    labels = [record.label for record in records]
    predictions = [record.prediction for record in records]
    scores = [record.final_probability for record in records]
    metrics = summarize_binary_metrics(labels, predictions)
    edge_routes = sum(1 for record in records if record.route != "local")
    raw_sequence_share_rate = sum(1 for record in records if record.raw_sequence_shared) / len(records)
    return PolicySummary(
        policy=policy,
        accuracy=round(metrics["accuracy"], 6),
        precision=round(metrics["precision"], 6),
        recall=round(metrics["recall"], 6),
        f1=round(metrics["f1"], 6),
        mcc=round(metrics["mcc"], 6),
        auc=round(auc_score(labels, scores), 6),
        map=round(average_precision(labels, scores), 6),
        mean_latency_ms=round(statistics.mean(record.latency_ms for record in records), 3),
        p95_latency_ms=round(percentile([record.latency_ms for record in records], 0.95), 3),
        mean_total_gco2e=round(statistics.mean(record.total_gco2e for record in records), 6),
        edge_route_rate=round(edge_routes / len(records), 6),
        raw_sequence_share_rate=round(raw_sequence_share_rate, 6),
    )

