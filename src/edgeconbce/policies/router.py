from __future__ import annotations

from edgeconbce.metrics.carbon import edge_latency_ms, embodied_gco2e_per_request, energy_kwh, local_latency_ms
from edgeconbce.metrics.ranking import bounded
from edgeconbce.schemas import NotebookCompatConfig, SequenceSample


def choose_route(
    policy: str,
    sample: SequenceSample,
    local_probability: float,
    edge_probability: float,
    features: dict[str, float],
    proxies: dict[str, float],
    config: NotebookCompatConfig,
) -> str:
    uncertainty = 1.0 - abs(local_probability - 0.5) * 2.0
    quality_risk = max(0.0, edge_probability - local_probability)
    local_total = (
        energy_kwh(config.local_device.operational_power_w, local_latency_ms(len(sample.sequence)))
        * config.grid_intensity_gco2_per_kwh
        + embodied_gco2e_per_request(config.local_device)
    )
    edge_total = (
        energy_kwh(config.edge_device.operational_power_w, edge_latency_ms(len(sample.sequence)))
        * config.grid_intensity_gco2_per_kwh
        + embodied_gco2e_per_request(config.edge_device)
    )
    if policy == "local_only":
        return "local"
    if policy == "edge_only":
        return "edge_full"
    if policy == "carbon_first":
        if edge_total < local_total and (uncertainty > 0.45 or proxies["contact_complexity"] > 0.24):
            return "edge_full"
        return "local"
    if policy != "adaptive_split":
        raise ValueError(f"Unsupported policy: {policy}")
    score = (
        config.weights.uncertainty * uncertainty
        + config.weights.sequence_length * features["length_norm"]
        + config.weights.contact_complexity * proxies["contact_complexity"]
        + config.weights.quality_risk * quality_risk
        + config.weights.operational_carbon * bounded(local_total / max(edge_total, 1e-9))
        - config.weights.local_confidence * abs(local_probability - 0.5) * 2.0
    )
    return "split_to_edge" if score >= 0.43 else "local"

