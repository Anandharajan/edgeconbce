from __future__ import annotations

from edgeconbce.metrics.ranking import bounded, sigmoid


def edge_model_probability(
    features: dict[str, float],
    proxies: dict[str, float],
    local_probability: float,
) -> float:
    score = (
        -0.50
        + 0.70 * proxies["surface_proxy"]
        + 0.82 * proxies["contact_complexity"]
        + 0.48 * proxies["loop_propensity"]
        + 0.28 * features["aromatic_ratio"]
        - 0.30 * proxies["compactness_risk"]
    )
    refined = sigmoid(2.0 * score)
    return bounded((0.30 * local_probability) + (0.70 * refined))

