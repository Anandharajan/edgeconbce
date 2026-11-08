from __future__ import annotations

from edgeconbce.metrics.ranking import sigmoid


def local_model_probability(features: dict[str, float]) -> float:
    score = (
        -0.58
        + 0.85 * features["charged_ratio"]
        + 0.72 * features["polar_ratio"]
        + 0.46 * features["aromatic_ratio"]
        + 0.26 * features["transition_rate"]
        + 0.16 * features["length_norm"]
        - 0.42 * features["hydrophobic_ratio"]
    )
    return sigmoid(score)

