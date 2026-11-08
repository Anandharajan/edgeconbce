from __future__ import annotations

from edgeconbce.features.pacvf import AROMATIC, CHARGED, pacvf_like_features
from edgeconbce.metrics.ranking import bounded


def long_range_contact_proxy(sequence: str) -> float:
    hits = 0
    possible = 0
    for index in range(len(sequence) - 7):
        aa_left = sequence[index]
        aa_right = sequence[index + 7]
        if (aa_left in AROMATIC and aa_right in CHARGED) or (aa_left in CHARGED and aa_right in AROMATIC):
            hits += 1
        elif aa_left == "C" and aa_right == "C":
            hits += 1
        possible += 1
    return hits / max(1, possible)


def structure_proxy_features(sequence: str, max_sequence_length: int) -> dict[str, float]:
    base = pacvf_like_features(sequence, max_sequence_length)
    surface_proxy = bounded(base["polar_ratio"] + base["charged_ratio"] - (0.35 * base["hydrophobic_ratio"]))
    contact_complexity = bounded(
        0.45 * long_range_contact_proxy(sequence)
        + 0.25 * base["transition_rate"]
        + 0.20 * base["aromatic_ratio"]
        + 0.10 * base["cysteine_ratio"]
    )
    loop_propensity = bounded(0.60 * base["flexible_ratio"] + 0.40 * base["polar_ratio"])
    compactness_risk = bounded(base["hydrophobic_ratio"] - (0.50 * surface_proxy))
    return {
        "surface_proxy": surface_proxy,
        "contact_complexity": contact_complexity,
        "loop_propensity": loop_propensity,
        "compactness_risk": compactness_risk,
    }

