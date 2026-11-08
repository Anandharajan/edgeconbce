from __future__ import annotations

from collections import Counter
from typing import Iterable


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
HYDROPHOBIC = set("AILMFWVY")
POLAR = set("STNQ")
CHARGED = set("KRHDE")
AROMATIC = set("FWYH")
FLEXIBLE = set("GSPDN")


def fraction(sequence: str, residues: Iterable[str]) -> float:
    residue_set = set(residues)
    return sum(1 for aa in sequence if aa in residue_set) / len(sequence)


def edge_charge_ratio(sequence: str) -> float:
    window = max(6, len(sequence) // 10)
    edge_region = sequence[:window] + sequence[-window:]
    return fraction(edge_region, CHARGED)


def transition_rate(sequence: str) -> float:
    groups: list[str] = []
    for aa in sequence:
        if aa in CHARGED:
            groups.append("charged")
        elif aa in POLAR:
            groups.append("polar")
        elif aa in HYDROPHOBIC:
            groups.append("hydrophobic")
        else:
            groups.append("other")
    changes = sum(1 for left, right in zip(groups, groups[1:]) if left != right)
    return changes / max(1, len(sequence) - 1)


def pacvf_like_features(sequence: str, max_sequence_length: int) -> dict[str, float]:
    counts = Counter(sequence)
    length = len(sequence)
    return {
        "length_norm": length / max_sequence_length,
        "diversity_ratio": len(counts) / len(AMINO_ACIDS),
        "aromatic_ratio": fraction(sequence, AROMATIC),
        "charged_ratio": fraction(sequence, CHARGED),
        "polar_ratio": fraction(sequence, POLAR),
        "hydrophobic_ratio": fraction(sequence, HYDROPHOBIC),
        "flexible_ratio": fraction(sequence, FLEXIBLE),
        "cysteine_ratio": counts.get("C", 0) / length,
        "edge_charge_ratio": edge_charge_ratio(sequence),
        "transition_rate": transition_rate(sequence),
    }

