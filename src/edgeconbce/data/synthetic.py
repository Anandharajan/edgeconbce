from __future__ import annotations

from typing import Iterable
import random

from edgeconbce.features.pacvf import AMINO_ACIDS, AROMATIC, CHARGED, FLEXIBLE, HYDROPHOBIC, POLAR, pacvf_like_features
from edgeconbce.features.structure_proxy import structure_proxy_features
from edgeconbce.metrics.ranking import bounded, sigmoid
from edgeconbce.schemas import NotebookCompatConfig, SequenceSample


PROFILE_POOLS = {
    "easy_positive": "DDDDEEEEKKKKRRRRNNQQSSTTGGPPFWYHAC",
    "hard_positive": "ACDEFGHIKLMNPQRSTVWYDDKKRRGGPPFW",
    "easy_negative": "AAAAVVVVLLLLIIIIMMMFFFWYYCCCSTNQ",
    "hard_negative": "ACDEFGHIKLMNPQRSTVWYAVILMF",
    "ambiguous": AMINO_ACIDS,
}


def validate_sequence(sequence: str) -> str:
    cleaned = sequence.strip().upper()
    invalid = sorted(set(cleaned) - set(AMINO_ACIDS))
    if invalid:
        raise ValueError(f"Invalid amino-acid symbols: {invalid}")
    return cleaned


def hidden_ground_truth_signal(features: dict[str, float], proxies: dict[str, float]) -> float:
    score = (
        -0.62
        + 1.05 * proxies["surface_proxy"]
        + 1.15 * proxies["contact_complexity"]
        + 0.75 * proxies["loop_propensity"]
        + 0.38 * features["aromatic_ratio"]
        + 0.22 * features["edge_charge_ratio"]
        + 0.24 * features["length_norm"]
        - 0.65 * proxies["compactness_risk"]
    )
    return sigmoid(2.0 * score)


def inject_motif(sequence: str, motif: str, start: int) -> str:
    return sequence[:start] + motif + sequence[start + len(motif) :]


def generate_candidate_sequence(rng: random.Random, length: int, mode: str) -> str:
    sequence = "".join(rng.choice(PROFILE_POOLS[mode]) for _ in range(length))
    if mode in {"easy_positive", "hard_positive"} and length > 18:
        for motif in ("KWG", "RGD", "CPY"):
            start = rng.randint(0, length - len(motif))
            sequence = inject_motif(sequence, motif, start)
    if mode in {"easy_negative", "hard_negative"} and length > 15:
        motif = "AVLII"
        start = rng.randint(0, length - len(motif))
        sequence = inject_motif(sequence, motif, start)
    return validate_sequence(sequence)


def generate_sample(index: int, rng: random.Random, config: NotebookCompatConfig) -> SequenceSample:
    modes = ("easy_positive", "hard_positive", "easy_negative", "hard_negative", "ambiguous")
    mode = modes[index % len(modes)]
    target_positive = mode in {"easy_positive", "hard_positive"}
    sequence = ""
    label = False
    hidden_signal = 0.0
    for _ in range(64):
        if mode == "easy_positive":
            length = rng.randint(90, 220)
        elif mode == "hard_positive":
            length = rng.randint(180, config.max_sequence_length)
        elif mode == "easy_negative":
            length = rng.randint(config.min_sequence_length, 180)
        elif mode == "hard_negative":
            length = rng.randint(140, 300)
        else:
            length = rng.randint(120, 280)
        sequence = generate_candidate_sequence(rng, length, mode)
        features = pacvf_like_features(sequence, config.max_sequence_length)
        proxies = structure_proxy_features(sequence, config.max_sequence_length)
        hidden_signal = hidden_ground_truth_signal(features, proxies)
        label = hidden_signal >= config.decision_threshold
        if mode == "ambiguous" and 0.45 <= hidden_signal <= 0.68:
            break
        if mode != "ambiguous" and label == target_positive:
            break
    return SequenceSample(
        sample_id=f"sample_{index:03d}",
        sequence=sequence,
        label=label,
        hidden_signal=hidden_signal,
        mode=mode,
    )


def build_dataset(config: NotebookCompatConfig) -> list[SequenceSample]:
    rng = random.Random(config.seed)
    return [generate_sample(index, rng, config) for index in range(config.num_samples)]

