"""How much does a residue's representation move when it is sent as a fragment?

``region_routing`` scores a routed region with remote scores that were computed
on the whole sequence. A deployment that actually sends only a region cannot do
that: a protein language model reads context, so the same residue inside a
200-residue antigen and inside a 30-residue fragment is not guaranteed the same
representation, and therefore not the same score.

That gap is the one assumption the routing result rests on, and this module
measures it instead of arguing about it. For every region the routing policy
would send, the residues inside it are embedded twice with the same encoder:
once in full sequence context, once in the fragment the policy would transmit,
padded with a given number of flanking residues. The similarity between the two
representations bounds how far a fragment-sent score could drift from the score
the routing evaluation assumed.

Flank width is swept because it is the knob a deployment actually has. Sending
more flanking context recovers more of the full-sequence representation and
discloses more residues, so the sweep is itself a disclosure-fidelity trade-off,
measured on the same antigens as the routing result.

The encoder here is the same ESM-2 checkpoint the local PLM tier uses. It is not
BepiPred-3.0, whose weights this project does not redistribute, so the numbers
describe how a frozen PLM representation behaves under fragmentation rather than
how BepiPred-3.0's output behaves. Stated plainly: this measures the mechanism
the assumption depends on, in the same family of model, not the deployed scorer.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from edgeconbce.pipeline.nested_fusion import (
    FusionInputs,
    apply_ecdf,
    build_bp3_inputs,
    fit_ecdf,
    threshold_for_positive_rate,
)
from edgeconbce.pipeline.region_routing import (
    DEFAULT_MIN_REGION,
    DEFAULT_SMOOTH_WINDOW,
    uncertainty_regions,
)

DEFAULT_FLANKS = (0, 10, 25, 50)
DEFAULT_BUDGET = 0.20
DEFAULT_CHECKPOINT = "esm2_t6_8M_UR50D"


def load_sequences(bepipred_raw_scores: str | Path) -> "OrderedDict[str, str]":
    """Reconstruct each antigen's sequence from the per-residue score table."""
    seqs: "OrderedDict[str, list[str]]" = OrderedDict()
    with open(bepipred_raw_scores, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seqs.setdefault(row["Accession"], []).append(row["Residue"].strip())
    return OrderedDict((k, "".join(v)) for k, v in seqs.items())


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as half-open [start, end) spans."""
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def _routed_masks(
    inputs: FusionInputs,
    budget: float,
    reference_threshold: float | None,
    window: int,
    min_len: int,
) -> dict[int, np.ndarray]:
    """The regions the routing policy would send, per antigen.

    The local transform and operating point come from that antigen's training
    folds, exactly as in ``region_routing``, so the regions measured here are
    the regions that evaluation actually routed.
    """
    groups = inputs.groups
    antigens = sorted({int(a) for a in groups})
    masks: dict[int, np.ndarray] = {}
    for antigen in antigens:
        test_mask = groups == int(antigen)
        train_mask = ~test_mask
        local_ecdf = fit_ecdf(inputs.candidate_scores[train_mask])
        local_train = apply_ecdf(local_ecdf, inputs.candidate_scores[train_mask])
        if reference_threshold is not None:
            rate = float(
                (inputs.reference_scores[train_mask] >= reference_threshold).mean()
            )
        else:
            rate = float(inputs.labels[train_mask].mean())
        point = threshold_for_positive_rate(local_train, rate)
        local_v = apply_ecdf(local_ecdf, inputs.candidate_scores[test_mask])
        masks[antigen] = uncertainty_regions(
            local_v, point, budget, window=window, min_len=min_len
        )
    return masks


def _embed(model: Any, batch_converter: Any, layer: int, sequence: str) -> np.ndarray:
    """Per-residue representations, BOS/EOS stripped."""
    import torch

    _, _, tokens = batch_converter([("x", sequence)])
    with torch.no_grad():
        out = model(tokens, repr_layers=[layer])["representations"][layer]
    return out[0, 1 : len(sequence) + 1].numpy().astype(np.float64)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def measure(
    inputs: FusionInputs,
    sequences: "OrderedDict[str, str]",
    *,
    budget: float = DEFAULT_BUDGET,
    flanks: Sequence[int] = DEFAULT_FLANKS,
    reference_threshold: float | None = None,
    window: int = DEFAULT_SMOOTH_WINDOW,
    min_len: int = DEFAULT_MIN_REGION,
    checkpoint: str = DEFAULT_CHECKPOINT,
) -> dict[str, Any]:
    """Embed every routed region in full context and as a fragment, and compare."""
    import esm

    loader = getattr(esm.pretrained, checkpoint)
    model, alphabet = loader()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    layer = int(model.num_layers)

    masks = _routed_masks(inputs, budget, reference_threshold, window, min_len)
    ids = list(inputs.sequence_ids)

    full_cache: dict[int, np.ndarray] = {}
    per_flank: dict[int, list[float]] = {int(f): [] for f in flanks}
    region_lengths: list[int] = []
    disclosed_at_flank: dict[int, list[float]] = {int(f): [] for f in flanks}
    regions_total = 0

    for antigen, mask in masks.items():
        name = ids[antigen] if antigen < len(ids) else None
        if name is None or name not in sequences:
            continue
        sequence = sequences[name]
        if len(sequence) != len(mask):
            continue
        if antigen not in full_cache:
            full_cache[antigen] = _embed(model, batch_converter, layer, sequence)
        full = full_cache[antigen]

        spans = _runs(mask)
        regions_total += len(spans)
        for start, end in spans:
            region_lengths.append(end - start)
            for flank in flanks:
                f = int(flank)
                lo = max(0, start - f)
                hi = min(len(sequence), end + f)
                fragment = sequence[lo:hi]
                if len(fragment) == 0:
                    continue
                frag_repr = _embed(model, batch_converter, layer, fragment)
                inside = slice(start - lo, start - lo + (end - start))
                sims = _cosine(full[start:end], frag_repr[inside])
                per_flank[f].extend(float(s) for s in sims if np.isfinite(s))
                disclosed_at_flank[f].append((hi - lo) / len(sequence))

    summary = []
    for flank in flanks:
        f = int(flank)
        sims = np.asarray(per_flank[f], dtype=np.float64)
        disc = np.asarray(disclosed_at_flank[f], dtype=np.float64)
        if sims.size == 0:
            continue
        summary.append(
            {
                "flank_residues": f,
                "mean_cosine_to_full_context": float(sims.mean()),
                "median_cosine": float(np.median(sims)),
                "p05_cosine": float(np.percentile(sims, 5)),
                "min_cosine": float(sims.min()),
                "share_above_0_99": float((sims >= 0.99).mean()),
                "share_above_0_95": float((sims >= 0.95).mean()),
                "residues_compared": int(sims.size),
                "mean_fragment_share_of_sequence": float(disc.mean()),
            }
        )

    return {
        "project": "EdgeConBCE",
        "analysis": "region_context_sensitivity",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": (
            "How far does a routed residue's representation move when the "
            "region is sent as a fragment instead of in full sequence context?"
        ),
        "encoder": checkpoint,
        "encoder_note": (
            "the local PLM tier's checkpoint, not BepiPred-3.0; this measures "
            "the mechanism the routing assumption depends on, in the same "
            "family of model, not the deployed scorer"
        ),
        "routing_budget": float(budget),
        "regions_measured": int(regions_total),
        "mean_region_length": (
            float(np.mean(region_lengths)) if region_lengths else float("nan")
        ),
        "by_flank": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--carbon-predictions", required=True)
    p.add_argument("--bepipred-raw-scores", required=True)
    p.add_argument("--reference-threshold", type=float, default=None)
    p.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    p.add_argument("--flanks", type=int, nargs="+", default=list(DEFAULT_FLANKS))
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)

    inputs = build_bp3_inputs(a.carbon_predictions, a.bepipred_raw_scores)
    sequences = load_sequences(a.bepipred_raw_scores)
    report = measure(
        inputs,
        sequences,
        budget=a.budget,
        flanks=a.flanks,
        reference_threshold=a.reference_threshold,
        checkpoint=a.checkpoint,
    )
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
