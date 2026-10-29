# Claim

Uncertainty-triggered routing between a local and a remote conformational
B-cell epitope predictor buys a laboratory two things it can verify — roughly a
threefold latency reduction and a large reduction in how many antigen sequences
leave the device — at no quality cost relative to always-remote execution, but
it does not reduce modelled carbon; and the fusion advantage this project
previously reported over BepiPred-3.0 is an artefact of selecting the fusion
weight and threshold on the evaluation split, disappearing on two datasets once
selection is nested.

## Measurable success criteria

1. On BP3C50ID across >= 10 seeds, adaptive-split shows no MCC deficit against
   edge-only whose 95% interval excludes zero, while reducing mean latency and
   raw-sequence share rate with intervals that do exclude zero. **Met.**
2. The direction of the modelled-carbon comparison between adaptive-split and
   both baselines is established with intervals excluding zero. **Met** — and
   the direction is the opposite of the project's original expectation.
3. The fusion effect is re-estimated under leave-one-sequence-out nested
   selection on two datasets, with the published protocol reproduced exactly so
   that any difference is attributable to the protocol. **Met.**

## Scope deliberately excluded

- No claim to state-of-the-art predictive accuracy. Standalone CARBON-BCE-R is
  below BepiPred-3.0 on BP3C50ID and the paper says so.
- No claim about metered energy or deployed emissions; carbon is modelled.
- No claim about real multi-host deployment; transport is a probe.
