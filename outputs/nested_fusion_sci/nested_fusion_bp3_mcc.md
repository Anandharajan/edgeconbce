# nested_fusion_evaluation: CARBON-BCE-R fused onto BepiPred-3.0 (official local rerun)

Generated at: 2026-10-25T03:41:16.136514+00:00

Leave-one-sequence-out nested selection. For each held-out antigen the fusion weight and the decision threshold are chosen using only the remaining antigens and then applied once to the held-out antigen.

Split: `BP3C50ID external test` — 15 sequences, 2670 residues, 507 positive residues.

Reference operating threshold: `0.1512` (reference predictor threshold, recomputed per training split).

Selection metric: `mcc`. Weights chosen per fold: median 0.90, range [0.55, 0.90].

## Pooled out-of-fold performance

| metric | fusion | reference | delta |
|---|---:|---:|---:|
| roc_auc | 0.770084 | 0.772576 | -0.002492 |
| average_precision | 0.414890 | 0.412587 | +0.002303 |
| mcc | 0.322691 | 0.324224 | -0.001532 |
| f1 | 0.463048 | 0.463935 | -0.000887 |

## Per-sequence paired differences (fusion minus reference)

| metric | mean delta | 95% CI | wins/losses | Wilcoxon p | sign-test p |
|---|---:|---|---:|---:|---:|
| roc_auc | -0.000516 | [-0.003293, +0.002093] | 7/7 | 0.9250 | 1.0000 |
| average_precision | -0.003605 | [-0.013545, +0.004091] | 9/5 | 0.9250 | 0.4240 |
| mcc | -0.032903 | [-0.095474, +0.008814] | 7/8 | 0.5614 | 1.0000 |
| f1 | -0.009483 | [-0.034310, +0.009416] | 7/8 | 0.7615 | 1.0000 |
