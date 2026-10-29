# nested_fusion_evaluation: CARBON-BCE-R fused onto EdgeConBCE local_only

Generated at: 2026-10-25T03:41:32.799037+00:00

Leave-one-sequence-out nested selection. For each held-out antigen the fusion weight and the decision threshold are chosen using only the remaining antigens and then applied once to the held-out antigen.

Split: `IEDB reduced bounded subset` — 47 sequences, 14408 residues, 2169 positive residues.

Reference operating threshold: `0.51` (reference predictor threshold, recomputed per training split).

Selection metric: `mcc`. Weights chosen per fold: median 0.95, range [0.85, 0.95].

## Pooled out-of-fold performance

| metric | fusion | reference | delta |
|---|---:|---:|---:|
| roc_auc | 0.645573 | 0.645160 | +0.000414 |
| average_precision | 0.248535 | 0.246334 | +0.002201 |
| mcc | 0.160583 | 0.162942 | -0.002360 |
| f1 | 0.320994 | 0.322455 | -0.001461 |

## Per-sequence paired differences (fusion minus reference)

| metric | mean delta | 95% CI | wins/losses | Wilcoxon p | sign-test p |
|---|---:|---|---:|---:|---:|
| roc_auc | +0.000796 | [-0.001045, +0.002723] | 25/22 | 0.4851 | 0.7709 |
| average_precision | -0.002942 | [-0.012803, +0.004245] | 26/21 | 0.1560 | 0.5601 |
| mcc | +0.001434 | [-0.003675, +0.006612] | 24/15 | 0.4183 | 0.1996 |
| f1 | -0.000993 | [-0.003551, +0.001380] | 21/17 | 0.9249 | 0.6271 |
