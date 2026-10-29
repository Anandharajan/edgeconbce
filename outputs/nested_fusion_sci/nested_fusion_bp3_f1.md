# nested_fusion_evaluation: CARBON-BCE-R fused onto BepiPred-3.0 (official local rerun)

Generated at: 2026-10-25T02:59:02.333806+00:00

Leave-one-sequence-out nested selection. For each held-out antigen the fusion weight and the decision threshold are chosen using only the remaining antigens and then applied once to the held-out antigen.

Split: `BP3C50ID external test` — 15 sequences, 2670 residues, 507 positive residues.

Selection metric: `f1`. Weights chosen per fold: median 0.85, range [0.75, 0.95].

## Pooled out-of-fold performance

| metric | fusion | reference | delta |
|---|---:|---:|---:|
| roc_auc | 0.773884 | 0.770516 | +0.003368 |
| average_precision | 0.421235 | 0.414392 | +0.006843 |
| mcc | 0.321655 | 0.315893 | +0.005762 |
| f1 | 0.463529 | 0.460000 | +0.003529 |

## Per-sequence paired differences (fusion minus reference)

| metric | mean delta | 95% CI | wins/losses | Wilcoxon p | sign-test p |
|---|---:|---|---:|---:|---:|
| roc_auc | +0.000147 | [-0.002222, +0.002566] | 7/7 | 0.8261 | 1.0000 |
| average_precision | -0.001818 | [-0.009176, +0.004868] | 8/6 | 0.6832 | 0.7905 |
| mcc | -0.000320 | [-0.010708, +0.010147] | 3/3 | 0.7532 | 1.0000 |
| f1 | -0.000126 | [-0.006468, +0.006049] | 3/3 | 0.9165 | 1.0000 |
