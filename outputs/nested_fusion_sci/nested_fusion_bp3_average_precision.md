# nested_fusion_evaluation: CARBON-BCE-R fused onto BepiPred-3.0 (official local rerun)

Generated at: 2026-10-25T02:58:58.659256+00:00

Leave-one-sequence-out nested selection. For each held-out antigen the fusion weight and the decision threshold are chosen using only the remaining antigens and then applied once to the held-out antigen.

Split: `BP3C50ID external test` — 15 sequences, 2670 residues, 507 positive residues.

Selection metric: `average_precision`. Weights chosen per fold: median 0.95, range [0.95, 1.00].

## Pooled out-of-fold performance

| metric | fusion | reference | delta |
|---|---:|---:|---:|
| roc_auc | 0.771038 | 0.770516 | +0.000522 |
| average_precision | 0.409197 | 0.414392 | -0.005195 |
| mcc | 0.319734 | 0.315893 | +0.003841 |
| f1 | 0.462353 | 0.460000 | +0.002353 |

## Per-sequence paired differences (fusion minus reference)

| metric | mean delta | 95% CI | wins/losses | Wilcoxon p | sign-test p |
|---|---:|---|---:|---:|---:|
| roc_auc | +0.000053 | [-0.001015, +0.001036] | 7/5 | 0.6379 | 0.7744 |
| average_precision | -0.001470 | [-0.005505, +0.002138] | 7/5 | 0.5829 | 0.7744 |
| mcc | +0.000747 | [-0.008057, +0.008780] | 3/2 | 0.6858 | 1.0000 |
| f1 | +0.000475 | [-0.004803, +0.005305] | 3/2 | 0.8927 | 1.0000 |
