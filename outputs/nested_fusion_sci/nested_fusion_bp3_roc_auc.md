# nested_fusion_evaluation: CARBON-BCE-R fused onto BepiPred-3.0 (official local rerun)

Generated at: 2026-10-25T02:58:55.115735+00:00

Leave-one-sequence-out nested selection. For each held-out antigen the fusion weight and the decision threshold are chosen using only the remaining antigens and then applied once to the held-out antigen.

Split: `BP3C50ID external test` — 15 sequences, 2670 residues, 507 positive residues.

Selection metric: `roc_auc`. Weights chosen per fold: median 0.85, range [0.80, 0.90].

## Pooled out-of-fold performance

| metric | fusion | reference | delta |
|---|---:|---:|---:|
| roc_auc | 0.769639 | 0.770516 | -0.000876 |
| average_precision | 0.411683 | 0.414392 | -0.002709 |
| mcc | 0.325496 | 0.315893 | +0.009604 |
| f1 | 0.465882 | 0.460000 | +0.005882 |

## Per-sequence paired differences (fusion minus reference)

| metric | mean delta | 95% CI | wins/losses | Wilcoxon p | sign-test p |
|---|---:|---|---:|---:|---:|
| roc_auc | -0.000743 | [-0.003862, +0.002020] | 6/7 | 0.9165 | 1.0000 |
| average_precision | -0.002770 | [-0.014829, +0.008099] | 9/5 | 0.9250 | 0.4240 |
| mcc | -0.003092 | [-0.012911, +0.006598] | 3/4 | 0.3105 | 1.0000 |
| f1 | -0.001767 | [-0.007620, +0.003775] | 3/4 | 0.4990 | 1.0000 |
