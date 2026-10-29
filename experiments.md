# Experimental design

Each question below maps to a claim in the manuscript. Thresholds were fixed
before the corresponding analysis was run; where that is not true it is said so.

| # | Question | Independent variable | Metrics | Seeds / units | Status |
|---|---|---|---|---|---|
| E1 | Does uncertainty routing match always-remote quality? | routing policy | MCC, ROC-AUC, AP, F1 | 10 seeds | done |
| E2 | What does routing cost in latency and disclosure? | routing policy | mean latency, raw-sequence share, bytes | 10 seeds | done |
| E3 | Does routing reduce modelled carbon? | routing policy | modelled gCO2e | 10 seeds | done — direction opposite to the project's prior expectation |
| E4 | Which PLM components contribute? | PLM policy variants | MCC, ROC-AUC | 10 seeds | done — conformal guard contributes nothing measurable |
| E5 | Does the published fusion advantage survive honest selection? | selection protocol | per-antigen ΔROC-AUC, ΔMCC, ΔAP, ΔF1 | 15 antigens (BP3C50ID) | done |
| E6 | Does E5 replicate on a second dataset? | selection protocol | as E5 | 47 antigens (IEDB subset) | done |
| E7 | Is the difference in E5 caused by the protocol or by reimplementation? | implementation | exact metric agreement with the published grid | all 21 grid weights | done — exact to 6 d.p. |
| E8 | Which predictor does the selection procedure prefer? | dataset | selected weight per fold | 15 + 47 folds | done |

## Pre-specified decision rules

- A comparison whose 95% interval contains zero is reported as no detected
  difference, never as a trend or a partial success.
- The fusion selection metric is MCC, fixed in advance because MCC is the
  metric the original result was tuned on; the other three metrics are
  reported as secondary and are not used to choose a configuration.
- Resampling unit for fusion analyses is the antigen, not the residue, because
  residues within an antigen are not independent.

## Not attempted, and why

- **Metered energy.** Requires instrumented hardware that was not available.
  Carbon stays modelled throughout and is labelled as such.
- **Genuine two-host deployment.** Requires a second machine; the existing
  evidence is an HTTP probe and the manuscript bounds its transport claims to
  that.
- **Larger ESM-2 checkpoints.** No GPU was reachable from the analysis
  environment. `docs/colab_larger_plm_checkpoint.ipynb` runs this on a T4 and
  regenerates every artifact; the PLM conclusions are bounded to the 8M
  checkpoint until it is run.
- **BepiPred-3.0 on the BP3 training split.** Would allow fusion weights to be
  selected on genuinely separate data rather than by nesting within the test
  split. It needs the DTU service and was not run.
