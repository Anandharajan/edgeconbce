# EdgeConBCE

Code and result artifacts for:

> Anandharajan TRV. *Local–edge routing for conformational B-cell epitope
> prediction: what survives leakage-free selection.*

The paper measures what a laboratory gains by routing antigens between a local
predictor and a remote one according to the local model's own uncertainty —
predictive quality, latency, modelled carbon and how many raw sequences leave
the device — and re-examines an earlier unpublished finding of our own that
fusing our local predictor with BepiPred-3.0 improved on BepiPred-3.0 alone.
Under leave-one-antigen-out nested cross-validation that advantage is no
longer detectable, and the MCC estimate changes sign.

## Reproducing the paper

```bash
./regenerate.sh
```

This rebuilds `paper/generated_numbers.tex`, every table, every figure and the
PDF from the stored result artifacts. It does not re-run the benchmark: the
multi-seed GPU runs are inputs here.

Every number in the manuscript is a LaTeX macro emitted by
`tools/make_numbers.py` from a result artifact. None is typed by hand, and
`paper/build.sh` fails the build if any macro is left unresolved.

## Layout

| Path | What it holds |
|---|---|
| `src/edgeconbce/` | benchmark harness, routing policies, cost model |
| `src/edgeconbce/pipeline/` | the analyses the paper reports, one module each |
| `tools/` | number, table and figure generators, plus the audit scripts |
| `tests/` | unit tests, including the leakage test for nested selection |
| `outputs/` | stored result artifacts the paper is generated from |
| `data/` | the IEDB evaluation-subset manifest (accessions only) |
| `paper/` | manuscript source |

## Data

**This repository contains no antigen sequences.** The BP3C50ID splits and the
`IEDBSeqsNotSharedAt20ID` collection are redistributed by the Technical
University of Denmark alongside BepiPred-3.0 and are not republished here.
What is released is derived data: per-antigen accessions, lengths, labels,
predicted probabilities, latencies and costs.

`data/iedb_subset_manifest.csv` lists the 47 antigens of the IEDB evaluation
subset by accession. The rule by which that subset was drawn from the source
collection was not recorded and has not been recovered; the manifest
identifies the set so it can be reconstructed, but it does not reproduce the
draw. The 205 training antigens are counted in the artifacts but not
identified, and we do not reconstruct them after the fact. This is stated in
the paper's limitations as well.

## Caveats worth knowing before reusing this

- Carbon is **modelled, not metered**, from device power, an amortised embodied
  term and measured latency. The embodied term depends on two device-utilisation
  fractions that are not in the recorded configuration; supply them to reproduce
  the absolute figures. The per-policy ordering follows from the cost identity
  for any positive constants.
- The remote tier is **co-located** in the benchmark. There is no network in the
  latency loop.
- The primary split holds **15 antigens**. The quality comparisons are not
  powered to resolve differences of the size at issue, and the paper says so.
- The headline test-derived-minus-nested gap is **not purely a selection
  effect**. The test-derived figure pools residues into one metric while the
  nested estimate averages per-antigen differences, so the two differ in how
  they aggregate as well as in where the tuning parameters were fitted.
  `selection_bias_decomposition` holds aggregation fixed and splits the gap:
  0.0388 MCC of the 0.0444 total is selection (87%) and 0.0056 is aggregation.
  Use the selection component, not the total, if you are comparing against a
  reported effect size.
- The selection-bias curve's rightmost cell is the **full antigen split**,
  where one subset exists, so it is a single deterministic evaluation rather
  than a mean over draws. It is excluded from the trend, and the size-trend
  macros read only cells with more than one replicate.
- The weight-grid families are equally spaced and therefore **not nested**
  across sizes (0.25 lies in a five-point grid but not an eleven-point one), so
  the maximal-selection inequality does not order every pair of grid sizes and
  the measured bias is not monotone in grid size in every cell.

## Licence

Code is released under the MIT Licence (`LICENSE`). Result artifacts are
released under CC BY 4.0.
