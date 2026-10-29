# EdgeConBCE

Code and result artifacts for:

> Anandharajan TRV. *Local–edge routing for conformational B-cell epitope
> prediction: what survives leakage-free selection.*

The paper measures what a laboratory gains by routing antigens between a local
predictor and a remote one according to the local model's own uncertainty —
predictive quality, latency, modelled carbon and how many raw sequences leave
the device — and re-examines an earlier unpublished finding of our own that
fusing our local predictor with BepiPred-3.0 improved on BepiPred-3.0 alone.
Under leave-one-sequence-out nested selection that advantage is no longer
detectable.

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
| `src/edgeconbce/pipeline/` | the four analyses the paper reports |
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

## Licence

Code is released under the MIT Licence (`LICENSE`). Result artifacts are
released under CC BY 4.0.
