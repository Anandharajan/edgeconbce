# Venue

**Target:** Computational Biology and Chemistry (Elsevier).

## Why this journal

The journal publishes computational life-science methods including sequence
analysis, bioinformatics and machine-learning models for biological data. It
published Attique et al., "DeepBCE: Evaluation of deep learning models for
identification of immunogenic B-cell epitopes", vol. 104, art. 107874, 2023
(doi:10.1016/j.compbiolchem.2023.107874), which is the work that motivated this
project — so the topic is demonstrably in scope.

The framing that fits is a benchmark, evaluation-methodology and negative-result
paper. A standalone state-of-the-art method claim would not survive review and
is not made.

## Requirements

Elsevier operates "Your Paper Your Way": the first submission does not need to
be in the journal template, so this manuscript builds with the standard
`article` class and an Elsevier-like layout. The accepted version should be
recompiled with `elsarticle.cls`, which was not available in the build
environment used here.

[PENDING] The following must be read off the journal's official Guide for
Authors before submission and recorded here; they were not verifiable from the
build environment and must not be guessed:
- permitted article types and which one this is submitted as
- word or page limit, and abstract length limit
- reference style and maximum reference count
- figure resolution and colour-charge policy
- the exact wording required for the data-availability and AI-use declarations

## Policy compatibility

- Data: BP3C50ID is redistributed by DTU and is not republished here. The
  IEDB-derived subset's redistribution terms are unconfirmed ([PENDING]).
- Code: the harness and all analysis code are in the repository and can be
  deposited publicly; a DOI is [PENDING].
- Ethics: sequence data only; no human or animal subjects.
