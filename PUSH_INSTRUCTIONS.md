# Publishing this repository

The repository is fully prepared and committed locally as a single commit
authored by Anandharajan TRV. It has not been pushed, because this session
could not create a GitHub repository.

## Steps

1. Create an **empty** public repository at
   <https://github.com/new>, owner `Anandharajan`, name `edgeconbce`.
   Do not add a README, .gitignore or licence — this repo already has them.

2. From the `CB&C` folder on your machine:

```bash
cd edgeconbce_release
git remote add origin https://github.com/Anandharajan/edgeconbce.git
git branch -M main
git push -u origin main
```

3. Confirm <https://github.com/Anandharajan/edgeconbce> loads, because the
   manuscript's data-availability statement and the cover letter both cite
   that exact URL.

4. At acceptance, archive the repository on Zenodo and replace the URL in
   `paper/sections/07_declarations.tex` with the DOI.

## What is in here, and what is deliberately not

Included: the benchmark harness and routing policies, the four analysis
modules, the number/table/figure generators, the unit tests, the stored
result artifacts, the IEDB evaluation-subset manifest (accessions only), and
the manuscript source.

Excluded on purpose: **all antigen sequences**. The BP3C50ID splits and the
`IEDBSeqsNotSharedAt20ID` collection are redistributed by the Technical
University of Denmark alongside BepiPred-3.0. Nothing in this repository
republishes them, and `.gitignore` blocks `*.fasta` so they cannot be added by
accident. The paper's data-availability statement says this, so keep it true.
