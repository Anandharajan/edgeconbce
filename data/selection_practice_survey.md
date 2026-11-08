# Selection-on-the-evaluation-split in combined B-cell epitope predictors

**Search date: 2026-11-08.** Scope: published B-cell epitope prediction papers, ~2018 onward,
that combine two or more predictors or models (score fusion, ensembles, stacking, weighted
voting, consensus). For each paper we ask: **was the combination parameter — mixing weight,
ensemble coefficient, decision threshold, or voting rule — selected using the same data on
which the headline performance is reported?**

Every row below was assessed from the paper's own full text (Europe PMC / PMC Open Access
BioC full text, or publisher HTML). No row is based on a title, abstract-only, or secondary
description. Rows with no retrievable methods passage are listed as *not assessable*.

Categories are kept strictly separate:

- **Leakage evident** — the text states that the combination parameter / voting rule /
  model-combination choice was made by inspecting performance on the same data reported as
  the headline result.
- **Leakage-free evident** — the text states a separate selection basis: a validation split,
  the training set, nested/cross-validated selection, or a combination rule fixed a priori
  with no tuning.
- **Unreported / cannot tell** — the paper does not say on which data the parameter or
  threshold was chosen. This is *not* counted as leakage.
- **Not applicable** — in-scope combination of tools, but no quantitative predictive
  performance is reported, so there is nothing to select on.

---

## Assessed papers

| # | Citation | What was combined | Parameter(s) at issue | Selection data | Evaluation data for headline | Classification | Verbatim evidence |
|---|----------|-------------------|-----------------------|----------------|------------------------------|----------------|-------------------|
| 1 | Manavalan B, Govindaraj RG, Shin TH, Kim MO, Lee G. **iBCE-EL**. *Front Immunol* 9:1695 (2018). doi:10.3389/fimmu.2018.01695 (PMC6072840) | 6 candidate ensembles of ML models (GB, ERT, RF, SVM, AB, NN, k-NN); equal-weight mean of predicted probabilities | (i) probability cut-off; (ii) *which* ensemble combination to adopt | Cut-off: grid search on MCC, **data not named**. Ensemble choice: the benchmarking **and the independent data set** | 5-fold CV on benchmarking set (MCC 0.454) and the independent 20% set (MCC 0.463) | **Leakage evident** (ensemble selection) | "we optimized the probability cut-off values (P) with respect to MCC using the grid search to define the class"; "However, we checked its transferability on an independent data set and selected a model that showed consistent performance both on benchmarking and independent data sets … According to this criterion, EM1 was selected as the best model and was labeled as iBCE-EL." |
| 2 | Hasan MM, Khatun MS, Kurata H. **iLBE**. *Genomics Proteomics Bioinformatics* 18(5):593–600 (2020). doi:10.1016/j.gpb.2019.04.004 (PMC8377379) | Four RF sub-models (PSSM, PKAF, AIP, AFC) fused by logistic regression | LR regression coefficients β; score thresholds | LR coefficient fitting data **not stated**; thresholds taken from training-set results | AUC 0.809 on training dataset (10-fold CV) plus an independent dataset | **Unreported / cannot tell** (combination coefficients); threshold itself leakage-free | "the respective RF scores evaluated from the four features (PSSM, PKAF, AIP, and AFC) were combined using a LR algorithm … the final prediction score P was calculated by: where βn is the regression coefficient" — no statement of the data used to fit β. Threshold: "The high-, moderate-, and low-level thresholds were determined based on RF scores of 0.485, 0.410, and 0.360, respectively, which corresponded to Sp levels of 0.866, 0.747, and 0.636 in the training set results" |
| 3 | Liu T, Shi K, Li W. **DLBEpitope**: Deep learning methods improve linear B-cell epitope prediction. *BioData Mining* 13:1 (2020). doi:10.1186/s13040-020-00211-0 (PMC7371472) | 11 feedforward DNN classifiers trained on 11 negative subsamples; votes summed | Vote-sum combination rule; peptide length | Combination rule fixed a priori (unweighted sum of 11 votes, no weights, no tuned cut-off; headline metric is AUC). Optimal epitope **length** was chosen on test-set AUC | AUC on hold-out independent test sets and two public test sets | **Leakage-free evident** (combination rule untuned) — with the caveat noted | "These classifiers were combined as an ensemble model to score candidate epitope. If a peptide was predicted as epitope by a classifier, one was assigned; otherwise, zero was given … The sum of these 11 values was taken as an index"; caveat: "We used the AUC value from the independent test set to evaluate the performance of the ensemble model." |
| 4 | Galanis KA, Nastou KC, Papandreou NC, Petichakis GN, Pigis DG, Iconomidou VA. **Linear B-cell epitope prediction … performance review of methods available via command-line interface**. *Int J Mol Sci* 22(6):3210 (2021). doi:10.3390/ijms22063210 (PMC8004178) | Consensus voting classifier over BcePred, BepiPred, ABCpred, COBEpro, SVMTriP, LBtope, LBEEP | Vote threshold; which predictors to include; window size | The same `Consensus_NR` / `Consensus_R` benchmark sets; the paper states no training set was used | MCC/accuracy of the consensus on `Consensus_R` and `Consensus_NR` | **Leakage evident** | "The consensus threshold chosen, **after testing**, is defined as the hit overlap of at least half out of 'n' selected methods"; "Using the Consensus_NR data set we implemented many iterations of the consensus method utilizing many different method combinations, in order to find the optimum. As expected, LBEEP's presence undermined the consensus predictor's performance, and it was therefore omitted from the final version"; "since all the predictors tested in this work were previously developed, only the latter [test set] was necessary" |
| 5 | La Marca AF, Lopes RDS, Lotufo ADP, Bartholomeu DC, Minussi CR. **BepFAMN**: A method for linear B-cell epitope predictions based on Fuzzy-ARTMAP artificial neural network. *Sensors* 22:4027 (2022). doi:10.3390/s22114027 (PMC9185646) | 3 Fuzzy-ARTMAP knowledge models per CV fold (15 total) combined by a "competition" rule | Voting rule | Fixed a priori: plurality vote of 3 runs; no tuned weight or threshold | 5-fold CV on DB_bac/DB_vir/DB_prot/DB_all, plus the independent BepiPred-2.0 test set | **Leakage-free evident** | "for each partition built by cross-validation, on each dataset, the algorithm is executed three times and, in the end, the result that occurs most frequently, for each input pattern, is the ANN prediction." |
| 6 | Ozger ZB, Cihan P. **A novel ensemble fuzzy classification model in SARS-CoV-2 B-cell epitope identification for development of protein-based vaccine**. *Appl Soft Comput* (2022):108280. doi:10.1016/j.asoc.2021.108280 (PMC8673934) | 5 fuzzy rule-based classifiers (GBML, GCCL, CHI, SLAVE, W) combined by majority voting | Voting rule (majority of 5); later 4V/5V/6V sensitivity on unlabelled SARS-CoV-2 data; training class balance | Combination rule fixed a priori for the labelled SARS-CoV evaluation. However, training-set class composition and the model carried forward were chosen from test-set results | Mean accuracy over 6 random train/test splits of the labelled SARS-CoV set | **Leakage-free evident** (combination rule untuned) — with the caveat noted | "The proposed ensemble model combines the decisions classifiers GCCL, GBML, CHI, W, and SLAVE by using a majority voting scheme … the class of each sample in the test set was decided by the majority voting method."; caveat: "The model with the high test set accuracy for SARS-CoV was also applied to the SARS-CoV-2 dataset"; "For the model to learn the classes correctly, it was decided experimentally how many samples from the negative class should be present in the training set." |
| 7 | Shashkova TI, Umerenkov D, Salnikov M, Strashnov PV, Konstantinova AV, Lebed I, Shcherbinin DN, et al. **SEMA**: Antigen B-cell conformational epitope prediction using deep transfer learning. *Front Immunol* 13:960985 (2022). doi:10.3389/fimmu.2022.960985 (PMC9523212) | Ensemble of five independently fine-tuned models, results averaged | Averaging weights (equal, untuned); classification threshold for MCC/PPV/sensitivity | Averaging: equal weights, untuned. Threshold: "optimal cut-off provided by ROC AUC analysis" — **data not named** (a 9:1 train/validation split exists but is not tied to the threshold) | Retrospective test set split by PDB release date (post-2020-01-01) | **Unreported / cannot tell** (threshold) | "The final models were obtained as an ensemble of five models fine-tuned independently from the same pretrained checkpoint."; "To calculate MCC, PPV and sensitivity we converted prediction values to a binary values applying a threshold. Threshold was set as an optimal cut-off provided by ROC AUC analysis corresponding to the highest true positive rate together with the lowest false positive rate." — no dataset stated |
| 8 | Clifford JN, Høie MH, Deleuran S, Peters B, Nielsen M, Marcatili P. **BepiPred-3.0**. *Protein Science* 31(12):e4497 (2022). doi:10.1002/pro.4497 (PMC9679979) | Majority voting over five cross-validation fold models | Per-fold classification threshold; majority-vote rule | Each fold model's threshold maximised MCC **on its own validation split** | IEDB external test sets | **Leakage-free evident** | "For the remaining threshold-dependent metrics, a majority voting scheme was based on the individual predictions from the five models. The classification threshold used for each fold model was one that maximized the MCC score on respective validation splits." |
| 9 | Israeli S, Louzoun Y. **CALIBER**: Single-residue linear and conformational B cell epitopes prediction using random and ESM-2 based projections. *Brief Bioinform* 25:bbae084 (2024). doi:10.1093/bib/bbae084 (PMC10940830) | BiLSTM + GCN, and a "boosting" combination of both | Boosting combination rule; threshold for BAC/MCC | Model hyperparameters by grid search on internal validation AUC (leakage-free). The boosting combination rule is referred to but **not described**, and no threshold-selection statement exists | A single held-out test set | **Unreported / cannot tell** (combination rule and threshold) | "All the model hyperparameters were optimized using a grid search using an internal validation AUC as a score."; "we tested a GCN, or the combination of a GCN with the model above, as well as boosting both models (See methods)" — the Methods contain no boosting subsection and no statement of how the classification threshold for BAC/MCC was set |
| 10 | Li J, Zhao Z, Tai C, Sun T, Tan L, Li X, He W, Li H, Zhang J. **VirusImmu**: a novel ensemble machine learning approach for viral immunogenicity prediction. *Brief Funct Genomics* (2025):elaf008. doi:10.1093/bfgp/elaf008 (PMC12051847) | Soft-voting ensemble of RF + XGBoost + kNN | Mixing weights (0.05 / 0.75 / 0.20) | Enumerated over a grid and selected **on the training set** | 50 random 80:20 test splits and an independently collected external test set (AUC) | **Leakage-free evident** | "we enumerated all possible (a total of 232) weights with each weight increasing from 0 to 1 at the incremental step of 0.05 … The optimal soft-voting ensemble model (VirusImmu) obtained in the training set (weight: 0.05 for XGBoost, 0.75 for RF, and 0.20 for kNN, AUC = 0.793) achieved better performance (average AUC: 0.782) … on test sets" |
| 11 | Gurcan F. **Linear B-cell epitope prediction for SARS and COVID-19 vaccine design: integrating balanced ensemble learning models and resampling strategies**. *PeerJ Comput Sci* (2025):e2970. doi:10.7717/peerj-cs.2970 (PMC12193457) | Ten off-the-shelf ensemble classifiers (bagging/boosting/balancing) × nine resampling schemes | Which resampler × classifier configuration to report; classifier hyperparameters | Hyperparameters by GridSearchCV inside the training data (leakage-free). But the winning configuration is chosen by comparing scores on the **single 20% split that is also the headline**; no third split exists | The same 20% stratified hold-out ("validation set") | **Leakage evident** (configuration selection) | "The train_test_split function was utilized to split the resampled data into training and testing subsets, with 80% of the data used for training and 20% reserved for testing."; "The results highlight that SMOTE-ENN generally produces the highest performance across the classifiers, with ExtraTrees achieving the highest ROC AUC score of 0.9899." — the reported best configuration is the argmax over the reported split |
| 12 | Wang S, Geng A, Luo Z, Li Y, Xu J, Meng Y, Wei L, Zou Q, Zhang Z, Wang T, Cui F. **MsgaBpred**: a B-cell epitope predictor integrating AlphaFold3-predicted structures with multi-scale GCNs and protein language models. *PLoS Comput Biol* (2026):e1014195. doi:10.1371/journal.pcbi.1014195 (PMC13123994) | Fusion of ESM-C + ESM-IF1 + DSSP representations; multi-scale GCN kernels concatenated and fused; additive attention | Classification threshold for Pre/Rec/F1/MCC/BACC | Threshold chosen by maximising F1 **on the validation set** | Held-out test split and the external Epitope3D benchmark | **Leakage-free evident** | "Threshold-based metrics (Pre, Rec, F1, MCC, BACC) are calculated using a classification threshold that is determined by maximizing the F1-score on the validation set." |

### Papers in scope as tool-combinations but reporting no predictive performance (not applicable)

| # | Citation | What was combined | Finding | Verbatim evidence |
|---|----------|-------------------|---------|-------------------|
| 13 | Abesamis LMI, Aliping EGA, Armada FKGH, Danao MF, Del Valle PDB, Regencia ZJG, Baja ES, Ligsay A, et al. **In silico comparative analysis of predicted B cell epitopes against dengue virus (serotypes 1–4) isolated from the Philippines**. *Vaccines* 10:1259 (2022). doi:10.3390/vaccines10081259 (PMC9415047) | Kolaskar–Tongaonkar antigenicity + Parker hydrophilicity + Emini surface accessibility (IEDB), intersected and ranked | Tool-default thresholds; no tuning; no sensitivity/specificity/AUC reported anywhere, so there is no headline performance to select on | "The threshold value was based on the provided value of the immunogenicity prediction tools in IEDB." |
| 14 | Omony JB, Wanyana A, Mugimba KK, Kirunda H, Nakavuma JL, Otim-Onapa M, Byarugaba DK. **Epitope peptide-based predication and other functional regions of antigenic F and HN proteins of waterfowl and poultry avian avulavirus serotype-1 isolates from Uganda**. *Front Vet Sci* 8:610375 (2021). doi:10.3389/fvets.2021.610375 (PMC8240872) | BepiPred-2.0 + Emini + Kolaskar–Tongaonkar + VaxiJen | Default thresholds used as published; no predictive performance reported | "For linear B-cell epitopes, BepiPred was used with default threshold value of 0.4." |

### Screened and excluded as not in scope (read, but they do not combine two or more predictors)

| Citation | Why excluded |
|----------|--------------|
| *A new approach for determining SARS-CoV-2 epitopes using machine learning-based in silico methods*, *Comput Biol Chem* (2022), PMC9055767 | Benchmarks individual ML algorithms (incl. RF/bagging) and carries the single best one forward; no fusion of predictors and no combination parameter |
| **BLMPred**, *Comput Struct Biotechnol J* (2026), PMC12795683 | Trains many models by 10-fold CV and selects a **single** SVM; no score fusion |
| **epitope1D**, *Brief Bioinform* 25 (2023), PMC10199762 | Multiple feature families but a single RF classifier chosen; no combination of predictors |
| Bioinformatics Advances (2026), PMC13197123 | Compares five single classifiers on Influenza A; no combination |

### Not assessable

| Citation | Why |
|----------|-----|
| Consensus-based computational mapping of NS1 antigenic determinants in dengue and Zika viruses, *BMC Infect Dis* (2026), PMC12910838 | Explicitly a consensus-of-tools paper and clearly in scope, but the full text could not be retrieved through Europe PMC full-text XML, the PMC Open-Access BioC service, or direct fetch (PMC returned a bot-check page). No methods passage was read, so no judgement is recorded. |

---

## Counts

In-scope papers assessed with a headline performance figure: **12**

| Category | Count | Papers |
|----------|-------|--------|
| Leakage evident | **3** | #1 iBCE-EL, #4 Galanis consensus, #11 PeerJ CS balanced ensemble |
| Leakage-free evident | **6** | #3 DLBEpitope, #5 BepFAMN, #6 ensemble fuzzy, #8 BepiPred-3.0, #10 VirusImmu, #12 MsgaBpred |
| Unreported / cannot tell | **3** | #2 iLBE, #7 SEMA, #9 CALIBER |

Additional rows: **2** in-scope tool-combination papers report no quantitative performance
(not applicable); **4** papers were read and excluded as not combining predictors;
**1** paper was in scope but **not assessable** (full text unobtainable).

Of the six "leakage-free evident" papers, only **three** (#8 BepiPred-3.0, #10 VirusImmu,
#12 MsgaBpred) state a *positive, explicit* selection procedure on data separate from the
evaluation data. The other three (#3, #5, #6) are leakage-free because their combination rule
is fixed a priori (an unweighted vote) and nothing was tuned at all — a different and weaker
kind of cleanliness. Two of those three (#3, #6) additionally make *other* design choices
(epitope length; training class balance; which trained model to carry forward) by inspecting
performance on the data they report, which is recorded in their rows.

So the defensible headline is: among 12 combined B-cell epitope predictors whose methods we
read, **3 demonstrably selected a combination parameter on the data they report performance
on, 3 demonstrably did not, 3 are clean only because nothing was tuned, and 3 do not report
enough to tell.** The single most common failure is not leakage but silence: no paper in the
"unreported" group states which split its decision threshold came from, and that silence is
itself consistent with either practice.

---

## Limitations

**How papers were found.** Candidates were identified on 2026-11-08 by (a) Europe PMC
structured queries over title/abstract and over full-text METHODS sections, restricted to
`FIRST_PDATE 2018-01-01 … 2026-11-08` and `OPEN_ACCESS:Y`, using the terms *B-cell epitope* /
*B cell epitope* crossed with *ensemble*, *consensus*, *stacking*, *voting*, *soft voting*,
*majority voting*, *weighted voting*, *machine learning*, *deep learning*, *classifier*; and
(b) general web search on the same phrases. Full texts were read via the PMC Open Access BioC
service and, where that failed, Europe PMC full-text XML or publisher HTML. One broad query
returned 1,128 hits, overwhelmingly applied multi-epitope vaccine-design papers that run
several IEDB tools but report no predictive performance; these were screened out by title and
abstract and are not represented here except for two illustrative rows (#13, #14).

**Selection bias in this sample.** The sample is biased in at least five ways, all of which
push against treating these counts as subfield prevalence estimates. (1) **Open access only.**
Papers behind paywalls — including much of the Elsevier/Springer methods literature — are
structurally absent, except where a PMC deposit existed. (2) **Keyword-dependent.** A paper
that fuses predictors without using the words "ensemble", "consensus", "stacking" or "voting"
in its title, abstract or methods was invisible to these queries. (3) **Recency skew.** Europe
PMC relevance ordering and the growth of open access put 2024–2026 papers over-proportionally
in the candidate pool; the three "leakage-free evident with explicit procedure" cases are all
from 2022 or later, so the sample may understate historical leakage and overstate current
hygiene. (4) **Prominence skew.** Well-cited tool papers (BepiPred-3.0, iBCE-EL, SEMA) are
over-represented relative to low-visibility applied ensembles, and prominent tool papers are
likelier to have been through methodologically demanding review. (5) **Small n.** With 12
assessable papers, each count carries roughly ±1–2 papers of sampling noise; no proportion here
should be quoted to better than "about a quarter / about a half / about a quarter".

**Not assessable, and why.** One in-scope paper (PMC12910838) could not be read at all: PMC
served a bot-check page and both Europe PMC full-text XML and the BioC service returned errors
for it. It is recorded with no judgement. Several near-miss candidates found in the queries
were not pursued because their abstracts made clear they train a single model; those that were
read and rejected are listed above. Within the assessed rows, three papers are classified
"unreported / cannot tell" precisely because the relevant sentence does not exist — that is an
observation about the text, not an inference about the authors' practice, and it is deliberately
kept out of the leakage count.

**Classification judgement.** Two rows required a judgement call that a reader may reasonably
make differently. Row #11 tunes hyperparameters properly by cross-validation inside the training
data but has no third split, so the *reported* best configuration is the maximum over the
reported split; we call that leakage evident because the quantity carried into the abstract was
selected on the evaluation data, even though no mixing weight as such was tuned. Row #1's
cut-off search does not name its data; the leakage classification rests on the separate,
explicit statement that the *ensemble composition* was chosen using the independent set. Rows
#3, #5 and #6 are "leakage-free" only in the narrow sense that the combination rule itself was
never tuned; both #3 and #6 select other hyperparameters on their reported test data, and a
paper arguing a broader thesis about selection hygiene should say so rather than count them as
clean practice.
