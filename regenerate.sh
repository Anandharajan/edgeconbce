#!/usr/bin/env bash
# One command that rebuilds every number, table, figure and the PDF from the
# stored result artifacts. Run from the repository root.
#
#   ./regenerate.sh
#
# It does not re-run the benchmark: the 10-seed GPU runs are inputs here. To
# regenerate those, see docs/colab_larger_plm_checkpoint.ipynb.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH:-}"
PY="${PYTHON:-python3}"

OUT=outputs
REPEATED="$OUT/bp3_repeated_eval_sci/bp3_real_multiseed_sci_aggregate_results.json"
BEPIPRED_SUMMARY="$OUT/external_baselines/bepipred_submission/evaluation/bepipred3_local_rerun_summary.json"
BEPIPRED_RAW="$OUT/external_baselines/bepipred_submission/bepipred3_results/raw_output.csv"
CARBON_BP3="$OUT/carbon_bce_r_colab_bp3/carbon_bce_r_test_predictions.json"
CARBON_IEDB="$OUT/carbon_bce_r_iedb_bounded_gpu/carbon_bce_r_test_predictions.json"
IEDB_REAL="$OUT/iedb_reduced_subset_bounded/iedb_reduced_subset_bounded_results.json"
NESTED="$OUT/nested_fusion_sci"
BIAS="$OUT/selection_bias_sci"

for f in "$REPEATED" "$BEPIPRED_SUMMARY" "$BEPIPRED_RAW" "$CARBON_BP3" "$CARBON_IEDB" "$IEDB_REAL"; do
  [ -f "$f" ] || { echo "missing required artifact: $f" >&2; exit 1; }
done

echo "== unit tests =="
$PY -m unittest discover -s tests -q

echo "== leakage-free fusion evaluation =="
$PY -m edgeconbce.pipeline.nested_fusion --dataset bp3 \
  --carbon-predictions "$CARBON_BP3" --bepipred-raw-scores "$BEPIPRED_RAW" \
  --reference-threshold 0.1512 --output-dir "$NESTED"
$PY -m edgeconbce.pipeline.nested_fusion --dataset iedb \
  --carbon-predictions "$CARBON_IEDB" --real-results "$IEDB_REAL" \
  --reference-policy local_only --reference-threshold 0.51 --output-dir "$NESTED"

echo "== replication of the published selection protocol =="
$PY -m edgeconbce.pipeline.replication_check \
  --carbon-predictions-bp3 "$CARBON_BP3" --bepipred-raw-scores "$BEPIPRED_RAW" \
  --carbon-predictions-iedb "$CARBON_IEDB" --iedb-real-results "$IEDB_REAL" \
  --published-report-bp3 "$OUT/fusion_baseline_sci/fusion_report.json" \
  --output "$NESTED/replication_check.json"

echo "== antigen-level resampling of the routing policies =="
$PY -m edgeconbce.pipeline.policy_resampling \
  --runs-dir "$OUT/bp3_repeated_eval_sci/runs" \
  --bootstrap 2000 \
  --output "$NESTED/policy_resampling.json"

echo "== carbon cost-model decomposition =="
$PY -m edgeconbce.pipeline.carbon_decomposition --repeated-eval "$REPEATED" \
  --real-results "$IEDB_REAL" --output "$NESTED/carbon_decomposition.json"

echo "== how far selection on the evaluation split moves the estimate =="
$PY -m edgeconbce.pipeline.selection_bias_curve \
  --carbon-predictions "$CARBON_BP3" --bepipred-raw-scores "$BEPIPRED_RAW" \
  --reference-threshold 0.1512 --replicates 100 \
  --output "$BIAS/selection_bias_curve.json"

echo "== region-level routing against whole-antigen routing =="
$PY -m edgeconbce.pipeline.region_routing \
  --carbon-predictions "$CARBON_BP3" --bepipred-raw-scores "$BEPIPRED_RAW" \
  --reference-threshold 0.1512 --bootstrap 2000 \
  --output "$OUT/region_routing/region_routing_bp3_mcc.json"

echo "== predicting the inflation of a grid scan without rerunning it =="
$PY -m edgeconbce.pipeline.selection_bias_correction \
  --carbon-predictions "$CARBON_BP3" --bepipred-raw-scores "$BEPIPRED_RAW" \
  --selection-bias-curve "$BIAS/selection_bias_curve.json" \
  --reference-threshold 0.1512 --resamples 300 \
  --output "$BIAS/selection_bias_correction.json"

echo "== how much of the headline gap is selection, how much is aggregation =="
$PY -m edgeconbce.pipeline.selection_bias_decomposition \
  --carbon-predictions "$CARBON_BP3" --bepipred-raw-scores "$BEPIPRED_RAW" \
  --reference-threshold 0.1512 \
  --output "$BIAS/selection_bias_decomposition.json"

echo "== how published combined predictors select their combination parameter =="
$PY -m edgeconbce.pipeline.selection_practice_survey \
  --survey data/selection_practice_survey.md \
  --output "$BIAS/selection_practice_survey.json"

echo "== carbon break-even across remote model scales =="
$PY -m edgeconbce.pipeline.carbon_scale_sensitivity \
  --carbon-decomposition "$NESTED/carbon_decomposition.json" \
  --output "$BIAS/carbon_scale_sensitivity.json"

echo "== manuscript numbers, tables, figures =="
$PY tools/make_numbers.py --repeated-eval "$REPEATED" --bepipred-summary "$BEPIPRED_SUMMARY" \
  --replication "$NESTED/replication_check.json" \
  --nested-bp3 "$NESTED/nested_fusion_bp3_mcc.json" --nested-iedb "$NESTED/nested_fusion_iedb_mcc.json" \
  --carbon-decomposition "$NESTED/carbon_decomposition.json" \
  --policy-resampling "$NESTED/policy_resampling.json" \
  --provenance-audit "$NESTED/provenance_audit.json" \
  --carbon-scale-sensitivity "$BIAS/carbon_scale_sensitivity.json" \
  --selection-survey "$BIAS/selection_practice_survey.json" \
  --selection-bias-curve "$BIAS/selection_bias_curve.json" \
  --selection-bias-decomposition "$BIAS/selection_bias_decomposition.json" \
  --iedb-real-results "$IEDB_REAL" --output paper/generated_numbers.tex
$PY tools/make_tables.py --repeated-eval "$REPEATED" \
  --nested-bp3 "$NESTED/nested_fusion_bp3_mcc.json" --nested-iedb "$NESTED/nested_fusion_iedb_mcc.json" \
  --replication "$NESTED/replication_check.json" \
  --policy-resampling "$NESTED/policy_resampling.json" \
  --scenario-summary "$OUT/scenario_analysis_sci/scenario_comparison_summary.csv" \
  --output-dir paper/tables
$PY tools/make_figures.py --repeated-eval "$REPEATED" \
  --policy-resampling "$NESTED/policy_resampling.json" \
  --nested-bp3 "$NESTED/nested_fusion_bp3_mcc.json" --nested-iedb "$NESTED/nested_fusion_iedb_mcc.json" \
  --selection-bias-curve "$BIAS/selection_bias_curve.json" \
  --carbon-scale-sensitivity "$BIAS/carbon_scale_sensitivity.json" \
  --output-dir paper/figures

echo "== manuscript =="
./paper/build.sh

echo "== audits =="
cat paper/main.tex paper/sections/*.tex > /tmp/edgeconbce_full.tex
$PY tools/claim_trace.py /tmp/edgeconbce_full.tex | head -1
$PY tools/originality_scan.py /tmp/edgeconbce_full.tex --sources docs/ | $PY -c \
  "import json,sys; d=json.load(sys.stdin); print('source overlap passages:', len(d['overlap']))"
echo "done"
