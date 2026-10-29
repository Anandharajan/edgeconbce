"""Emit the manuscript's LaTeX tables from result artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

POLICIES = ("local_only", "edge_only", "adaptive_split", "adaptive_carbon_guard",
            "plm_local_only", "plm_edge_only", "plm_region_constrained_edge",
            "plm_adaptive_split", "plm_adaptive_conformal_guard")
LABEL = {
    "local_only": "local-only", "edge_only": "edge-only", "adaptive_split": "adaptive-split",
    "adaptive_carbon_guard": "carbon-guard", "plm_local_only": "PLM local-only",
    "plm_edge_only": "PLM edge-only", "plm_region_constrained_edge": "PLM region-constrained",
    "plm_adaptive_split": "PLM adaptive-split", "plm_adaptive_conformal_guard": "PLM conformal-guard",
}
COMPARISONS = (
    ("adaptive_split_minus_local_only", "adaptive-split vs local-only"),
    ("adaptive_split_minus_edge_only", "adaptive-split vs edge-only"),
    ("adaptive_carbon_guard_minus_local_only", "carbon-guard vs local-only"),
    ("adaptive_carbon_guard_minus_adaptive_split", "carbon-guard vs adaptive-split"),
    ("plm_region_constrained_edge_minus_plm_local_only", "PLM region-constrained vs PLM local-only"),
    ("plm_region_constrained_edge_minus_plm_edge_only", "PLM region-constrained vs PLM edge-only"),
    ("plm_local_only_minus_local_only", "PLM local-only vs local-only"),
)


def ci(row: dict, metric: str, places: int = 3) -> str:
    mean, lo, hi = row.get(f"{metric}_mean"), row.get(f"{metric}_ci_low"), row.get(f"{metric}_ci_high")
    if mean is None:
        return "--"
    return f"{mean:.{places}f} \\tiny[{lo:.{places}f}, {hi:.{places}f}]"


def table_policies(d: dict) -> str:
    agg = {r["policy"]: r for r in d["aggregate_policy_metrics"]}
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Routing-policy performance on the BP3C50ID external test split, "
        r"aggregated over " + str(len(d["seeds"])) + r" seeds. Each cell gives the mean with its "
        r"95\% bootstrap confidence interval. Carbon is a model-based estimate, not a metered "
        r"measurement (Section~\ref{sec:cost-accounting}).}",
        r"\label{tab:policy-summary}",
        r"\small",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Policy & MCC & ROC-AUC & Latency (ms) & gCO$_2$e & Raw-sequence share \\",
        r"\midrule",
    ]
    for i, pol in enumerate(POLICIES):
        row = agg.get(pol)
        if row is None:
            continue
        if pol == "plm_local_only":
            lines.append(r"\midrule")
        share = ci(row, "raw_sequence_share_rate") if row.get("raw_sequence_share_rate_mean") is not None else "--"
        lines.append(
            f"{LABEL[pol]} & {ci(row,'mcc')} & {ci(row,'roc_auc')} & {ci(row,'mean_latency_ms',1)} "
            f"& {ci(row,'mean_total_gco2e')} & {share} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def table_paired(d: dict) -> str:
    deltas = {r["comparison"]: r for r in d["paired_delta_summary"]}
    sig = {(r["comparison"], r["metric"]): r for r in d["paired_significance"]}
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Paired per-seed differences between routing policies "
        r"(" + str(len(d["seeds"])) + r" seeds). $\Delta$ is the mean difference with its 95\% "
        r"bootstrap interval; $p$ is a two-sided exact sign test over seeds. An interval that "
        r"contains zero is reported as no detected difference.}",
        r"\label{tab:paired}", r"\small",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Comparison & $\Delta$MCC [95\% CI] & $p$ & $\Delta$latency (ms) & $\Delta$raw share \\",
        r"\midrule",
    ]
    for key, label in COMPARISONS:
        row = deltas.get(key)
        if row is None:
            continue
        s = sig.get((key, "mcc"), {})
        p = s.get("p_value_two_sided_sign_test")
        p_str = f"{p:.4f}" if p is not None else "--"
        lines.append(
            f"{label} & {row['mcc_mean']:+.3f} \\tiny[{row['mcc_ci_low']:+.3f}, {row['mcc_ci_high']:+.3f}] "
            f"& {p_str} & {row['mean_latency_ms_mean']:+.1f} & {row['raw_sequence_share_rate_mean']:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def table_fusion(nested: dict[str, dict], replication: dict) -> str:
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{The fusion effect under two selection protocols. The test-derived protocol "
        r"chooses the mixing weight and the decision threshold on the same split it reports; the "
        r"nested protocol chooses both inside the training folds of a leave-one-sequence-out loop. "
        r"Intervals are 95\% bootstrap intervals over per-antigen differences.}",
        r"\label{tab:fusion}", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
        r"Split & Protocol & Weight & $\Delta$ROC-AUC & $\Delta$MCC & Antigens improved \\",
        r"\midrule",
    ]
    key_map = {"BP3C50ID external test": "bp3", "IEDB bounded subset": "iedb"}
    for name, d in nested.items():
        rep = replication.get(key_map[name], {})
        sel = rep.get("selected", {})
        td = rep.get("delta_vs_reference", {})
        lines.append(
            f"{name} & test-derived & {sel.get('weight', float('nan')):.2f} "
            f"& {td.get('roc_auc', float('nan')):+.4f} & {td.get('mcc', float('nan')):+.4f} & -- \\\\"
        )
        w = d["selected_weights"]
        auc = d["paired_statistics"]["roc_auc"]
        mcc = d["paired_statistics"]["mcc"]
        n = len(d["per_fold"])
        # Estimates on one row, intervals beneath: four-decimal deltas with
        # bracketed intervals overrun the text width on a single line.
        lines.append(
            f" & nested (LOSO) & {w['median']:.2f} "
            f"& {auc['mean_delta']:+.4f} & {mcc['mean_delta']:+.4f} "
            f"& {auc['wins']}/{n} \\\\"
        )
        lines.append(
            f" & & \\tiny[{w['min']:.2f}, {w['max']:.2f}] "
            f"& \\tiny[{auc['ci_low']:+.4f}, {auc['ci_high']:+.4f}] "
            f"& \\tiny[{mcc['ci_low']:+.4f}, {mcc['ci_high']:+.4f}] & \\\\"
        )
        lines.append(r"\addlinespace")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


ANTIGEN_COMPARISONS = (
    ("adaptive_split_minus_edge_only", "adaptive-split vs edge-only", True),
    ("adaptive_split_minus_local_only", "adaptive-split vs local-only", False),
    ("adaptive_carbon_guard_minus_adaptive_split", "carbon-guard vs adaptive-split", False),
    ("plm_region_constrained_edge_minus_plm_local_only", "PLM region-constrained vs PLM local-only", False),
    ("plm_adaptive_split_minus_plm_edge_only", "PLM adaptive-split vs PLM edge-only", True),
)
QUALITY = (("mcc", "MCC"), ("roc_auc", "ROC-AUC"), ("average_precision", "AP"), ("f1", "F1"))


def _pair_key(resampled: dict, name: str) -> tuple[str, int]:
    """Locate a comparison, allowing for the bootstrap's alphabetical pair order."""
    if name in resampled["paired"]:
        return name, 1
    a, b = name.split("_minus_")
    flipped = f"{b}_minus_{a}"
    if flipped in resampled["paired"]:
        return flipped, -1
    raise KeyError(name)


def table_policies_antigen(resampled: dict) -> str:
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Routing-policy performance on the BP3C50ID external split under "
        r"antigen-level resampling. Intervals are 95\% two-way bootstrap intervals over "
        r"antigens and seeds (" + str(resampled["n_antigens"]) + r" antigens, "
        + str(resampled["n_seeds"]) + r" seeds, " + str(resampled["bootstrap_repeats"]) +
        r" resamples), so they carry the uncertainty that matters for a claim about other "
        r"antigens. Carbon is a model-based estimate, not a metered measurement "
        r"(Section~\ref{sec:cost-accounting}).}",
        r"\label{tab:policy-summary}", r"\small",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Policy & MCC & ROC-AUC & Latency (ms) & Raw-sequence share \\",
        r"\midrule",
    ]
    for pol in POLICIES:
        row = resampled["per_policy"].get(pol)
        if row is None:
            continue
        if pol == "plm_local_only":
            lines.append(r"\midrule")
        def cell(metric, places=3):
            v = row[metric]
            return f"{v['mean']:.{places}f} \\tiny[{v['ci_low']:.{places}f}, {v['ci_high']:.{places}f}]"
        lines.append(
            f"{LABEL[pol]} & {cell('mcc')} & {cell('roc_auc')} & {cell('latency_ms', 1)} "
            f"& {cell('raw_sequence_shared')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def table_paired_antigen(resampled: dict) -> str:
    """Every quality metric for every comparison, so no metric can be omitted selectively."""
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Paired differences between routing policies under antigen-level "
        r"resampling, with 95\% two-way bootstrap intervals. All four quality metrics are "
        r"shown for every comparison. An interval containing zero is reported as no detected "
        r"difference; intervals excluding zero are marked $^{\ast}$.}",
        r"\label{tab:paired}", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"Comparison & $\Delta$MCC & $\Delta$ROC-AUC & $\Delta$AP & $\Delta$F1 \\",
        r"\midrule",
    ]
    for name, label, _ in ANTIGEN_COMPARISONS:
        try:
            key, sign = _pair_key(resampled, name)
        except KeyError:
            continue
        block = resampled["paired"][key]
        # The estimate and its interval go on separate rows: four metrics each
        # carrying a bracketed interval do not fit the text width on one line.
        ests, ivals = [], []
        for metric, _pretty in QUALITY:
            v = block[metric]
            mean, lo, hi = sign * v["mean"], sign * v["ci_low"], sign * v["ci_high"]
            if sign < 0:
                lo, hi = hi, lo
            star = r"$^{\ast}$" if (lo > 0) or (hi < 0) else ""
            ests.append(f"{mean:+.3f}{star}")
            ivals.append(f"\\tiny[{lo:+.3f}, {hi:+.3f}]")
        lines.append(f"{label} & " + " & ".join(ests) + r" \\")
        lines.append(" & " + " & ".join(ivals) + r" \\")
        lines.append(r"\addlinespace")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


SCENARIO_LABEL = {
    "baseline": "baseline",
    "edge_unavailable": "remote tier unavailable",
    "high_grid_1p5x": "grid intensity $\\times 1.5$",
    "low_grid_0p5x": "grid intensity $\\times 0.5$",
    "high_network_plus_50ms": "network $+\\SI{50}{\\milli\\second}$",
    "low_network_plus_5ms": "network $+\\SI{5}{\\milli\\second}$",
}


def table_scenarios(csv_path: Path) -> str:
    """Robustness of the headline comparisons to the scenario axes that were varied."""
    import csv as _csv
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(_csv.DictReader(handle))
    wanted = {
        "adaptive_split_minus_local_only": "vs local-only",
        "adaptive_split_minus_edge_only": "vs edge-only",
    }
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Sensitivity of the adaptive-split comparisons to the scenario axes "
        r"available in the harness. Values are mean $\Delta$MCC over seeds. The grid-intensity "
        r"axes do not touch predictive quality by construction and are included to show which "
        r"axes were varied; the remote-tier-unavailable row is the informative one, where "
        r"routing degrades exactly to always-local behaviour. The dominant carbon parameter, "
        r"the embodied amortisation constant, is not among the available axes.}",
        r"\label{tab:scenarios}", r"\small",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Scenario & $\Delta$MCC " + wanted["adaptive_split_minus_local_only"]
        + r" & $\Delta$MCC " + wanted["adaptive_split_minus_edge_only"] + r" \\",
        r"\midrule",
    ]
    by_scenario: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["comparison"] in wanted:
            by_scenario.setdefault(row["scenario"], {})[row["comparison"]] = row["mcc_mean"]
    for scenario in ("baseline", "edge_unavailable", "high_network_plus_50ms",
                     "low_network_plus_5ms", "high_grid_1p5x", "low_grid_0p5x"):
        vals = by_scenario.get(scenario)
        if not vals:
            continue
        a = vals.get("adaptive_split_minus_local_only")
        b = vals.get("adaptive_split_minus_edge_only")
        fmt_v = lambda v: f"{float(v):+.3f}" if v is not None else "--"
        lines.append(f"{SCENARIO_LABEL.get(scenario, scenario)} & {fmt_v(a)} & {fmt_v(b)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_provenance(d: dict) -> str:
    prov = d["measurement_provenance"]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Provenance of each reported cost quantity. Only latency is measured; the "
        r"energy and carbon figures are derived from fixed device power and a grid-intensity "
        r"constant, and are therefore model outputs rather than measurements.}",
        r"\label{tab:provenance}", r"\small",
        r"\begin{tabular}{ll}", r"\toprule", r"Quantity & Provenance \\", r"\midrule",
    ]
    for key, label in (("latency", "Latency"), ("energy", "Energy"),
                       ("operational_carbon", "Operational carbon"),
                       ("embodied_carbon", "Embodied carbon")):
        lines.append(f"{label} & {prov[key].replace('_', ' ')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repeated-eval", required=True)
    p.add_argument("--nested-bp3", required=True)
    p.add_argument("--nested-iedb", required=True)
    p.add_argument("--replication", required=True)
    p.add_argument("--policy-resampling")
    p.add_argument("--scenario-summary")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    d = json.loads(Path(a.repeated_eval).read_text(encoding="utf-8"))
    nested = {"BP3C50ID external test": json.loads(Path(a.nested_bp3).read_text(encoding="utf-8")),
              "IEDB bounded subset": json.loads(Path(a.nested_iedb).read_text(encoding="utf-8"))}
    replication = json.loads(Path(a.replication).read_text(encoding="utf-8"))

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resampled = json.loads(Path(a.policy_resampling).read_text(encoding="utf-8")) if a.policy_resampling else None
    for stem, text in ((("tab_policy_summary", table_policies_antigen(resampled)) if resampled
                        else ("tab_policy_summary", table_policies(d))),
                       (("tab_paired", table_paired_antigen(resampled)) if resampled
                        else ("tab_paired", table_paired(d))),
                       ("tab_fusion", table_fusion(nested, replication)),
                       ("tab_provenance", table_provenance(d))) + (
                           (("tab_scenarios", table_scenarios(Path(a.scenario_summary))),)
                           if a.scenario_summary else ()):
        (out / f"{stem}.tex").write_text(text, encoding="utf-8")
        print("wrote", out / f"{stem}.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
