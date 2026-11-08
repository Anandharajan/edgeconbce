"""Generate every manuscript figure from result artifacts.

One function per figure, each reading a summary file and writing a vector PDF.
Colours come from a fixed categorical order validated for colour-vision
deficiency; marker shape repeats the identity so the figures survive greyscale
printing, and every comparison carries its 95% interval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical order; validated (light surface) for deuteranopia/tritanopia
# separation. Never cycled - a new series takes the next unused slot.
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h")
INK, MUTED, GRID = "#15202b", "#5a6876", "#d9dfe6"

HEURISTIC = ("local_only", "edge_only", "adaptive_split", "adaptive_carbon_guard")
PLM = ("plm_local_only", "plm_edge_only", "plm_region_constrained_edge",
       "plm_adaptive_split", "plm_adaptive_conformal_guard")
LABEL = {
    "local_only": "local-only", "edge_only": "edge-only",
    "adaptive_split": "adaptive-split", "adaptive_carbon_guard": "carbon-guard",
    "plm_local_only": "PLM local-only", "plm_edge_only": "PLM edge-only",
    "plm_region_constrained_edge": "PLM region-constrained", "plm_adaptive_split": "PLM adaptive-split",
    "plm_adaptive_conformal_guard": "PLM conformal-guard",
}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 8.5,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.alpha": 0.9,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "legend.frameon": False, "legend.fontsize": 7.5,
        "lines.linewidth": 1.6, "pdf.fonttype": 42,
    })


def _agg(path: Path) -> dict[str, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {r["policy"]: r for r in d["aggregate_policy_metrics"]}


# Metric names differ between the seed-level artifact and the antigen-level one.
ANTIGEN_ALIAS = {"mean_latency_ms": "latency_ms", "mean_total_gco2e": "total_gco2e",
                 "raw_sequence_share_rate": "raw_sequence_shared"}


def _antigen(path: Path) -> dict[str, dict]:
    """Per-policy antigen-level intervals, flattened to the seed-artifact key names."""
    d = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for policy, metrics in d["per_policy"].items():
        row: dict[str, float] = {}
        for metric, v in metrics.items():
            for name in {metric} | {k for k, a in ANTIGEN_ALIAS.items() if a == metric}:
                row[f"{name}_mean"] = v["mean"]
                row[f"{name}_ci_low"] = v["ci_low"]
                row[f"{name}_ci_high"] = v["ci_high"]
        out[policy] = row
    return out


def _err(row, metric):
    """Asymmetric error-bar lengths from a mean and its 95% interval."""
    mean = row[f"{metric}_mean"]
    lo, hi = row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]
    return mean, [[max(mean - lo, 0.0)], [max(hi - mean, 0.0)]]


def tradeoff(agg, out: Path, xmetric: str, xlabel: str, title: str, logx: bool = False) -> Path:
    """Quality against a cost axis, one panel per policy family.

    Identity is carried by a shared legend plus marker shape rather than by
    per-point labels: with nine policies in two panels the labels collide, and
    two PLM policies occupy the same point exactly.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
    handles: list = []
    for ax, family, fam_name in ((axes[0], HEURISTIC, "Heuristic family"), (axes[1], PLM, "PLM family")):
        for i, pol in enumerate(family):
            row = agg.get(pol)
            if row is None:
                continue
            x, xerr = _err(row, xmetric)
            y, yerr = _err(row, "mcc")
            bar = ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=MARKERS[i], color=PALETTE[i % len(PALETTE)],
                              markersize=6.5, markeredgecolor="white", markeredgewidth=0.9,
                              elinewidth=1.0, capsize=2.5, label=LABEL[pol], zorder=3)
            handles.append((ax, bar))
        ax.set_title(fam_name, color=MUTED, loc="left")
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale("log")
        ax.margins(x=0.16, y=0.20)
        ax.legend(loc="lower right", handletextpad=0.4, labelspacing=0.3, borderaxespad=0.4)
    axes[0].set_ylabel("MCC")
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out)
    plt.close(fig)
    return out


def carbon_panel(agg, out: Path) -> Path:
    """Carbon per policy, ordered as reported; the routing overhead is the point."""
    order = [p for p in HEURISTIC + PLM if p in agg]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    xs = np.arange(len(order))
    for i, pol in enumerate(order):
        mean, err = _err(agg[pol], "mean_total_gco2e")
        fam = 0 if pol in HEURISTIC else 2
        ax.bar(xs[i], mean, width=0.62, color=PALETTE[fam], edgecolor="white", linewidth=1.4, zorder=2)
        ax.errorbar(xs[i], mean, yerr=err, fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=3)
        ax.annotate(f"{mean:.3f}", (xs[i], mean), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=6.8, color=MUTED)
    base = agg["local_only"]["mean_total_gco2e_mean"]
    ax.axhline(base, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
    ax.annotate("local-only reference", (0.42, base), textcoords="offset points",
                xytext=(0, -11), ha="left", fontsize=6.8, color=MUTED)
    ax.set_xticks(xs, [LABEL[p] for p in order], rotation=28, ha="right")
    ax.set_ylabel(r"estimated gCO$_2$e per antigen")
    ax.set_title("Estimated carbon per routing policy (model-based, not metered)",
                 loc="left", color=INK, fontsize=9.5)
    ax.set_ylim(0, max(agg[p]["mean_total_gco2e_ci_high"] for p in order) * 1.18)
    handles = [plt.Line2D([], [], color=PALETTE[0], lw=6, label="heuristic family"),
               plt.Line2D([], [], color=PALETTE[2], lw=6, label="PLM family")]
    ax.legend(handles=handles, loc="upper left", ncols=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fusion_forest(nested_paths: dict[str, Path], tuned: dict[str, dict], out: Path) -> Path:
    """Per-sequence fusion effect with 95% intervals, against the tuned-on-test estimate."""
    metrics = ("roc_auc", "average_precision", "mcc", "f1")
    pretty = {"roc_auc": "ROC-AUC", "average_precision": "avg. precision", "mcc": "MCC", "f1": "F1"}
    fig, axes = plt.subplots(1, len(nested_paths), figsize=(7.0, 2.8), sharex=False)
    if len(nested_paths) == 1:
        axes = [axes]
    for ax, (name, path) in zip(axes, nested_paths.items()):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        ys = np.arange(len(metrics))[::-1]
        for y, metric in zip(ys, metrics):
            s = d["paired_statistics"].get(metric, {})
            if not s.get("n_folds"):
                continue
            ax.plot([s["ci_low"], s["ci_high"]], [y, y], color=PALETTE[0], linewidth=1.8,
                    solid_capstyle="round", zorder=2)
            ax.plot([s["mean_delta"]], [y], MARKERS[0], color=PALETTE[0], markersize=6,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            t = tuned.get(name, {}).get(metric)
            if t is not None:
                ax.plot([t], [y], MARKERS[1], color=PALETTE[1], markersize=6,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.axvline(0.0, color=INK, linewidth=1.0, zorder=1)
        ax.set_yticks(ys, [pretty[m] for m in metrics])
        ax.set_title(name, loc="left", color=MUTED)
        ax.set_xlabel("fusion minus reference")
        ax.margins(x=0.18)
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=PALETTE[0], marker=MARKERS[0], lw=1.8, markeredgecolor="white",
                          label="leave-one-sequence-out (95% CI)"),
               plt.Line2D([], [], color=PALETTE[1], marker=MARKERS[1], lw=0, markeredgecolor="white",
                          label="tuned on the evaluation split")]
    fig.legend(handles=handles, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Fusion effect under honest versus test-derived selection", x=0.01, ha="left",
                 fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(out)
    plt.close(fig)
    return out


def weight_stability(nested_paths: dict[str, Path], out: Path) -> Path:
    """Selected fusion weight per held-out fold, fitted on training folds only."""
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    for i, (name, path) in enumerate(nested_paths.items()):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        w = d["selected_weights"]["per_fold"]
        jitter = (np.random.default_rng(7 + i).random(len(w)) - 0.5) * 0.16
        ax.scatter(w, np.full(len(w), i) + jitter, s=26, color=PALETTE[i], marker=MARKERS[i],
                   edgecolor="white", linewidth=0.7, zorder=3, label=name)
        med = d["selected_weights"]["median"]
        ax.plot([med, med], [i - 0.26, i + 0.26], color=INK, linewidth=1.6, zorder=4)
        ax.annotate(f"median {med:.2f}", (med, i + 0.30), fontsize=7, color=MUTED, ha="center")
    ax.set_yticks(range(len(nested_paths)), list(nested_paths))
    ax.set_xlabel("weight on the reference predictor, chosen on training folds")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.55, len(nested_paths) - 0.35)
    ax.grid(axis="y", visible=False)
    ax.set_title("Both datasets select a strongly reference-dominant fusion weight",
                 loc="left", color=INK, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def per_sequence_strip(nested_paths: dict[str, Path], out: Path, metric: str = "roc_auc") -> Path:
    """Per-antigen fusion effect, so the reader sees the spread behind the mean."""
    fig, axes = plt.subplots(1, len(nested_paths), figsize=(7.0, 2.6), sharey=False)
    if len(nested_paths) == 1:
        axes = [axes]
    for ax, (i, (name, path)) in zip(axes, enumerate(nested_paths.items())):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        deltas = np.array([r[f"delta_{metric}"] for r in d["per_fold"]])
        order = np.argsort(deltas)
        colors = [PALETTE[2] if v > 0 else PALETTE[1] for v in deltas[order]]
        ax.bar(np.arange(len(deltas)), deltas[order], color=colors, width=0.78,
               edgecolor="white", linewidth=0.8, zorder=2)
        ax.axhline(0, color=INK, linewidth=1.0, zorder=3)
        wins = int((deltas > 0).sum())
        ax.set_title(name, loc="left", color=MUTED, fontsize=8.2)
        ax.annotate(f"{wins} of {len(deltas)} antigens improve", (0.5, 0.03), xycoords="axes fraction",
                    ha="center", fontsize=7.2, color=MUTED)
        ax.set_xlabel("antigen, ordered by effect")
        ax.set_xticks([])
        ax.margins(y=0.2)
    axes[0].set_ylabel(f"Δ {metric.replace('_', ' ')}")
    handles = [plt.Line2D([], [], color=PALETTE[2], lw=6, label="fusion better"),
               plt.Line2D([], [], color=PALETTE[1], lw=6, label="reference better")]
    fig.legend(handles=handles, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Per-antigen fusion effect under leave-one-sequence-out selection",
                 x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(out)
    plt.close(fig)
    return out



def selection_bias_panel(curve_path: Path, scale_path: Path, out: Path) -> Path:
    """Left: how selection bias grows as the design shrinks. Right: where the
    carbon ordering crosses as the remote tier grows."""
    style()
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    scale = json.loads(scale_path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    ax = axes[0]
    grids = sorted({r["grid_points"] for r in curve["rows"]})
    for g in grids:
        rows = sorted((r for r in curve["rows"] if r["grid_points"] == g),
                      key=lambda r: r["antigens"])
        if len(rows) < 2:
            continue
        ax.plot([r["antigens"] for r in rows], [r["mean_bias"] for r in rows],
                marker="o", ms=3, lw=1.2, label=f"{g} weights")
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("antigens in the evaluation split")
    ax.set_ylabel(f"selection bias ({curve['select_metric'].upper()})")
    ax.set_title("Bias from selecting on the evaluation split", fontsize=8)
    ax.legend(fontsize=6, frameon=False)

    ax = axes[1]
    rows = scale["checkpoints"]
    x = [r["operational_scale_factor"] for r in rows]
    y = [r["break_even_route_rate"] for r in rows]
    ax.plot(x, y, marker="o", ms=4, lw=1.2, color="#1f77b4")
    ax.axhline(scale["recorded"]["route_rate"], color="#d62728", lw=1.0, ls="--",
               label="policy's routed fraction")
    for r in rows:
        ax.annotate(r["remote_checkpoint"].replace("esm2_", "").replace("_UR50D", ""),
                    (r["operational_scale_factor"], r["break_even_route_rate"]),
                    fontsize=5, xytext=(2, -6), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("remote operational cost (x measured)")
    ax.set_ylabel("break-even routed fraction")
    ax.set_title("Where routing stops costing carbon", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repeated-eval", required=True)
    p.add_argument("--nested-bp3", required=True)
    p.add_argument("--nested-iedb", required=True)
    p.add_argument("--policy-resampling",
                   help="antigen-level intervals; used for the policy figures when given")
    p.add_argument("--published-fusion-iedb")
    p.add_argument("--selection-bias-curve")
    p.add_argument("--carbon-scale-sensitivity")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    style()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    agg = _antigen(Path(a.policy_resampling)) if a.policy_resampling else _agg(Path(a.repeated_eval))
    interval_label = ("95% bootstrap interval over antigens and seeds"
                      if a.policy_resampling else "95% bootstrap interval over seeds")
    nested = {"BP3C50ID external test": Path(a.nested_bp3),
              "IEDB bounded subset": Path(a.nested_iedb)}

    # The published, test-derived fusion estimate, for contrast on the forest plot.
    tuned: dict[str, dict] = {}
    if a.published_fusion_iedb:
        d = json.loads(Path(a.published_fusion_iedb).read_text(encoding="utf-8"))
        tuned["IEDB bounded subset"] = {r["metric"]: r["mean_delta"]
                                        for r in d.get("sequence_bootstrap_delta_vs_baseline", [])}

    written = [
        tradeoff(agg, out / "fig_quality_latency.pdf", "mean_latency_ms",
                 "mean latency per antigen (ms, log scale)",
                 f"Predictive quality against latency ({interval_label})", logx=True),
        tradeoff(agg, out / "fig_quality_datamovement.pdf", "raw_sequence_share_rate",
                 "fraction of antigens whose raw sequence leaves the device",
                 f"Predictive quality against raw-sequence exposure ({interval_label})"),
        carbon_panel(agg, out / "fig_carbon.pdf"),
        fusion_forest(nested, tuned, out / "fig_fusion_forest.pdf"),
        weight_stability(nested, out / "fig_weight_stability.pdf"),
        per_sequence_strip(nested, out / "fig_per_sequence.pdf"),
    ]
    if a.selection_bias_curve and a.carbon_scale_sensitivity:
        written.append(
            selection_bias_panel(
                Path(a.selection_bias_curve),
                Path(a.carbon_scale_sensitivity),
                out / "fig_selection_bias.pdf",
            )
        )
    for path in written:
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
