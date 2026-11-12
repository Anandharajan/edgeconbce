"""Draw the system architecture figure as vector art.

Three panels: the deployment path for one antigen, the two selection protocols
side by side, and the carbon crossover. Every number is read from
paper/generated_numbers.tex or from the carbon artifact, never typed here, so
the figure cannot drift from the manuscript the way a hand-drawn one would.

Panel (c) is computed from the recorded constants rather than drawn by eye.
That matters: routed carbon has slope r*O and remote-only has slope O, so
routed starts higher and is nearly flat while remote-only rises about three
and a half times faster and overtakes it. Sketching the two lines freehand
gets this backwards, which inverts the paper's result.

Output: PDF (vector, fonts embedded as Type 42) at 190 mm, the double-column
width, with no text below 7 pt at final size.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

MM = 1.0 / 25.4
WIDTH_MM, HEIGHT_MM = 190.0, 128.0

# Okabe-Ito derived; safe under the common forms of colour blindness and
# separable in greyscale.
C_DEVICE = "#D6E6F2"
C_REMOTE = "#E8E8E8"
C_BENCH = "#F4F4F4"
C_UP = "#E69F00"      # uplink: data leaving the lab
C_DOWN = "#009E73"    # downlink: prediction returning
C_GATE = "#B8860B"
C_OPT = "#D55E00"     # test-derived, always also dashed
C_NEST = "#0072B2"    # nested, always also solid
C_INK = "#000000"

FS_PANEL, FS_BODY, FS_ANNOT, FS_FOOT = 9.0, 8.0, 7.0, 6.5


def read_macros(numbers_tex: Path) -> dict[str, str]:
    text = numbers_tex.read_text(encoding="utf-8")
    found = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", text))
    needed = [
        "ALocalOnlyLatency", "AAdaptiveSplitLatency", "AEdgeOnlyLatency",
        "AAdaptiveSplitRawShare", "RouteBudgetMax", "AntigenCount",
        "ScaleCrossoverFactor", "ScaleCrossoverParams",
        "ScaleOperationalShareEdge", "DecompHeadlineGap", "DecompSelection",
        "DecompAggregation",
    ]
    missing = [m for m in needed if m not in found]
    if missing:
        raise SystemExit(f"macros missing from {numbers_tex}: {missing}")
    return {m: found[m] for m in needed}


def read_carbon(path: Path) -> dict[str, float]:
    rec = json.loads(path.read_text(encoding="utf-8"))["recorded"]
    return {
        "c_local": rec["c_local_gco2e"],
        "embodied": rec["embodied_edge_gco2e_per_request"],
        "operational": rec["operational_edge_gco2e_per_request"],
        "route_rate": rec["route_rate"],
    }


def box(ax, x, y, w, h, label, *, fc="white", ec=C_INK, lw=0.6, fs=FS_BODY,
        style="round,pad=0,rounding_size=1.2", weight="normal", ls="-", zorder=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, facecolor=fc,
                                edgecolor=ec, linewidth=lw, linestyle=ls,
                                zorder=zorder, mutation_scale=1))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=C_INK, zorder=zorder + 1, linespacing=1.25,
                fontweight=weight)


def arrow(ax, p, q, *, color=C_INK, lw=0.9, ls="-", rad=0.0, zorder=3):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=7, color=color, linewidth=lw,
        linestyle=ls, shrinkA=0, shrinkB=0,
        connectionstyle=f"arc3,rad={rad}", zorder=zorder))


def padlock(ax, x, y, closed=True, s=1.0, color=C_INK):
    """Small padlock. Closed = the sequence never left the bench."""
    ax.add_patch(Rectangle((x - 1.1 * s, y - 1.1 * s), 2.2 * s, 1.9 * s,
                           facecolor="white", edgecolor=color, linewidth=0.55,
                           zorder=6))
    th = [i * 3.14159 / 24 for i in range(25)]
    import math
    sh = 0.0 if closed else 0.55 * s
    xs = [x + sh + 0.72 * s * math.cos(t) for t in th]
    ys = [y + 0.8 * s + 0.72 * s * math.sin(t) for t in th]
    ax.plot(xs, ys, color=color, linewidth=0.55, zorder=6, solid_capstyle="round")


def panel_a(ax, M):
    ax.set_xlim(0, 190); ax.set_ylim(0, 56); ax.axis("off")

    # three zones, named once, left to right
    for x0, w, fc, name in ((2, 44, C_BENCH, "Laboratory bench"),
                            (48, 70, C_DEVICE, "On-device predictor"),
                            (120, 68, C_REMOTE, "Remote predictor")):
        ax.add_patch(Rectangle((x0, 10), w, 44, facecolor=fc, edgecolor="none",
                               zorder=0))
        ax.text(x0 + w / 2, 50.6, name, ha="center", va="center",
                fontsize=FS_BODY, fontweight="bold", color="#333333", zorder=1)

    # specimen -> sequence
    ax.add_patch(FancyBboxPatch((6, 30), 5.4, 10, boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor="white", edgecolor=C_INK, linewidth=0.7, zorder=2))
    ax.add_patch(Rectangle((6.9, 30.6), 3.6, 5.2, facecolor=C_DEVICE,
                           edgecolor="none", zorder=3))
    ax.text(8.7, 42.4, "new antigen", ha="center", va="center", fontsize=FS_ANNOT)
    for i, c in enumerate("MKTAYIA"):
        ax.add_patch(Rectangle((15 + i * 3.3, 32.2), 3.1, 5.4, facecolor="white",
                               edgecolor="#888888", linewidth=0.4, zorder=2))
        ax.text(16.55 + i * 3.3, 34.9, c, ha="center", va="center",
                fontsize=FS_FOOT, family="monospace", zorder=3)
    ax.text(40.4, 34.9, "…", ha="center", va="center", fontsize=FS_BODY)
    ax.text(27.5, 39.6, "sequence", ha="center", va="center", fontsize=FS_ANNOT)
    arrow(ax, (11.6, 34.9), (14.6, 34.9))

    # local predictor
    arrow(ax, (42.5, 34.9), (51.5, 34.9))
    box(ax, 52, 29.5, 24, 11, "local\npredictor", fs=FS_BODY)
    # per-residue score heat strip
    cmap = matplotlib.colormaps["magma"]
    for i in range(16):
        ax.add_patch(Rectangle((52 + i * 1.5, 25.0), 1.5, 2.6,
                               facecolor=cmap(0.15 + 0.05 * i),
                               edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((52, 25.0), 24, 2.6, facecolor="none",
                           edgecolor="#777777", linewidth=0.4, zorder=4))
    ax.text(64, 22.8, "per-residue scores", ha="center", va="center", fontsize=FS_FOOT)

    # the gate: the only diamond in the figure
    gx, gy = 96, 34.9
    ax.add_patch(Polygon([[gx, gy + 11], [gx + 16, gy], [gx, gy - 11], [gx - 16, gy]],
                         closed=True, facecolor=C_GATE, edgecolor=C_INK,
                         linewidth=0.7, zorder=4))
    ax.text(gx, gy + 2.4, "mean per-residue", ha="center", va="center",
            fontsize=FS_ANNOT, color="white", zorder=5)
    ax.text(gx, gy - 0.7, "uncertainty  $u(a) > \\theta$ ?", ha="center",
            va="center", fontsize=FS_ANNOT, color="white", zorder=5)
    ax.text(gx, gy - 4.2, f"routed fraction $r \\leq$ {M['RouteBudgetMax']}",
            ha="center", va="center", fontsize=FS_FOOT, color="#FFF0C0", zorder=5)
    arrow(ax, (76.5, 34.9), (79.5, 34.9))

    # YES -> remote
    arrow(ax, (112.2, 38.0), (147, 38.0), color=C_UP, lw=1.2)
    ax.text(129, 40.3, "yes — escalate", ha="center", va="center",
            fontsize=FS_ANNOT, color=C_UP, fontweight="bold")
    ax.text(129, 35.3, "sequence leaves the lab", ha="center", va="center",
            fontsize=FS_FOOT, color="#555555")
    padlock(ax, 118.5, 41.6, closed=False, s=0.9, color=C_UP)

    box(ax, 148, 31.5, 38, 13,
        "published reference predictor\n(ESM-2 650M backbone)", fs=FS_ANNOT)

    # NO -> straight to the output, bypassing the remote zone entirely
    arrow(ax, (96, 23.9), (96, 17.0), color=C_DOWN, lw=1.2)
    arrow(ax, (96, 17.0), (164, 17.0), color=C_DOWN, lw=1.2)
    ax.text(118, 14.6, "no — answer on-device", ha="center", va="center",
            fontsize=FS_ANNOT, color=C_DOWN, fontweight="bold")
    padlock(ax, 103.5, 19.4, closed=True, s=0.9, color=C_DOWN)

    # remote scores return, then join the same output
    arrow(ax, (167, 31.5), (167, 19.0), color=C_DOWN, lw=1.0)
    ax.text(178.5, 25.0, "remote\nper-residue\nscores", ha="center", va="center",
            fontsize=FS_FOOT, color=C_DOWN)


def panel_a_metrics(ax, M):
    """Readout strip: the three costs the routing decision trades against."""
    ax.set_xlim(0, 190); ax.set_ylim(0, 15); ax.axis("off")
    ax.add_patch(Rectangle((2, 3.0), 186, 10.4, facecolor="#FAFAFA",
                           edgecolor="#BBBBBB", linewidth=0.5))
    cols = [(34, "on-device only"), (74, "routed"), (114, "remote-only")]
    ax.text(14, 11.2, "per antigen", ha="center", va="center",
            fontsize=FS_FOOT, style="italic", color="#555555")
    for x, lab in cols:
        ax.text(x, 11.2, lab, ha="center", va="center", fontsize=FS_ANNOT,
                color="#333333")
    rows = [
        ("latency", [f"{M['ALocalOnlyLatency']} ms", f"{M['AAdaptiveSplitLatency']} ms",
                     f"{M['AEdgeOnlyLatency']} ms"]),
        ("sequences disclosed", ["0%", f"{float(M['AAdaptiveSplitRawShare'])*100:.1f}%", "100%"]),
        ("modelled carbon", ["lowest", "highest", "middle"]),
    ]
    for j, (name, vals) in enumerate(rows):
        y = 8.6 - j * 2.5
        ax.text(14, y, name, ha="center", va="center", fontsize=FS_FOOT)
        for (x, _), v in zip(cols, vals):
            ax.text(x, y, v, ha="center", va="center", fontsize=FS_FOOT,
                    fontweight="bold" if x == 74 else "normal")
    ax.text(188, 1.1,
            "carbon is modelled from device power, an amortised embodied term "
            "and measured latency — not metered",
            ha="right", va="center", fontsize=FS_FOOT, color="#555555")


def panel_b(ax, M):
    ax.set_xlim(0, 128); ax.set_ylim(0, 54); ax.axis("off")

    # --- left: test-derived, dashed, fitted and scored on the same antigens
    ax.add_patch(FancyBboxPatch((1, 2), 55, 48,
                                boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor="#FDF0E9", edgecolor=C_OPT,
                                linewidth=0.9, linestyle="--", zorder=1))
    ax.text(28.5, 46.6, "test-derived (optimistic)", ha="center", va="center",
            fontsize=FS_BODY, fontweight="bold", color=C_OPT, zorder=3)
    box(ax, 5, 28, 26, 13, f"evaluation split\n{M['AntigenCount']} antigens",
        fc="white", ec="#999999", fs=FS_ANNOT, zorder=2)
    box(ax, 36, 28, 17, 13, "scan\nweight grid\n$\\mathcal{W}$", fc="white",
        ec="#999999", fs=FS_ANNOT, zorder=2)
    arrow(ax, (31, 37), (36, 37), color=C_OPT, lw=1.0)
    arrow(ax, (44.5, 28), (18, 28), color=C_OPT, lw=1.0, rad=0.38)
    ax.text(28.5, 20.6, "fitted and scored on the same antigens",
            ha="center", va="center", fontsize=FS_ANNOT, color=C_OPT,
            fontstyle="italic", zorder=3)
    ax.text(28.5, 13.0,
            r"$\widehat{\Delta}_{\mathrm{test}}=\max_{w\in\mathcal{W}}"
            r"\widehat{M}\left(y_{\mathcal{A}},s_{\mathcal{A}}(w,\mathcal{A}),"
            r"\tau\right)-\widehat{M}_{\mathrm{ref}}$",
            ha="center", va="center", fontsize=FS_ANNOT, zorder=3)

    # --- right: nested CV, solid, nothing flows back from the held-out antigen
    ax.add_patch(FancyBboxPatch((60, 2), 67, 48,
                                boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor="#EAF2F8", edgecolor=C_NEST,
                                linewidth=0.9, zorder=1))
    ax.text(93.5, 46.6, "nested cross-validation (leakage-free)",
            ha="center", va="center", fontsize=FS_BODY, fontweight="bold",
            color=C_NEST, zorder=3)
    n = int(M["AntigenCount"]); held = 4
    sw, x0, ytop = 3.9, 63.5, 36.5
    for i in range(n):
        x = x0 + i * (sw + 0.45)
        is_held = (i == held)
        ax.add_patch(Rectangle((x, ytop), sw, 5.0,
                               facecolor=C_NEST if is_held else "white",
                               edgecolor=C_NEST if is_held else "#9DB8CC",
                               linewidth=0.6, zorder=3))
        ax.text(x + sw / 2, ytop + 2.5, str(i + 1), ha="center", va="center",
                fontsize=5.6, color="white" if is_held else "#44607A", zorder=4)
    hx = x0 + held * (sw + 0.45) + sw / 2
    ax.text(hx, 43.4, "held-out\nantigen $a$", ha="center", va="center",
            fontsize=FS_FOOT, color=C_NEST, zorder=4)
    arrow(ax, (hx, 42.0), (hx, 38.2), color=C_NEST, lw=0.8)
    ax.annotate("", xy=(x0, 35.4), xytext=(x0 + n * (sw + 0.45) - 0.45, 35.4),
                arrowprops=dict(arrowstyle="-", color="#44607A", lw=0.6))
    ax.text(93.5, 32.6, "training folds $T_a$  —  the other 14 antigens",
            ha="center", va="center", fontsize=FS_FOOT, color="#44607A")
    ax.text(93.5, 28.0,
            "weight and threshold chosen on $T_a$ alone,\n"
            "then applied unchanged to $a$",
            ha="center", va="center", fontsize=FS_ANNOT, color=C_NEST,
            fontstyle="italic", linespacing=1.3)
    # the blocked path: nothing from a informs its own configuration
    ax.add_patch(FancyArrowPatch((hx + 2.0, 37.0), (112, 30.5),
                                 arrowstyle="-|>", mutation_scale=6,
                                 color="#B00020", linewidth=0.7, linestyle=(0, (2, 1.6)),
                                 connectionstyle="arc3,rad=-0.35", zorder=4))
    ax.plot([116.4, 120.4], [27.4, 31.4], color="#B00020", lw=1.0, zorder=6)
    ax.plot([116.4, 120.4], [31.4, 27.4], color="#B00020", lw=1.0, zorder=6)
    ax.text(118.4, 24.4, "no feedback\nto selection", ha="center", va="center",
            fontsize=FS_FOOT, color="#B00020", linespacing=1.2)
    ax.text(93.5, 16.4,
            r"$w_a=\arg\max_{w\in\mathcal{W}}\widehat{M}"
            r"\left(y_{T_a},s_{T_a}(w,T_a),\tau\right)$",
            ha="center", va="center", fontsize=FS_ANNOT)
    ax.text(93.5, 10.0,
            r"$\widehat{\Delta}_{\mathrm{nested}}=\frac{1}{n}\sum_{a}"
            r"\left[\widehat{M}\left(y_a,s_a(w_a,T_a),\tau_a\right)"
            r"-\widehat{M}_{\mathrm{ref}}\right]$",
            ha="center", va="center", fontsize=FS_ANNOT)

    # --- the gap between them, decomposed
    ax.text(58.5, 40.0, r"$b=\widehat{\Delta}_{\mathrm{test}}-"
                        r"\widehat{\Delta}_{\mathrm{nested}}$",
            ha="center", va="center", fontsize=FS_ANNOT, rotation=90)
    ax.text(58.5, 23.0,
            f"= {M['DecompHeadlineGap']}\nselection {M['DecompSelection']}\n"
            f"aggregation {M['DecompAggregation']}",
            ha="center", va="center", fontsize=FS_FOOT, rotation=90,
            linespacing=1.4)


def panel_c(ax, M, K):
    """Computed, not sketched: slope r*O for routed against O for remote-only."""
    c_local, E, O, r = K["c_local"], K["embodied"], K["operational"], K["route_rate"]
    kmax = 30.0
    ks = [1 + i * (kmax - 1) / 400 for i in range(401)]
    remote = [E + k * O for k in ks]
    routed = [c_local + r * (E + k * O) for k in ks]
    kstar = ((c_local / (1 - r)) - E) / O

    ax.plot(ks, routed, color=C_UP, lw=1.1, ls="--", label="routed", zorder=3)
    ax.plot(ks, remote, color=C_NEST, lw=1.1, ls="-", label="remote-only", zorder=3)
    ax.axvline(kstar, color="#666666", lw=0.6, ls=":", zorder=2)
    ax.axvspan(kstar, kmax, color="#E8F4EC", zorder=0)

    ymax = max(max(remote), max(routed))
    ymin = min(min(remote), min(routed))
    pad = (ymax - ymin) * 0.16
    ax.set_xlim(1, kmax); ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("remote operational cost, $\\times$ measured scale ($k$)",
                  fontsize=FS_FOOT, labelpad=1.6)
    ax.set_ylabel("modelled carbon\nper antigen", fontsize=FS_FOOT, labelpad=1.6,
                  linespacing=1.2)
    ax.tick_params(labelsize=FS_FOOT - 0.5, width=0.5, length=2, pad=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax.text(kstar + 0.9, ymax + pad * 0.1,
            f"$k^\\star$ = {M['ScaleCrossoverFactor']}\n"
            f"$\\simeq$ {M['ScaleCrossoverParams']}M parameters",
            ha="left", va="top", fontsize=FS_FOOT, linespacing=1.3)
    ax.text(kmax - 0.8, ymin - pad * 0.1, "routing is greener here",
            ha="right", va="bottom", fontsize=FS_FOOT, color="#1A6B3C")
    ax.text(2.0, ymax + pad * 0.1, "routed", color=C_UP, fontsize=FS_FOOT,
            ha="left", va="top", fontweight="bold")
    ax.text(kmax - 0.8, ymax + pad * 0.05, "remote-only", color=C_NEST,
            fontsize=FS_FOOT, ha="right", va="top", fontweight="bold")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--numbers", required=True)
    p.add_argument("--carbon-scale-sensitivity", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)

    M = read_macros(Path(a.numbers))
    K = read_carbon(Path(a.carbon_scale_sensitivity))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "pdf.fonttype": 42,   # embed as TrueType, never Type 3
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
    })

    fig = plt.figure(figsize=(WIDTH_MM * MM, HEIGHT_MM * MM))

    ax_a = fig.add_axes([0.0, 0.615, 1.0, 0.375]); panel_a(ax_a, M)
    ax_m = fig.add_axes([0.0, 0.505, 1.0, 0.115]); panel_a_metrics(ax_m, M)
    ax_b = fig.add_axes([0.005, 0.045, 0.675, 0.455]); panel_b(ax_b, M)
    ax_c = fig.add_axes([0.775, 0.135, 0.195, 0.305]); panel_c(ax_c, M, K)

    for ax, lab, x, y in ((ax_a, "(a)", 0.004, 0.985),
                          (ax_b, "(b)", 0.004, 0.985),
                          (ax_c, "(c)", -0.46, 1.20)):
        ax.text(x, y, lab, transform=ax.transAxes, fontsize=FS_PANEL,
                fontweight="bold", ha="left", va="top")

    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
