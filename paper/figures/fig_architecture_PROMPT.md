# Generation brief — system architecture figure (EdgeConBCE)

A prompt pack for Google Gemini. It specifies one publication-grade architecture
figure for a computational-biology manuscript.

**Read this first.** Two modes follow. Use **Mode B** for the figure you submit.

| | Mode A — image generation | Mode B — code generation |
|---|---|---|
| Ask Gemini for | a raster concept image | SVG or TikZ source |
| Output | PNG/JPEG, fixed resolution | vector, infinitely scalable |
| Text | frequently garbled, invented glyphs | exact, because you typed it |
| Line weights | uncontrolled | exact, in points |
| Meets Elsevier 1000 dpi line-art / embedded-font rules | **No** | **Yes** |
| Use it for | agreeing the layout with co-authors | the submitted figure |

Image generators cannot currently be trusted with the dense small text, Greek
symbols and subscripted notation this figure needs; they also invent plausible
extra boxes. Mode A is for deciding what the figure should look like. Mode B is
how you get a figure a journal will accept. Both prompts below describe the same
diagram, so Mode A's output is a faithful preview of Mode B's.

---

## 0. Specification the figure must satisfy

Collected from publisher artwork rules. Check your target journal before
submitting; these are the common values.

| Property | Requirement |
|---|---|
| Width | 190 mm (double column). Single column is 90 mm — too narrow for this figure |
| Max height | 230 mm; aim for 110–130 mm so it sits in one column-break |
| Format | vector **EPS** or **PDF**; if raster is unavoidable, TIFF at **1000 dpi** for line art |
| Fonts | Arial or Helvetica only, **embedded**; no font below **7 pt** at final printed size |
| Line weights | **0.25–1.5 pt**. Hairlines below 0.25 pt drop out in print |
| Colour | RGB. Must stay readable converted to **greyscale** |
| Palette | colourblind-safe; never encode meaning in red-vs-green alone |
| Content | no data outside the illustration area; tight crop; no drop shadows, no 3-D, no gradients, no clip-art bevels |

Sources: Elsevier artwork instructions (EPS line art 1000 dpi, line weights
0.1–1.5 pt, Arial/Helvetica/Times, fonts embedded); common Elsevier column
widths 90 mm / 190 mm, minimum 6 pt.

---

## 1. Visual grammar to follow

Modelled on Figure 1 of Matsubara, Levorato & Restuccia, *Split Computing and
Early Exiting for Deep Learning Applications* (ACM Comput. Surv., 2022) — a
figure in this exact problem domain that a reader understands at a glance.
Reuse its five devices:

1. **Persistent vertical zones.** The canvas is divided into columns that mean
   the same thing everywhere in the figure. Each zone is named **once**, by a
   glyph plus a label in a strip along the bottom. The reader learns the
   geography once and then reads any row.
2. **Size encodes capacity.** The remote compute block is drawn visibly larger
   than the on-device block. A capability that is unused in a given row is drawn
   as an **empty** block, not omitted — absence is information.
3. **Paired directional arrows.** Uplink and downlink are two separate
   horizontal arrows, one above the other, in different colours, each labelled
   with what it carries.
4. **Show the payload, don't just name it.** Each arrow carries a small
   thumbnail of the actual thing in transit. This is what makes the figure
   legible without the caption.
5. **A concrete specimen.** The input is a real, specific object, not an
   abstract "data" cylinder.

---

## 2. What the figure depicts — the real-world scene

**Do not label this as a generic ML pipeline.** It is a scene in a laboratory,
and a reader who has never seen the title should still follow it.

> An immunology lab has expressed a new viral surface protein. They need to know
> which residues on it antibodies will actually bind, so they can shortlist
> peptides for a vaccine candidate. They have a small predictor on the bench
> laptop and access to a much larger published predictor in the cloud. Sending
> the sequence off-site costs time, energy, and disclosure of an unpublished
> construct. So the laptop tries first, and only escalates the antigens it is
> unsure about.

That tension — **answer locally, or pay to ask the big model** — is the subject
of the figure. Everything else serves it.

### Glyph vocabulary (use these, consistently)

| Concept | Glyph | Notes |
|---|---|---|
| New antigen specimen | capped sample vial, or ribbon-diagram protein fold | the concrete input |
| Amino-acid sequence | short horizontal strip of letter tiles `M K T A Y I A …` | monospace, 6–8 tiles, then ellipsis |
| Bench laptop (on-device) | laptop outline | small compute block beside it |
| Remote model (cloud) | server rack, or cloud outline | larger compute block beside it |
| Network hop | two horizontal arrows | uplink and downlink |
| Uncertainty gate | **diamond** | the only diamond in the figure |
| Epitope map (output) | protein fold with a patch of residues highlighted | the answer the lab wanted |
| Shortlisted peptides | 2–3 small peptide bars | the actionable deliverable |
| Latency | stopwatch | meter strip |
| Carbon | leaf, or CO₂ cloud | meter strip |
| Disclosure | padlock | meter strip; **closed** = sequence stayed on-site |

Draw glyphs as **flat monochrome line icons**, 0.5 pt stroke. No skeuomorphism,
no shading, no 3-D.

### Colour roles (colourblind-safe, greyscale-separable)

| Role | Colour | Hex |
|---|---|---|
| On-device zone | pale blue | `#D6E6F2` |
| Remote zone | pale grey | `#E8E8E8` |
| Uplink (data leaving the lab) | orange | `#E69F00` |
| Downlink (prediction returning) | teal | `#009E73` |
| The gate and its threshold | dark amber | `#B8860B` |
| Leakage-free / corrected path | blue | `#0072B2` |
| Optimistic / test-derived path | vermilion | `#D55E00` |

Blue vs vermilion for the two estimators is the Okabe–Ito safe pair. Never rely
on hue alone: the optimistic path is also **dashed**, the corrected path
**solid**.

---

## 3. Panel layout

Two panels stacked, plus one inset. Total 190 mm wide.

```
┌─ (a) ──────────────────────────────────────────────────────────────┐
│  DEPLOYMENT: what happens to one antigen                           │
│  three zones, left to right, with the gate in the middle           │
├─ (b) ──────────────────────────────────────────────────────────────┤
│  EVALUATION: how the fusion weight is chosen, two protocols        │
│  side by side, same data, different access                 ┌─────┐ │
│                                                            │ (c) │ │
│                                                            └─────┘ │
└────────────────────────────────────────────────────────────────────┘
```

### Panel (a) — Deployment path

Three vertical zones, named by glyph + label in a bottom strip:
**Laboratory bench** · **On-device predictor** · **Remote predictor**

Flow, left to right:

1. **Vial glyph** → labelled `new antigen`.
2. → **sequence strip** of letter tiles.
3. → enters the pale-blue on-device zone: **laptop glyph** + a *small* compute
   block labelled `local predictor`.
4. → produces per-residue scores **and** an uncertainty value. Draw the scores
   as a short horizontal heat-strip (light→dark), not a chart.
5. → **diamond gate**, annotated:
   `mean per-residue uncertainty u(a) > θ ?`
   with a second line in smaller type: `subject to routed fraction r ≤ 0.50`
6. **Gate NO branch** (down-left, solid teal, labelled `no — answer on-device`)
   → straight to the output, carrying a **closed padlock**.
7. **Gate YES branch** (right, solid orange, labelled `yes — escalate`) → crosses
   into the remote zone over the **uplink arrow**, which carries a thumbnail of
   the sequence strip and the label `sequence leaves the lab`. Put an **open
   padlock** on this arrow.
8. Remote zone: **server glyph** + a *visibly larger* compute block labelled
   `published reference predictor (ESM-2 650M backbone)`.
9. **Downlink arrow** (teal, returning left) labelled `remote per-residue scores`.
10. Both branches converge on → **epitope map** (protein fold, highlighted
    residue patch) → **shortlisted peptides**.

**Meter strip** along the bottom of panel (a), three small readouts with glyphs.
Show the routed policy against the two extremes, so the trade-off is visible:

| | on-device only | routed | remote only |
|---|---|---|---|
| ⏱ latency per antigen | 0.4 ms | **13.6 ms** | 42.4 ms |
| 🔒 sequences disclosed | 0% | **28.5%** | 100% |
| 🍃 modelled carbon | lowest | highest | middle |

Set the three routed values in **bold**. Add a one-line footnote inside the
panel, 6–7 pt: `carbon is modelled from device power, an amortised embodied
term and measured latency — not metered`.

> That carbon row is the figure's most interesting fact and must not be
> smoothed over: routing is the *most* carbon-intensive of the three at the
> scale measured. Panel (c) is where that resolves.

### Panel (b) — Why the evaluation protocol matters

Two sub-blocks side by side, same inputs, different data access. This is the
paper's actual contribution and the panel a reviewer will study.

**Left, vermilion, dashed border — `test-derived (optimistic)`**
A single box containing all 15 antigens, labelled `evaluation split`. One arrow
loops from the box, through a `scan weight grid 𝒲` element, and back into the
same box. **Label that loop `fitted and scored on the same antigens`.** The loop
returning to its own source is the whole point — make it unmistakable.
Annotate with:

```
Δ̂_test = max  M̂( y_A , s_A(w, A) , τ(s_A(w,A), ρ_A) ) − M̂(reference)
         w∈𝒲
```

**Right, blue, solid border — `nested cross-validation (leakage-free)`**
A row of 15 small antigen squares. One square is pulled out and marked
`held-out antigen a`; the remaining 14 are bracketed as `training folds T_a`.
Arrows show the weight and threshold being chosen **only** from `T_a`, then
applied, unchanged, to `a`. Annotate:

```
w_a = argmax M̂( y_{T_a} , s_{T_a}(w, T_a) , τ(·, ρ_{T_a}) )
      w∈𝒲
Δ̂_nested = (1/n) Σ_a [ M̂(y_a , s_a(w_a,T_a) , τ_a) − M̂(reference) ]
```

Add a small padlock-with-slash icon on the arrow from `a` back into the
selection step, crossed out, to show nothing flows that way.

**Between the two blocks**, a vertical brace spanning both, labelled:

```
b = Δ̂_test − Δ̂_nested        selection: 0.0388   aggregation: 0.0056
```

and below it, in 7 pt:
`E[ max_w M̂(w) ] ≥ E[ M̂(w₀) ] = M(w₀)   — the scan reports a maximum, so it exceeds any pre-committed w₀`

### Panel (c) — Carbon crossover inset

Small plot, ~45 × 35 mm, bottom-right of panel (b).

- x-axis: `remote operational cost, ×  measured scale (k)`, log or linear, 1 to 30
- y-axis: `modelled carbon per antigen`
- Two lines: `routed` and `remote-only`. They **cross** at `k* = 13.0`.
- Mark the crossing with a vertical dotted rule and label:
  `k* = 13.0  ≈ 104M parameters`
- Shade right of the crossing very lightly and label `routing is greener here`
- Annotate with the closed form:

```
C_edge(k) = E + kO ,    k* = (1/O)( C_local/(1−r) − E )
```

- One 6 pt note: `embodied term held fixed; operational charge is 1.7% of the remote total`

---

## 4. Typography and finish

- All body labels **8 pt Arial**; sub-annotations and equations **7 pt**; the
  in-panel footnotes **6 pt**. Nothing smaller.
- Panel letters **(a) (b) (c)**, bold, 9 pt, top-left of each panel, outside the
  plotting area.
- Equations in the figure: plain upright Arial with italic variables. Do not set
  them in a serif maths face; it will clash with the labels.
- Arrowheads: filled triangles, 2 pt wide. Every arrow must visibly touch both
  the element it leaves and the element it enters.
- Leave ≥ 3 mm clear inside every box between text and border.
- No title inside the figure. The caption carries the title.

---

## 5. Mode A prompt — concept image

Paste everything between the rules into Gemini, with an image-capable model.

---

Create a single scientific figure for a computational-biology journal, 190 mm
wide, flat 2-D vector style, white background, Arial labels, no 3-D, no
shadows, no gradients, no decorative elements.

Depict a laboratory workflow for predicting which residues of a newly expressed
viral antigen antibodies will bind, where a small predictor on a bench laptop
tries first and escalates only uncertain cases to a large published model in
the cloud.

Panel (a), top: left to right — a sample vial labelled "new antigen"; an
amino-acid sequence strip of letter tiles; a pale-blue zone containing a laptop
icon and a small box "local predictor"; a per-residue score heat-strip; a
diamond decision node reading "mean per-residue uncertainty u(a) > θ ?". From
the diamond, a teal "no — answer on-device" arrow goes directly to the output
carrying a closed-padlock icon; an orange "yes — escalate" arrow crosses right
into a pale-grey zone containing a server-rack icon and a visibly LARGER box
"published reference predictor", with a closed padlock becoming an OPEN padlock
on the crossing arrow labelled "sequence leaves the lab". A teal return arrow
brings "remote per-residue scores" back. Both paths converge on a protein
ribbon diagram with one highlighted surface patch, labelled "epitope map", then
three small peptide bars labelled "shortlisted peptides". Along the bottom of
panel (a), a three-column readout with a stopwatch, a padlock and a leaf icon
comparing on-device-only, routed, and remote-only.

Panel (b), bottom: two blocks side by side. Left, dashed vermilion border,
titled "test-derived (optimistic)": one box of 15 antigens with an arrow
looping out through "scan weight grid" and back into the SAME box, labelled
"fitted and scored on the same antigens". Right, solid blue border, titled
"nested cross-validation (leakage-free)": a row of 15 small squares with one
pulled out as "held-out antigen", the other 14 bracketed as "training folds",
and arrows showing the weight chosen only from the training folds then applied
to the held-out one. A vertical brace between the two blocks labelled "b =
Δtest − Δnested".

Bottom-right, a small inset line plot with two lines crossing at a point marked
"k* = 13.0", x-axis "remote operational cost (× measured)", y-axis "modelled
carbon per antigen".

Label panels (a), (b), (c) in bold at their top-left. Keep every word legible
and correctly spelled. Use only: pale blue #D6E6F2, pale grey #E8E8E8, orange
#E69F00, teal #009E73, blue #0072B2, vermilion #D55E00, black text.

---

**Expect to iterate.** Common failures and the correction to send back:

| Failure | Correction |
|---|---|
| Text garbled or misspelled | "Redraw with far fewer words. Keep only the panel letters and these exact labels: …" |
| Invented extra boxes | "Remove every element I did not list. The figure has exactly N boxes." |
| 3-D / glossy / drop shadows | "Flat 2-D line art only, uniform stroke weight, no shading of any kind." |
| Arrows not touching boxes | "Every arrow must start on a box edge and end on a box edge." |
| Both estimator blocks look alike | "Left block dashed vermilion, right block solid blue. The left one's arrow must return into its own source box." |

---

## 6. Mode B prompt — publication asset

This is the one that produces a submittable figure. Gemini writes code; you
compile it; the text is exact because you supplied it.

---

Write complete, self-contained **SVG** for a scientific figure, 190 mm wide and
about 120 mm tall, using `viewBox="0 0 190 120"` so units are millimetres.

Hard requirements:
- Arial only. Body text `font-size="2.8"` (≈8 pt), annotations `2.5`, footnotes
  `2.1`. Nothing smaller.
- All strokes between `0.09` and `0.53` (0.25–1.5 pt).
- `fill="none"` plus explicit stroke on every shape; no gradients, no filters,
  no `<image>`, no external references, no embedded raster.
- Define one arrowhead `<marker>` and reuse it; filled triangle about 0.7 wide.
- Every text element positioned with explicit x/y and `text-anchor`; never rely
  on default flow.
- Palette, exactly: `#D6E6F2`, `#E8E8E8`, `#E69F00`, `#009E73`, `#0072B2`,
  `#D55E00`, `#000000`.
- Group each panel in `<g id="panel-a">`, `<g id="panel-b">`, `<g id="panel-c">`
  so I can edit them separately.

Then build the three panels exactly as specified in sections 3 and 4 of the
brief I am pasting below. Reproduce every label string verbatim, including the
equations — set them as plain text with `<tspan baseline-shift="sub">` for
subscripts, do not attempt MathML.

[paste sections 2, 3 and 4 of this file here]

Return only the SVG, with no commentary.

---

### Converting to a submittable file

```bash
# SVG → PDF (vector, fonts embedded as outlines)
inkscape fig_architecture.svg --export-type=pdf \
         --export-filename=fig_architecture.pdf --export-text-to-path

# PDF → EPS, if the journal insists on EPS
pdftops -eps fig_architecture.pdf fig_architecture.eps

# Only if a raster is demanded — 1000 dpi line art
inkscape fig_architecture.svg --export-type=png --export-dpi=1000 \
         --export-filename=fig_architecture.png
```

### Acceptance checklist

Print the figure at 100% on A4 and check by eye:

- [ ] Every word legible at printed size, nothing under 7 pt
- [ ] Photocopy it in black and white — are the two estimator blocks still
      distinguishable? (dashed vs solid, not colour alone)
- [ ] Every arrow touches both endpoints
- [ ] The diamond is the only diamond
- [ ] The remote compute block is visibly larger than the local one
- [ ] The optimistic block's arrow visibly returns into its own source box
- [ ] Every number matches the manuscript: 0.4 / 13.6 / 42.4 ms, 28.5%,
      k\* = 13.0, 104M, 0.0388 / 0.0056
- [ ] No title inside the figure
- [ ] Nothing in the figure contradicts the caption

---

## 7. Draft caption

> **Routing antigens between a bench predictor and a published remote model,
> and the protocol used to evaluate it.** **(a)** For each newly expressed
> antigen the on-device predictor runs first; antigens whose mean per-residue
> uncertainty exceeds a fitted threshold are escalated to the remote model,
> subject to a routed-fraction budget. Escalation is what discloses the
> sequence, so the padlock opens only on the uplink. The readout compares the
> routed policy against always-local and always-remote execution on latency,
> disclosure and modelled carbon; carbon is modelled from device power, an
> amortised embodied term and measured latency, and is not metered.
> **(b)** The mixing weight and decision threshold can be chosen two ways. The
> test-derived protocol scans the weight grid on the same antigens it reports,
> so its arrow returns into its own source. Nested cross-validation chooses
> both on the training folds alone and applies them unchanged to the held-out
> antigen. The gap between the two estimates is 0.0444 MCC, of which 0.0388 is
> selection and 0.0056 is the difference between pooling residues and averaging
> over antigens. **(c)** Modelled carbon favours always-remote execution only
> below a crossover in remote operational cost; above k\* ≈ 13.0, corresponding
> to roughly 104M remote parameters, routing becomes the less carbon-intensive
> policy.

---

## 8. Honest notes

- **Mode A output is not submittable.** It is a raster, its text will not
  survive scrutiny at 7 pt, and generators invent details. Use it to agree the
  layout, then generate the real asset with Mode B.
- **Check every number against `generated_numbers.tex` before submitting.** The
  manuscript's discipline is that no number is typed by hand; a figure is the
  easiest place for that discipline to break, because the figure is not rebuilt
  by `regenerate.sh`. Every number in this brief traces to a macro, and all
  eleven were verified against `generated_numbers.tex` when it was written:

  | In the figure | Macro | Value |
  |---|---|---|
  | 0.4 ms | `\ALocalOnlyLatency` | 0.4 |
  | 13.6 ms | `\AAdaptiveSplitLatency` | 13.6 |
  | 42.4 ms | `\AEdgeOnlyLatency` | 42.4 |
  | 28.5% disclosed | `\AAdaptiveSplitRawShare` | 0.285 (a fraction — render as %) |
  | r ≤ 0.50 | `\RouteBudgetMax` | 0.50 |
  | 15 antigens | `\AntigenCount` | 15 |
  | k\* = 13.0 | `\ScaleCrossoverFactor` | 13.0 |
  | ≈104M parameters | `\ScaleCrossoverParams` | 104 |
  | 1.7% operational share | `\ScaleOperationalShareEdge` | 1.7 |
  | 0.0444 headline gap | `\DecompHeadlineGap` | 0.0444 |
  | 0.0388 selection | `\DecompSelection` | 0.0388 |
  | 0.0056 aggregation | `\DecompAggregation` | 0.0056 |

  To re-verify after any pipeline re-run:

  ```bash
  cd paper && grep -oE 'newcommand\{\\(ALocalOnlyLatency|AAdaptiveSplitLatency|\
AEdgeOnlyLatency|AAdaptiveSplitRawShare|RouteBudgetMax|AntigenCount|\
ScaleCrossoverFactor|ScaleCrossoverParams|ScaleOperationalShareEdge|\
DecompHeadlineGap|DecompSelection|DecompAggregation)\}\{[^}]*\}' \
    generated_numbers.tex
  ```
- **Panel (a)'s carbon row is counter-intuitive on purpose.** Routing is the
  most carbon-intensive of the three policies at the scale measured, and panel
  (c) is what resolves it. Do not let a cleaner-looking figure quietly reverse
  that.
