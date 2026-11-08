"""Count how published combined B-cell epitope predictors select their
combination parameter.

The paper's own fusion result is one instance of selecting a mixing weight on
the data the result is reported on. On its own that is an anecdote about our
own unpublished analysis. This module turns the accompanying survey into
counts, so the manuscript can say how common the practice is in the published
literature rather than only that it happened to us.

The survey itself lives in ``data/selection_practice_survey.md`` and stays the
single source of truth: it carries, for every paper, the verbatim sentence the
classification rests on. This module only parses it. Nothing here decides a
classification, and no count in the manuscript is typed by hand.

Three categories are kept apart on purpose:

``leakage evident``
    The text states the combination parameter was chosen by inspecting the
    data the headline result is reported on.
``leakage-free evident``
    The text states a separate basis: a validation split, the training set,
    nested selection, or a rule fixed a priori.
``unreported / cannot tell``
    The paper does not say. This is an observation about the text, not an
    accusation, and it is never folded into the leakage count.

A paper's row may discuss more than one parameter and mention several of these
phrases in passing. The classification is the **bolded** label that opens the
classification cell, which is why this parser reads that and not the rest of
the sentence.
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CATEGORIES = ("leakage evident", "leakage-free evident", "unreported / cannot tell")

#: The classification cell opens with its label in bold.
_LEADING_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CLASSIFICATION_COLUMN = 6
_MIN_COLUMNS = 8


def parse_rows(markdown: str) -> list[dict[str, str]]:
    """Every numbered row of the assessed-papers table, with its label."""
    rows: list[dict[str, str]] = []
    for line in markdown.splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) < _MIN_COLUMNS or not cells[0].isdigit():
            continue
        cell = cells[_CLASSIFICATION_COLUMN]
        match = _LEADING_BOLD.match(cell)
        if match is None:
            raise ValueError(
                f"row {cells[0]}: classification cell does not open with a "
                f"bold label, so it cannot be counted: {cell[:80]!r}"
            )
        label = match.group(1).strip().lower()
        if label not in CATEGORIES:
            raise ValueError(
                f"row {cells[0]}: unknown classification {label!r}; "
                f"expected one of {CATEGORIES}"
            )
        rows.append({"row": cells[0], "citation": cells[1], "classification": label})
    return rows


def build_report(markdown: str) -> dict[str, Any]:
    rows = parse_rows(markdown)
    counts = {c: sum(1 for r in rows if r["classification"] == c) for c in CATEGORIES}
    assessed = len(rows)
    evident = counts["leakage evident"] + counts["leakage-free evident"]
    return {
        "project": "EdgeConBCE",
        "analysis": "selection_practice_survey",
        "question": (
            "Among published B-cell epitope predictors that combine two or "
            "more models, how often is the combination parameter selected on "
            "the same data the headline result is reported on?"
        ),
        "source": "data/selection_practice_survey.md",
        "assessed_papers": assessed,
        "counts": counts,
        "papers_stating_a_basis": evident,
        "leakage_share_of_those_stating_a_basis": (
            counts["leakage evident"] / evident if evident else None
        ),
        "caveat": (
            "Open-access, keyword-reachable sample of "
            f"{assessed} papers. Not a prevalence estimate for the subfield: "
            "paywalled methods papers are structurally absent, the queries are "
            "keyword-dependent, and the sample skews recent and prominent. "
            "Papers that do not state a basis are reported separately and are "
            "never counted as leakage."
        ),
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    report = build_report(Path(args.survey).read_text(encoding="utf-8"))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {out}: {report['assessed_papers']} assessed, "
        f"{report['counts']['leakage evident']} leakage evident"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
