"""The survey counts must come from the survey, and must not blur categories."""
from __future__ import annotations

from pathlib import Path

import pytest

from edgeconbce.pipeline.selection_practice_survey import build_report, parse_rows

SURVEY = Path(__file__).resolve().parents[1] / "data" / "selection_practice_survey.md"

HEADER = (
    "| # | Citation | Combined | Parameter | Selection | Evaluation | "
    "Classification | Evidence |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def _row(n: int, classification: str) -> str:
    return f"| {n} | Cite | X | w | train | test | {classification} | \"q\" |\n"


def test_classification_is_the_bold_label_not_the_prose() -> None:
    """A row may mention another category in its caveat; that must not count.

    This is the real shape of row 2 of the survey: classified unreported, but
    the same cell goes on to say the threshold itself was leakage-free.
    """
    md = HEADER + _row(
        1, "**Unreported / cannot tell** (coefficients); threshold itself leakage-free"
    )
    counts = build_report(md)["counts"]
    assert counts["unreported / cannot tell"] == 1
    assert counts["leakage-free evident"] == 0


def test_unknown_label_is_refused_rather_than_silently_dropped() -> None:
    with pytest.raises(ValueError, match="unknown classification"):
        parse_rows(HEADER + _row(1, "**Probably fine**"))


def test_unbolded_classification_is_refused() -> None:
    with pytest.raises(ValueError, match="does not open with a bold label"):
        parse_rows(HEADER + _row(1, "Leakage evident"))


def test_unreported_is_excluded_from_the_leakage_share() -> None:
    md = HEADER + _row(1, "**Leakage evident**") + _row(
        2, "**Leakage-free evident**"
    ) + _row(3, "**Unreported / cannot tell**")
    report = build_report(md)
    assert report["assessed_papers"] == 3
    assert report["papers_stating_a_basis"] == 2
    assert report["leakage_share_of_those_stating_a_basis"] == pytest.approx(0.5)


def test_real_survey_parses_and_totals_agree() -> None:
    report = build_report(SURVEY.read_text(encoding="utf-8"))
    assert report["assessed_papers"] == sum(report["counts"].values())
    assert report["assessed_papers"] >= 10
    assert report["counts"]["leakage evident"] >= 1
