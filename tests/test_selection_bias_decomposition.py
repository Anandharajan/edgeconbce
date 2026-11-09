"""The headline gap must split into selection plus aggregation, exactly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgeconbce.pipeline import selection_bias_decomposition as sbd

ARTIFACT = Path("outputs/selection_bias_sci/selection_bias_decomposition.json")


@pytest.fixture(scope="module")
def report() -> dict:
    if not ARTIFACT.exists():
        pytest.skip("decomposition artifact not generated")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_components_sum_to_the_headline_gap(report: dict) -> None:
    total = report["selection_component"] + report["aggregation_component"]
    assert total == pytest.approx(report["headline_gap"], abs=1e-12)


def test_headline_gap_is_pooled_minus_nested(report: dict) -> None:
    gap = report["test_derived_pooled"] - report["nested_per_antigen"]
    assert gap == pytest.approx(report["headline_gap"], abs=1e-12)


def test_selection_component_holds_aggregation_fixed(report: dict) -> None:
    """Selection is the per-antigen test-derived figure minus the nested one."""
    sel = report["test_derived_per_antigen"] - report["nested_per_antigen"]
    assert sel == pytest.approx(report["selection_component"], abs=1e-12)


def test_aggregation_is_not_counted_as_selection(report: dict) -> None:
    """Guards the defect this module exists to fix."""
    assert report["aggregation_component"] != pytest.approx(0.0, abs=1e-9)
    assert report["selection_component"] < report["headline_gap"]


def test_full_split_is_deterministic(report: dict) -> None:
    assert report["replicates"] == 1
    assert report["deterministic"] is True


def test_decompose_is_reproducible() -> None:
    """Re-running on the same inputs returns the same numbers."""
    carbon = Path("outputs/carbon_bce_r_colab_bp3/carbon_bce_r_test_predictions.json")
    raw = Path(
        "outputs/external_baselines/bepipred_submission/"
        "bepipred3_results/raw_output.csv"
    )
    if not (carbon.exists() and raw.exists()):
        pytest.skip("source predictions not present")
    inputs = sbd.build_bp3_inputs(str(carbon), str(raw))
    a = sbd.decompose(inputs, metric="mcc", grid_points=21, reference_threshold=0.1512)
    b = sbd.decompose(inputs, metric="mcc", grid_points=21, reference_threshold=0.1512)
    assert a["headline_gap"] == pytest.approx(b["headline_gap"], abs=0.0)
    assert a["selection_component"] == pytest.approx(b["selection_component"], abs=0.0)
