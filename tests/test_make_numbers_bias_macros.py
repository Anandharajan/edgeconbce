"""The bias-curve macros must not quote an unreplicated cell as an estimate.

At the full antigen count there is only one possible subset, so that cell is a
single deterministic run. Reading the grid-density comparison there would put
an n = 1 figure into the manuscript as though it were a mean.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mk", ROOT / "tools" / "make_numbers.py")
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)


def _curve(tmp_path: Path) -> Path:
    """Two sizes: one resampled, one deterministic with an implausible value."""
    rows = []
    for grid, bias in ((3, 0.01), (41, 0.02)):
        rows.append({
            "antigens": 12, "grid_points": grid, "mean_bias": bias,
            "share_bias_positive": 0.9, "mean_predictor_agreement": 0.6,
            "replicates_requested": 100, "replicates_used": 100,
        })
    for grid, bias in ((3, 0.0), (41, 0.0)):
        rows.append({
            "antigens": 15, "grid_points": grid, "mean_bias": bias,
            "share_bias_positive": 0.0, "mean_predictor_agreement": 0.6,
            "replicates_requested": 100, "replicates_used": 1,
        })
    p = tmp_path / "curve.json"
    p.write_text(json.dumps({"select_metric": "mcc", "rows": rows}), encoding="utf-8")
    return p


def test_grid_macros_come_from_a_replicated_cell(tmp_path: Path) -> None:
    m = mk.MacroSet()
    mk.add_selection_bias_curve(m, _curve(tmp_path))
    out = dict(m.items()) if hasattr(m, "items") else m._macros  # noqa: SLF001
    assert out["BiasGridRefN"] == "12", "must not read the deterministic cell"
    assert out["BiasCoarseGridBias"] == "0.0100"
    assert out["BiasFineGridBias"] == "0.0200"


def test_denser_grid_is_reported_as_the_costlier_one(tmp_path: Path) -> None:
    m = mk.MacroSet()
    mk.add_selection_bias_curve(m, _curve(tmp_path))
    out = dict(m.items()) if hasattr(m, "items") else m._macros  # noqa: SLF001
    assert int(out["BiasFineGridPoints"]) > int(out["BiasCoarseGridPoints"])


def test_real_curve_grid_cell_is_replicated() -> None:
    path = ROOT / "outputs" / "selection_bias_sci" / "selection_bias_curve.json"
    if not path.exists():
        pytest.skip("curve artifact not generated in this checkout")
    d = json.loads(path.read_text(encoding="utf-8"))
    m = mk.MacroSet()
    mk.add_selection_bias_curve(m, path)
    out = dict(m.items()) if hasattr(m, "items") else m._macros  # noqa: SLF001
    ref = int(out["BiasGridRefN"])
    used = [r["replicates_used"] for r in d["rows"] if r["antigens"] == ref]
    assert min(used) > 1, f"n={ref} cell is not resampled"


def test_size_trend_macros_ignore_unreplicated_full_split(tmp_path: Path) -> None:
    """The full-split cell runs once; never quote it as a mean or a share."""
    rows = []
    for n, bias, share, reps in (
        (4, 0.0542, 0.97, 100),
        (12, 0.0321, 1.00, 100),
        (15, 0.0444, 1.00, 1),      # full split: one deterministic draw
    ):
        rows.append({
            "antigens": n, "grid_points": 21, "mean_bias": bias,
            "share_bias_positive": share, "mean_predictor_agreement": 0.6,
            "replicates_requested": 100, "replicates_used": reps,
        })
    path = tmp_path / "curve_sizes.json"
    path.write_text(
        json.dumps({"select_metric": "mcc", "rows": rows}), encoding="utf-8"
    )
    m = mk.MacroSet()
    mk.add_selection_bias_curve(m, path)
    out = dict(m.items()) if hasattr(m, "items") else m._macros  # noqa: SLF001
    assert out["BiasSmallestN"] == "4"
    assert out["BiasLargestN"] == "12", "must not read the full-split cell"
    assert out["BiasAtLargestN"] == "0.0321"
    assert out["BiasPositiveShareLargestN"] == "100"


def test_full_split_macros_are_separate_from_the_trend(tmp_path: Path) -> None:
    """The deterministic cell gets its own macros, distinct from the endpoint."""
    rows = []
    for n, bias, reps in ((4, 0.0542, 100), (12, 0.0315, 100), (15, 0.0444, 1)):
        rows.append({
            "antigens": n, "grid_points": 21, "mean_bias": bias,
            "share_bias_positive": 1.0, "mean_predictor_agreement": 0.6,
            "replicates_requested": 100, "replicates_used": reps,
        })
    path = tmp_path / "curve_full.json"
    path.write_text(
        json.dumps({"select_metric": "mcc", "rows": rows}), encoding="utf-8"
    )
    m = mk.MacroSet()
    mk.add_selection_bias_curve(m, path)
    out = dict(m.items()) if hasattr(m, "items") else m._macros  # noqa: SLF001
    assert out["BiasFullSplitN"] == "15"
    assert out["BiasFullSplitValue"] == "0.0444"
    # and the trend endpoint is a different cell entirely
    assert out["BiasAtLargestN"] != out["BiasFullSplitValue"]
