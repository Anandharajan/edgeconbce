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
