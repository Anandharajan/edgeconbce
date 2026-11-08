from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from edgeconbce.config import load_run_config
from edgeconbce.schemas import LegacyBP3RealV3Config, NotebookCompatConfig


class ConfigLoadTests(TestCase):
    def test_load_notebook_compat_config(self) -> None:
        config = load_run_config(Path("configs/notebook_compat.yaml"))
        self.assertIsInstance(config, NotebookCompatConfig)
        self.assertEqual(config.mode, "notebook_compat")
        self.assertEqual(len(config.policies), 4)

    def test_load_real_benchmark_config(self) -> None:
        config = load_run_config(Path("configs/bp3_real_v3.yaml"))
        self.assertIsInstance(config, LegacyBP3RealV3Config)
        self.assertEqual(config.mode, "legacy_bp3_real_v3")
        self.assertEqual(config.smoothing_radius_grid, (0, 2, 3))

