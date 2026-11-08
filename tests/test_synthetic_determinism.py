from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from edgeconbce.config import load_run_config
from edgeconbce.data.synthetic import build_dataset
from edgeconbce.schemas import NotebookCompatConfig


class SyntheticDeterminismTests(TestCase):
    def test_synthetic_dataset_is_deterministic_for_same_seed(self) -> None:
        config = load_run_config(Path("configs/notebook_compat.yaml"))
        self.assertIsInstance(config, NotebookCompatConfig)
        dataset_a = build_dataset(config)
        dataset_b = build_dataset(config)
        self.assertEqual([item.sample_id for item in dataset_a], [item.sample_id for item in dataset_b])
        self.assertEqual([item.sequence for item in dataset_a], [item.sequence for item in dataset_b])
        self.assertEqual([item.label for item in dataset_a], [item.label for item in dataset_b])

