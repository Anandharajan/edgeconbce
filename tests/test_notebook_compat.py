from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest import TestCase
import shutil
import uuid

from edgeconbce.config import load_run_config
from edgeconbce.pipeline.notebook_compat import run_benchmark
from edgeconbce.schemas import NotebookCompatConfig


class NotebookCompatTests(TestCase):
    def test_notebook_benchmark_writes_expected_artifact(self) -> None:
        config = load_run_config(Path("configs/notebook_compat.yaml"))
        self.assertIsInstance(config, NotebookCompatConfig)
        temp_dir = Path("outputs") / "_test_tmp" / f"notebook_{uuid.uuid4().hex}"
        temp_output = temp_dir / "notebook_results.json"
        try:
            run_config = replace(config, output_json_path=str(temp_output), num_samples=12)
            result = run_benchmark(run_config)
            self.assertTrue(temp_output.exists())
            self.assertEqual(result["dataset"]["num_samples"], 12)
            self.assertEqual(len(result["policy_results"]), len(run_config.policies))
            for policy_result in result["policy_results"]:
                self.assertEqual(len(policy_result["records"]), 12)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
