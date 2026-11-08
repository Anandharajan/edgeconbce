from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest import TestCase
import shutil
import uuid

from edgeconbce.config import load_run_config
from edgeconbce.pipeline.notebook_compat import run_benchmark
from edgeconbce.schemas import NotebookCompatConfig, validate_notebook_bundle


class ResultSchemaTests(TestCase):
    def test_notebook_bundle_validator_accepts_generated_bundle(self) -> None:
        config = load_run_config(Path("configs/notebook_compat.yaml"))
        self.assertIsInstance(config, NotebookCompatConfig)
        temp_dir = Path("outputs") / "_test_tmp" / f"schema_{uuid.uuid4().hex}"
        temp_output = temp_dir / "bundle.json"
        try:
            run_config = replace(config, output_json_path=str(temp_output), num_samples=10)
            result = run_benchmark(run_config)
            validate_notebook_bundle(result, run_config)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
