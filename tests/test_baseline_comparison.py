from __future__ import annotations

from pathlib import Path
from unittest import TestCase
import json
import shutil
import uuid

from edgeconbce.pipeline.baseline_comparison import generate_baseline_comparison


class BaselineComparisonTests(TestCase):
    def test_generate_baseline_comparison_from_mock_repeated_results(self) -> None:
        temp_dir = Path("outputs") / "_test_tmp" / f"baseline_{uuid.uuid4().hex}"
        repeated_path = temp_dir / "repeated.json"
        extra_path = temp_dir / "extra_baselines.json"
        second_extra_path = temp_dir / "second_extra_baselines.json"
        output_dir = temp_dir / "comparison"
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            repeated_payload = {
                "aggregate_policy_metrics": [
                    {
                        "policy": "adaptive_split",
                        "mcc_mean": 0.2,
                        "f1_mean": 0.4,
                        "roc_auc_mean": 0.7,
                        "average_precision_mean": 0.3,
                        "mean_latency_ms_mean": 10.0,
                        "mean_total_gco2e_mean": 0.15,
                        "raw_sequence_share_rate_mean": 0.2,
                    }
                ]
            }
            repeated_path.write_text(json.dumps(repeated_payload), encoding="utf-8")
            extra_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "model": "BepiPred-3.0 local rerun",
                                "source_type": "local_rerun",
                                "dataset": "BP3C50ID external test (15 antigens)",
                                "roc_auc": 0.71,
                                "mcc": 0.24,
                                "f1": 0.4,
                                "average_precision": None,
                                "latency_ms": None,
                                "total_gco2e": None,
                                "raw_sequence_share_rate": None,
                                "notes": "Mock imported comparator.",
                                "source_url": None,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            second_extra_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "model": "CARBON-BCE-R local",
                                "source_type": "local_rerun",
                                "dataset": "BP3C50ID external test (15 antigens)",
                                "roc_auc": 0.75,
                                "mcc": 0.3,
                                "f1": 0.45,
                                "average_precision": 0.4,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            reports = generate_baseline_comparison(
                repeated_results_path=repeated_path,
                output_dir=output_dir,
                extra_baselines_path=[extra_path, second_extra_path],
            )

            self.assertTrue(Path(reports["reports"]["csv"]).exists())
            self.assertTrue(Path(reports["reports"]["markdown"]).exists())
            payload = json.loads(Path(reports["reports"]["json"]).read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(payload["rows"]), 4)
            self.assertIn("local_rerun", {row["source_type"] for row in payload["rows"]})
            self.assertIn("CARBON-BCE-R local", {row["model"] for row in payload["rows"]})
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
