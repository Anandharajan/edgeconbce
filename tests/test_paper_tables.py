from __future__ import annotations

from pathlib import Path
from unittest import TestCase
import json
import shutil
import uuid

from edgeconbce.pipeline.paper_tables import generate_paper_tables


class PaperTablesTests(TestCase):
    def test_generate_paper_tables_from_mock_artifacts(self) -> None:
        temp_dir = Path("outputs") / "_test_tmp" / f"paper_{uuid.uuid4().hex}"
        repeated_path = temp_dir / "repeated.json"
        real_path = temp_dir / "real.json"
        output_dir = temp_dir / "tables"
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            repeated_payload = {
                "run_name": "mock_repeated",
                "aggregate_policy_metrics": [
                    {
                        "policy": "adaptive_split",
                        "mcc_mean": 0.2,
                        "mcc_ci_low": 0.1,
                        "mcc_ci_high": 0.3,
                        "f1_mean": 0.4,
                        "f1_ci_low": 0.3,
                        "f1_ci_high": 0.5,
                        "roc_auc_mean": 0.7,
                        "roc_auc_ci_low": 0.6,
                        "roc_auc_ci_high": 0.8,
                        "average_precision_mean": 0.25,
                        "average_precision_ci_low": 0.2,
                        "average_precision_ci_high": 0.3,
                        "mean_latency_ms_mean": 10.0,
                        "mean_latency_ms_ci_low": 8.0,
                        "mean_latency_ms_ci_high": 12.0,
                        "mean_total_gco2e_mean": 0.15,
                        "mean_total_gco2e_ci_low": 0.1,
                        "mean_total_gco2e_ci_high": 0.2,
                        "route_rate_mean": 0.3,
                        "route_rate_ci_low": 0.2,
                        "route_rate_ci_high": 0.4,
                        "raw_sequence_share_rate_mean": 0.3,
                        "raw_sequence_share_rate_ci_low": 0.2,
                        "raw_sequence_share_rate_ci_high": 0.4,
                        "mean_bytes_transferred_mean": 42.0,
                        "mean_bytes_transferred_ci_low": 40.0,
                        "mean_bytes_transferred_ci_high": 44.0,
                    }
                ],
                "paired_deltas": {
                    "adaptive_split_minus_local_only": [
                        {
                            "comparison": "adaptive_split_minus_local_only",
                            "mcc": 0.1,
                            "f1": 0.1,
                            "roc_auc": 0.05,
                            "average_precision": 0.03,
                            "mean_latency_ms": -5.0,
                            "mean_total_gco2e": 0.01,
                            "raw_sequence_share_rate": 0.3,
                            "mean_bytes_transferred": 42.0,
                        }
                    ]
                },
                "measurement_provenance": {
                    "latency": "measured_from_mock",
                    "energy": "estimated_from_mock",
                },
            }
            real_payload = {
                "measurement_provenance": {
                    "bytes_transferred": "estimated_from_mock",
                },
                "external_test": {
                    "adaptive_split": {
                        "per_sequence": [
                            {"length": 42, "raw_sequence_shared": True},
                            {"length": 10, "raw_sequence_shared": False},
                        ]
                    }
                }
            }
            repeated_path.write_text(json.dumps(repeated_payload), encoding="utf-8")
            real_path.write_text(json.dumps(real_payload), encoding="utf-8")

            reports = generate_paper_tables(
                repeated_results_path=repeated_path,
                real_results_path=real_path,
                output_dir=output_dir,
            )

            self.assertTrue(Path(reports["markdown"]).exists())
            self.assertTrue(Path(reports["policy_csv"]).exists())
            self.assertTrue(Path(reports["comparison_csv"]).exists())
            self.assertTrue(Path(reports["data_movement_csv"]).exists())
            self.assertTrue(Path(reports["provenance_csv"]).exists())
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
