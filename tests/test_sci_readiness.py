from __future__ import annotations

from pathlib import Path
from unittest import TestCase
import json
import shutil
import uuid

from edgeconbce.pipeline.sci_readiness import REQUIRED_POLICIES, generate_sci_readiness_report


class SciReadinessTests(TestCase):
    def test_generate_sci_readiness_report_classifies_external_blockers(self) -> None:
        temp_dir = Path("outputs") / "_test_tmp" / f"sci_{uuid.uuid4().hex}"
        repeated_path = temp_dir / "repeated.json"
        baseline_path = temp_dir / "baseline.json"
        transport_path = temp_dir / "transport.json"
        real_path = temp_dir / "real.json"
        output_dir = temp_dir / "readiness"
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            repeated_path.write_text(
                json.dumps(
                    {
                        "seeds": list(range(10)),
                        "aggregate_policy_metrics": [{"policy": policy} for policy in sorted(REQUIRED_POLICIES)],
                        "measurement_provenance": {
                            "latency": "measured_from_mock",
                            "energy": "estimated_from_mock",
                            "operational_carbon": "estimated_from_mock",
                            "embodied_carbon": "estimated_from_mock",
                        },
                    }
                ),
                encoding="utf-8",
            )
            baseline_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "model": "BepiPred-3.0 FFNN (ESM-1b)",
                                "source_type": "published_same_split",
                                "dataset": "BP3C50ID external test (15 antigens)",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            transport_path.write_text(json.dumps({"summary_rows": [{"policy": "edge_only"}]}), encoding="utf-8")
            real_path.write_text(
                json.dumps(
                    {
                        "config": {"test_fasta_path": "BP3C50ID_external_test_set.fasta"},
                        "measurement_provenance": {"bytes_transferred": "estimated_from_mock"},
                    }
                ),
                encoding="utf-8",
            )

            report = generate_sci_readiness_report(
                repeated_results_path=repeated_path,
                baseline_results_path=baseline_path,
                transport_results_path=transport_path,
                real_results_path=real_path,
                output_dir=output_dir,
            )

            self.assertTrue(Path(report["reports"]["csv"]).exists())
            self.assertTrue(Path(report["reports"]["markdown"]).exists())
            statuses = {row["item"]: row["status"] for row in report["rows"]}
            self.assertEqual(statuses["10-seed repeated evaluation"], "pass")
            self.assertEqual(statuses["Required policy coverage"], "pass")
            self.assertEqual(statuses["Official or local BepiPred-3.0 rerun"], "blocked_external")
            self.assertEqual(statuses["Dataset breadth beyond BP3C50ID"], "blocked_external")
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
