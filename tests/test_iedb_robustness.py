from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from edgeconbce.pipeline.iedb_robustness import generate_iedb_robustness


class IedbRobustnessTests(unittest.TestCase):
    def test_generate_iedb_robustness_report(self) -> None:
        temp_dir = Path("outputs") / "_test_tmp" / f"iedb_robustness_{uuid.uuid4().hex}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            real_path = temp_dir / "real.json"
            carbon_path = temp_dir / "carbon.json"
            output_dir = temp_dir / "out"
            per_sequence = [
                {
                    "sequence_id": "seq1",
                    "length": 4,
                    "labels": [0, 1, 1, 0],
                    "adaptive_probabilities": [0.2, 0.7, 0.8, 0.3],
                    "adaptive_predictions": [0, 1, 1, 0],
                }
            ]
            real_path.write_text(
                json.dumps(
                    {
                        "config": {"test_fasta_path": "mock.fasta"},
                        "external_test": {
                            "local_only": {"per_sequence": per_sequence},
                            "adaptive_split": {"per_sequence": per_sequence},
                        },
                    }
                ),
                encoding="utf-8",
            )
            carbon_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "sequence_id": "seq1",
                                "length": 4,
                                "labels": [0, 1, 1, 0],
                                "probabilities": [0.1, 0.9, 0.6, 0.2],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            report = generate_iedb_robustness(
                real_results_path=real_path,
                carbon_predictions_path=carbon_path,
                output_dir=output_dir,
                bootstrap_repeats=10,
            )

            self.assertTrue(Path(report["reports"]["markdown"]).exists())
            self.assertTrue(Path(report["reports"]["sensitivity_csv"]).exists())
            self.assertEqual(len(report["weight_sensitivity"]), 11)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
