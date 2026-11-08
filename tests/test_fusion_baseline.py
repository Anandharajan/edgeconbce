from __future__ import annotations

import csv
import json
import shutil
import unittest
import uuid
from pathlib import Path

from edgeconbce.pipeline.fusion_baseline import generate_fusion_baseline


class FusionBaselineTests(unittest.TestCase):
    def test_generate_fixed_fusion_baseline(self) -> None:
        temp_dir = Path("outputs") / "_test_tmp" / f"fusion_{uuid.uuid4().hex}"
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            carbon_path = temp_dir / "carbon.json"
            bepi_raw_path = temp_dir / "bepi.csv"
            bepi_summary_path = temp_dir / "bepi_summary.json"
            output_dir = temp_dir / "out"

            carbon_path.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "sequence_id": "seq1",
                                "length": 4,
                                "labels": [0, 1, 1, 0],
                                "probabilities": [0.1, 0.8, 0.7, 0.2],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with bepi_raw_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Accession", "Residue", "BepiPred-3.0 score"])
                writer.writeheader()
                for score in [0.05, 0.9, 0.6, 0.1]:
                    writer.writerow({"Accession": "seq1", "Residue": "A", "BepiPred-3.0 score": score})
            bepi_summary_path.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")

            report = generate_fusion_baseline(
                carbon_predictions_path=carbon_path,
                bepipred_raw_scores_path=bepi_raw_path,
                bepipred_summary_path=bepi_summary_path,
                output_dir=output_dir,
                bepipred_weight=0.9,
                bootstrap_repeats=10,
            )

            self.assertTrue(Path(report["reports"]["markdown"]).exists())
            self.assertTrue(Path(report["reports"]["baseline_json"]).exists())
            self.assertTrue(Path(report["reports"]["sensitivity_csv"]).exists())
            self.assertTrue(Path(report["reports"]["bootstrap_csv"]).exists())
            self.assertEqual(report["algorithm"]["bepipred_positive_count"], 2)
            self.assertEqual(report["summary"]["predicted_positive_residues"], 2)
            self.assertEqual(len(report["weight_sensitivity"]), 11)
            self.assertIn("sequence_bootstrap_delta_vs_bepipred", report)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
