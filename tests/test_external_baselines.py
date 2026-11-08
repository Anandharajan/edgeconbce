from __future__ import annotations

from pathlib import Path
from unittest import TestCase
import json
import shutil
import uuid

from edgeconbce.pipeline.external_baselines import evaluate_bepipred_output, write_unlabeled_submission_fasta


class ExternalBaselineTests(TestCase):
    def test_prepare_unlabeled_fasta_and_evaluate_prediction_fasta(self) -> None:
        temp_dir = Path("outputs") / "_test_tmp" / f"external_{uuid.uuid4().hex}"
        truth_fasta = temp_dir / "truth.fasta"
        prediction_fasta = temp_dir / "prediction.fasta"
        unlabeled_fasta = temp_dir / "unlabeled.fasta"
        output_dir = temp_dir / "reports"
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            truth_fasta.write_text(">seq1\nAaA\n>seq2\ncCC\n", encoding="utf-8")
            prediction_fasta.write_text(">seq1\nAaA\n>seq2\nccC\n", encoding="utf-8")

            written = write_unlabeled_submission_fasta(truth_fasta_path=truth_fasta, output_fasta_path=unlabeled_fasta)
            self.assertEqual(written, unlabeled_fasta)
            self.assertIn("AAA", unlabeled_fasta.read_text(encoding="utf-8"))

            report = evaluate_bepipred_output(
                truth_fasta_path=truth_fasta,
                prediction_fasta_path=prediction_fasta,
                output_dir=output_dir,
            )

            self.assertTrue(Path(report["reports"]["extra_baselines_json"]).exists())
            payload = json.loads(Path(report["reports"]["extra_baselines_json"]).read_text(encoding="utf-8"))
            row = payload["rows"][0]
            self.assertEqual(row["source_type"], "local_rerun")
            self.assertGreater(row["mcc"], 0.0)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
