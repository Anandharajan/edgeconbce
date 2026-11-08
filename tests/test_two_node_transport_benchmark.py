from __future__ import annotations

from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from unittest import TestCase
import json
import shutil
import uuid

from edgeconbce.deployment.edge_server import EdgeRequestHandler
from edgeconbce.pipeline.two_node_transport_benchmark import generate_two_node_transport_benchmark


class TwoNodeTransportBenchmarkTests(TestCase):
    def test_generate_two_node_transport_benchmark_from_mock_real_results(self) -> None:
        temp_dir = Path("outputs") / "_test_tmp" / f"two_node_{uuid.uuid4().hex}"
        fasta_path = temp_dir / "test.fasta"
        real_path = temp_dir / "real.json"
        output_dir = temp_dir / "transport"
        server = ThreadingHTTPServer(("127.0.0.1", 0), EdgeRequestHandler)
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            fasta_path.write_text(">seq1\nACDEFG\n", encoding="utf-8")
            real_path.write_text(
                json.dumps(
                    {
                        "config": {"test_fasta_path": str(fasta_path)},
                        "external_test": {
                            "edge_only": {
                                "per_sequence": [
                                    {
                                        "sequence_id": "seq1",
                                        "length": 6,
                                        "raw_sequence_shared": True,
                                        "local_probabilities": [0.1, 0.2, 0.3],
                                        "mean_local_uncertainty": 0.4,
                                        "positive_prediction_count": 1,
                                    }
                                ]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = generate_two_node_transport_benchmark(
                real_results_path=real_path,
                edge_url=f"http://127.0.0.1:{server.server_port}/edge",
                output_dir=output_dir,
            )

            self.assertEqual(report["analysis"], "measured_two_node_transport")
            self.assertTrue(Path(report["reports"]["summary_json"]).exists())
            self.assertEqual(len(report["summary_rows"]), 2)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5.0)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
