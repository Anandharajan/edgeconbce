from __future__ import annotations

from http.server import ThreadingHTTPServer
from threading import Thread
from unittest import TestCase

from edgeconbce.deployment.edge_server import EdgeRequestHandler
from edgeconbce.deployment.transport import benchmark_loopback_payload, post_json_payload


class TransportBenchmarkTests(TestCase):
    def test_loopback_payload_returns_measured_bytes_and_latency(self) -> None:
        response = benchmark_loopback_payload({"sequence_id": "seq1", "sequence": "ACDEFG"})
        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.request_bytes, 0)
        self.assertGreater(response.response_bytes, 0)
        self.assertGreaterEqual(response.roundtrip_ms, 0.0)

    def test_post_json_payload_to_edge_server(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), EdgeRequestHandler)
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            response = post_json_payload(
                {"sequence_id": "seq1", "sequence": "ACDEFG", "policy": "edge_only"},
                url=f"http://127.0.0.1:{server.server_port}/edge",
            )
            self.assertEqual(response.status_code, 200)
            self.assertGreater(response.request_bytes, 0)
            self.assertGreater(response.response_bytes, 0)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5.0)
