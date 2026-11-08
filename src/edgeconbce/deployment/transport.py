from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from time import perf_counter
from typing import Any
import json

import requests


@dataclass(frozen=True)
class LoopbackResponse:
    request_bytes: int
    response_bytes: int
    roundtrip_ms: float
    status_code: int


@dataclass(frozen=True)
class TransportResponse:
    request_bytes: int
    response_bytes: int
    roundtrip_ms: float
    status_code: int


class _EchoHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        payload = json.dumps(
            {
                "received_bytes": len(body),
                "path": self.path,
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def encode_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def post_json_payload(payload: dict[str, Any], *, url: str, timeout_seconds: float = 15.0) -> TransportResponse:
    body = encode_payload(payload)
    start = perf_counter()
    response = requests.post(url, data=body, headers={"Content-Type": "application/json"}, timeout=timeout_seconds)
    roundtrip_ms = (perf_counter() - start) * 1000.0
    response.raise_for_status()
    return TransportResponse(
        request_bytes=len(body),
        response_bytes=len(response.content),
        roundtrip_ms=round(float(roundtrip_ms), 6),
        status_code=int(response.status_code),
    )


def benchmark_loopback_payload(payload: dict[str, Any], *, path: str = "/edge") -> LoopbackResponse:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    request_bytes = 0
    response_bytes = 0
    status_code = 0
    roundtrip_ms = 0.0
    try:
        body = encode_payload(payload)
        request_bytes = len(body)
        url = f"http://127.0.0.1:{server.server_port}{path}"
        start = perf_counter()
        response = requests.post(url, data=body, headers={"Content-Type": "application/json"}, timeout=15.0)
        roundtrip_ms = (perf_counter() - start) * 1000.0
        response.raise_for_status()
        status_code = int(response.status_code)
        response_bytes = len(response.content)
        return LoopbackResponse(
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            roundtrip_ms=round(float(roundtrip_ms), 6),
            status_code=status_code,
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5.0)
