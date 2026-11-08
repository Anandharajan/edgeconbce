from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class EdgeRequestHandler(BaseHTTPRequestHandler):
    server_version = "EdgeConBCEEdge/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in {"", "/health"}:
            self.send_error(404)
            return
        self._write_json({"status": "ok", "service": "edgeconbce-edge", "time_utc": datetime.now(timezone.utc).isoformat()})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/edge":
            self.send_error(404)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON payload")
            return
        response = {
            "status": "ok",
            "received_bytes": len(body),
            "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
            "sequence_id": payload.get("sequence_id") if isinstance(payload, dict) else None,
            "policy": payload.get("policy") if isinstance(payload, dict) else None,
            "time_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._write_json(response)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _write_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(*, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), EdgeRequestHandler)
    try:
        print(f"EdgeConBCE edge server listening on http://{host}:{port}/edge", flush=True)
        server.serve_forever()
    finally:
        server.server_close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal EdgeConBCE edge endpoint for two-node transport measurement.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
