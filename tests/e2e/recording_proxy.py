#!/usr/bin/env python3
"""Recording proxy that captures requests to OpenAI-compatible APIs.

Starts an HTTP server that forwards requests to the real upstream,
records them, and saves to a JSONL file for comparison.

Usage:
    python recording_proxy.py --upstream https://real-api/v1 --port 9999 --output requests.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import httpx


def run_proxy(upstream: str, port: int, output: str) -> None:
    records: list[dict] = []
    client = httpx.Client(base_url=upstream, timeout=120.0)

    class ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)

            try:
                request_json = json.loads(body.decode("utf-8"))
            except Exception:
                request_json = {"_raw": body.decode("utf-8", errors="replace")}

            # Record the request
            record = {
                "timestamp": time.time(),
                "method": "POST",
                "path": self.path,
                "request": request_json,
            }

            # Forward to upstream
            headers = {}
            for key in ("Authorization", "Content-Type", "Accept"):
                val = self.headers.get(key)
                if val:
                    headers[key] = val

            try:
                resp = client.post(
                    self.path,
                    content=body,
                    headers=headers,
                )

                record["response_status"] = resp.status_code
                record["response_headers"] = dict(resp.headers)

                self.send_response(resp.status_code)
                for k, v in resp.headers.items():
                    if k.lower() not in ("content-encoding", "transfer-encoding", "content-length"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(resp.content)))
                self.end_headers()
                self.wfile.write(resp.content)
                self.wfile.flush()

            except Exception as e:
                record["error"] = str(e)
                self.send_response(502)
                self.end_headers()
                self.wfile.write(f"Proxy error: {e}".encode())

            records.append(record)

            # Write incrementally
            with open(output, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), ProxyHandler)
    print(f"Recording proxy: http://127.0.0.1:{port} -> {upstream}", file=sys.stderr)
    print(f"Recording to: {output}", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        client.close()
        print(f"\nRecorded {len(records)} requests to {output}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--output", default="requests.jsonl")
    args = parser.parse_args()
    run_proxy(args.upstream, args.port, args.output)
