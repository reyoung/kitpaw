#!/usr/bin/env python3
"""Recording proxy that handles GET + POST, for Codex Rust comparison."""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import httpx

upstream = sys.argv[1]
output = sys.argv[2]
port = int(sys.argv[3]) if len(sys.argv) > 3 else 19905
client = httpx.Client(base_url=upstream, timeout=120.0)

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def _proxy(self, method):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        headers = {k: self.headers[k] for k in ("Authorization","Content-Type","Accept") if self.headers.get(k)}
        try: req_json = json.loads(body) if body else {}
        except: req_json = {}
        record = {"ts": time.time(), "method": method, "path": self.path, "request": req_json}
        try:
            resp = client.request(method, self.path, content=body, headers=headers)
            record["status"] = resp.status_code
            self.send_response(resp.status_code)
            for k, v in resp.headers.items():
                if k.lower() not in ("content-encoding","transfer-encoding","content-length"):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(resp.content)))
            self.end_headers()
            self.wfile.write(resp.content)
        except Exception as e:
            record["error"] = str(e)
            self.send_response(502)
            self.end_headers()
        with open(output, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    def do_GET(self): self._proxy("GET")
    def do_POST(self): self._proxy("POST")
    def log_message(self, *a): pass

print(f"Proxy :{port} -> {upstream}", file=sys.stderr)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
