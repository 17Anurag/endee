"""
Lightweight mock of the Endee REST API for local development without Docker.
Implements: POST /indexes, GET /indexes, DELETE /indexes/{name},
            POST /indexes/{name}/upsert, POST /indexes/{name}/search

Run:  python mock_endee_server.py
      (listens on http://localhost:8080)
"""
from __future__ import annotations

import math
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from typing import Any

# In-memory store: { index_name: { "dimension": int, "metric": str, "vectors": {id: {...}} } }
STORE: dict[str, Any] = {}


def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class EndeeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[mock-endee] {fmt % args}")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw)

    def _send(self, code: int, body: Any) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/indexes":
            result = [
                {"name": k, "dimension": v["dimension"], "metric": v["metric"]}
                for k, v in STORE.items()
            ]
            self._send(200, result)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        parts = self.path.strip("/").split("/")

        # POST /indexes  — create index
        if parts == ["indexes"]:
            body = self._read_body()
            name = body.get("name")
            if not name:
                return self._send(400, {"error": "name required"})
            if name in STORE:
                return self._send(409, {"error": "index already exists"})
            STORE[name] = {
                "dimension": body.get("dimension", 384),
                "metric": body.get("metric", "cosine"),
                "vectors": {},
            }
            self._send(200, {"name": name, "status": "created"})

        # POST /indexes/{name}/upsert
        elif len(parts) == 3 and parts[0] == "indexes" and parts[2] == "upsert":
            idx_name = parts[1]
            if idx_name not in STORE:
                return self._send(404, {"error": "index not found"})
            body = self._read_body()
            for vec in body.get("vectors", []):
                STORE[idx_name]["vectors"][vec["id"]] = vec
            self._send(200, {"upserted": len(body.get("vectors", []))})

        # POST /indexes/{name}/search
        elif len(parts) == 3 and parts[0] == "indexes" and parts[2] == "search":
            idx_name = parts[1]
            if idx_name not in STORE:
                return self._send(404, {"error": "index not found"})
            body = self._read_body()
            query = body.get("vector", [])
            top_k = body.get("top_k", 5)

            scores = []
            for vid, vdata in STORE[idx_name]["vectors"].items():
                sim = cosine_similarity(query, vdata.get("values", []))
                scores.append({"id": vid, "score": sim, "metadata": vdata.get("metadata", {})})

            scores.sort(key=lambda x: x["score"], reverse=True)
            self._send(200, {"matches": scores[:top_k]})

        else:
            self._send(404, {"error": "not found"})

    def do_DELETE(self):
        parts = self.path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "indexes":
            name = parts[1]
            if name in STORE:
                del STORE[name]
                self._send(200, {"deleted": name})
            else:
                self._send(404, {"error": "index not found"})
        else:
            self._send(404, {"error": "not found"})


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8080), EndeeHandler)
    print("[mock-endee] Endee mock server running on http://localhost:8080")
    server.serve_forever()
