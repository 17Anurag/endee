"""
Thin HTTP client for the Endee vector database REST API.
Endee exposes its API on port 8080 by default.
Docs: https://docs.endee.io
"""
from __future__ import annotations

import os
import requests
from typing import Any, Optional


class EndeeClient:
    """Minimal client wrapping Endee's HTTP REST API."""

    def __init__(self, host: Optional[str] = None, api_key: Optional[str] = None):
        self.host = (host or os.getenv("ENDEE_HOST", "http://localhost:8080")).rstrip("/")
        self.api_key = api_key or os.getenv("ENDEE_API_KEY", "")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self.session.headers.update({"Content-Type": "application/json"})

    def _url(self, path: str) -> str:
        return f"{self.host}{path}"

    def _raise(self, resp: requests.Response) -> None:
        if not resp.ok:
            raise RuntimeError(f"Endee API error {resp.status_code}: {resp.text}")

    # ------------------------------------------------------------------ #
    # Index management
    # ------------------------------------------------------------------ #

    def create_index(self, name: str, dimension: int, metric: str = "cosine") -> dict:
        """Create a new vector index."""
        payload = {"name": name, "dimension": dimension, "metric": metric}
        resp = self.session.post(self._url("/indexes"), json=payload)
        self._raise(resp)
        return resp.json()

    def list_indexes(self) -> list[dict]:
        resp = self.session.get(self._url("/indexes"))
        self._raise(resp)
        return resp.json()

    def delete_index(self, name: str) -> dict:
        resp = self.session.delete(self._url(f"/indexes/{name}"))
        self._raise(resp)
        return resp.json()

    # ------------------------------------------------------------------ #
    # Vector operations
    # ------------------------------------------------------------------ #

    def upsert(self, index: str, vectors: list[dict[str, Any]]) -> dict:
        """
        Upsert vectors into an index.

        Each vector dict must have:
          - id   (str)
          - values (list[float])
          - metadata (dict, optional)
        """
        payload = {"vectors": vectors}
        resp = self.session.post(self._url(f"/indexes/{index}/upsert"), json=payload)
        self._raise(resp)
        return resp.json()

    def search(
        self,
        index: str,
        query_vector: list,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list:
        """
        Search for the top-k nearest neighbours.

        Returns a list of matches with id, score, and metadata.
        """
        payload: dict[str, Any] = {"vector": query_vector, "top_k": top_k}
        if filters:
            payload["filter"] = filters
        resp = self.session.post(self._url(f"/indexes/{index}/search"), json=payload)
        self._raise(resp)
        data = resp.json()
        # Endee returns {"matches": [...]} or similar — handle both shapes
        return data.get("matches", data) if isinstance(data, dict) else data
