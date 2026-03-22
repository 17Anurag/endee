"""
Embedding utility using sentence-transformers (runs fully locally, no API key needed).
Model: all-MiniLM-L6-v2  — 384-dimensional, fast, good quality.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
from typing import Optional

_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Return a list of embedding vectors for the given texts."""
    model = get_model()
    return model.encode(texts, convert_to_numpy=True).tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
