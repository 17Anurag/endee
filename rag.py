"""
RAG (Retrieval-Augmented Generation) core logic.

Retrieval:  Endee vector search
Generation: OpenAI GPT (falls back to retrieval-only if no API key is set)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List

from dotenv import load_dotenv

from embedder import embed_one
from endee_client import EndeeClient

load_dotenv()

INDEX_NAME = os.getenv("ENDEE_INDEX_NAME", "rag_documents")
TOP_K = 4


@dataclass
class RetrievedChunk:
    id: str
    score: float
    title: str
    text: str


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: List[RetrievedChunk]
    retrieval_only: bool


# ------------------------------------------------------------------ #
# Retrieval
# ------------------------------------------------------------------ #

def retrieve(question: str, client: EndeeClient, top_k: int = TOP_K) -> List[RetrievedChunk]:
    query_vec = embed_one(question)
    matches = client.search(INDEX_NAME, query_vec, top_k=top_k)

    chunks = []
    for m in matches:
        meta = m.get("metadata", {})
        chunks.append(
            RetrievedChunk(
                id=m.get("id", ""),
                score=m.get("score", 0.0),
                title=meta.get("title", "Unknown"),
                text=meta.get("text", ""),
            )
        )
    return chunks


# ------------------------------------------------------------------ #
# Generation (OpenAI)
# ------------------------------------------------------------------ #

def _build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {c.title}\n{c.text}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant documents found."
    # Retrieval-only: return all retrieved chunks as the answer
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {c.title}\n{c.text}")
    return "\n\n".join(parts)


# ------------------------------------------------------------------ #
# Public interface
# ------------------------------------------------------------------ #

def ask(question: str, client: Optional[EndeeClient] = None) -> RAGResponse:
    if client is None:
        client = EndeeClient()

    chunks = retrieve(question, client)
    retrieval_only = True
    answer = generate_answer(question, chunks)

    return RAGResponse(
        question=question,
        answer=answer,
        sources=chunks,
        retrieval_only=retrieval_only,
    )
