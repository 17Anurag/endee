"""
FastAPI REST interface for the RAG system.

Run:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /             → web UI
    POST /ask          { "question": "..." }
    POST /ingest       { "title": "...", "category": "...", "content": "..." }
    GET  /documents    → list ingested documents
    GET  /health
"""

import os
import uuid
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dotenv import load_dotenv
from rag import ask, RAGResponse
from embedder import embed, EMBEDDING_DIM
from endee_client import EndeeClient
from ingest import chunk_text

load_dotenv()

INDEX_NAME = os.getenv("ENDEE_INDEX_NAME", "rag_documents")

app = FastAPI(
    title="RAG with Endee",
    description="Retrieval-Augmented Generation powered by Endee vector database",
    version="1.0.0",
)

_static = Path(__file__).parent / "static"
if _static.exists():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")

# In-memory document registry (title, category, chunk_count, date)
_documents: list[dict] = [
    {"title": "Python Programming Language", "category": "technology", "chunks": 3, "date": "22/3/2026"},
    {"title": "Machine Learning Overview",   "category": "ai",         "chunks": 3, "date": "22/3/2026"},
    {"title": "Vector Databases",            "category": "technology", "chunks": 3, "date": "22/3/2026"},
    {"title": "Retrieval-Augmented Generation (RAG)", "category": "ai", "chunks": 3, "date": "22/3/2026"},
    {"title": "Large Language Models",       "category": "ai",         "chunks": 3, "date": "22/3/2026"},
    {"title": "Docker Containers",           "category": "technology", "chunks": 3, "date": "22/3/2026"},
]


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return (_static / "index.html").read_text(encoding="utf-8")


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 4


class SourceOut(BaseModel):
    id: str
    score: float
    title: str
    text: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
    retrieval_only: bool


class IngestRequest(BaseModel):
    title: str
    category: str = "general"
    content: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents():
    return {"documents": _documents, "total": len(_documents)}


@app.post("/ingest")
def ingest_document(req: IngestRequest):
    if not req.title.strip() or not req.content.strip():
        raise HTTPException(status_code=400, detail="Title and content are required")
    try:
        client = EndeeClient()
        chunks = chunk_text(req.content)
        texts = [c for c in chunks]
        vectors = embed(texts)
        endee_vectors = [
            {
                "id": str(uuid.uuid4()),
                "values": vec,
                "metadata": {"text": chunk, "title": req.title, "chunk_index": i},
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        client.upsert(INDEX_NAME, endee_vectors)
        today = date.today().strftime("%-d/%-m/%Y") if os.name != "nt" else date.today().strftime("%#d/%#m/%Y")
        _documents.append({
            "title": req.title,
            "category": req.category.lower(),
            "chunks": len(chunks),
            "date": today,
        })
        return {"ingested": len(chunks), "title": req.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        resp: RAGResponse = ask(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AnswerResponse(
        question=resp.question,
        answer=resp.answer,
        sources=[
            SourceOut(id=s.id, score=s.score, title=s.title, text=s.text)
            for s in resp.sources
        ],
        retrieval_only=resp.retrieval_only,
    )
