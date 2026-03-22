"""
Document ingestion pipeline.

Usage:
    python ingest.py                        # ingest built-in sample docs
    python ingest.py --file path/to/doc.txt # ingest a plain-text file
    python ingest.py --reset                # drop and recreate the index first

Steps:
  1. Load documents (plain text, split into chunks)
  2. Embed each chunk with sentence-transformers
  3. Upsert vectors + metadata into Endee
"""

import argparse
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

from embedder import embed, EMBEDDING_DIM
from endee_client import EndeeClient

load_dotenv()
console = Console()

INDEX_NAME = os.getenv("ENDEE_INDEX_NAME", "rag_documents")
CHUNK_SIZE = 300   # characters per chunk
CHUNK_OVERLAP = 50


# ------------------------------------------------------------------ #
# Sample knowledge base (used when no --file is provided)
# ------------------------------------------------------------------ #

SAMPLE_DOCS = [
    {
        "title": "Python Programming Language",
        "text": (
            "Python is a high-level, general-purpose programming language. Its design philosophy "
            "emphasises code readability with the use of significant indentation. Python is "
            "dynamically typed and garbage-collected. It supports multiple programming paradigms, "
            "including structured, object-oriented and functional programming. It is often described "
            "as a 'batteries included' language due to its comprehensive standard library. Guido van "
            "Rossum began working on Python in the late 1980s as a successor to the ABC programming "
            "language. Python 3.0 was released in 2008 and is the current major version."
        ),
    },
    {
        "title": "Machine Learning Overview",
        "text": (
            "Machine learning (ML) is a field of study in artificial intelligence concerned with the "
            "development and study of statistical algorithms that can learn from data and generalise "
            "to unseen data, and thus perform tasks without explicit instructions. Recently, "
            "generative artificial intelligence has become a notable application of machine learning. "
            "Supervised learning algorithms build a mathematical model of a set of data that contains "
            "both the inputs and the desired outputs. Unsupervised learning algorithms take a set of "
            "data that contains only inputs, and find structure in the data. Reinforcement learning "
            "trains agents to take actions in an environment to maximise a reward signal."
        ),
    },
    {
        "title": "Vector Databases",
        "text": (
            "A vector database is a type of database that stores data as high-dimensional vectors, "
            "which are mathematical representations of features or attributes. Each vector has a "
            "certain number of dimensions, which can range from tens to thousands, depending on the "
            "complexity of the data. Vector databases are used in AI applications for semantic "
            "search, recommendation systems, and retrieval-augmented generation (RAG). They support "
            "approximate nearest neighbour (ANN) search algorithms such as HNSW and IVF to find "
            "similar vectors efficiently. Endee is a high-performance open-source vector database "
            "designed to handle up to 1 billion vectors on a single node."
        ),
    },
    {
        "title": "Retrieval-Augmented Generation (RAG)",
        "text": (
            "Retrieval-Augmented Generation (RAG) is an AI framework that combines information "
            "retrieval with text generation. In a RAG pipeline, a retriever first fetches relevant "
            "documents from a knowledge base using vector similarity search. The retrieved context "
            "is then passed to a large language model (LLM) together with the user's question, "
            "allowing the LLM to generate a grounded, factual answer. RAG reduces hallucinations "
            "because the model can cite retrieved passages rather than relying solely on parametric "
            "memory. It is widely used in enterprise Q&A systems, chatbots, and copilots."
        ),
    },
    {
        "title": "Large Language Models",
        "text": (
            "A large language model (LLM) is a type of machine learning model designed for natural "
            "language processing tasks such as language generation. LLMs are language models with "
            "many parameters, and are trained on large quantities of text. They are based on the "
            "transformer architecture, introduced in the paper 'Attention Is All You Need' (2017). "
            "Examples include GPT-4, Claude, Gemini, and Llama. LLMs can perform tasks such as "
            "question answering, summarisation, translation, and code generation. Fine-tuning and "
            "prompt engineering are common techniques to adapt LLMs to specific domains."
        ),
    },
    {
        "title": "Docker Containers",
        "text": (
            "Docker is a platform for developing, shipping, and running applications in containers. "
            "Containers allow developers to package an application with all its dependencies into a "
            "standardised unit. Unlike virtual machines, containers share the host OS kernel and are "
            "therefore lightweight and fast to start. Docker uses a layered filesystem and a "
            "declarative Dockerfile to build images. Docker Compose allows multi-container "
            "applications to be defined and run with a single command. Endee can be run as a Docker "
            "container using the official image from the Endee GitHub repository."
        ),
    },
]


# ------------------------------------------------------------------ #
# Chunking
# ------------------------------------------------------------------ #

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


# ------------------------------------------------------------------ #
# Main ingestion logic
# ------------------------------------------------------------------ #

def ingest_documents(docs: list[dict], client: EndeeClient) -> None:
    all_chunks: list[dict] = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "title": doc.get("title", "Untitled"),
                    "chunk_index": i,
                }
            )

    console.print(f"[cyan]Embedding {len(all_chunks)} chunks...[/cyan]")
    texts = [c["text"] for c in all_chunks]
    vectors = embed(texts)

    # Build Endee upsert payload
    endee_vectors = [
        {
            "id": chunk["id"],
            "values": vec,
            "metadata": {
                "text": chunk["text"],
                "title": chunk["title"],
                "chunk_index": chunk["chunk_index"],
            },
        }
        for chunk, vec in zip(all_chunks, vectors)
    ]

    # Upsert in batches of 100
    batch_size = 100
    for i in track(range(0, len(endee_vectors), batch_size), description="Upserting to Endee"):
        batch = endee_vectors[i : i + batch_size]
        client.upsert(INDEX_NAME, batch)

    console.print(f"[green]✓ Ingested {len(endee_vectors)} vectors into index '{INDEX_NAME}'[/green]")


def ensure_index(client: EndeeClient, reset: bool = False) -> None:
    if reset:
        try:
            client.delete_index(INDEX_NAME)
            console.print(f"[yellow]Dropped existing index '{INDEX_NAME}'[/yellow]")
        except Exception:
            pass

    try:
        client.create_index(INDEX_NAME, dimension=EMBEDDING_DIM, metric="cosine")
        console.print(f"[green]Created index '{INDEX_NAME}' (dim={EMBEDDING_DIM})[/green]")
    except Exception as e:
        # Index may already exist — that's fine
        console.print(f"[dim]Index already exists or creation skipped: {e}[/dim]")


def load_text_file(path: str) -> list[dict]:
    content = Path(path).read_text(encoding="utf-8")
    return [{"title": Path(path).stem, "text": content}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Endee")
    parser.add_argument("--file", help="Path to a plain-text file to ingest")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the index")
    args = parser.parse_args()

    client = EndeeClient()
    ensure_index(client, reset=args.reset)

    docs = load_text_file(args.file) if args.file else SAMPLE_DOCS
    console.print(f"[cyan]Ingesting {len(docs)} document(s)...[/cyan]")
    ingest_documents(docs, client)


if __name__ == "__main__":
    main()
