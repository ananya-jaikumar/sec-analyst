"""
pipeline/vectorstore.py
Chunks SEC filing text, embeds with sentence-transformers,
and builds a FAISS vector index for semantic search.

Model: all-MiniLM-L6-v2 (22MB, runs locally, no API needed)
"""

import os
import re
import json
import joblib
import numpy as np
from typing import List

import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR         = os.path.join(os.path.dirname(__file__), "..", "data")

_embedder = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("  Loading embedding model (first run downloads ~22MB)...")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


# ── Chunking ───────────────────────────────────────────────────────────────

SECTION_HEADERS = [
    "ITEM 1.", "ITEM 1A.", "ITEM 1B.", "ITEM 2.", "ITEM 3.", "ITEM 4.",
    "ITEM 5.", "ITEM 6.", "ITEM 7.", "ITEM 7A.", "ITEM 8.", "ITEM 9.",
    "ITEM 9A.", "ITEM 10.", "ITEM 11.", "ITEM 12.", "ITEM 13.", "ITEM 14.",
    "RISK FACTORS", "BUSINESS", "MANAGEMENT", "FINANCIAL STATEMENTS",
    "QUANTITATIVE AND QUALITATIVE", "LEGAL PROCEEDINGS",
]

SECTION_LABELS = {
    "ITEM 1.":   "Business Overview",
    "ITEM 1A.":  "Risk Factors",
    "ITEM 7.":   "MD&A",
    "ITEM 7A.":  "Market Risk",
    "ITEM 8.":   "Financial Statements",
    "RISK FACTORS": "Risk Factors",
    "BUSINESS":  "Business Overview",
    "MANAGEMENT": "MD&A",
}


def detect_section(text_chunk: str) -> str:
    """Try to identify which 10-K section a chunk belongs to."""
    upper = text_chunk.upper()
    for header, label in SECTION_LABELS.items():
        if header in upper[:200]:
            return label
    return "General"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    Split text into overlapping chunks of ~chunk_size words.
    Works with both single and double newline separators.
    Returns list of dicts with keys: text, chunk_id, section, word_count
    """
    # Split on any newline (single or double) — SEC filings often use single \n
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 20]

    chunks = []
    current_words = []

    for line in lines:
        words = line.split()
        current_words.extend(words)

        if len(current_words) >= chunk_size:
            chunk_text = " ".join(current_words)
            chunks.append({
                "text":       chunk_text,
                "chunk_id":   len(chunks),
                "section":    detect_section(chunk_text),
                "word_count": len(current_words),
            })
            # Keep overlap
            current_words = current_words[-overlap:]

    # Flush remainder
    if len(current_words) > 50:
        chunk_text = " ".join(current_words)
        chunks.append({
            "text":       chunk_text,
            "chunk_id":   len(chunks),
            "section":    detect_section(chunk_text),
            "word_count": len(current_words),
        })

    return chunks


# ── Vector Store ───────────────────────────────────────────────────────────

class FilingVectorStore:
    """
    FAISS-backed vector store for a single SEC filing.
    Supports semantic search with top-k retrieval.
    """

    def __init__(self):
        self.index   = None
        self.chunks  = []
        self.meta    = {}

    def build(self, text: str, filing_meta: dict) -> None:
        """Chunk text, embed, and build FAISS index."""
        print(f"  Chunking filing text...")
        self.chunks = chunk_text(text)
        self.meta   = filing_meta
        print(f"  Created {len(self.chunks)} chunks")

        embedder = _get_embedder()
        texts    = [c["text"] for c in self.chunks]

        print(f"  Embedding {len(texts)} chunks...")
        embeddings = embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        dim         = embeddings.shape[1]
        self.index  = faiss.IndexFlatIP(dim)   # Inner product = cosine sim (normalized)
        self.index.add(embeddings.astype(np.float32))
        print(f"  FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        """Semantic search — returns top_k most relevant chunks."""
        if self.index is None:
            raise RuntimeError("Vector store not built. Call .build() first.")

        embedder = _get_embedder()
        q_vec = embedder.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(score)
                results.append(chunk)

        return results

    def save(self, path: str) -> None:
        """Persist index + chunks to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        joblib.dump({"chunks": self.chunks, "meta": self.meta},
                    os.path.join(path, "chunks.joblib"))
        print(f"  Saved vector store → {path}")

    @classmethod
    def load(cls, path: str) -> "FilingVectorStore":
        """Load a persisted vector store from disk."""
        store        = cls()
        store.index  = faiss.read_index(os.path.join(path, "index.faiss"))
        data         = joblib.load(os.path.join(path, "chunks.joblib"))
        store.chunks = data["chunks"]
        store.meta   = data["meta"]
        return store


def get_store_path(company: str, form: str, date: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{company}_{form}_{date}")
    return os.path.join(DATA_DIR, "stores", safe)


def store_exists(company: str, form: str, date: str) -> bool:
    path = get_store_path(company, form, date)
    return os.path.exists(os.path.join(path, "index.faiss"))
