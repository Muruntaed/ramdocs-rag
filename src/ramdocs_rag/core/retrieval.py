"""Hybrid retrieval (BM25 + dense cosine) inside a per-question pool.

Ported from the legacy prototype with minimal adaptation: ``RAMDoc``
instead of ``InferenceDoc``; the embedder is a lazy module-level
singleton (one MiniLM model per process).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from .types import RAMDoc, RetrievedDoc

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_EMBEDDINGS_MODEL)


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


@dataclass(frozen=True)
class RetrievalConfig:
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    top_k: int = 8


def retrieve(
    query: str, docs: list[RAMDoc], cfg: RetrievalConfig | None = None
) -> list[RetrievedDoc]:
    """Return the top-K documents ranked by the hybrid score. Used since v1.0."""
    cfg = cfg or RetrievalConfig()
    if not docs:
        return []

    # BM25
    tokenised = [_tokenize(d.text) for d in docs]
    bm25 = BM25Okapi(tokenised)
    bm25_raw = np.asarray(bm25.get_scores(_tokenize(query)), dtype=np.float64)

    # Dense
    embedder = _get_embedder()
    doc_emb = embedder.encode(
        [d.text for d in docs], normalize_embeddings=True, show_progress_bar=False
    )
    q_emb = embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    dense_raw = np.asarray(doc_emb @ q_emb, dtype=np.float64)

    bm25_n = _minmax(bm25_raw)
    dense_n = _minmax(dense_raw)
    combined = cfg.bm25_weight * bm25_n + cfg.dense_weight * dense_n

    order = np.argsort(-combined)[: cfg.top_k]
    return [
        RetrievedDoc(doc_id=docs[i].doc_id, text=docs[i].text, score=float(combined[i]))
        for i in order
    ]
