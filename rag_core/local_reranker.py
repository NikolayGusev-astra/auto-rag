"""
Local Cross-Encoder Reranker — ms-marco-MiniLM-L6-v2 (~80MB).

Runs locally via sentence-transformers, no LM Studio round-trip.
Fallback-compatible with bge-reranker in fuser.py.

Usage:
    from local_reranker import LocalReranker, rerank
    reranked = rerank(query, chunks, topn=5)
    # or
    r = LocalReranker(device="cpu")
    results = r.rerank(query, documents, topn=5)
"""
from __future__ import annotations

import os
import sys
import logging

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_model = None
_device = None
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"


def _get_model(device: str = "cpu"):
    """Lazy-load cross-encoder model (downloads ~80MB on first call)."""
    global _model, _device
    if _model is not None and _device == device:
        return _model

    from sentence_transformers import CrossEncoder

    logger.info("Loading local reranker %s on %s ...", _MODEL_NAME, device)
    _model = CrossEncoder(_MODEL_NAME, device=device)
    _device = device
    logger.info("Local reranker loaded.")
    return _model


class LocalReranker:
    """Local cross-encoder reranker. Constructor takes query + rerank_field;
    actual reranking happens in .rerank(results, topn=N)."""

    def __init__(self, query: str, rerank_field: str = "text",
                 model_name: str | None = None,
                 device: str = "cpu"):
        self.query = query
        self.rerank_field = rerank_field
        self._model_name = model_name or _MODEL_NAME
        self._device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker %s on %s", self._model_name, self._device)
            self._model = CrossEncoder(self._model_name, device=self._device)

    def rerank(self, results: list[dict], topn: int = 5) -> list[dict]:
        """Rerank list of dicts by self.rerank_field field. Returns top-N."""
        if not results:
            return []

        self._ensure_model()
        pairs = [(self.query, r.get(self.rerank_field, "")) for r in results]
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        for r, s in zip(results, scores):
            r["_rerank_score"] = float(s)

        results.sort(key=lambda x: x["_rerank_score"], reverse=True)
        return results[:topn]


def rerank(
    query: str,
    chunks: list[dict],
    topn: int = 5,
    text_field: str = "text",
    device: str = "cpu",
) -> list[dict]:
    """Functional API: rerank chunks in-place, return top-N.

    Compatible with fuser.py _rerank() output format.
    """
    if not chunks:
        return []

    model = _get_model(device)
    pairs = [(query, c.get(text_field, "")) for c in chunks]
    scores = model.predict(pairs, show_progress_bar=False)

    for c, s in zip(chunks, scores):
        c["_rerank_score"] = float(s)

    chunks.sort(key=lambda x: x["_rerank_score"], reverse=True)
    return chunks[:topn]