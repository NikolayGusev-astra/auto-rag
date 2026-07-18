"""Унифицированный reranker service через LM Studio.

LM Studio может работать с reranker моделями:
1. Jina-style: POST /v1/embeddings с input=[{query, document}, ...] -> data[i].score
2. Cohere-style: POST /v1/rerank с query + documents -> results[i].relevance_score

Автоматически определяем поддерживаемый формат.
"""
from __future__ import annotations

import os
import logging
import threading
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

RERANKER_URL = os.getenv("RAG_RERANK_URL", "http://localhost:1234/v1/embeddings")
RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "text-embedding-bge-reranker-v2-m3")
RERANKER_TIMEOUT = int(os.getenv("RAG_RERANKER_TIMEOUT", "30"))
RERANKER_BATCH = int(os.getenv("RAG_RERANKER_BATCH", "16"))
RERANKER_MAX_PAIRS = int(os.getenv("RAG_RERANKER_MAX_PAIRS", "100"))
RERANKER_MAX_RETRIES = int(os.getenv("RAG_RERANKER_RETRIES", "2"))


class RerankerService:
    """Singleton reranker service через LM Studio."""
    _instance: Optional[RerankerService] = None
    _lock = threading.Lock()

    def __init__(self):
        self._api_format: Optional[str] = None

    @classmethod
    def get(cls) -> RerankerService:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _detect_api_format(self) -> str:
        """Определить формат reranker API в LM Studio."""
        if self._api_format is not None:
            return self._api_format

        rerank_url = RERANKER_URL.replace("/v1/embeddings", "/v1/rerank")
        try:
            r = requests.options(rerank_url, timeout=2)
            if r.status_code in (200, 204, 405):
                self._api_format = "cohere"
                return "cohere"
        except requests.RequestException as exc:
            logger.debug("Cohere-style reranker endpoint unavailable: %s", exc)

        self._api_format = "jina"
        return "jina"

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Rerank documents against query.

        Returns: list of (original_index, score) sorted by score desc, top_k items.
        """
        if not documents:
            return []

        if len(documents) > RERANKER_MAX_PAIRS:
            documents = documents[:RERANKER_MAX_PAIRS]

        api_format = self._detect_api_format()

        if api_format == "cohere":
            scores = self._rerank_cohere(query, documents)
        else:
            scores = self._rerank_jina(query, documents)

        if not scores:
            return [(i, 0.5) for i in range(min(top_k, len(documents)))]

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: float(x[1]), reverse=True)
        return indexed[:top_k]

    def _rerank_jina(self, query: str, documents: list[str]) -> list[float]:
        """Jina-style: POST /v1/embeddings с input=[{query, document}]."""
        all_scores: list[float] = []
        for i in range(0, len(documents), RERANKER_BATCH):
            batch = documents[i : i + RERANKER_BATCH]
            pairs = [{"query": query, "document": doc[:512]} for doc in batch]

            last_error = None
            for attempt in range(RERANKER_MAX_RETRIES + 1):
                try:
                    r = requests.post(
                        RERANKER_URL,
                        json={"model": RERANKER_MODEL, "input": pairs},
                        timeout=RERANKER_TIMEOUT,
                    )
                    r.raise_for_status()
                    data = r.json()
                    batch_scores = [d.get("score", 0.0) for d in data.get("data", [])]
                    all_scores.extend(batch_scores)
                    break
                except (requests.Timeout, requests.ConnectionError) as e:
                    last_error = e
                    if attempt < RERANKER_MAX_RETRIES:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    all_scores.extend([0.5] * len(batch))
                except Exception:
                    all_scores.extend([0.5] * len(batch))
                    break

        return all_scores

    def _rerank_cohere(self, query: str, documents: list[str]) -> list[float]:
        """Cohere-style: POST /v1/rerank с query + documents."""
        rerank_url = RERANKER_URL.replace("/v1/embeddings", "/v1/rerank")

        last_error = None
        for attempt in range(RERANKER_MAX_RETRIES + 1):
            try:
                r = requests.post(
                    rerank_url,
                    json={
                        "model": RERANKER_MODEL,
                        "query": query,
                        "documents": [doc[:512] for doc in documents],
                        "top_n": len(documents),
                    },
                    timeout=RERANKER_TIMEOUT,
                )
                r.raise_for_status()
                data = r.json()
                results = data.get("results", [])
                scores = [0.0] * len(documents)
                for res in results:
                    idx = res.get("index", 0)
                    score = res.get("relevance_score", 0.0)
                    if idx < len(scores):
                        scores[idx] = float(score)
                return scores
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                if attempt < RERANKER_MAX_RETRIES:
                    time.sleep(0.5 * (attempt + 1))
                    continue
            except Exception:
                break

        return [0.5] * len(documents)

    def rerank_chunks(
        self,
        query: str,
        chunks: list[dict],
        text_field: str = "text",
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank list of chunk dicts, return top_k sorted."""
        texts = [c.get(text_field, c.get("content", "")) for c in chunks]
        ranked = self.rerank(query, texts, top_k=top_k)
        result = []
        for idx, score in ranked:
            chunk = chunks[idx].copy()
            chunk["rerank_score"] = float(score)
            result.append(chunk)
        return result

    @property
    def api_format(self) -> str:
        return self._detect_api_format()

    def stats(self) -> dict:
        return {
            "lm_studio_url": RERANKER_URL,
            "model": RERANKER_MODEL,
            "api_format": self._detect_api_format(),
        }


def rerank_chunks(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    return RerankerService.get().rerank_chunks(query, chunks, top_k=top_k)
