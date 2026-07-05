"""Унифицированный embedding service через LM Studio HTTP.

Backend: LM Studio на localhost:1234 (есть на каждом узле федерации).
Features:
  - Batching с дедупликацией (не эмбеддить одинаковые тексты дважды)
  - Disk cache (sqlite) — эмбеддинги сохраняются между рестартами
  - Health check LM Studio перед использованием
  - Auto-retry на transient errors
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from typing import Optional

import requests

EMBEDDING_URL = os.getenv("RAG_EMBEDDING_URL", "http://localhost:1234/v1/embeddings")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-baai-bge-m3-568m")
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1024"))
EMBEDDING_BATCH = int(os.getenv("RAG_EMBEDDING_BATCH", "32"))
EMBEDDING_TIMEOUT = int(os.getenv("RAG_EMBEDDING_TIMEOUT", "30"))
EMBEDDING_CACHE_DB = os.getenv(
    "RAG_EMBEDDING_CACHE",
    os.path.expanduser("~/.cache/auto-rag/embeddings.sqlite"),
)
EMBEDDING_MAX_RETRIES = int(os.getenv("RAG_EMBEDDING_RETRIES", "2"))


class EmbeddingService:
    """Singleton embedding service через LM Studio HTTP."""
    _instance: Optional[EmbeddingService] = None
    _lock = threading.Lock()

    def __init__(self):
        self._cache_conn: Optional[sqlite3.Connection] = None
        self._cache_lock = threading.Lock()
        self._lm_studio_available: Optional[bool] = None
        self._last_health_check = 0.0
        self._init_cache()

    @classmethod
    def get(cls) -> EmbeddingService:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _init_cache(self) -> None:
        os.makedirs(os.path.dirname(EMBEDDING_CACHE_DB), exist_ok=True)
        self._cache_conn = sqlite3.connect(EMBEDDING_CACHE_DB, check_same_thread=False)
        self._cache_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                text_hash TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._cache_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model)"
        )
        self._cache_conn.commit()

    def check_lm_studio(self, force: bool = False) -> bool:
        """Health check LM Studio. Кешируем на 30s."""
        if not force and self._lm_studio_available is not None:
            if time.time() - self._last_health_check < 30:
                return self._lm_studio_available

        try:
            r = requests.get(
                EMBEDDING_URL.replace("/v1/embeddings", "/v1/models"),
                timeout=2,
            )
            self._lm_studio_available = r.status_code == 200
        except Exception:
            self._lm_studio_available = False
        self._last_health_check = time.time()
        return self._lm_studio_available

    def _cache_get(self, text: str) -> Optional[list[float]]:
        h = hashlib.md5(text.encode()).hexdigest()
        with self._cache_lock:
            cur = self._cache_conn.execute(
                "SELECT embedding FROM embeddings WHERE text_hash=? AND model=?",
                (h, EMBEDDING_MODEL),
            )
            row = cur.fetchone()
        if row:
            import numpy as np
            return np.frombuffer(row[0], dtype=np.float32).tolist()
        return None

    def _cache_put(self, text: str, emb: list[float]) -> None:
        import numpy as np
        h = hashlib.md5(text.encode()).hexdigest()
        with self._cache_lock:
            self._cache_conn.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, model, embedding, created_at) VALUES (?, ?, ?, ?)",
                (h, EMBEDDING_MODEL, np.array(emb, dtype=np.float32).tobytes(), time.time()),
            )
            self._cache_conn.commit()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts with dedup + caching.

        Returns: list of embeddings (same order as input).
        """
        if not texts:
            return []

        unique_texts = list(dict.fromkeys(texts))
        text_to_emb: dict[str, list[float]] = {}

        to_embed: list[str] = []
        for t in unique_texts:
            cached = self._cache_get(t)
            if cached is not None:
                text_to_emb[t] = cached
            else:
                to_embed.append(t)

        if to_embed:
            new_embs = self._embed_via_lm_studio(to_embed)
            for t, emb in zip(to_embed, new_embs):
                text_to_emb[t] = emb
                self._cache_put(t, emb)

        return [text_to_emb[t] for t in texts]

    def _embed_via_lm_studio(self, texts: list[str]) -> list[list[float]]:
        """HTTP к LM Studio с retry."""
        if not self.check_lm_studio():
            return [[0.0] * EMBEDDING_DIM for _ in texts]

        last_error = None
        for attempt in range(EMBEDDING_MAX_RETRIES + 1):
            try:
                r = requests.post(
                    EMBEDDING_URL,
                    json={"model": EMBEDDING_MODEL, "input": texts},
                    timeout=EMBEDDING_TIMEOUT,
                )
                r.raise_for_status()
                data = r.json()
                return [d["embedding"] for d in data["data"]]
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                if attempt < EMBEDDING_MAX_RETRIES:
                    time.sleep(0.5 * (attempt + 1))
                    continue
            except Exception as e:
                return [[0.0] * EMBEDDING_DIM for _ in texts]

        return [[0.0] * EMBEDDING_DIM for _ in texts]

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def stats(self) -> dict:
        cur = self._cache_conn.execute(
            "SELECT COUNT(*), SUM(LENGTH(embedding)) FROM embeddings WHERE model=?",
            (EMBEDDING_MODEL,),
        )
        count, total_size = cur.fetchone()
        return {
            "lm_studio_available": self.check_lm_studio(),
            "lm_studio_url": EMBEDDING_URL,
            "model": EMBEDDING_MODEL,
            "cached_count": count or 0,
            "cache_size_mb": round((total_size or 0) / (1024 * 1024), 1),
        }


def get_embedding(text: str) -> list[float]:
    return EmbeddingService.get().embed(text)


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    return EmbeddingService.get().embed_batch(texts)
