"""Унифицированный embedding service.

Backend priority:
  1. LM Studio на localhost:1234 (primary — есть на каждом узле федерации)
  2. sentence-transformers local (CPU fallback — когда LM Studio недоступен)
  3. Zero vectors (последний рубеж — поиск не работает, но не падает)

Features:
  - Batching с дедупликацией (не эмбеддить одинаковые тексты дважды)
  - Disk cache (sqlite) — эмбеддинги сохраняются между рестартами
  - Health check LM Studio перед использованием
  - Auto-retry на transient errors
  - CPU fallback через sentence-transformers (bge-m3, 1024d)
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from typing import Optional

import numpy as np
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

ST_FALLBACK_MODEL = os.getenv("RAG_ST_FALLBACK_MODEL", "BAAI/bge-m3")
ST_FALLBACK_DEVICE = os.getenv("RAG_ST_FALLBACK_DEVICE", "auto")
ST_FALLBACK_BATCH = int(os.getenv("RAG_ST_FALLBACK_BATCH", "8"))
ST_FALLBACK_ENABLED = os.getenv("RAG_ST_FALLBACK_ENABLED", "true").lower() == "true"


class EmbeddingService:
    """Singleton embedding service.

    Backend priority:
      1. LM Studio HTTP (если доступен)
      2. sentence-transformers local (CPU/GPU fallback)
      3. Zero vectors (последний рубеж)
    """
    _instance: Optional["EmbeddingService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._cache_conn: Optional[sqlite3.Connection] = None
        self._cache_lock = threading.Lock()
        self._lm_studio_available: Optional[bool] = None
        self._last_health_check = 0.0
        self._st_model = None
        self._st_device = None
        self._st_load_attempted = False
        self._init_cache()

    @classmethod
    def get(cls) -> "EmbeddingService":
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
            return np.frombuffer(row[0], dtype=np.float32).tolist()
        return None

    def _cache_put(self, text: str, emb: list[float]) -> None:
        if all(v == 0.0 for v in emb):
            return
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
            new_embs = self._embed_uncached(to_embed)
            for t, emb in zip(to_embed, new_embs):
                text_to_emb[t] = emb
                self._cache_put(t, emb)

        return [text_to_emb[t] for t in texts]

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Embed texts — LM Studio primary, sentence-transformers fallback."""
        if self.check_lm_studio():
            try:
                return self._embed_via_lm_studio(texts)
            except Exception as e:
                print(f"[embedding] LM Studio failed: {e}, trying local fallback")

        if ST_FALLBACK_ENABLED:
            try:
                return self._embed_via_sentence_transformers(texts)
            except Exception as e:
                print(f"[embedding] sentence-transformers fallback failed: {e}")

        print("[embedding] All backends failed, returning zero vectors")
        return [[0.0] * EMBEDDING_DIM for _ in texts]

    def _embed_via_lm_studio(self, texts: list[str]) -> list[list[float]]:
        """HTTP к LM Studio с retry."""
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
                raise
            except Exception as e:
                raise

        raise RuntimeError(f"LM Studio failed after {EMBEDDING_MAX_RETRIES} retries: {last_error}")

    def _embed_via_sentence_transformers(self, texts: list[str]) -> list[list[float]]:
        """CPU/GPU fallback через sentence-transformers."""
        if self._st_model is None:
            if self._st_load_attempted:
                raise RuntimeError("sentence-transformers model load previously failed")

            self._st_load_attempted = True
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                if ST_FALLBACK_DEVICE == "auto":
                    self._st_device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self._st_device = ST_FALLBACK_DEVICE

                print(f"[embedding] Loading sentence-transformers {ST_FALLBACK_MODEL} on {self._st_device}...")
                t0 = time.time()
                self._st_model = SentenceTransformer(
                    ST_FALLBACK_MODEL,
                    device=self._st_device,
                )
                print(f"[embedding] Loaded in {time.time()-t0:.1f}s")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as e:
                self._st_model = None
                raise RuntimeError(f"Failed to load sentence-transformers: {e}")

        all_embs = []
        for i in range(0, len(texts), ST_FALLBACK_BATCH):
            batch = texts[i:i + ST_FALLBACK_BATCH]
            embs = self._st_model.encode(
                batch,
                batch_size=len(batch),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embs.extend(embs.tolist())
        return all_embs

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    @property
    def backend(self) -> str:
        """Текущий backend: 'lm_studio' | 'sentence_transformers' | 'none'."""
        if self.check_lm_studio():
            return "lm_studio"
        if self._st_model is not None:
            return "sentence_transformers"
        return "none"

    def stats(self) -> dict:
        cur = self._cache_conn.execute(
            "SELECT COUNT(*), SUM(LENGTH(embedding)) FROM embeddings WHERE model=?",
            (EMBEDDING_MODEL,),
        )
        count, total_size = cur.fetchone()
        return {
            "backend": self.backend,
            "lm_studio_available": self.check_lm_studio(),
            "lm_studio_url": EMBEDDING_URL,
            "model": EMBEDDING_MODEL,
            "st_fallback_enabled": ST_FALLBACK_ENABLED,
            "st_fallback_model": ST_FALLBACK_MODEL if ST_FALLBACK_ENABLED else None,
            "st_fallback_device": self._st_device,
            "cached_count": count or 0,
            "cache_size_mb": round((total_size or 0) / (1024 * 1024), 1),
        }


def get_embedding(text: str) -> list[float]:
    return EmbeddingService.get().embed(text)


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    return EmbeddingService.get().embed_batch(texts)
