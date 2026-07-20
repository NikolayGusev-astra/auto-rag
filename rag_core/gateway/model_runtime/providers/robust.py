"""Robust embedding provider with LM Studio → CPU fallback chain."""
from __future__ import annotations

import logging
from collections.abc import Sequence

import httpx

from rag_core.gateway.model_providers import EmbeddingCapabilities

logger = logging.getLogger(__name__)


class RobustEmbeddingProvider:
    """Tries LM Studio first, falls back to local CPU, degrades gracefully."""

    def __init__(
        self,
        lm_studio_url: str = "http://localhost:1234/v1/embeddings",
        lm_studio_model: str = "text-embedding-baai-bge-m3-568m",
        expected_dim: int = 1024,
        cpu_model_id: str = "intfloat/multilingual-e5-large",
        cpu_dim: int = 1024,
    ) -> None:
        self._lm_url = lm_studio_url.rstrip("/")
        self._lm_model = lm_studio_model
        self._dim = expected_dim
        self._cpu_model_id = cpu_model_id
        self._cpu_dim = cpu_dim
        self._client: httpx.AsyncClient | None = None
        self._cpu_provider: object | None = None
        self._lm_available: bool | None = None  # None = unknown

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            provider_id="robust",
            model_id="fallback-chain",
            revision=None,
            local=False,
            offline_capable=True,  # CPU fallback
            max_batch_size=16,
            dimension=self._dim,
            normalized=True,
            similarity_metric="cosine",
        )

    async def embed_query(self, text: str) -> list[float] | None:
        # Try LM Studio
        if self._lm_available is not False:
            result = await self._try_lm_studio(text)
            if result is not None:
                return result

        # Try CPU
        if self._cpu_provider is None:
            self._cpu_provider = self._load_cpu()
        if self._cpu_provider is not None:
            try:
                return await self._cpu_provider.embed_query(text)
            except Exception as exc:
                logger.warning("CPU embedding failed: %s", exc)

        return None

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v for v in [await self.embed_query(t) for t in texts] if v is not None]

    async def _try_lm_studio(self, text: str) -> list[float] | None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._lm_url, trust_env=False, timeout=30
            )
        try:
            resp = await self._client.post(
                "/embeddings",
                json={"model": self._lm_model, "input": [text]},
            )
            resp.raise_for_status()
            vector = resp.json()["data"][0]["embedding"]
            if len(vector) != self._dim:
                logger.warning("LM Studio dim=%d != expected=%d", len(vector), self._dim)
                self._lm_available = False
                return None
            self._lm_available = True
            return vector
        except Exception as exc:
            if self._lm_available is not False:
                logger.info("LM Studio unavailable, trying CPU fallback: %s", exc)
            self._lm_available = False
            return None

    def _load_cpu(self) -> object | None:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self._cpu_model_id)
            logger.info("CPU embedding provider loaded: %s", self._cpu_model_id)

            class _CPUWrapper:
                def __init__(self, m):
                    self._m = m

                async def embed_query(self, t: str) -> list[float]:
                    return self._m.encode([t], normalize_embeddings=True)[0].tolist()

            return _CPUWrapper(model)
        except ImportError:
            logger.info("sentence-transformers not installed; CPU fallback unavailable")
        except Exception as exc:
            logger.warning("CPU fallback load failed: %s", exc)
        return None
