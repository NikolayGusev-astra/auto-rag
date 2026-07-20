from __future__ import annotations

from collections.abc import Sequence

import httpx

from rag_core.gateway.model_providers import EmbeddingCapabilities


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        expected_dim: int,
        api_key: str | None = None,
    ):
        self._base = base_url.rstrip("/")
        self._endpoint = "" if self._base.endswith("/embeddings") else "/embeddings"
        self._model = model
        self._dim = expected_dim
        self._api_key = api_key
        self._client = None

    def _ensure(self) -> None:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            self._client = httpx.AsyncClient(base_url=self._base, headers=headers, trust_env=False, timeout=30)

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            provider_id="openai-compatible",
            model_id=self._model,
            revision=None,
            local=False,
            offline_capable=False,
            max_batch_size=16,
            dimension=self._dim,
            normalized=True,
            similarity_metric="cosine",
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed_query(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        self._ensure()
        response = await self._client.post(
            self._endpoint, json={"model": self._model, "input": [text]}
        )
        response.raise_for_status()
        vector = response.json()["data"][0]["embedding"]
        if len(vector) != self._dim:
            raise ValueError(
                f"Embedding dim {len(vector)} != expected {self._dim} for model {self._model}. "
                "Refusing to use incompatible index."
            )
        return vector
