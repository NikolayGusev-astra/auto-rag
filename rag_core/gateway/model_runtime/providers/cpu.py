from __future__ import annotations

from collections.abc import Sequence

from rag_core.gateway.model_providers import EmbeddingCapabilities, EmbeddingProfile


def make_cpu_profile(
    model_id: str,
    dim: int,
    revision: str | None = None,
    normalized: bool = True,
    metric: str = "cosine",
    pre: str = "query-passages-v1",
) -> EmbeddingProfile:
    return EmbeddingProfile(
        provider_family="sentence-transformers",
        model_id=model_id,
        model_revision=revision,
        dimension=dim,
        normalized=normalized,
        distance_metric=metric,
        preprocessing_revision=pre,
    )


class SentenceTransformersEmbeddingProvider:
    def __init__(self, model_id: str, dim: int, revision: str | None = None):
        self._model_id = model_id
        self._dim = dim
        self._revision = revision
        self._model = None

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            provider_id="sentence-transformers",
            model_id=self._model_id,
            revision=self._revision,
            local=True,
            offline_capable=True,
            max_batch_size=32,
            dimension=self._dim,
            normalized=True,
            similarity_metric="cosine",
        )

    def _ensure(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "sentence-transformers not installed; CPU embedding provider unavailable."
                ) from error
            self._model = SentenceTransformer(self._model_id, revision=self._revision)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._ensure()
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return vectors.tolist() if hasattr(vectors, "tolist") else vectors

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]


class OnnxEmbeddingProvider:
    """Placeholder ONNX provider; concrete runtime loading is deferred."""

    def __init__(self, model_path: str, dim: int):
        self._model_path = model_path
        self._dim = dim

    @property
    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(
            provider_id="onnx",
            model_id=self._model_path,
            revision=None,
            local=True,
            offline_capable=True,
            max_batch_size=32,
            dimension=self._dim,
            normalized=True,
            similarity_metric="cosine",
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError("ONNX embedding implementation is deferred")

    async def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("ONNX embedding implementation is deferred")
