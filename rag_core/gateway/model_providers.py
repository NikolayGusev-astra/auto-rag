from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, kw_only=True)
class ModelCapabilities:
    provider_id: str
    model_id: str
    revision: str | None
    local: bool
    offline_capable: bool
    max_batch_size: int
    max_input_tokens: int | None = None


@dataclass(frozen=True)
class EmbeddingCapabilities(ModelCapabilities):
    dimension: int
    normalized: bool
    similarity_metric: str


@dataclass(frozen=True)
class EmbeddingProfile:
    provider_family: str
    model_id: str
    model_revision: str | None
    dimension: int
    normalized: bool
    distance_metric: str
    preprocessing_revision: str


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def capabilities(self) -> EmbeddingCapabilities: ...
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class RerankerProvider(Protocol):
    async def rerank(self, query: str, evidence: Sequence[Any], limit: int) -> list[Any]: ...


@runtime_checkable
class LanguageModelProvider(Protocol):
    async def complete(self, request: Any) -> Any: ...
