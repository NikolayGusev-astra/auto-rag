from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeCapabilities:
    embeddings: bool
    lexical_search: bool
    reranking: bool
    query_rewrite: bool
    generation: bool
    offline: bool


class ProviderRegistry:
    def __init__(
        self,
        *,
        embeddings=None,
        lexical: bool = True,
        reranking: bool = False,
        query_rewrite: bool = False,
        generation: bool = False,
        offline: bool = True,
    ):
        self._embeddings = embeddings
        self._lexical = lexical
        self._reranking = reranking
        self._query_rewrite = query_rewrite
        self._generation = generation
        self._offline = offline

    def negotiate(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            embeddings=self._embeddings is not None,
            lexical_search=self._lexical,
            reranking=self._reranking,
            query_rewrite=self._query_rewrite,
            generation=self._generation,
            offline=self._offline,
        )

    def embedding_provider(self):
        return self._embeddings
