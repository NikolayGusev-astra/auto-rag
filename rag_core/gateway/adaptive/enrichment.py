from __future__ import annotations

from rag_core.gateway.adaptive.contracts import MemoryEpisode
from rag_core.gateway.models import Evidence


class MemvidEnricher:
    def __init__(self) -> None:
        self._episodes: list[MemoryEpisode] = []

    @property
    def episodes(self) -> tuple[MemoryEpisode, ...]:
        return tuple(self._episodes)

    def build_episode(
        self,
        query: str,
        evidence: list[Evidence],
        *,
        successful: bool | None = None,
        index_revision: str | None = None,
        embedding_profile_id: str | None = None,
    ) -> MemoryEpisode:
        episode = MemoryEpisode(
            id=f"ep-{abs(hash(query))}",
            query=query,
            summary=query[:200],
            route=tuple(sorted({item.source for item in evidence})),
            document_ids=tuple(item.document_id for item in evidence),
            source_uris=tuple(item.uri for item in evidence if item.uri),
            entities=(),
            successful=successful,
            created_at=None,
            index_revision=index_revision,
            embedding_profile_id=embedding_profile_id,
        )
        return episode

    def persist_episode(self, episode: MemoryEpisode) -> None:
        self._episodes.append(episode)
