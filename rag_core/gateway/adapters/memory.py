from __future__ import annotations

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.adaptive.contracts import MemoryEvidence
from rag_core.gateway.models import Evidence, EvidenceOrigin


class MemoryConnector(SourceConnector):
    """Expose explicitly recalled episodic memory as gateway evidence."""

    source = "agent_memory"
    retrieval_kind = "memory"

    def __init__(self, episodes: list[dict] | None = None) -> None:
        self._episodes = episodes or []

    def as_memory_evidence(self, idx: int) -> MemoryEvidence:
        episode = self._episodes[idx]
        return MemoryEvidence(
            episode_id=episode.get("episode_id", f"e{idx}"),
            summary=episode.get("answer", ""),
            source_document_ids=tuple(episode.get("document_ids", [])),
            source_uris=tuple(episode.get("source_uris", [])),
            route=tuple(episode.get("route", [])),
            score=float(episode.get("score", 0.0)),
            created_at=None,
            embedding_profile_id=episode.get("embedding_profile_id"),
        )

    def is_compatible(self, active_profile_id: str | None) -> bool:
        return all(
            episode.get("embedding_profile_id") in (None, active_profile_id)
            for episode in self._episodes
        )

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        return [
            Evidence(
                id=f"memory:{index}",
                document_id=f"memory:{index}",
                title="episodic",
                text=episode.get("answer", ""),
                source=self.source,
                origin=EvidenceOrigin.AGENT_MEMORY,
                retrieval_score=float(episode.get("score", 0.0)),
            )
            for index, episode in enumerate(self._episodes)
        ]

    async def fetch(self, ref: object) -> object:
        raise NotImplementedError("fetch is not implemented for memory")

    async def sync_changes(self, cursor: str | None) -> object:
        raise NotImplementedError("sync_changes is not implemented for memory")

    async def health(self) -> dict[str, object]:
        return {"source": self.source, "available": True}
