from __future__ import annotations

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.models import Evidence, EvidenceOrigin


class MemoryConnector(SourceConnector):
    """Expose explicitly recalled episodic memory as gateway evidence."""

    source = "agent_memory"

    def __init__(self, episodes: list[dict] | None = None) -> None:
        self._episodes = episodes or []

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
