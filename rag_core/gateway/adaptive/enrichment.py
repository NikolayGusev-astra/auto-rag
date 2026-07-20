from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rag_core.gateway.adaptive.contracts import MemoryEpisode
from rag_core.gateway.models import Evidence


class MemvidEnricher:
    def __init__(self, persist_path: Path | None = None) -> None:
        self._path = Path(persist_path) if persist_path is not None else None
        self._episodes: list[MemoryEpisode] = []
        if self._path is not None:
            self._load()

    @property
    def episodes(self) -> tuple[MemoryEpisode, ...]:
        return tuple(self._episodes)

    @property
    def path(self) -> Path | None:
        return self._path

    def build_episode(
        self,
        query: str,
        evidence: list[Evidence],
        *,
        successful: bool | None = None,
        index_revision: str | None = None,
        embedding_profile_id: str | None = None,
    ) -> MemoryEpisode:
        reranker_scores = [item.reranker_score for item in evidence if item.reranker_score is not None]
        episode = MemoryEpisode(
            id=f"ep-{abs(hash(query))}",
            query=query,
            summary=query[:200],
            route=tuple(sorted({item.source for item in evidence})),
            document_ids=tuple(item.document_id for item in evidence),
            source_uris=tuple(item.uri for item in evidence if item.uri),
            entities=(),
            successful=successful,
            created_at=datetime.now(),
            index_revision=index_revision,
            embedding_profile_id=embedding_profile_id,
            reranker_score=sum(reranker_scores) / len(reranker_scores) if reranker_scores else None,
        )
        return episode

    def persist_episode(self, episode: MemoryEpisode) -> None:
        self._episodes.append(episode)
        if self._path is not None:
            self._append_jsonl(episode)

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                for field in ("route", "document_ids", "source_uris", "entities"):
                    data[field] = tuple(data[field])
                if data["created_at"] is not None:
                    data["created_at"] = datetime.fromisoformat(data["created_at"])
                self._episodes.append(MemoryEpisode(**data))

    def _append_jsonl(self, episode: MemoryEpisode) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(episode)
        data["created_at"] = episode.created_at.isoformat() if episode.created_at else None
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")
