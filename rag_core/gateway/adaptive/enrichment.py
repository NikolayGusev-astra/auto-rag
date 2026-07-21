from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rag_core.gateway.adaptive.contracts import MemoryEpisode
from rag_core.gateway.models import Evidence, EvidenceOrigin


_TERM = re.compile(r"\w+", re.UNICODE)


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

    def search_episodes(self, query: str, topk: int) -> list[Evidence]:
        """Return episodes whose recorded query overlaps the requested terms."""
        if topk <= 0:
            return []
        query_terms = set(_TERM.findall(query.lower()))
        if not query_terms:
            return []

        matches: list[tuple[float, MemoryEpisode]] = []
        for episode in self._episodes:
            episode_terms = set(_TERM.findall(f"{episode.query} {episode.summary}".lower()))
            overlap = len(query_terms & episode_terms)
            if overlap:
                matches.append((overlap / len(query_terms), episode))
        matches.sort(key=lambda match: (match[0], match[1].created_at or datetime.min), reverse=True)
        return [
            Evidence(
                id=episode.id,
                document_id=episode.id,
                title=f"Memory: {episode.query}",
                text=episode.summary,
                source="memvid",
                uri=episode.source_uris[0] if episode.source_uris else None,
                origin=EvidenceOrigin.LOCAL_SNAPSHOT,
                retrieval_score=score,
                reranker_score=episode.reranker_score,
                metadata={
                    "episode_id": episode.id,
                    "document_ids": episode.document_ids,
                    "route": episode.route,
                },
            )
            for score, episode in matches[:topk]
        ]

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
