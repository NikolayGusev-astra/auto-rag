"""Retrieval over a source's currently published local index revision."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.models import Evidence, EvidenceOrigin
from rag_core.gateway.sync.engine import SyncEngine, _await_synchronously


_TERM = re.compile(r"\w+", re.UNICODE)


class LocalSnapshotConnector(SourceConnector):
    """Search the immutable artifacts of a source's active revision."""

    source = "local_snapshot"

    def __init__(self, engine: SyncEngine, source: str) -> None:
        self._engine = engine
        self._snapshot_source = source

    async def sync_changes(self, cursor: str | None) -> object:
        raise NotImplementedError("local snapshots are synchronized through SyncEngine")

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_web: bool = False,
        query_vector: Sequence[float] | None = None,
    ) -> list[Evidence]:
        del include_web
        revision = self._engine.active_revision(self._snapshot_source)
        if revision is None:
            return []

        path = Path(revision)
        documents = {item["id"]: item for item in _read_jsonl(path / "docs.jsonl")}
        chunks = {item["id"]: item for item in _read_jsonl(path / "chunks.jsonl")}
        lexical = json.loads((path / "lexical.json").read_text(encoding="utf-8"))
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))

        terms = _TERM.findall(query.lower())
        if not terms:
            return []
        postings = [set(lexical.get(term, ())) for term in set(terms)]
        if not postings or any(not ids for ids in postings):
            return []
        candidate_ids = set.intersection(*postings)
        if not candidate_ids:
            return []

        term_counts = Counter(terms)
        lexical_scores = {
            chunk_id: sum(
                query_count * Counter(_TERM.findall(chunks[chunk_id]["text"].lower())).get(term, 0)
                for term, query_count in term_counts.items()
            )
            for chunk_id in candidate_ids
            if chunk_id in chunks
        }
        vector_scores = self._vector_scores(path, manifest, query_vector, candidate_ids)
        ranked_ids = sorted(
            lexical_scores,
            key=lambda chunk_id: (vector_scores.get(chunk_id, lexical_scores[chunk_id]), lexical_scores[chunk_id], chunk_id),
            reverse=True,
        )[:top_k]
        return [
            _evidence(chunks[chunk_id], documents[chunks[chunk_id]["document_id"]], self._snapshot_source,
                      vector_scores.get(chunk_id, lexical_scores[chunk_id]))
            for chunk_id in ranked_ids
            if chunks[chunk_id]["document_id"] in documents
        ]

    def retrieve(self, query: str, **kwargs: Any) -> list[Evidence]:
        return _await_synchronously(self.search(query, **kwargs))

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        return await self.search(request.query, top_k=request.topk, include_web=request.include_web)

    async def fetch(self, ref: object) -> object:
        raise NotImplementedError("fetch is not implemented for local snapshots")

    async def health(self) -> dict[str, object]:
        return {"source": self.source, "available": self._engine.active_revision(self._snapshot_source) is not None}

    @staticmethod
    def _vector_scores(
        path: Path,
        manifest: dict[str, Any],
        query_vector: Sequence[float] | None,
        candidate_ids: set[str],
    ) -> dict[str, float]:
        vectors_path = path / "vectors.jsonl"
        if query_vector is None or not vectors_path.is_file() or not manifest.get("embedding_profile"):
            return {}
        return {
            item["id"]: score
            for item in _read_jsonl(vectors_path)
            if item.get("id") in candidate_ids
            if (score := _cosine(query_vector, item.get("vector", ()))) is not None
        }


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    return None if denominator == 0 else sum(a * b for a, b in zip(left, right)) / denominator


def _evidence(chunk: dict[str, Any], document: dict[str, Any], source: str, score: float) -> Evidence:
    return Evidence(
        id=chunk["id"],
        document_id=chunk["document_id"],
        title=document.get("title", ""),
        text=chunk["text"],
        source=source,
        uri=document.get("uri"),
        origin=EvidenceOrigin.LOCAL_SNAPSHOT,
        retrieval_score=score,
        metadata={"chunk_id": chunk["id"]},
    )
