from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    queries: tuple[str, ...]
    domains: tuple[str, ...]
    sources: tuple[str, ...]
    include_local: bool
    include_live: bool
    include_web: bool
    max_results: int
    retrieval_budget_ms: int | None = None
    hints: dict[str, Any] = field(default_factory=dict)

    @property
    def top_k(self) -> int:
        return self.max_results


@dataclass(frozen=True)
class RoutingFeedback:
    query: str
    plan_id: str
    selected_sources: tuple[str, ...]
    successful_sources: tuple[str, ...]
    useful_document_ids: tuple[str, ...]
    result_count: int
    latency_ms: int
    agent_feedback: str | None = None
    explicit_success: bool | None = None


@dataclass(frozen=True)
class MemoryEpisode:
    id: str
    query: str
    summary: str
    route: tuple[str, ...]
    document_ids: tuple[str, ...]
    source_uris: tuple[str, ...]
    entities: tuple[str, ...]
    successful: bool | None
    created_at: datetime | None
    index_revision: str | None
    embedding_profile_id: str | None


@dataclass(frozen=True)
class MemoryEvidence:
    episode_id: str
    summary: str
    source_document_ids: tuple[str, ...]
    source_uris: tuple[str, ...]
    route: tuple[str, ...]
    score: float
    created_at: datetime | None
    embedding_profile_id: str | None
