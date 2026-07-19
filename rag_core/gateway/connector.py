from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SearchRequest:
    query: str
    topk: int = 5
    domain: str | None = None
    collection: str | None = None
    include_web: bool = False
    continuation_token: str | None = None


@runtime_checkable
class SourceConnector(Protocol):
    source: str
    retrieval_kind: str = "live"

    async def search_live(self, request: SearchRequest) -> list:
        ...

    async def fetch(self, ref) -> object:
        ...

    async def sync_changes(self, cursor: str | None) -> object:
        ...

    async def health(self) -> object:
        ...
