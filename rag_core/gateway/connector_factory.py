"""Small, local connector factory for workstation gateway startup."""
from __future__ import annotations

import os
from typing import Any

from rag_core.gateway.config_schema import GatewayConfig, SourceConfig
from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.connectors.snapshot import LocalSnapshotConnector
from rag_core.gateway.connectors.confluence_connector import ConfluenceConnector
from rag_core.gateway.connectors.jira_connector import JiraConnector
from rag_core.gateway.secrets import resolve_credential
from rag_core.gateway.sync.engine import SyncEngine


class ConnectorStub:
    """Offline-safe placeholder for optional live connectors not bundled locally."""

    retrieval_kind = "live"

    def __init__(self, source: str, kind: str, credential: str | None = None, **extra: Any) -> None:
        self.source = source
        self.kind = kind
        self.credential = credential
        self.extra = extra

    async def search_live(self, request: SearchRequest) -> list:
        raise NotImplementedError(f"{self.kind} connector is unavailable on this workstation")

    async def fetch(self, ref: object) -> object:
        raise NotImplementedError

    async def sync_changes(self, cursor: str | None) -> object:
        raise NotImplementedError

    async def health(self) -> dict[str, object]:
        return {"source": self.source, "available": False, "reason": f"{self.kind} connector unavailable"}


class ConnectorMap(dict[str, SourceConnector]):
    """Connector mapping with non-fatal startup notes for unsupported sources."""

    def __init__(self) -> None:
        super().__init__()
        self.diagnostics: list[str] = []


def build_connectors(config: GatewayConfig) -> dict[str, SourceConnector]:
    connectors = ConnectorMap()
    engine = SyncEngine(config.knowledge_root)
    if config.local_snapshot:
        connectors["local_snapshot"] = LocalSnapshotConnector(engine, "local_snapshot")
    for name, source in config.sources.items():
        if not source.enabled:
            continue
        connector = _build_source(name, source, connectors.diagnostics)
        if connector is not None:
            connectors[name] = connector
    return connectors


def _build_source(name: str, source: SourceConfig, diagnostics: list[str]) -> SourceConnector | None:
    if source.kind not in {"jira", "confluence", "wiki", "mcp"}:
        diagnostics.append(f"skipped {name}: unsupported connector kind {source.kind!r}")
        return None
    try:
        credential = resolve_credential(source.credential_ref)
    except KeyError:
        credential = None
        diagnostics.append(f"{name}: credential environment variable is unavailable; source is offline")
    if source.kind in {"jira", "confluence"}:
        base_url = _base_url(source)
        if not credential or not base_url:
            missing = "credential" if not credential else "base URL"
            diagnostics.append(f"{name}: {missing} is unavailable; source is offline")
            return ConnectorStub(name, source.kind, credential, **source.extra)
        if source.kind == "jira":
            return JiraConnector(base_url, credential, source=name)
        return ConfluenceConnector(base_url, credential, source=name)
    return ConnectorStub(name, source.kind, credential, **source.extra)


def _base_url(source: SourceConfig) -> str | None:
    configured = source.extra.get("base_url")
    if isinstance(configured, str) and configured:
        return configured
    environment_key = "JIRA_BASE_URL" if source.kind == "jira" else "CONFLUENCE_BASE_URL"
    return os.environ.get(environment_key)
