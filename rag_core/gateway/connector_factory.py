"""Small, local connector factory for workstation gateway startup."""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from rag_core.gateway.config_schema import GatewayConfig, SourceConfig
from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.connectors.snapshot import LocalSnapshotConnector
from rag_core.gateway.connectors.confluence_connector import ConfluenceConnector
from rag_core.gateway.connectors.hub_connector import HubConnector
from rag_core.gateway.connectors.jira_connector import JiraConnector
from rag_core.gateway.connectors.mcp_proxy import GenericMcpConnector
from rag_core.gateway.connectors.zvec_http import ZVecHttpConnector
from rag_core.gateway.secrets import resolve_credential
from rag_core.gateway.sync.engine import SyncEngine

McpSessionFactory = Callable[[], Any]
_mcp_session_factories: dict[str, McpSessionFactory] = {}


def register_mcp_session_factory(server: str, factory: McpSessionFactory) -> None:
    _mcp_session_factories[server] = factory


async def discover_hermes_mcp_tools(
    server: str, session_factory: McpSessionFactory
) -> dict[str, GenericMcpConnector]:
    session = session_factory()
    if hasattr(session, "__await__"):
        session = await session
    response = await session.list_tools()
    tools = response.get("tools", []) if isinstance(response, Mapping) else response.tools
    discovered: dict[str, GenericMcpConnector] = {}
    for tool in tools:
        name = tool.get("name") if isinstance(tool, Mapping) else tool.name
        if name:
            discovered[f"mcp:{server}:{name}"] = GenericMcpConnector(name, server, session_factory)
    return discovered


class ConnectorStub:
    retrieval_kind = "live"

    def __init__(self, source: str, kind: str, credential: str | None = None, **extra: Any) -> None:
        self.source = source
        self.kind = kind
        self.credential = credential
        self.extra = extra

    async def search_live(self, request: SearchRequest) -> list:
        raise NotImplementedError(f"{self.kind} connector is unavailable on this workstation")

    async def health(self) -> dict[str, object]:
        return {"source": self.source, "available": False, "detail": "offline"}


def build_connectors(config: GatewayConfig) -> dict[str, SourceConnector]:
    connectors: dict[str, SourceConnector] = {}
    diagnostics: list[str] = []
    if config.local_snapshot:
        connectors["local_snapshot"] = LocalSnapshotConnector(
            config=config, engine=SyncEngine(config=config)
        )
    for name, source in config.sources.items():
        if not source.enabled:
            diagnostics.append(f"skipped {name}: disabled")
            continue
        connector = _build_source(name, source, diagnostics)
        if connector is not None:
            connectors[name] = connector
    return connectors


def _build_source(name: str, source: SourceConfig, diagnostics: list[str]) -> SourceConnector | None:
    if source.kind not in {"jira", "confluence", "hub", "wiki", "mcp", "mcp-proxy", "zvec"}:
        diagnostics.append(f"skipped {name}: unsupported connector kind {source.kind!r}")
        return None
    if source.kind == "zvec":
        base_url = source.extra.get("url", "http://127.0.0.1:8678")
        return ZVecHttpConnector(base_url=base_url)
    if source.kind == "mcp-proxy":
        tool = source.extra.get("tool")
        server = source.extra.get("server")
        if not isinstance(tool, str) or not isinstance(server, str):
            diagnostics.append(f"{name}: mcp-proxy requires string extra.tool and extra.server")
            return None
        session_factory = _mcp_session_factories.get(server)
        if session_factory is None:
            diagnostics.append(f"{name}: Hermes MCP server {server!r} is unavailable")
            return ConnectorStub(name, source.kind, **source.extra)
        return GenericMcpConnector(tool, server, session_factory)
    try:
        credential = resolve_credential(source.credential_ref)
    except KeyError:
        credential = None
        diagnostics.append(f"{name}: credential environment variable is unavailable; source is offline")
    if source.kind in {"jira", "confluence", "hub"}:
        base_url = _base_url(source)
        if not credential or not base_url:
            missing = "credential" if not credential else "base URL"
            diagnostics.append(f"{name}: {missing} is unavailable; source is offline")
            return ConnectorStub(name, source.kind, credential, **source.extra)
        if source.kind == "jira":
            return JiraConnector(base_url, credential, source=name)
        if source.kind == "confluence":
            return ConfluenceConnector(base_url, credential, source=name)
        return HubConnector(base_url, credential, source=name)
    if source.kind == "wiki":
        return _wiki_connector(credential, diagnostics, source)
    if source.kind == "mcp":
        return ConnectorStub(name, source.kind, credential, **source.extra)
    return None


def _base_url(source: SourceConfig) -> str:
    return source.extra.get("base_url", os.environ.get(f"{source.name.upper()}_BASE_URL", ""))


def _wiki_connector(
    credential: str | None, diagnostics: list[str], source: SourceConfig
) -> SourceConnector:
    return ConnectorStub(source.name, "wiki", credential, **source.extra)
