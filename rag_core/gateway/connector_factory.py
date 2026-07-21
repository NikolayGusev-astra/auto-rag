"""Small, local connector factory for workstation gateway startup."""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from rag_core.gateway.config_schema import GatewayConfig, SourceConfig
from rag_core.gateway.adaptive.enrichment import MemvidEnricher
from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.connectors.snapshot import LocalSnapshotConnector
from rag_core.gateway.connectors.confluence_connector import ConfluenceConnector
from rag_core.gateway.connectors.hub_connector import HubConnector
from rag_core.gateway.connectors.jira_connector import JiraConnector
from rag_core.gateway.connectors.mcp_proxy import GenericMcpConnector
from rag_core.gateway.connectors.memvid_connector import MemvidConnector
from rag_core.gateway.connectors.zvec_http import ZVecHttpConnector
from rag_core.gateway.connectors.web_search import WebSearchConnector
from rag_core.gateway.connectors.searxng import SearXNGConnector
from rag_core.gateway.connectors.web_browser import CamoufoxConnector
from rag_core.gateway.connectors.lodestone_connector import LodestoneConnector
from rag_core.gateway.secrets import resolve_credential
from rag_core.gateway.sync.engine import SyncEngine


McpSessionFactory = Callable[[], Any]
_mcp_session_factories: dict[str, McpSessionFactory] = {}


def register_mcp_session_factory(server: str, factory: McpSessionFactory) -> None:
    """Register a Hermes-managed MCP session factory for one server."""
    _mcp_session_factories[server] = factory


async def discover_hermes_mcp_tools(
    server: str, session_factory: McpSessionFactory
) -> dict[str, GenericMcpConnector]:
    """Discover a Hermes MCP server's tools and expose each as a connector."""
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
        self.retrieval_config: dict[str, float] = {}


def build_connectors(
    config: GatewayConfig, *, enricher: MemvidEnricher | None = None
) -> dict[str, SourceConnector]:
    connectors = ConnectorMap()
    connectors.retrieval_config = {
        "exact_id_boost": config.exact_id_boost,
        "exact_slug_title_boost": config.exact_slug_title_boost,
    }
    engine = SyncEngine(config.knowledge_root)
    memvid_config = config.sources.get("memvid")
    if enricher is not None and (memvid_config is None or memvid_config.enabled):
        connectors["memvid"] = MemvidConnector(enricher)
    if config.local_snapshot:
        connectors["local_snapshot"] = LocalSnapshotConnector(engine, "local_snapshot")
    for name, source in config.sources.items():
        if not source.enabled or source.kind == "memvid":
            continue
        connector = _build_source(name, source, connectors.diagnostics, enricher=enricher)
        if connector is not None:
            connectors[name] = connector
    return connectors


def _build_source(
    name: str,
    source: SourceConfig,
    diagnostics: list[str],
    *,
    enricher: MemvidEnricher | None = None,
) -> SourceConnector | None:
    if source.kind not in {"jira", "confluence", "hub", "wiki", "mcp", "mcp-proxy", "zvec", "web-search", "searxng", "web-browser", "memvid", "lodestone"}:
        diagnostics.append(f"skipped {name}: unsupported connector kind {source.kind!r}")
        return None
    if source.kind == "memvid":
        if enricher is None:
            diagnostics.append(f"{name}: Memvid enricher is unavailable")
            return None
        return MemvidConnector(enricher)
    if source.kind == "web-search":
        return WebSearchConnector()
    if source.kind == "searxng":
        base_url = source.extra.get("url", "http://localhost:8888")
        return SearXNGConnector(base_url=base_url)
    if source.kind == "web-browser":
        return CamoufoxConnector()
    if source.kind == "lodestone":
        token = source.extra.get("token", "")
        return LodestoneConnector(token=token, source=name)
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
    return ConnectorStub(name, source.kind, credential, **source.extra)


def _base_url(source: SourceConfig) -> str | None:
    configured = source.extra.get("base_url")
    if isinstance(configured, str) and configured:
        return configured
    environment_key = {
        "jira": "JIRA_BASE_URL",
        "confluence": "CONFLUENCE_BASE_URL",
        "hub": "HUB_BASE_URL",
    }.get(source.kind)
    if environment_key is None:
        return None
    return os.environ.get(environment_key)
