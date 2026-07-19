"""Minimal stdio JSON-RPC gateway for MCP-compatible clients.

The project intentionally has no MCP SDK dependency; this module speaks the
small line-delimited JSON request/response protocol required by local clients.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.mcp_handlers import handle_search
from rag_core.gateway.sync.engine import SyncEngine


def _configured_connectors() -> dict[str, SourceConnector]:
    """Return configured gateway connectors.

    Connector credentials and concrete adapter selection are deployment-owned;
    an empty configuration is a valid offline gateway profile.
    """
    return {}


def _sync_engine() -> SyncEngine:
    root = Path(os.getenv("RAG_GATEWAY_SYNC_ROOT", ".auto-rag-gateway"))
    return SyncEngine(root)


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "value"):
        return value.value
    return str(value)


async def dispatch(
    message: Mapping[str, Any],
    connectors: Mapping[str, SourceConnector] | None = None,
    sync_engine: SyncEngine | None = None,
) -> dict[str, Any]:
    """Dispatch one gateway tool request and return a JSON-serializable result."""
    active_connectors = dict(
        connectors if connectors is not None else message.get("connectors", _configured_connectors())
    )
    engine = sync_engine or _sync_engine()
    method = message.get("method")
    params = message.get("params") or {}

    if method == "search":
        allowed = {"query", "topk", "domain", "collection", "include_web", "continuation_token"}
        request = SearchRequest(**{key: value for key, value in params.items() if key in allowed})
        return await handle_search(request, active_connectors)
    if method == "fetch":
        source = active_connectors[params["source"]]
        item = await source.fetch(params["ref"])
        return {"result": item}
    if method == "sync":
        source = active_connectors[params["source"]]
        revision = await engine.sync_source(source, params.get("cursor"))
        return {"source": source.source, "revision": str(revision.path), "cursor": revision.cursor}
    if method == "sync_status":
        return engine.sync_status(params["source"])
    if method == "list_sources":
        return {"sources": sorted(active_connectors)}
    if method == "source_status":
        source = active_connectors[params["source"]]
        health = await source.health()
        return {"source": source.source, "health": health}
    raise ValueError(f"unknown gateway method: {method}")


def serve_stdio(
    connectors: Mapping[str, SourceConnector] | None = None,
    sync_engine: SyncEngine | None = None,
) -> None:
    """Serve newline-delimited JSON requests until stdin closes."""
    active_connectors = dict(connectors or _configured_connectors())
    engine = sync_engine or _sync_engine()
    for line in sys.stdin:
        try:
            message = json.loads(line)
            response = asyncio.run(dispatch(message, active_connectors, engine))
            if "id" in message:
                response = {"jsonrpc": "2.0", "id": message["id"], "result": response}
        except Exception as error:
            response = {"error": {"message": str(error)}}
        print(json.dumps(response, default=_json_value), flush=True)

