"""MCP stdio gateway backed by the official MCP Python SDK.

The default command starts an MCP server.  ``--legacy-jsonl`` retains the
pre-SDK newline-delimited protocol for local debugging only.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.stdio import stdio_server  # Imported from the official transport module.
except ImportError as error:  # pragma: no cover - exercised in installations without the extra
    FastMCP = None  # type: ignore[assignment,misc]
    stdio_server = None  # type: ignore[assignment,misc]
    _MCP_IMPORT_ERROR = error
else:
    _MCP_IMPORT_ERROR = None

from rag_core.gateway.adaptive.enrichment import MemvidEnricher
from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.config_loader import load_config
from rag_core.gateway.connector_factory import build_connectors
from rag_core.gateway.mcp_handlers import handle_search
from rag_core.gateway.model_runtime.providers import OpenAICompatibleEmbeddingProvider
from rag_core.gateway.rerank_adapter import RerankAdapter
from rag_core.gateway.scheduler import apply_low_cpu_priority
from rag_core.gateway.sync.engine import SyncEngine


MCP_SDK_INSTALL_MESSAGE = "MCP transport requires the official MCP SDK. Install it with: pip install 'auto-rag[gateway]'"


def _factory_connectors(
    config_path: Path | None = None, *, enricher: MemvidEnricher | None = None
) -> dict[str, SourceConnector]:
    return build_connectors(load_config(config_path), enricher=enricher)


def _sync_engine() -> SyncEngine:
    root = Path(os.getenv("RAG_GATEWAY_SYNC_ROOT", ".auto-rag-gateway"))
    return SyncEngine(root)


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "value"):
        return value.value
    return str(value)


async def dispatch_legacy(
    message: Mapping[str, Any],
    connectors: Mapping[str, SourceConnector] | None = None,
    sync_engine: SyncEngine | None = None,
) -> dict[str, Any]:
    """Dispatch one request from the deprecated JSON-lines debug protocol."""
    active_connectors = dict(
        connectors if connectors is not None else message.get("connectors", _factory_connectors())
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
        return {"result": await source.fetch(params["ref"])}
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
        return {"source": source.source, "health": await source.health()}
    raise ValueError(f"unknown gateway method: {method}")


# Compatibility for callers that explicitly use the old in-process dispatcher.
dispatch = dispatch_legacy


def serve_legacy_jsonl(
    connectors: Mapping[str, SourceConnector] | None = None,
    sync_engine: SyncEngine | None = None,
    config_path: Path | None = None,
) -> None:
    """Serve the deprecated newline-delimited debug protocol until stdin closes."""
    active_connectors = dict(connectors if connectors is not None else _factory_connectors(config_path))
    engine = sync_engine or _sync_engine()
    for line in sys.stdin:
        try:
            message = json.loads(line)
            response = asyncio.run(dispatch_legacy(message, active_connectors, engine))
            if "id" in message:
                response = {"jsonrpc": "2.0", "id": message["id"], "result": response}
        except Exception as error:
            response = {"error": {"message": str(error)}}
        print(json.dumps(response, default=_json_value), flush=True)


def create_mcp_server(
    connectors: Mapping[str, SourceConnector] | None = None,
    sync_engine: SyncEngine | None = None,
    config_path: Path | None = None,
) -> FastMCP:
    """Create the SDK-managed MCP server and register gateway tools."""
    if FastMCP is None:
        raise ImportError(MCP_SDK_INSTALL_MESSAGE) from _MCP_IMPORT_ERROR

    enricher = MemvidEnricher(Path(os.getenv("RAG_ENRICHMENT_PATH", ".auto-rag-gateway/episodes.jsonl")))
    active_connectors = dict(
        connectors if connectors is not None else _factory_connectors(config_path, enricher=enricher)
    )
    engine = sync_engine or _sync_engine()
    embedding_url = os.getenv(
        "EMBED_URL", os.getenv("RAG_EMBED_URL", "http://localhost:1234/v1/embeddings")
    )
    embedding_model = os.getenv("EMBED_MODEL", os.getenv("RAG_EMBED_MODEL", "bge-m3"))
    cpu_model = os.getenv("CPU_EMBED_MODEL", "intfloat/multilingual-e5-large")
    from rag_core.gateway.model_runtime.providers.robust import RobustEmbeddingProvider
    embedding_provider = RobustEmbeddingProvider(
        lm_studio_url=embedding_url,
        lm_studio_model=embedding_model,
        expected_dim=1024,
        cpu_model_id=cpu_model,
        cpu_dim=1024,
    )
    reranker = RerankAdapter(embedding_provider)
    server = FastMCP("auto-rag-gateway")

    @server.tool()
    async def search(query: str, top_k: int = 5, include_web: bool = False) -> dict[str, object]:
        """Retrieve evidence for a query from configured gateway connectors."""
        request = SearchRequest(query=query, topk=top_k, include_web=include_web)
        return await handle_search(
            request, active_connectors, enricher=enricher, reranker=reranker
        )

    @server.tool()
    async def sync(source: str) -> dict[str, str | None]:
        """Synchronize one configured source into its local gateway index."""
        connector = active_connectors[source]
        revision = await engine.sync_source(connector)
        return {"source": connector.source, "revision": str(revision.path), "cursor": revision.cursor}

    return server


def serve_mcp_stdio(
    connectors: Mapping[str, SourceConnector] | None = None,
    sync_engine: SyncEngine | None = None,
    config_path: Path | None = None,
) -> None:
    """Start the standard MCP stdio transport provided by the SDK."""
    create_mcp_server(connectors, sync_engine, config_path).run(transport="stdio")


def main() -> None:
    """Start the MCP server, or the explicitly requested legacy debug transport."""
    parser = argparse.ArgumentParser(description="Run the auto-rag gateway over MCP stdio.")
    parser.add_argument(
        "--legacy-jsonl",
        action="store_true",
        help="run the deprecated newline-delimited JSON debug transport",
    )
    parser.add_argument("--config", type=Path, help="path to local gateway TOML configuration")
    args = parser.parse_args()
    if os.getenv("RAG_GATEWAY_LOW_CPU", "false").lower() in {"1", "true", "yes"}:
        apply_low_cpu_priority()
    if args.legacy_jsonl:
        serve_legacy_jsonl(config_path=args.config)
        return
    serve_mcp_stdio(config_path=args.config)


if __name__ == "__main__":
    main()
