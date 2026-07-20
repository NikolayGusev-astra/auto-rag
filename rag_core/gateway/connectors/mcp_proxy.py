"""Generic adapter for search-capable tools exposed by an MCP server."""
from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.models import Evidence, EvidenceOrigin


class GenericMcpConnector(SourceConnector):
    """Wrap an arbitrary Hermes MCP tool in the gateway connector contract."""

    retrieval_kind = "live"

    def __init__(
        self,
        mcp_tool_name: str,
        mcp_server_name: str,
        mcp_session_factory: Callable[[], Any],
    ) -> None:
        self.source = f"mcp:{mcp_server_name}"
        self._tool = mcp_tool_name
        self._session = mcp_session_factory

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        session = await _open_session(self._session)
        response = await session.call_tool(
            self._tool, {"query": request.query, "topk": request.topk}
        )
        return [
            _evidence(item, self.source, self._tool, index)
            for index, item in enumerate(_tool_content(response))
            if _content_text(item)
        ]

    async def health(self) -> bool:
        try:
            session = await _open_session(self._session)
            await session.list_tools()
        except Exception:
            return False
        return True


async def _open_session(factory: Callable[[], Any]) -> Any:
    session = factory()
    if inspect.isawaitable(session):
        return await session
    return session


def _tool_content(response: Any) -> list[Any]:
    content = _get(response, "content", [])
    return content if isinstance(content, list) else []


def _content_text(item: Any) -> str:
    text = _get(item, "text", "")
    return text if isinstance(text, str) else str(text)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _evidence(item: Any, source: str, tool: str, index: int) -> Evidence:
    text = _content_text(item)
    uri = _get(item, "uri")
    identifier = hashlib.sha256(f"{source}:{tool}:{index}:{text}".encode()).hexdigest()[:16]
    return Evidence(
        id=f"{source}:{identifier}",
        document_id=f"{source}:{identifier}",
        title=tool,
        text=text,
        source=source,
        uri=uri if isinstance(uri, str) else None,
        origin=EvidenceOrigin.LIVE_CORPORATE,
        retrieval_score=1.0 / (index + 1),
        metadata={"mcp_tool": tool},
    )
