"""Live Lodestone retrieval through MCP HTTP endpoint.

Connects to Astra's internal corporate RAG (Lodestone) at
https://lodestone.ai.astra-team.ru/mcp/ using the bearer token from
~/.hermes/config.yaml.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import httpx
import yaml

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.models import Evidence, EvidenceOrigin, SyncBatch


def _load_lodestone_token() -> str:
    cfg_path = pathlib.Path.home() / ".hermes" / "config.yaml"
    if not cfg_path.exists():
        return os.environ.get("LODESTONE_TOKEN", "")
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        lode = (cfg.get("mcp_servers") or {}).get("lodestone") or {}
        headers = lode.get("headers") or {}
        return headers.get("Authorization", "").removeprefix("Bearer ")
    except Exception:
        return os.environ.get("LODESTONE_TOKEN", "")


class LodestoneConnector:
    retrieval_kind = "live"

    def __init__(self, token: str = "", source: str = "lodestone") -> None:
        self._token = token or _load_lodestone_token()
        self._url = "https://lodestone.ai.astra-team.ru/mcp/"
        self.source = source

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        if not self._token:
            return []

        session_id = await self._mcp_init()
        if not session_id:
            return []

        # Call lodestone_query tool
        result = await self._mcp_call(
            session_id, "lodestone_query",
            {"query": request.query[:200]},
        )
        if not result:
            return []

        return self._parse_results(result, request.query)

    async def _mcp_init(self) -> str:
        payload = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "auto-rag-lodestone", "version": "1.0"},
            },
        }
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=30.0) as client:
                resp = await client.post(
                    self._url, json=payload,
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                )
                resp.raise_for_status()
                sid = resp.headers.get("mcp-session-id", "")
                if not sid:
                    try:
                        body = resp.json()
                        sid = body.get("sessionId", "") or body.get("result", {}).get("sessionId", "")
                    except Exception:
                        pass
                # Send initialized notification
                if sid:
                    notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
                    await client.post(
                        self._url, json=notif,
                        headers={
                            "Authorization": f"Bearer {self._token}",
                            "Content-Type": "application/json",
                            "Accept": "application/json, text/event-stream",
                            "mcp-session-id": sid,
                        },
                    )
                return sid or ""
        except Exception:
            return ""

    async def _mcp_call(self, session_id: str, tool: str, args: dict[str, Any]) -> str:
        payload = {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        }
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=60.0) as client:
                resp = await client.post(
                    self._url, json=payload,
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                        "mcp-session-id": session_id,
                    },
                )
                resp.raise_for_status()
                body = resp.content
                # Parse SSE
                for event_block in body.split(b"\n\n"):
                    if not event_block.strip():
                        continue
                    for line_bytes in event_block.split(b"\n"):
                        line_bytes = line_bytes.strip()
                        if line_bytes.startswith(b"data: "):
                            try:
                                d = json.loads(line_bytes[6:].decode("utf-8", errors="replace"))
                                if d.get("id") == 2:
                                    result = d.get("result", {})
                                    content = result.get("content", "")
                                    if isinstance(content, list):
                                        return "\n\n".join(
                                            (c.get("text", "") if isinstance(c, dict) else str(c))
                                            for c in content
                                        )
                                    return str(content)
                            except json.JSONDecodeError:
                                continue
                # Fallback: try JSON body directly
                try:
                    d = resp.json()
                    result = d.get("result", {})
                    content = result.get("content", "")
                    if isinstance(content, list):
                        return "\n\n".join(
                            (c.get("text", "") if isinstance(c, dict) else str(c))
                            for c in content
                        )
                    return str(content)
                except Exception:
                    return ""
        except Exception:
            return ""

    def _parse_results(self, raw: str, query: str) -> list[Evidence]:
        if not raw.strip():
            return []
        # Single evidence document for the Lodestone response
        return [
            Evidence(
                id=f"lodestone:{hash(query) & 0x7FFFFFFF}",
                document_id=str(hash(query) & 0x7FFFFFFF),
                title=f"Lodestone: {query[:80]}",
                text=raw[:8000],
                source="lodestone",
                uri="https://lodestone.ai.astra-team.ru/",
                origin=EvidenceOrigin.LIVE_CORPORATE,
                metadata={},
            )
        ]

    async def health(self) -> dict[str, object]:
        if not self._token:
            return {"source": self.source, "available": False, "reason": "no token"}
        try:
            sid = await self._mcp_init()
            if sid:
                return {"source": self.source, "available": True}
            return {"source": self.source, "available": False, "reason": "init failed"}
        except Exception as exc:
            return {"source": self.source, "available": False, "reason": str(exc)}

    async def sync_changes(self, cursor: str | None) -> SyncBatch:
        del cursor
        return SyncBatch(added=[])

    async def fetch(self, ref: object) -> object:
        del ref
        raise NotImplementedError
