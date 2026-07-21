"""Live Lodestone retrieval through MCP HTTP endpoint.

Connects to Astra's internal corporate RAG at the configured endpoint.
Token is resolved via the gateway's standard credential pipeline
(``credential_ref`` in gateway.toml), NOT by reading Hermes config directly.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import httpx

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.models import Evidence, EvidenceOrigin, SyncBatch

_DEFAULT_ENDPOINT = "https://lodestone.corp.example/mcp/"


class LodestoneConnector:
    retrieval_kind = "live"

    def __init__(
        self,
        token: str = "",
        endpoint: str = "",
        source: str = "lodestone",
    ) -> None:
        self._token = token
        self._url = (endpoint or _DEFAULT_ENDPOINT).rstrip("/") + "/"
        self.source = source

    # ── search_live ────────────────────────────────────────────────

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        if not self._token:
            return []

        session_id = await self._mcp_init()
        if not session_id:
            return []

        raw = await self._mcp_call(
            session_id, "lodestone_query",
            {"query": request.query[:200]},
        )
        if not raw.strip():
            return []

        return self._parse_structured(raw, request.query)[: request.topk]

    # ── MCP protocol ───────────────────────────────────────────────

    async def _mcp_init(self) -> str:
        payload = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "auto-rag-lodestone", "version": "1.0"},
            },
        }
        hdr = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=30.0) as client:
                resp = await client.post(self._url, json=payload, headers=hdr)
                resp.raise_for_status()
                sid = resp.headers.get("mcp-session-id", "")
                if not sid:
                    try:
                        body = resp.json()
                        sid = body.get("sessionId", "") or body.get("result", {}).get("sessionId", "")
                    except Exception:
                        pass
                if sid:
                    notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
                    await client.post(
                        self._url, json=notif,
                        headers={**hdr, "mcp-session-id": sid},
                    )
                return sid or ""
        except Exception:
            return ""

    async def _mcp_call(self, session_id: str, tool: str, args: dict[str, Any]) -> str:
        payload = {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        }
        hdr = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": session_id,
        }
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=60.0) as client:
                resp = await client.post(self._url, json=payload, headers=hdr)
                resp.raise_for_status()
                body = resp.content
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
                                        return "\n".join(
                                            (c.get("text", "") if isinstance(c, dict) else str(c))
                                            for c in content
                                        )
                                    return str(content)
                            except json.JSONDecodeError:
                                continue
                # Fallback: try JSON body
                try:
                    d = resp.json()
                    result = d.get("result", {})
                    content = result.get("content", "")
                    if isinstance(content, list):
                        return "\n".join(
                            (c.get("text", "") if isinstance(c, dict) else str(c))
                            for c in content
                        )
                    return str(content)
                except Exception:
                    return ""
        except Exception:
            return ""

    # ── structured parsing ─────────────────────────────────────────

    _RESULT_RE = re.compile(
        r"###\s*Result\s+(\d+)\s*[—–-]\s*source_id:\s*(\S+)\s*[—–-]\s*score:\s*([\d.]+)",
        re.IGNORECASE,
    )
    _TITLE_RE = re.compile(r"\*\*Title:\*\*\s*(.+?)(?:\n|$)")
    _URL_RE = re.compile(r"\*\*Source URL.*?:\*\*\s*(https?://\S+)")

    def _parse_structured(self, raw: str, query: str) -> list[Evidence]:
        results: list[Evidence] = []

        # Try structured splitting
        blocks = re.split(r"\n---\n", raw)
        if len(blocks) < 2:
            # Single block — return as one evidence
            digest = _stable_digest(query)
            return [Evidence(
                id=f"lodestone:{digest}",
                document_id=digest,
                title=f"Lodestone: {query[:80]}",
                text=raw[:8000],
                source="lodestone",
                uri=self._url,
                origin=EvidenceOrigin.LIVE_CORPORATE,
                metadata={"results_count": 1},
            )]

        for block in blocks:
            if not block.strip() or "Lodestone Search Results" in block:
                continue
            result_match = self._RESULT_RE.search(block)
            if not result_match:
                continue
            result_num = result_match.group(1)
            source_id = result_match.group(2)
            title_match = self._TITLE_RE.search(block)
            title = title_match.group(1).strip() if title_match else f"Result {result_num}"
            url_match = self._URL_RE.search(block)
            uri = url_match.group(1).strip() if url_match else self._url
            digest = _stable_digest(f"{source_id}:{title}")

            # Drop instruction lines
            body = re.sub(r"\n*lodestone_document\(.*?\).*", "", block)
            body = re.sub(r"\n*To retrieve full document.*", "", body)

            results.append(Evidence(
                id=f"lodestone:{digest}",
                document_id=digest,
                title=title,
                text=body[:8000],
                source="lodestone",
                uri=uri,
                origin=EvidenceOrigin.LIVE_CORPORATE,
                metadata={
                    "lodestone_source_id": source_id,
                    "lodestone_score": float(result_match.group(3)),
                    "lodestone_result_num": int(result_num),
                },
            ))

        if not results and raw.strip():
            digest = _stable_digest(query)
            results.append(Evidence(
                id=f"lodestone:{digest}",
                document_id=digest,
                title=f"Lodestone: {query[:80]}",
                text=raw[:8000],
                source="lodestone",
                uri=self._url,
                origin=EvidenceOrigin.LIVE_CORPORATE,
                metadata={"results_count": 1},
            ))

        return results

    # ── health / sync / fetch ──────────────────────────────────────

    async def health(self) -> dict[str, object]:
        if not self._token:
            return {"source": self.source, "available": False, "reason": "no token"}
        try:
            sid = await self._mcp_init()
            if sid:
                return {"source": self.source, "available": True}
            return {"source": self.source, "available": False, "reason": "init failed"}
        except Exception as exc:
            return {"source": self.source, "available": False, "reason": str(exc)[:200]}

    async def sync_changes(self, cursor: str | None) -> SyncBatch:
        return SyncBatch(added=[])

    async def fetch(self, ref: object) -> object:
        raise NotImplementedError


def _stable_digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]
