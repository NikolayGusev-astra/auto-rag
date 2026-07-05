"""
Async MCP Client — aiohttp-based, без блокирующих requests.

Поддерживает:
  - HTTP MCP (SSE) — Context7, Lodestone
  - REST MCP — Jira (JQL), Confluence (CQL)
  - Все запросы параллельные через один aiohttp.ClientSession
"""

import json
import os
import sys
import uuid
from typing import Any

import aiohttp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rag_config import MCP_SERVERS, MCP_TIMEOUT


def server_name_from_url(url: str) -> str:
    """Извлечь имя сервера из URL."""
    from urllib.parse import urlparse
    netloc = urlparse(url).netloc
    return netloc.split(".")[0] if netloc else "mcp"


class AsyncMCPClient:
    """Async MCP client. Создаётся раз на весь pipeline, переиспользует session."""

    def __init__(self, session: aiohttp.ClientSession | None = None, timeout: int = 30):
        self.timeout = timeout
        self._own_session = session is None
        self._session: aiohttp.ClientSession | None = session

    async def __aenter__(self):
        if self._own_session:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self._own_session and self._session:
            await self._session.close()

    async def query(
        self, server_name: str, query: str, max_results: int = 5
    ) -> list[dict[str, Any]]:
        """Query MCP server by name."""
        cfg = MCP_SERVERS.get(server_name)
        if not cfg:
            return []
        
        server_type = cfg.get("type", "stdio")
        try:
            if server_name == "context7":
                return await self._query_context7(cfg, query, max_results)
            elif server_type == "http":
                return await self._query_http(cfg, query, max_results)
            elif server_type == "rest":
                return await self._query_rest(cfg, query, max_results)
            elif server_type == "stdio":
                return await self._query_stdio(cfg, query, max_results)
            return []
        except Exception as e:
            return []

    async def _query_rest(
        self, cfg: dict, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """REST API — Jira (JQL) и Confluence (CQL)."""
        url = cfg.get("url", cfg.get("base_url", ""))
        headers = dict(cfg.get("headers", {}))
        template = cfg.get("rest_query", "")
        
        formatted = self._format_query(template, query, max_results)
        full_url = url.rstrip("/") + formatted
        
        if not self._session:
            return []
        
        try:
            async with self._session.get(
                full_url, headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                data = await resp.json()
            
            chunks = []
            items = data.get("issues", data.get("results", []))
            for item in items[:max_results]:
                if "fields" in item:
                    key = item.get("key", "")
                    summary = item.get("fields", {}).get("summary", "")
                    desc = item.get("fields", {}).get("description", "") or ""
                    text = f"[{key}] {summary}\n{desc[:800]}"
                else:
                    title = item.get("title", "")
                    excerpt = item.get("excerpt", "") or item.get("body", {}).get("storage", {}).get("value", "")
                    space = item.get("space", {}).get("key", "")
                    text = f"[{space}] {title}\n{excerpt[:800]}"
                chunks.append(self._chunk(server_name_from_url(url), text, query, source=server_name_from_url(url)))
            return chunks
        except Exception:
            return []

    async def _query_http(
        self, cfg: dict, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """HTTP MCP (SSE) — Lodestone и другие SSE-серверы."""
        url = cfg.get("url", cfg.get("base_url", ""))
        headers = dict(cfg.get("headers", {}))
        headers.setdefault("Accept", "application/json, text/event-stream")
        
        if not self._session:
            return []
        
        # Init session
        sid, ok = await self._http_mcp_init(url, headers)
        if not ok:
            return []
        
        h2 = dict(headers)
        h2["mcp-session-id"] = sid
        
        # Notify
        await self._http_mcp_send(url, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}, h2)
        
        # Call tool
        tool_name = cfg.get("tool", "search")
        result = await self._http_mcp_call(url, h2, "tools/call", {
            "name": tool_name,
            "arguments": {"query": query, "limit": max_results}
        })
        
        return self._parse_mcp_result(result, url, query, max_results)

    async def _query_context7(
        self, cfg: dict, query: str, max_results: int
    ) -> list[dict[str, Any]]:
        """Context7 two-step: resolve library → query docs."""
        url = cfg.get("url", cfg.get("base_url", ""))
        headers = dict(cfg.get("headers", {}))
        headers.setdefault("Accept", "application/json, text/event-stream")
        
        if not self._session:
            return []
        
        sid, ok = await self._http_mcp_init(url, headers)
        if not ok:
            return []
        
        h2 = dict(headers)
        h2["mcp-session-id"] = sid
        await self._http_mcp_send(url, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}, h2)
        
        # Step 1: resolve library
        lib_candidates = self._extract_library(query)
        lib_id = None
        for lib in lib_candidates[:3]:
            result = await self._http_mcp_call(url, h2, "resolve-library-id",
                {"query": query[:200], "libraryName": lib})
            if result:
                import re
                ids = re.findall(r'/[a-zA-Z0-9_/-]+', str(result))
                for rid in ids:
                    rid = rid.strip().strip('.')
                    if len(rid) > 5:
                        lib_id = rid
                        break
                if lib_id:
                    break
        
        if not lib_id:
            first_word = query.split()[0] if query.split() else ""
            if first_word and len(first_word) > 2:
                result = await self._http_mcp_call(url, h2, "resolve-library-id",
                    {"query": query[:200], "libraryName": first_word})
                if result:
                    import re
                    ids = re.findall(r'/[a-zA-Z0-9_/-]+', str(result))
                    for rid in ids:
                        rid = rid.strip().strip('.')
                        if len(rid) > 5:
                            lib_id = rid
                            break
        
        if not lib_id:
            return []
        
        # Step 2: query docs
        result = await self._http_mcp_call(url, h2, "query-docs",
            {"libraryId": lib_id, "query": query[:200]})
        
        return self._parse_mcp_result(result, "context7", query, max_results)

    async def _query_stdio(self, cfg: dict, query: str, max_results: int) -> list[dict]:
        """Stdio MCP — запускаем subprocess (блокирующий, но редко)."""
        import asyncio
        cmd = cfg.get("command", [])
        if isinstance(cmd, str):
            cmd = cmd.split()
        env = os.environ.copy()
        env.update(cfg.get("env", {}))

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=env,
        )
        
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        proc.stdin.write((json.dumps(init) + "\n").encode())
        await proc.stdin.drain()
        await proc.stdout.readline()
        
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        proc.stdin.write((json.dumps(notif) + "\n").encode())
        await proc.stdin.drain()
        
        tool_name = cfg.get("tool", "search")
        call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": tool_name, "arguments": {"query": query, "limit": max_results}}}
        proc.stdin.write((json.dumps(call) + "\n").encode())
        await proc.stdin.drain()
        
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=self.timeout)
        proc.kill()
        
        try:
            resp = json.loads(line.decode())
            return self._parse_mcp_result(resp, "stdio", query, max_results)
        except Exception:
            return []

    # ── HTTP MCP helpers ──

    async def _http_mcp_init(self, url: str, headers: dict) -> tuple[str, bool]:
        """Initialize MCP session. Returns (session_id, ok)."""
        try:
            async with self._session.post(url, json={
                "jsonrpc": "2.0", "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
                "id": str(uuid.uuid4())
            }, headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)) as resp:
                data = await resp.json()
                sid = data.get("sessionId", "") or data.get("result", {}).get("sessionId", "")
                return sid, bool(sid)
        except Exception:
            return "", False

    async def _http_mcp_send(self, url: str, msg: dict, headers: dict) -> bool:
        """Send MCP message (notification)."""
        try:
            async with self._session.post(url, json=msg, headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def _http_mcp_call(self, url: str, headers: dict, method: str, params: dict) -> Any:
        """Call MCP tool method. Returns result or None."""
        msg = {"jsonrpc": "2.0", "method": method, "params": params, "id": str(uuid.uuid4())}
        headers = dict(headers)
        # SSE-серверы могут долго отвечать
        try:
            async with self._session.post(url, json=msg, headers=dict(headers),
                timeout=aiohttp.ClientTimeout(total=self.timeout)) as resp:
                data = await resp.json()
                result = data.get("result", data.get("content", data))
                # SSE может вернуть массив результатов
                if isinstance(result, list) and len(result) > 0:
                    # Может быть [{"type": "resource", "resource": {"text": "..."}}]
                    texts = []
                    for item in result:
                        if isinstance(item, dict):
                            text = item.get("text", "")
                            if not text and "resource" in item:
                                text = item["resource"].get("text", "")
                            if text:
                                texts.append(text)
                    if texts:
                        return "\n".join(texts)
                return result
        except Exception:
            return None

    def _parse_mcp_result(self, result: Any, name: str, query: str, max_results: int) -> list[dict]:
        chunks = []
        if isinstance(result, dict):
            for item in result.get("results", result.get("documents", []))[:max_results]:
                text = item.get("text", item.get("content", ""))
                if text:
                    chunks.append(self._chunk(name, text[:800], query, source=name))
        elif isinstance(result, list):
            for item in result[:max_results]:
                text = item.get("text", item.get("content", str(item))) if isinstance(item, dict) else str(item)
                if text and len(text) > 20:
                    chunks.append(self._chunk(name, text[:800], query, source=name))
        elif isinstance(result, str) and len(result) > 20:
            # Плоский текст — разбиваем на параграфы
            for para in result.split("\n\n")[:max_results]:
                if len(para.strip()) > 20:
                    chunks.append(self._chunk(name, para[:800], query, source=name))
        return chunks

    def _chunk(self, name: str, text: str, query: str, source: str = "") -> dict:
        return {
            "text": text,
            "source": source or name,
            "mcp_source": name,
            "query": query[:200],
        }

    def _format_query(self, template: str, query: str, max_results: int) -> str:
        """Format query template with {query}, {query_first3}, {query_and3}, {max}."""
        words = query.split()
        query_first3 = " ".join(words[:3]) if len(words) >= 3 else " ".join(words[:2]) if len(words) >= 2 else query
        
        # Escape quotes for JQL/CQL string literals
        def _esc(w: str) -> str:
            return w.replace('"', '\\"')
        
        # AND-формат: text~"w1" AND text~"w2" AND text~"w3"
        non_rus = [w for w in words if not any('\u0400' <= c <= '\u04FF' for c in w)]
        if len(non_rus) >= 2:
            query_and3 = " AND ".join(f'text~"{_esc(w)}"' for w in non_rus[:5])
        elif len(words) >= 2:
            query_and3 = " AND ".join(f'text~"{_esc(w)}"' for w in words[:5])
        else:
            query_and3 = f'text~"{_esc(query)}"'
        
        result = template.replace("{query_first3}", query_first3)
        result = result.replace("{query_and3}", query_and3)
        result = result.replace("{query}", query)
        result = result.replace("{max}", str(max_results))
        result = result.replace("__QUOTE__", '"')
        return result

    def _extract_library(self, query: str) -> list[str]:
        """Извлечь названия библиотек из запроса."""
        import re
        candidates = []
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_.-]{2,}\b', query)
        skip = {'the', 'and', 'for', 'with', 'how', 'what', 'setup', 'config', 'install', 'use', 'using', 'this', 'that'}
        for w in words:
            if w.lower() not in skip:
                 candidates.append(w)
        return candidates[:10]
