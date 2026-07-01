"""
MCP Client for RAG fallback chain — queries MCP servers (stdio/http/rest).
"""
import json
import os
import subprocess
import sys
import time
import uuid
from urllib.parse import quote_plus

import requests


class MCPClient:
    """Client for querying MCP servers in the CRAG fallback chain."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.last_error = None

    def query(self, server_name: str, config: dict, query: str, max_results: int = 3) -> list[dict]:
        server_type = config.get("type", "stdio")
        try:
            if server_name == "context7":
                return self._query_context7(server_name, config, query, max_results)
            if server_type == "stdio":
                return self._query_stdio(server_name, config, query, max_results)
            elif server_type == "http":
                return self._query_http(server_name, config, query, max_results)
            elif server_type == "rest":
                return self._query_rest(server_name, config, query, max_results)
            else:
                self.last_error = f"Unknown MCP type: {server_type}"
                return []
        except Exception as e:
            self.last_error = f"{server_name}: {e}"
            return []

    def _query_context7(self, name: str, cfg: dict, query: str, max_results: int) -> list[dict]:
        """Two-step Context7 query: resolve library -> query docs."""
        url = cfg["url"]
        headers = dict(cfg.get("headers", {}))
        headers.setdefault("Accept", "application/json, text/event-stream")

        # Init SSE session
        sid, ok = self._http_mcp_init(url, headers)
        if not ok:
            self.last_error = f"{name}: init failed"
            return []
        h2 = dict(headers)
        h2["mcp-session-id"] = sid

        # Notify
        self._http_mcp_send(url, {"jsonrpc":"2.0","method":"notifications/initialized","params":{}}, h2)

        # Step 1: Extract library name from query and resolve
        lib_id = None
        # Common library names to search for
        lib_candidates = self._extract_library_names(query)
        for lib in lib_candidates[:3]:
            result = self._http_mcp_call(url, h2, "resolve-library-id",
                {"query": query[:200], "libraryName": lib})
            if result:
                # Parse library ID from response
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
            # Fallback: try with first keyword
            first_word = query.split()[0] if query.split() else ""
            if first_word and len(first_word) > 2:
                result = self._http_mcp_call(url, h2, "resolve-library-id",
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
            self.last_error = f"{name}: could not resolve library from query"
            return []

        # Step 2: Query docs with resolved library
        result = self._http_mcp_call(url, h2, "query-docs",
            {"libraryId": lib_id, "query": query[:200]})
        if not result:
            self.last_error = f"{name}: docs query returned empty"
            return []

        chunks = self._parse_result(name, result, query)
        return chunks[:max_results]

    def _extract_library_names(self, query: str) -> list[str]:
        """Extract potential library names from query."""
        import re
        known = ["postgresql", "postgres", "redis", "nginx", "terraform", "docker",
                 "debian", "ubuntu", "salt", "systemd", "python", "ansible",
                 "hadoop", "kafka", "elasticsearch", "mongodb", "mysql",
                 "rabbitmq", "prometheus", "grafana", "kubernetes", "helm",
                 "letsencrypt", "certbot", "freeipa", "samba", "bind", "dhcpd",
                 "saltstack", "syslog-ng", "reprepro", "haproxy", "keepalived"]
        query_lower = query.lower()
        found = []
        for lib in known:
            if lib in query_lower:
                found.append(lib)
        return found[:5]

    def _http_mcp_init(self, url, headers) -> tuple[str, bool]:
        """MCP HTTP init with SSE. Returns (session_id, ok)."""
        import requests
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                          "clientInfo": {"name": "rag-mcp", "version": "1.0"}}}
        try:
            r = requests.post(url, json=init, headers=headers, timeout=self.timeout, stream=True)
            r.raise_for_status()
            sid = r.headers.get("mcp-session-id", "")
            r.close()
            return sid, bool(sid)
        except Exception as e:
            self.last_error = f"HTTP init: {e}"
            return "", False

    def _http_mcp_send(self, url, payload, headers) -> dict | None:
        """Send MCP message and read SSE response."""
        import requests
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=self.timeout, stream=True)
            r.raise_for_status()
            body = r.content
            r.close()
            # Parse SSE
            for event_block in body.split(b"\n\n"):
                if not event_block.strip():
                    continue
                for line_bytes in event_block.split(b"\n"):
                    line_bytes = line_bytes.strip()
                    if line_bytes.startswith(b"data: "):
                        try:
                            return json.loads(line_bytes[6:].decode("utf-8", errors="replace"))
                        except json.JSONDecodeError:
                            continue
            return None
        except Exception:
            return None

    def _http_mcp_call(self, url, headers, tool, args) -> dict | list | None:
        """Call MCP tool and return result content."""
        call_id = hash(tool + str(args)) % 10000 + 3
        payload = {"jsonrpc": "2.0", "id": call_id, "method": "tools/call",
                   "params": {"name": tool, "arguments": args}}
        try:
            import requests
            r = requests.post(url, json=payload, headers=headers, timeout=self.timeout, stream=True)
            r.raise_for_status()
            body = r.content
            r.close()
            for event_block in body.split(b"\n\n"):
                if not event_block.strip():
                    continue
                for line_bytes in event_block.split(b"\n"):
                    line_bytes = line_bytes.strip()
                    if line_bytes.startswith(b"data: "):
                        try:
                            d = json.loads(line_bytes[6:].decode("utf-8", errors="replace"))
                            if d.get("id") == call_id:
                                return d.get("result", {}).get("content", "")
                        except json.JSONDecodeError:
                            continue
            return None
        except Exception:
            return None

    def _query_stdio(self, name: str, cfg: dict, query: str, max_results: int) -> list[dict]:
        cmd = [cfg["command"]] + cfg.get("args", [])
        env = {**os.environ, **cfg.get("env", {}), "PYTHONUNBUFFERED": "1"}
        try:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env, text=True, bufsize=1,
            )
        except FileNotFoundError as e:
            self.last_error = f"{name}: binary not found: {e}"
            return []

        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "rag-mcp", "version": "1.0"}}}
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        tools_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        for msg in [init, notif, tools_req]:
            proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            proc.stdin.flush()

        tools = []
        start = time.time()
        while time.time() - start < self.timeout:
            line = proc.stdout.readline()
            if not line:
                break
            try:
                resp = json.loads(line.strip())
                if resp.get("id") == 2:
                    tools = resp.get("result", {}).get("tools", [])
                    break
            except json.JSONDecodeError:
                continue

        if not tools:
            proc.kill()
            self.last_error = f"{name}: no tools discovered"
            return []

        target_tool = cfg.get("query_tool", "")
        tool_def = None
        for t in tools:
            tn = t.get("name", "")
            if target_tool and tn == target_tool:
                tool_def = t
                break
        if not tool_def:
            for t in tools:
                tn = t.get("name", "")
                if any(k in tn.lower() for k in ["search", "query", "find", "get"]):
                    tool_def = t
                    break
        if not tool_def:
            tool_def = tools[0]

        call_args = cfg.get("query_args", {})
        resolved_args = {}
        for k, v in call_args.items():
            if isinstance(v, str) and "{query}" in v:
                resolved_args[k] = v.replace("{query}", query)
            else:
                resolved_args[k] = v
        if not resolved_args:
            resolved_args = {"query": query[:200]}

        call_req = {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": tool_def["name"], "arguments": resolved_args}}
        proc.stdin.write(json.dumps(call_req, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        proc.stdin.close()  # Signal EOF to MCP server

        # Read result
        result_content = ""
        start = time.time()
        while time.time() - start < self.timeout:
            line = proc.stdout.readline()
            if not line:
                break
            try:
                resp = json.loads(line.strip())
                if resp.get("id") == 3:
                    result_content = resp.get("result", {}).get("content", "")
                    break
            except json.JSONDecodeError:
                continue

        proc.kill()
        proc.wait(timeout=3)
        return self._parse_result(name, result_content, query)[:max_results]

    def _query_http(self, name: str, cfg: dict, query: str, max_results: int) -> list[dict]:
        url = cfg["url"]
        headers = dict(cfg.get("headers", {}))
        headers.setdefault("Accept", "application/json, text/event-stream")
        tool = cfg.get("query_tool", "")

        # Step 1: Initialize via SSE
        init_payload = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "rag-mcp", "version": "1.0"}},
        }
        session_id = None
        tools_data = []
        init_ok = False

        try:
            resp = requests.post(url, json=init_payload, headers=headers, timeout=self.timeout, stream=True)
            resp.raise_for_status()
            session_id = resp.headers.get("mcp-session-id", "")
            body = resp.content
            resp.close()
            # Parse SSE events
            for event_block in body.split(b"\n\n"):
                if not event_block.strip():
                    continue
                data_line = None
                for line_bytes in event_block.split(b"\n"):
                    line_bytes = line_bytes.strip()
                    if line_bytes.startswith(b"data: "):
                        data_line = line_bytes[6:]
                        break
                if data_line is None:
                    continue
                try:
                    data = json.loads(data_line.decode("utf-8", errors="replace"))
                    if data.get("id") == 1 or "serverInfo" in data.get("result", {}):
                        init_ok = True
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            self.last_error = f"{name}: init error: {e}"
            return []

        if not init_ok:
            self.last_error = f"{name}: init failed"
            return []

        # Step 1b: Notify initialized + tools/list
        call_headers = dict(headers)
        if session_id:
            call_headers["mcp-session-id"] = session_id

        notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        try:
            for msg in [notif, list_req]:
                r = requests.post(url, json=msg, headers=call_headers, timeout=self.timeout, stream=True)
                body = r.content
                r.close()
                if msg.get("id") == 2:
                    for event_block in body.split(b"\n\n"):
                        if not event_block.strip():
                            continue
                        for line_bytes in event_block.split(b"\n"):
                            line_bytes = line_bytes.strip()
                            if line_bytes.startswith(b"data: "):
                                try:
                                    d = json.loads(line_bytes[6:].decode("utf-8", errors="replace"))
                                    if d.get("id") == 2 or d.get("result", {}).get("tools") is not None:
                                        tools_data = d.get("result", {}).get("tools", [])
                                except json.JSONDecodeError:
                                    continue
        except Exception as e:
            self.last_error = f"{name}: tools/list error: {e}"
            return []

        # Find tool to call
        if not tool and tools_data:
            tool = tools_data[0].get("name", "")
        if not tool:
            return []

        # Step 2: Call tool via SSE
        call_args = cfg.get("query_args", {})
        resolved_args = {}
        for k, v in call_args.items():
            if isinstance(v, str) and "{query}" in v:
                resolved_args[k] = v.replace("{query}", query)
            else:
                resolved_args[k] = v
        # Default args for lodestone_query
        if not resolved_args and tool == "lodestone_query":
            resolved_args = {"query": query[:200], "max_results": max_results, "sources": None}
        if not resolved_args:
            resolved_args = {"query": query[:200]}

        call_payload = {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": tool, "arguments": resolved_args},
        }
        call_headers = dict(headers)
        if session_id:
            call_headers["mcp-session-id"] = session_id

        try:
            resp = requests.post(url, json=call_payload, headers=call_headers, timeout=self.timeout, stream=True)
            resp.raise_for_status()
            result_content = None
            body = resp.content
            resp.close()
            for event_block in body.split(b"\n\n"):
                if not event_block.strip():
                    continue
                data_line = None
                for line_bytes in event_block.split(b"\n"):
                    line_bytes = line_bytes.strip()
                    if line_bytes.startswith(b"data: "):
                        data_line = line_bytes[6:]
                        break
                if data_line is None:
                    continue
                try:
                    data = json.loads(data_line.decode("utf-8", errors="replace"))
                    if data.get("id") == 3:
                        result_content = data.get("result", {}).get("content", "")
                except json.JSONDecodeError:
                    continue
            return self._parse_result(name, result_content, query)[:max_results]
        except Exception as e:
            self.last_error = f"{name}: query error: {e}"
            return []

    def _query_rest(self, name: str, cfg: dict, query: str, max_results: int) -> list[dict]:
        base_url = cfg["base_url"]
        headers = cfg.get("headers", {})
        rest_template = cfg.get("rest_query", "/rest/api/2/search?jql=text~\"{query}\"+ORDER+BY+created+DESC&maxResults={max}")
        url = base_url + rest_template.format(query=quote_plus(query), max=max_results)
        try:
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            chunks = []
            for issue in data.get("issues", [])[:max_results]:
                key = issue.get("key", "")
                summary = issue.get("fields", {}).get("summary", "")
                desc = issue.get("fields", {}).get("description", "") or ""
                text = f"[{key}] {summary}\n{desc[:500]}"
                chunks.append(self._chunk(name, text, query))
            return chunks
        except Exception as e:
            self.last_error = f"{name}: REST error: {e}"
            return []

    def _parse_result(self, name: str, result_content, query: str) -> list[dict]:
        chunks = []
        if not result_content:
            return chunks
        if isinstance(result_content, list):
            for item in result_content:
                text = ""
                if isinstance(item, dict):
                    text = item.get("text", "") or item.get("content", "") or str(item)
                elif isinstance(item, str):
                    text = item
                if text and len(text) > 10:
                    chunks.append(self._chunk(name, text, query))
        elif isinstance(result_content, str):
            try:
                data = json.loads(result_content)
                if isinstance(data, list):
                    for item in data[:10]:
                        if isinstance(item, dict):
                            text = (item.get("text", "") or item.get("content", "") or
                                    item.get("body", "") or item.get("snippet", "") or str(item))
                            if len(text) > 10:
                                chunks.append(self._chunk(name, text[:1000], query))
                        elif isinstance(item, str) and len(item) > 10:
                            chunks.append(self._chunk(name, item[:1000], query))
                elif isinstance(data, dict):
                    items = (data.get("results", []) or data.get("items", []) or
                             data.get("data", []) or data.get("issues", []))
                    if isinstance(items, list):
                        for item in items[:10]:
                            chunks.append(self._chunk(name, json.dumps(item, ensure_ascii=False)[:1000], query))
                    else:
                        chunks.append(self._chunk(name, result_content[:1000], query))
            except json.JSONDecodeError:
                paragraphs = [p.strip() for p in result_content.split("\n") if p.strip() and len(p.strip()) > 20]
                for p in paragraphs[:5]:
                    chunks.append(self._chunk(name, p[:1000], query))
                if not chunks and len(result_content) > 10:
                    chunks.append(self._chunk(name, result_content[:1000], query))
        return chunks

    def _chunk(self, source: str, text: str, query: str) -> dict:
        return {
            "text": text[:1500],
            "source": f"mcp/{source}",
            "heading": f"MCP: {source}",
            "type": "mcp",
            "tags": f"mcp,{source}",
            "score": 0.90,
            "cosine_score": 0.90,
            "boost": 0.0,
            "from_mcp": True,
            "mcp_source": source,
        }


def mcp_fallback_search(query: str, mc: MCPClient, servers: dict, chain: list[str],
                         max_results: int = 3) -> list[dict]:
    """Try MCP sources in chain order. Returns on first non-empty result."""
    for server_name in chain:
        if server_name not in servers:
            continue
        cfg = servers[server_name]
        sys.stderr.write(f"  >> MCP fallback: {server_name}... ")
        sys.stderr.flush()
        chunks = mc.query(server_name, cfg, query, max_results)
        if chunks:
            sys.stderr.write(f"OK ({len(chunks)} results)\n")
            return chunks
        err = mc.last_error or "no results"
        sys.stderr.write(f"FAIL: {err[:60]}\n")
    return []