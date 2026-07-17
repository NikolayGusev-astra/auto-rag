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

    _session_cache: dict[str, dict] = {}  # server_name -> {sid, url, headers, ts}
    _lib_cache: dict[str, str] = {}       # library_name -> library_id

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

    # ── Cache helpers ──────────────────────────────────────────────
    @classmethod
    def _get_cached_session(cls, name: str, url: str) -> str | None:
        """Get cached MCP session-id if still fresh (<5 min)."""
        entry = cls._session_cache.get(name)
        if entry and entry.get("url") == url and time.time() - entry.get("ts", 0) < 300:
            return entry.get("sid")
        return None

    @classmethod
    def _set_cached_session(cls, name: str, url: str, sid: str):
        cls._session_cache[name] = {"sid": sid, "url": url, "ts": time.time()}

    @classmethod
    def _get_cached_lib_id(cls, lib_name: str) -> str | None:
        return cls._lib_cache.get(lib_name.lower())

    @classmethod
    def _set_cached_lib_id(cls, lib_name: str, lib_id: str):
        cls._lib_cache[lib_name.lower()] = lib_id

    def _query_context7(self, name: str, cfg: dict, query: str, max_results: int) -> list[dict]:
        """Two-step Context7 query: resolve library -> query docs."""
        url = cfg["url"]
        headers = dict(cfg.get("headers", {}))
        headers.setdefault("Accept", "application/json, text/event-stream")

        # Init SSE session (cached)
        sid = self._get_cached_session(name, url)
        if not sid:
            sid, ok = self._http_mcp_init(url, headers)
            if not ok:
                self.last_error = f"{name}: init failed"
                return []
            self._set_cached_session(name, url, sid)
            h2 = dict(headers)
            h2["mcp-session-id"] = sid
            self._http_mcp_send(url, {"jsonrpc":"2.0","method":"notifications/initialized","params":{}}, h2)
        else:
            h2 = dict(headers)
            h2["mcp-session-id"] = sid

        # Step 1: Extract library name from query and resolve
        lib_id = None
        lib_candidates = self._extract_library_names(query)
        for lib in lib_candidates[:3]:
            # Check cache first
            cached = self._get_cached_lib_id(lib)
            if cached:
                lib_id = cached
                break
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
                    self._set_cached_lib_id(lib, lib_id)
                    break

        if not lib_id:
            # Fallback: try with first keyword
            first_word = query.split()[0] if query.split() else ""
            if first_word and len(first_word) > 2:
                cached = self._get_cached_lib_id(first_word)
                if cached:
                    lib_id = cached
                else:
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
                        if lib_id:
                            self._set_cached_lib_id(first_word, lib_id)

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
            sess = requests.Session()
            sess.trust_env = False
            r = sess.post(url, json=init, headers=headers, timeout=self.timeout, stream=True)
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
            sess = requests.Session()
            sess.trust_env = False
            r = sess.post(url, json=payload, headers=headers, timeout=self.timeout, stream=True)
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
        import requests
        call_id = hash(tool + str(args)) % 10000 + 3
        payload = {"jsonrpc": "2.0", "id": call_id, "method": "tools/call",
                   "params": {"name": tool, "arguments": args}}
        try:
            sess = requests.Session()
            sess.trust_env = False
            r = sess.post(url, json=payload, headers=headers, timeout=self.timeout, stream=True)
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
        # Audit S8: do NOT inherit the entire parent environment (it would
        # leak every secret — API keys, tokens — into the MCP subprocess).
        # Pass only a minimal safe baseline plus the server's own env.
        _SAFE_ENV_KEYS = (
            "PATH", "HOME", "USER", "USERNAME", "LOGNAME", "LANG", "LC_ALL",
            "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "COMSPEC", "SHELL",
            "PYTHONPATH", "LD_LIBRARY_PATH", "TERM",
        )
        base_env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
        env = {**base_env, **cfg.get("env", {}), "PYTHONUNBUFFERED": "1"}
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

        # Use cached session if available
        session_id = self._get_cached_session(name, url)
        tools_data = []
        init_ok = bool(session_id)

        # Init payload (always defined - used both for fresh init and cached session)
        init_payload = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "rag-mcp", "version": "1.0"}},
        }

        if not session_id:
            pass  # Just use the init_payload defined above

        try:
            # Create session without proxy (trust_env=False)
            session = requests.Session()
            session.trust_env = False
            resp = session.post(url, json=init_payload, headers=headers, timeout=self.timeout, stream=True)
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

        if session_id:
            self._set_cached_session(name, url, session_id)

        # Step 1b: Notify initialized + tools/list
        call_headers = dict(headers)
        if session_id:
            call_headers["mcp-session-id"] = session_id

        notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        try:
            for msg in [notif, list_req]:
                # Use no-proxy session
                sess = requests.Session()
                sess.trust_env = False
                r = sess.post(url, json=msg, headers=call_headers, timeout=self.timeout, stream=True)
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
            sess = requests.Session()
            sess.trust_env = False
            resp = sess.post(url, json=call_payload, headers=call_headers, timeout=self.timeout, stream=True)
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
        # Config from rag_config.py uses "url"; legacy configs may use "base_url".
        base_url = cfg.get("base_url") or cfg.get("url", "")
        headers = cfg.get("headers", {})
        rest_template = cfg.get("rest_query", "/rest/api/2/search?jql=text~\"{query}\"+ORDER+BY+created+DESC&maxResults={max}")
        # Поддержка {query_and3} — первые 3 значимых слова с AND вместо фразы
        # Escape backslash BEFORE double quotes to prevent JQL injection:
        # a trailing "\" would otherwise eat the closing quote (S4).
        def _esc(w: str) -> str:
            return w.replace("\\", "\\\\").replace('"', '\\"')

        query_and3 = query_first3 = query
        if "{query_and3}" in rest_template:
            words = [w for w in query.split() if len(w) > 2][:3]
            if words:
                # JQL requires AND with spaces; encode as %20AND%20, then
                # URL-encode the whole clause so it is safe inside the URL (S5).
                clause = " AND ".join(f'text~"{_esc(w)}"' for w in words)
                query_and3 = quote_plus(clause)
        elif "{query_first3}" in rest_template:
            words = [w for w in query.split() if len(w) > 2][:3]
            query_first3 = " ".join(words) if words else query[:50]
        else:
            query_first3 = query
        # JQL-инъекция: {query} идёт внутрь text~"{query}" — нужен
        # JQL double-quote + backslash escape (_esc) ДО URL-энкодинга, иначе
        # кавычки/бэкслеши в запросе ломают JQL (security MEDIUM, S4).
        query_safe = quote_plus(_esc(query))
        url = base_url + rest_template.format(
            query=query_safe,
            query_first3=quote_plus(query_first3),
            query_and3=query_and3,
            max=max_results,
        )
        try:
            sess = requests.Session()
            sess.trust_env = False
            resp = sess.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            chunks = []
            items = data.get("issues", data.get("results", []))
            for item in items[:max_results]:
                if "fields" in item:
                    # Jira format
                    key = item.get("key", "")
                    summary = item.get("fields", {}).get("summary", "")
                    desc = item.get("fields", {}).get("description", "") or ""
                    text = f"[{key}] {summary}\n{desc[:500]}"
                else:
                    # Confluence format
                    title = item.get("title", "")
                    excerpt = item.get("excerpt", "") or item.get("body", {}).get("storage", {}).get("value", "")
                    space = item.get("space", {}).get("key", "")
                    text = f"[{space}] {title}\n{excerpt[:500]}"
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