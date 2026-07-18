import asyncio
import os
from unittest import mock

import pytest

from rag_core.rag_mcp_client import MCPClient


class _FakeResp:
    def __init__(self, headers: dict, body: dict):
        self._headers = headers
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        return self._body

    @property
    def headers(self):
        return self._headers


def _make_client():
    c = MCPClient.__new__(MCPClient)
    c.timeout = 10
    c.last_error = None
    return c


@pytest.mark.asyncio
async def test_session_id_from_header():
    """MCP spec: sessionId в заголовке Mcp-Session-Id, а не в теле.
    До фикса клиент читал тело -> пустой sid -> initialize провален."""
    c = _make_client()

    fake_resp = mock.MagicMock()
    fake_resp.headers = {"Mcp-Session-Id": "abc-123"}
    fake_resp.raise_for_status.return_value = None
    fake_resp.close.return_value = None

    with mock.patch("rag_core.rag_mcp_client.requests.Session") as sess_cls:
        sess = sess_cls.return_value
        sess.post.return_value = fake_resp
        sid, ok = c._http_mcp_init("http://x/init", {})

    assert ok is True
    assert sid == "abc-123"


@pytest.mark.asyncio
async def test_session_id_fallback_to_body():
    """Tolerant-сервер кладёт sessionId в тело JSON — тоже работает (fallback)."""
    c = _make_client()

    fake_resp = mock.MagicMock()
    fake_resp.headers = {}
    fake_resp.json.return_value = {"jsonrpc": "2.0", "result": {"sessionId": "body-sid"}}
    fake_resp.raise_for_status.return_value = None
    fake_resp.close.return_value = None

    with mock.patch("rag_core.rag_mcp_client.requests.Session") as sess_cls:
        sess = sess_cls.return_value
        sess.post.return_value = fake_resp
        sid, ok = c._http_mcp_init("http://x/init", {})

    assert ok is True
    assert sid == "body-sid"


def test_stdio_mcp_does_not_inherit_parent_secrets(monkeypatch):
    """S8: stdio MCP processes receive an allowlisted environment only."""
    captured = {}

    def fake_popen(*args, **kwargs):
        captured.update(kwargs["env"])
        raise FileNotFoundError("test binary is intentionally absent")

    monkeypatch.setenv("JIRA_PAT", "must-not-reach-child")
    monkeypatch.setattr("rag_core.rag_mcp_client.subprocess.Popen", fake_popen)

    result = MCPClient()._query_stdio(
        "test", {"command": "missing-mcp", "env": {"MCP_TOKEN": "allowed"}}, "q", 1
    )

    assert result == []
    assert "JIRA_PAT" not in captured
    assert captured["MCP_TOKEN"] == "allowed"
    assert captured["PYTHONUNBUFFERED"] == "1"
