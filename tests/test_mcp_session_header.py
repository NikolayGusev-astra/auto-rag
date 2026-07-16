import asyncio
from unittest import mock

import pytest

from rag_v2.mcp import AsyncMCPClient


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


def _make_client(resp):
    c = AsyncMCPClient.__new__(AsyncMCPClient)
    c._session = mock.MagicMock()
    c._session.post = mock.MagicMock(return_value=resp)
    c.timeout = 10
    return c


@pytest.mark.asyncio
async def test_session_id_from_header():
    """MCP spec: sessionId в заголовке Mcp-Session-Id, а не в теле.
    До фикса клиент читал тело -> пустой sid -> initialize провален."""
    resp = _FakeResp(headers={"Mcp-Session-Id": "abc-123"}, body={"jsonrpc": "2.0", "result": {}})
    c = _make_client(resp)

    sid, ok = await c._http_mcp_init("http://x/init", {})
    assert ok is True
    assert sid == "abc-123"


@pytest.mark.asyncio
async def test_session_id_fallback_to_body():
    """Tolerant-сервер кладёт sessionId в тело JSON — тоже работает (fallback)."""
    resp = _FakeResp(headers={}, body={"jsonrpc": "2.0", "result": {"sessionId": "body-sid"}})
    c = _make_client(resp)

    sid, ok = await c._http_mcp_init("http://x/init", {})
    assert ok is True
    assert sid == "body-sid"
