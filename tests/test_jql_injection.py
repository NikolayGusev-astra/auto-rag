from unittest import mock

import pytest

import rag_mcp_client


def _make_client():
    c = rag_mcp_client.MCPClient.__new__(rag_mcp_client.MCPClient)
    c.timeout = 10
    c.last_error = None
    return c


def test_jql_injection_escaped():
    """Security MEDIUM: запрос с кавычками не ломает JQL.
    {query} идёт в text~"{query}" -> кавычки должны быть
    экранированы (_esc) до URL-кодинга."""
    captured = {}

    class _FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"issues": []}

    class _FakeSession:
        def __init__(self): self.trust_env = True
        def get(self, url, headers=None, timeout=None):
            captured["url"] = url
            return _FakeResp()

    c = _make_client()
    with mock.patch("rag_mcp_client.requests.Session", return_value=_FakeSession()):
        cfg = {
            "base_url": "http://jira.example.com",
            "rest_query": '/rest/api/2/search?jql=text~"{query}"&maxResults={max}',
        }
        c._query_rest("jira", cfg, 'вым "сломать" jql', 5)

    url = captured.get("url", "")
    # кавычки из запроса НЕ должны попасть в URL как сырые "
    assert '"сломать"' not in url, f"JQL-инъекция не экранирована: {url}"
    # _esc превращает " в \", затем quote_plus -> %5C%22
    assert "%5C%22" in url, f"escape-последовательность не найдена в URL: {url}"


def test_jql_simple_query_still_works():
    """Регрессия: обычный запрос без кавычек формирует валидный URL."""
    captured = {}

    class _FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"issues": []}

    class _FakeSession:
        def __init__(self): self.trust_env = True
        def get(self, url, headers=None, timeout=None):
            captured["url"] = url
            return _FakeResp()

    c = _make_client()
    with mock.patch("rag_mcp_client.requests.Session", return_value=_FakeSession()):
        cfg = {
            "base_url": "http://jira.example.com",
            "rest_query": '/rest/api/2/search?jql=text~"{query}"&maxResults={max}',
        }
        c._query_rest("jira", cfg, "postgresql replication", 5)

    url = captured.get("url", "")
    assert "postgresql" in url
    assert "replication" in url
    assert "%5C%22" not in url  # нет лишних escape для обычного текста
