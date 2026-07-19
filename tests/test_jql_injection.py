from unittest import mock

import pytest

import rag_core.rag_mcp_client as rag_mcp_client


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


def test_jql_backslash_escaped():
    """Security S4: trailing backslash must not eat the closing quote.

    A query word ending in '\\' previously became '\\"' after _esc, which
    broke the JQL string termination and reopened injection. Now '\\' is
    doubled before '"' is escaped.
    """
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
        c._query_rest("jira", cfg, 'word\\ "inject', 5)

    url = captured.get("url", "")
    # The backslash from the query must be escaped (doubled) before the
    # double-quote, i.e. %5C%5C, not a raw unescaped quote break.
    assert "%5C%5C" in url, f"backslash not escaped in URL: {url}"
    # A raw, unescaped stray quote from the query must NOT appear (early term).
    assert 'text~""inject"' not in url, f"JQL string terminated early: {url}"


def test_jql_and3_url_encoded():
    """Security S5: {query_and3} clause must be URL-encoded, not raw.

    The active Jira config uses jql={query_and3}; an unencoded clause
    injected spaces/quotes straight into the URL (param injection).
    """
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
            "rest_query": '/rest/api/2/search?jql={query_and3}&maxResults={max}',
        }
        c._query_rest("jira", cfg, "reset password admin", 5)

    url = captured.get("url", "")
    # quote_plus encodes spaces as '+' and quotes as %22 — both prove the
    # clause is URL-safe, not injected raw into the query string.
    assert "+AND+" in url, f"query_and3 AND clause not URL-encoded: {url}"
    assert "%22" in url, f"query_and3 quotes not encoded: {url}"
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