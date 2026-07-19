import re
from urllib.parse import unquote_plus

import pytest

import rag_core.rag_mcp_client as rag_mcp_client


@pytest.fixture
def client():
    return rag_mcp_client.MCPClient()


def format_rest_query(monkeypatch, client, template, query, max_results=5):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"issues": []}

    class FakeSession:
        trust_env = True

        def get(self, url, headers=None, timeout=None):
            captured["url"] = url
            return FakeResponse()

    monkeypatch.setattr(rag_mcp_client.requests, "Session", FakeSession)
    client._query_rest(
        "test",
        {"base_url": "http://example.test", "rest_query": template},
        query,
        max_results,
    )
    return captured["url"]


def test_benign_query(monkeypatch, client):
    url = format_rest_query(
        monkeypatch, client, "/rest/api/2/search?jql={query_and3}&maxResults={max}", "postgresql config"
    )
    jql = unquote_plus(url.split("jql=")[1].split("&")[0])
    assert 'text~"postgresql"' in jql
    assert 'text~"config"' in jql
    assert " AND " in jql


def test_jql_injection_attempt(monkeypatch, client):
    url = format_rest_query(
        monkeypatch, client, "/rest/api/2/search?jql={query_and3}&maxResults={max}", 'foo" OR summary!="'
    )
    jql = unquote_plus(url.split("jql=")[1].split("&")[0])
    assert '\\"' in jql
    matches = re.findall(r'text~"((?:[^"\\]|\\.)*)"', jql)
    assert jql.count('text~"') == len(matches)


def test_single_word_query(monkeypatch, client):
    url = format_rest_query(
        monkeypatch, client, "/rest/api/2/search?jql={query_and3}&maxResults={max}", "postgresql"
    )
    assert 'text~"postgresql"' in unquote_plus(url)


def test_confluence_template(monkeypatch, client):
    url = format_rest_query(
        monkeypatch,
        client,
        '/rest/api/content/search?cql=text~"{query_first3}"&limit={max}',
        "nginx reverse proxy ssl",
    )
    assert 'text~"nginx+reverse+proxy"' in url