import httpx
import pytest

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.connectors.zvec_http import ZVecHttpConnector
from rag_core.gateway.models import EvidenceOrigin


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_search_live_converts_zvec_response_to_local_evidence(monkeypatch):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs == {"timeout": 30, "trust_env": False}

        async def get(self, url, *, params):
            calls.append((url, params))
            return _Response(200, {"chunks": [{"id": "chunk-1", "text": "ZVec result", "score": 0.75}]})

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    connector = ZVecHttpConnector("http://zvec.test")

    result = await connector.search_live(SearchRequest(query="zvec", topk=3))

    assert calls == [("http://zvec.test/search", {"q": "zvec", "topk": 3})]
    assert result[0].id == "chunk-1"
    assert result[0].text == "ZVec result"
    assert result[0].source == "zvec"
    assert result[0].retrieval_score == 0.75
    assert result[0].origin is EvidenceOrigin.LOCAL_SNAPSHOT


@pytest.mark.asyncio
async def test_health_reports_http_success(monkeypatch):
    class Client:
        def __init__(self, **kwargs):
            pass

        async def get(self, url, *, params=None):
            assert url == "http://zvec.test/health"
            return _Response(200, {"status": "ok"})

    monkeypatch.setattr(httpx, "AsyncClient", Client)

    assert await ZVecHttpConnector("http://zvec.test").health() == {"available": True}


def test_factory_builds_zvec_connector_with_configured_url():
    from rag_core.gateway.config_schema import GatewayConfig, SourceConfig
    from rag_core.gateway.connector_factory import build_connectors

    connectors = build_connectors(
        GatewayConfig(sources={"zvec": SourceConfig(name="zvec", kind="zvec", extra={"url": "http://zvec.test"})})
    )

    assert isinstance(connectors["zvec"], ZVecHttpConnector)
    assert connectors["zvec"]._base == "http://zvec.test"
