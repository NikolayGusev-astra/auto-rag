from unittest.mock import AsyncMock, patch

import httpx
import pytest

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.connectors.hub_connector import HubConnector
from rag_core.gateway.models import EvidenceOrigin


@pytest.mark.asyncio
async def test_search_live_maps_collections_to_latest_versions():
    collections = httpx.Response(
        200,
        request=httpx.Request("GET", "https://hub.example.test/api/galaxy/v3/collections/"),
        json={"data": [{"namespace": {"name": "astra"}, "name": "network"}]},
    )
    versions = httpx.Response(
        200,
        request=httpx.Request(
            "GET", "https://hub.example.test/api/galaxy/v3/plugin/ansible/content/published/astra/network/"
        ),
        json={"data": [{"version": "2.4.0"}]},
    )
    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(side_effect=[collections, versions])) as get:
        result = await HubConnector("https://hub.example.test/", "secret").search_live(
            SearchRequest(query="network", topk=3)
        )

    assert get.await_args_list[0].kwargs == {"params": {"namespace": "astra", "name__icontains": "network"}}
    assert result[0].document_id == "astra.network"
    assert result[0].title == "network"
    assert result[0].text == "collection: network latest: 2.4.0"
    assert result[0].uri == "https://hub.example.test/ui/repo/published/astra/network"
    assert result[0].source == "hub"
    assert result[0].origin is EvidenceOrigin.LIVE_CORPORATE


@pytest.mark.asyncio
async def test_health_reports_unavailable_when_request_fails():
    with patch.object(httpx.AsyncClient, "get", new=AsyncMock(side_effect=httpx.HTTPError("offline"))):
        health = await HubConnector("https://hub.example.test", "secret").health()

    assert health == {"source": "hub", "available": False, "reason": "offline"}


def test_factory_builds_hub_from_environment(monkeypatch):
    from rag_core.gateway.config_schema import GatewayConfig, SourceConfig
    from rag_core.gateway.connector_factory import build_connectors

    monkeypatch.setenv("HUB_TOKEN", "from-env")
    monkeypatch.setenv("HUB_BASE_URL", "https://hub.example.test")
    connectors = build_connectors(
        GatewayConfig(sources={"hub": SourceConfig(name="hub", kind="hub", credential_ref="env:HUB_TOKEN")})
    )

    assert isinstance(connectors["hub"], HubConnector)
    assert connectors["hub"]._token == "from-env"


def test_factory_falls_back_to_stub_without_hub_credentials(monkeypatch):
    from rag_core.gateway.config_schema import GatewayConfig, SourceConfig
    from rag_core.gateway.connector_factory import ConnectorStub, build_connectors

    monkeypatch.delenv("HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUB_BASE_URL", raising=False)
    connectors = build_connectors(
        GatewayConfig(sources={"hub": SourceConfig(name="hub", kind="hub", credential_ref="env:HUB_TOKEN")})
    )

    assert isinstance(connectors["hub"], ConnectorStub)
