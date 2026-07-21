from __future__ import annotations

import pytest

from rag_core.gateway.adaptive.enrichment import MemvidEnricher
from rag_core.gateway.config_schema import GatewayConfig, SourceConfig
from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.connectors.memvid_connector import MemvidConnector
from rag_core.gateway.coordinator import RetrievalCoordinator
from rag_core.gateway.mcp_handlers import handle_search
from rag_core.gateway.models import Evidence, EvidenceOrigin


def test_search_episodes_returns_evidence():
    enricher = MemvidEnricher()
    episode = enricher.build_episode(
        "deploy payments service",
        [
            Evidence(
                "payments#c0",
                "payments",
                "Payments deployment",
                "Deploy the payments service.",
                "local",
                reranker_score=0.8,
            )
        ],
    )
    enricher.persist_episode(episode)

    results = enricher.search_episodes("payments deploy", topk=1)

    assert len(results) == 1
    assert results[0].source == "memvid"
    assert results[0].origin is EvidenceOrigin.LOCAL_SNAPSHOT
    assert results[0].document_id == episode.id
    assert results[0].reranker_score == 0.8


def test_empty_episodes_returns_empty():
    assert MemvidEnricher().search_episodes("payments deploy", topk=5) == []


@pytest.mark.asyncio
async def test_memvid_first_in_pipeline():
    enricher = MemvidEnricher()
    connectors = build_connectors(
        GatewayConfig(
            sources={"corporate": SourceConfig(name="corporate", kind="wiki")},
        ),
        enricher=enricher,
    )

    assert list(connectors)[0] == "memvid"
    assert isinstance(connectors["memvid"], MemvidConnector)
    assert await connectors["memvid"].search_live(SearchRequest(query="anything")) == []
    assert await connectors["memvid"].health() == {"available": False}


@pytest.mark.asyncio
async def test_handler_adds_memvid_to_connectors_when_enriching():
    enricher = MemvidEnricher()
    enricher.persist_episode(enricher.build_episode("deploy payments", []))

    response = await handle_search(SearchRequest(query="deploy"), {}, enricher=enricher)

    assert response["results"][0]["source"] == "memvid"


@pytest.mark.asyncio
async def test_memvid_runs_alongside_corporate_not_short_circuit():
    enricher = MemvidEnricher()
    enricher.persist_episode(enricher.build_episode("deploy payments", []))
    corporate_called = False

    class CorporateConnector:
        source = "corporate"

        async def health(self):
            return {"available": True}

        async def search_live(self, request):
            nonlocal corporate_called
            corporate_called = True
            return []

    coordinator = RetrievalCoordinator(
        {"memvid": MemvidConnector(enricher), "corporate": CorporateConnector()}
    )
    results = await coordinator.search(SearchRequest(query="deploy payments", topk=3, include_web=False))
    assert corporate_called, "corporate must run alongside memvid, not be skipped"
    assert any(result.source == "memvid" for result in results)
