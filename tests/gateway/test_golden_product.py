"""Product-level retrieval tests for gateway goldens.

These exercise the full gateway pipeline (coordinator → fuse → canonical dedup)
with mock connectors, verifying real diagnostic cases including INT-6515.
"""

from __future__ import annotations

import pytest

from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.coordinator import RetrievalCoordinator
from rag_core.gateway.models import Evidence, EvidenceOrigin


def _jira_evidence(key: str, summary: str, text: str = "", score: float = 0.9) -> Evidence:
    return Evidence(
        id=f"jira:{key}",
        document_id=key,
        title=summary,
        text=text or summary,
        source="jira",
        uri=f"https://jira.example.test/browse/{key}",
        origin=EvidenceOrigin.LIVE_CORPORATE,
        retrieval_score=score,
    )


def _confluence_evidence(page_id: str, title: str, text: str = "", score: float = 0.85) -> Evidence:
    return Evidence(
        id=f"confluence:{page_id}",
        document_id=page_id,
        title=title,
        text=text or title,
        source="confluence",
        uri=f"https://wiki.example.test/pages/viewpage.action?pageId={page_id}",
        origin=EvidenceOrigin.LIVE_CORPORATE,
        retrieval_score=score,
    )


def _snapshot_evidence(doc_id: str, title: str, text: str = "", score: float = 0.7) -> Evidence:
    return Evidence(
        id=f"snapshot:{doc_id}",
        document_id=doc_id,
        title=title,
        text=text or title,
        source="local_snapshot",
        uri=None,
        origin=EvidenceOrigin.LOCAL_SNAPSHOT,
        retrieval_score=score,
    )


class MockConnector:
    """Connector that returns preconfigured evidence."""
    source = "test"

    def __init__(self, evidence: list[Evidence]) -> None:
        self._evidence = evidence

    async def health(self) -> dict:
        return {"available": True}

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        return self._evidence


# ── INT-6515: ревизия проектных артефактов ────────────────────────


@pytest.mark.asyncio
async def test_int6515_jira_exact_key_retrieval():
    """Exact Jira key query must return INT-6515."""
    jira = MockConnector([
        _jira_evidence("INT-6515", "Ревизия проектных артефактов",
                       "Провести ревизию ПМИ, РП, ТЗ для продуктов ALD Pro, ACM, AA"),
    ])
    coordinator = RetrievalCoordinator({"jira": jira})
    results = await coordinator.search(SearchRequest(query="INT-6515"))

    assert len(results) >= 1
    assert results[0].document_id == "INT-6515"
    assert "ревиз" in results[0].text.lower()
    assert results[0].source == "jira"


@pytest.mark.asyncio
async def test_int6515_cross_source_dedup():
    """Snapshot and live Jira both return INT-6515 → canonical dedup keeps one."""
    # Cross-source: snapshot stores same Jira key → now normalizes to "jira:INT-6515"
    live = MockConnector([
        _jira_evidence("INT-6515", "Ревизия проектных артефактов", score=0.9),
    ])
    snapshot = MockConnector([
        _snapshot_evidence("INT-6515", "Ревизия проектных артефактов", score=0.5),
    ])
    coordinator = RetrievalCoordinator({"jira": live, "local_snapshot": snapshot})
    results = await coordinator.search(SearchRequest(query="INT-6515"))

    # Cross-source: both normalize to "jira:INT-6515", fuse must collapse to 1
    int6515_results = [r for r in results if "6515" in r.document_id]
    assert len(int6515_results) == 1, f"cross-source dedup failed, got {len(int6515_results)} INT-6515"
    assert int6515_results[0].retrieval_score >= 0.9  # boosted by exact_id_boost


@pytest.mark.asyncio
async def test_int6515_linked_confluence_pages():
    """INT-6515 has linked Confluence pages with report and task page."""
    jira = MockConnector([
        _jira_evidence("INT-6515", "Ревизия проектных артефактов", score=0.9),
    ])
    # Confluence returns linked pages
    confluence = MockConnector([
        _confluence_evidence("682752698", "Отчёт ревизии INT-6515",
                             "Результаты ревизии: ПМИ ALD Pro 3.3.0 проверен, замечания устранены"),
        _confluence_evidence("682748311", "Страница задачи INT-6515",
                             "Сводка по задаче: ревизия проектных артефактов для продуктов"),
    ])
    coordinator = RetrievalCoordinator({"jira": jira, "confluence": confluence})
    results = await coordinator.search(SearchRequest(query="INT-6515 ревизия проектных артефактов"))

    # All three are distinct documents — all should appear
    assert len(results) >= 3
    document_ids = {r.document_id for r in results}
    assert "INT-6515" in document_ids
    assert "682752698" in document_ids
    assert "682748311" in document_ids


# ── Source coverage for multi-connector retrieval ──────────────────


@pytest.mark.asyncio
async def test_multi_source_coverage():
    """Multiple sources return evidence — all appear in results."""
    jira = MockConnector([
        _jira_evidence("SIRIUS-195479", "Ошибки КД", score=0.9),
    ])
    confluence = MockConnector([
        _confluence_evidence("639737224", "ПМИ ALD Pro 3.3.0", score=0.85),
    ])
    hub = MockConnector([
        Evidence(
            id="hub:ald_pro", document_id="astra.ald_pro",
            title="ALD Pro коллекция", text="ALD Pro коллекция",
            source="hub", origin=EvidenceOrigin.LIVE_CORPORATE,
            retrieval_score=0.8,
        ),
    ])
    snapshot = MockConnector([
        _snapshot_evidence("ald-pro", "ALD Pro продукт", score=0.7),
    ])

    coordinator = RetrievalCoordinator({
        "jira": jira, "confluence": confluence, "hub": hub, "local_snapshot": snapshot,
    })
    results = await coordinator.search(SearchRequest(query="ALD Pro документация"))

    sources = {r.source for r in results}
    assert "jira" in sources
    assert "confluence" in sources
    assert "hub" in sources
    assert "local_snapshot" in sources
    assert len(results) >= 4


@pytest.mark.asyncio
async def test_parallel_partial_source_failure():
    """One connector fails — other results are still returned."""
    class FailingConnector:
        source = "lodestone"

        async def health(self):
            return {"available": True}

        async def search_live(self, request):
            raise RuntimeError("lodestone unavailable")

    jira = MockConnector([
        _jira_evidence("INT-6515", "Ревизия проектных артефактов", score=0.9),
    ])

    coordinator = RetrievalCoordinator({"jira": jira, "lodestone": FailingConnector()})
    results = await coordinator.search(SearchRequest(query="INT-6515"))

    # Jira should still return results despite lodestone failure
    assert len(results) >= 1
    assert results[0].document_id == "INT-6515"
    assert "lodestone" in coordinator.last_failed_sources


# ── Golden set regression on known diagnostic queries ─────────────

KNOWN_DIAGNOSTIC_QUERIES = [
    ("INT-6515", "INT-6515"),
    ("INT-6515 ревизия проектных артефактов", "INT-6515"),
    ("SIRIUS-195479 ошибки КД", "SIRIUS-195479"),
    ("репликация ALD Pro рассинхрон", "SIRIUS-195479"),
    ("SIRIUS-154603 обновление ALD Pro", "SIRIUS-154603"),
    ("PRESALE-11471 ЦБ РФ обновление", "PRESALE-11471"),
    ("639737224 ПМИ ALD Pro", "639737224"),
]


@pytest.mark.parametrize("query,expected_doc_id", KNOWN_DIAGNOSTIC_QUERIES)
@pytest.mark.asyncio
async def test_diagnostic_queries_hit_expected_document(query: str, expected_doc_id: str):
    """Known diagnostic queries must hit their expected document."""
    jira = MockConnector([
        _jira_evidence("INT-6515", "Ревизия проектных артефактов", score=0.9),
        _jira_evidence("SIRIUS-195479", "Ошибки КД", score=0.9),
        _jira_evidence("SIRIUS-154603", "Обновление ALD Pro", score=0.85),
        _jira_evidence("PRESALE-11471", "ЦБ РФ обновление 3.0", score=0.8),
    ])
    confluence = MockConnector([
        _confluence_evidence("639737224", "ПМИ ALD Pro 3.3.0", score=0.85),
    ])

    coordinator = RetrievalCoordinator({"jira": jira, "confluence": confluence})
    results = await coordinator.search(SearchRequest(query=query))

    document_ids = {r.document_id for r in results}
    assert expected_doc_id in document_ids, f"Query '{query}' did not return {expected_doc_id}"
