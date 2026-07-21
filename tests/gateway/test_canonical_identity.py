from dataclasses import replace

from rag_core.gateway.boosting import apply_exact_match_boost
from rag_core.gateway.deduplication import deduplicate_evidence
from rag_core.gateway.models import Evidence, EvidenceOrigin


def _evidence(
    document_id: str,
    *,
    source: str = "local_snapshot",
    origin: EvidenceOrigin = EvidenceOrigin.LOCAL_SNAPSHOT,
    canonical_id: str | None = None,
    title: str = "Runbook",
    metadata: dict | None = None,
    score: float = 0.5,
) -> Evidence:
    return Evidence(
        id=f"{source}:{document_id}",
        document_id=document_id,
        canonical_id=canonical_id,
        title=title,
        text="body",
        source=source,
        origin=origin,
        retrieval_score=score,
        metadata=metadata or {},
    )


def test_canonical_jira():
    evidence = _evidence("PROJ-123", source="jira", origin=EvidenceOrigin.LIVE_CORPORATE)

    assert evidence.canonical_id == "jira:PROJ-123"


def test_canonical_confluence():
    evidence = _evidence("123456", source="confluence", origin=EvidenceOrigin.LIVE_CORPORATE)

    assert evidence.canonical_id == "confluence:123456"


def test_canonical_wiki():
    evidence = _evidence("ignored", source="wiki", metadata={"slug": "gateway-runbook"})

    assert evidence.canonical_id == "wiki:gateway-runbook"


def test_dedup_same_canonical():
    snapshot = _evidence("PROJ-123", canonical_id="jira:PROJ-123", score=0.9)
    live = _evidence(
        "PROJ-123",
        source="jira",
        origin=EvidenceOrigin.LIVE_CORPORATE,
        canonical_id="jira:PROJ-123",
        score=0.1,
    )

    results = deduplicate_evidence([snapshot, live])

    assert len(results) == 1
    assert results[0].source == "jira"


def test_dedup_preserves_alternate_sources():
    snapshot = _evidence("PROJ-123", canonical_id="jira:PROJ-123", metadata={"chunk_id": "c1"})
    live = _evidence(
        "PROJ-123",
        source="jira",
        origin=EvidenceOrigin.LIVE_CORPORATE,
        canonical_id="jira:PROJ-123",
        metadata={"updated": "today"},
    )

    result = deduplicate_evidence([snapshot, live])[0]

    assert result.metadata["updated"] == "today"
    assert result.metadata["alternate_sources"] == ("local_snapshot",)
    assert result.metadata["alternate_metadata"] == ({"chunk_id": "c1"},)


def test_dedup_different_canonical():
    first = _evidence("PROJ-123", canonical_id="jira:PROJ-123")
    second = _evidence("PROJ-124", canonical_id="jira:PROJ-124")

    assert deduplicate_evidence([first, second]) == [first, second]


def test_boost_exact_id():
    evidence = _evidence("ALD-PRO", canonical_id="hub:ald-pro")

    result = apply_exact_match_boost("ald_pro", evidence)

    assert result.retrieval_score == 1.5


def test_boost_exact_slug():
    evidence = _evidence("ignored", metadata={"slug": "gateway-runbook"})

    result = apply_exact_match_boost("gateway_runbook", evidence)

    assert result.retrieval_score == 1.2


def test_boost_no_single_word():
    evidence = _evidence("ignored", title="Runbook")

    assert apply_exact_match_boost("runbook", evidence) == evidence


def test_boost_query_normalization():
    evidence = _evidence("ALD-PRO", canonical_id="hub:ald-pro")

    assert apply_exact_match_boost("ALD Pro", evidence).retrieval_score == 1.5
