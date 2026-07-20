from rag_core.gateway.adaptive.enrichment import MemvidEnricher
from rag_core.gateway.models import Evidence, EvidenceOrigin


def test_enricher_builds_episode_with_provenance():
    evidence = [Evidence("d1#c0", "d1", "t", "x", "local", origin=EvidenceOrigin.LOCAL_SNAPSHOT, retrieval_score=0.8)]
    episode = MemvidEnricher().build_episode(
        "how to deploy", evidence, successful=True, index_revision="rev1", embedding_profile_id="profA",
    )
    assert episode.document_ids == ("d1",)
    assert episode.successful is True
    assert episode.index_revision == "rev1"


def test_enricher_excludes_credentials():
    evidence = [Evidence(
        "d1#c0", "d1", "t", "x", "local", origin=EvidenceOrigin.LOCAL_SNAPSHOT,
        retrieval_score=0.8, metadata={"secret": "TOP"},
    )]
    episode = MemvidEnricher().build_episode("q", evidence, successful=True)
    assert "secret" not in episode.summary
    assert "TOP" not in episode.summary


def test_enricher_records_reranker_score_for_routing_learning():
    evidence = [
        Evidence("d1#c0", "d1", "t", "x", "local", reranker_score=0.6),
        Evidence("d2#c0", "d2", "t", "x", "web", reranker_score=1.0),
    ]

    episode = MemvidEnricher().build_episode("q", evidence)

    assert episode.reranker_score == 0.8
