"""Tests for the audit-driven modules (Evidence / QueryContext / verification /
compound / runtime). These protect the contracts the code-quality audit found
unguarded.
"""
import os
from unittest import mock

from rag_core.evidence import Evidence, SourceType, TrustLevel
from rag_core.query_context import QueryContext
from rag_core.compound import detect_compound
from rag_core.runtime import RagRuntime
from rag_core import verification


def test_evidence_calibration_discounts_untrusted():
    trusted = Evidence(text="x", source_type=SourceType.LOCAL_VECTOR,
                       source_id="z", retrieval_score=0.99,
                       trust_level=TrustLevel.TRUSTED_INTERNAL)
    untrusted = Evidence(text="x", source_type=SourceType.WEB,
                         source_id="w", retrieval_score=0.99,
                         trust_level=TrustLevel.UNTRUSTED)
    assert trusted.calibrated_score == 0.99
    assert untrusted.calibrated_score < trusted.calibrated_score
    assert untrusted.calibrated_score == 0.594


def test_query_context_cache_key_isolates_tenant_and_acl():
    a = QueryContext(query="q", domain="devops", tenant_id="tenantA",
                     principal_acl_hash="abc")
    b = QueryContext(query="q", domain="devops", tenant_id="tenantB",
                     principal_acl_hash="abc")
    c = QueryContext(query="q", domain="devops", tenant_id="tenantA",
                     principal_acl_hash="xyz")
    assert a.cache_key() != b.cache_key()
    assert a.cache_key() != c.cache_key()
    # collection + index revision participate
    d = QueryContext(query="q", domain="devops", collection="depl",
                     index_revision="v2", tenant_id="tenantA",
                     principal_acl_hash="abc")
    assert d.cache_key() != a.cache_key()


def test_verification_fail_closed_on_error():
    with mock.patch.object(verification, "_CACHE", {}), \
         mock.patch("rag_core.verification.requests.post",
                    side_effect=RuntimeError("model down")):
        res = verification.verify_relevance(
            "q", [{"text": "chunk"}], enabled=True,
            url="http://x", model="m", timeout=1)
    assert res.status == verification.VerificationStatus.UNAVAILABLE
    assert res.score is None


def test_verification_relevant_on_high_score():
    class _Resp:
        def json(self):
            return {"choices": [{"message": {"content": "0.9"}}]}
    with mock.patch.object(verification, "_CACHE", {}), \
         mock.patch("rag_core.verification.requests.post",
                    return_value=_Resp()):
        res = verification.verify_relevance(
            "q", [{"text": "chunk"}], enabled=True,
            url="http://x", model="m", timeout=1)
    assert res.status == verification.VerificationStatus.RELEVANT
    assert res.score == 0.9


def test_compound_detection_routes_product_and_infra():
    subs = detect_compound("настройка ald pro postgresql репликация", {})
    domains = {s["domain"] for s in subs}
    assert "rusbitech" in domains
    assert "devops" in domains
    # non-compound single-domain query -> empty
    assert detect_compound("что такое docker", {}) == []


def test_runtime_lifecycle_and_isolation():
    rt = RagRuntime(tenant_id="t1")
    rt.cache_set("k", {"v": 1})
    assert rt.cache_get("k") == {"v": 1}
    assert rt.cache_get("missing") is None
    # separate runtime = separate cache
    rt2 = RagRuntime(tenant_id="t2")
    assert rt2.cache_get("k") is None
    rt.shutdown()
    rt2.shutdown()