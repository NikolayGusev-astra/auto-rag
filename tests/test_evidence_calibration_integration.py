"""Integration test: untrusted federation/web scores must not outrank
equivalent trusted-local scores after calibration (P0/P1 audit closure).

This is the concrete proof that the original risk — a remote federation
chunk with raw score 0.99 sorting above a local ZVec chunk with 0.99 — is
mitigated end-to-end through _calibrate_chunks.
"""
from rag_core import rag_async
from rag_core.evidence import Evidence, SourceType, TrustLevel


def test_federated_099_loses_to_local_099():
    """A federated chunk scored 0.99 must rank BELOW a trusted-local 0.99."""
    local = {"text": "local doc content", "score": 0.99, "source": "zvec/wiki"}
    fed = {"text": "remote doc content", "score": 0.99,
           "source": "federated:node2", "url": "https://node2/doc"}

    local_cal = rag_async._calibrate_chunks([local], "zvec", "zvec")[0]
    fed_cal = rag_async._calibrate_chunks([fed], "federated", "federated:node2")[0]

    assert local_cal["score"] == 0.99
    assert fed_cal["score"] < local_cal["score"]
    # explicit ordering proof
    assert fed_cal["score"] < 0.7  # discounted well below trusted


def test_web_06_below_trusted_07():
    web = {"text": "web page", "score": 0.6, "source": "web"}
    mcp = {"text": "mcp doc", "score": 0.7, "source": "jira"}

    web_cal = rag_async._calibrate_chunks([web], "web", "web")[0]
    mcp_cal = rag_async._calibrate_chunks([mcp], "mcp", "jira")[0]

    # both keep their fixed gate scores, but trust tags differ and the
    # calibrated values remain ordered: web(0.36) < mcp(0.7)
    assert web_cal["_trust"] == TrustLevel.UNTRUSTED.value
    assert mcp_cal["_trust"] == TrustLevel.TRUSTED_INTERNAL.value
    assert web_cal["score"] < mcp_cal["score"]


def test_calibrated_pool_sorts_trust_aware():
    """Federation pool with high raw scores sorts trusted-local first."""
    pool_raw = [
        {"text": "remote A", "score": 0.99, "source": "federated:x"},
        {"text": "local B", "score": 0.85, "source": "zvec/wiki"},
        {"text": "remote C", "score": 0.95, "source": "federated:y"},
    ]
    calibrated = []
    for c in pool_raw:
        st = "federated" if c["source"].startswith("federated") else "zvec"
        calibrated.extend(rag_async._calibrate_chunks([c], st, c["source"]))
    calibrated.sort(key=lambda x: x["score"], reverse=True)
    top = calibrated[0]
    # top must be the trusted-local 0.85, not the federated 0.99
    assert top["source"] == "zvec/wiki"
    assert top["score"] == 0.85


def test_evidence_calibrated_score_matches_helper():
    """_calibrate_chunks must agree with Evidence.calibrated_score directly."""
    chunk = {"text": "x", "score": 0.8, "source": "web"}
    out = rag_async._calibrate_chunks([chunk], "web", "web")[0]
    ref = Evidence(text="x", source_type=SourceType.WEB, source_id="web",
                   retrieval_score=0.8, trust_level=TrustLevel.UNTRUSTED)
    assert out["score"] == ref.calibrated_score
