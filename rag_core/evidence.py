"""Evidence contract — common shape for retrieval results across sources.

P0/P1 fix (audit): previously ZVec score, MCP's hardcoded 0.7, web's 0.6 and a
remote federation score were compared as if they shared one meaning. They do
not. We introduce a typed `Evidence` carrying source type, trust level and the
raw retrieval score separately, so ranking can calibrate instead of mixing
uncalibrated numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    LOCAL_VECTOR = "local_vector"      # ZVec / Chroma
    MCP = "mcp"                        # context7/jira/confluence/etc
    WEB = "web"                        # SearXNG + trafilatura
    FEDERATION = "federation"          # remote RAG instance
    MEMORY = "memory"                  # episodic memory recall


class TrustLevel(str, Enum):
    """Authority of the source. Drives calibration, not raw score."""
    TRUSTED_INTERNAL = "trusted_internal"   # local index, internal MCP
    EXTERNAL_CURATED = "external_curated"   # known-good web (docs)
    UNTRUSTED = "untrusted"                # arbitrary web / federation


@dataclass
class Evidence:
    text: str
    source_type: SourceType
    source_id: str
    retrieval_score: float | None = None
    trust_level: TrustLevel = TrustLevel.UNTRUSTED
    tenant_id: str = "default"
    freshness_at: str | None = None
    content_hash: str = ""
    acl_context: dict = field(default_factory=dict)
    citation_uri: str | None = None

    @property
    def calibrated_score(self) -> float:
        """Map an uncalibrated retrieval score into a comparable 0..1 rank.

        Trusted internal sources keep their score; external/untrusted are
        discounted so a remote federation `0.99` cannot outrank a local hit
        without explicit trust.
        """
        base = self.retrieval_score if self.retrieval_score is not None else 0.0
        base = max(0.0, min(1.0, base))
        multiplier = {
            TrustLevel.TRUSTED_INTERNAL: 1.0,
            TrustLevel.EXTERNAL_CURATED: 0.85,
            TrustLevel.UNTRUSTED: 0.6,
        }[self.trust_level]
        return round(base * multiplier, 4)


def from_chunk(chunk: dict, source_type: SourceType,
               source_id: str = "", tenant_id: str = "default",
               trust: TrustLevel = TrustLevel.UNTRUSTED) -> Evidence:
    """Adapt a legacy result chunk dict into an Evidence."""
    return Evidence(
        text=chunk.get("text") or chunk.get("content") or "",
        source_type=source_type,
        source_id=source_id or str(chunk.get("source", "")),
        retrieval_score=chunk.get("score"),
        trust_level=trust,
        tenant_id=tenant_id,
        citation_uri=chunk.get("url"),
    )
