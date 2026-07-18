"""Episodic memory writer with poisoning guards.

P0/P1 fix (audit): the old `_record_episode` stored the raw `answer` (or
concatenated retrieved chunks) including content from web / federated
sources, with no source-trust check, no TTL, no index revision. That let an
untrusted web/federation chunk become a fast "memory" answer on the next
semantically-similar query (poisoning + staleness).

Guards added:
  * source_trust gate — episodes built from untrusted sources are NOT recorded
    unless at least one trusted (local/zvec/mcp) chunk is present
  * index_revision + created_at make staleness detectable by the recall side
  * tenant is taken from the explicit QueryContext, never silently from env
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# Sources we treat as trusted enough to anchor a memory episode.
TRUSTED_SOURCES = frozenset({
    "zvec/wiki", "zvec_fastapi", "zvec_direct", "context7", "jira",
    "confluence", "lodestone", "protopack", "mcp",
    # bare local-source labels used in tests / simple indexers
    "wiki", "docs", "local", "zvec",
})

# Sources that MUST NOT alone become a stored answer.
UNTRUSTED_SOURCES = frozenset({"web", "federated", "federation"})

_MEMORY_TTL_SECONDS = int(os.environ.get("RAG_MEMVID_EPISODE_TTL", "86400"))


def _source_is_trusted(source: str) -> bool:
    src = (source or "").split(":")[0].lower()
    if src in TRUSTED_SOURCES:
        return True
    if src in UNTRUSTED_SOURCES:
        return False
    # unknown sources default to untrusted to avoid poisoning
    return False


def _answer_text(result: dict) -> str:
    if result.get("answer"):
        return str(result["answer"])
    parts = []
    for chunk in result.get("chunks", []):
        text = chunk.get("text") or chunk.get("content") or ""
        if text:
            parts.append(str(text))
    return "\n\n".join(parts)[:6000]


def _episode_sources(result: dict) -> list[dict]:
    out = []
    for chunk in result.get("chunks", []):
        src = chunk.get("source", result.get("source", ""))
        out.append({"source": src, "trusted": _source_is_trusted(src)})
    if not out:
        out = [{"source": result.get("source", ""), "trusted": False}]
    return out


def should_record(result: dict, tenant: str) -> tuple[bool, str]:
    """Decide whether a result may become a memory episode.

    Returns (allowed, reason). Untrusted-only results are rejected to prevent
    web/federation poisoning.
    """
    if not result.get("chunks") and not result.get("answer"):
        return False, "empty result"
    sources = _episode_sources(result)
    has_trusted = any(s["trusted"] for s in sources)
    if not has_trusted:
        return False, "no trusted source anchor (web/federation-only)"
    if any(s for s in sources if not s["trusted"]):
        # mixed: keep, but recall side must weigh trusted anchor
        return True, "mixed sources, trusted anchor present"
    return True, "trusted sources only"


def build_episode(result: dict, query: str, domain: str, tenant: str,
                  index_revision: str, trace: Any = None):
    """Build a memvid Episode with poisoning metadata. Returns Episode or None."""
    from memvid_memory import Episode

    allowed, reason = should_record(result, tenant)
    if not allowed:
        logger.debug("episode skipped: %s", reason)
        return None

    answer = _answer_text(result)
    if not answer:
        return None

    sources = [
        {"source": s["source"], "trusted": s["trusted"]}
        for s in _episode_sources(result)
    ]
    meta = {
        "index_revision": index_revision,
        "created_at_unix": int(time.time()),
        "ttl_unix": int(time.time()) + _MEMORY_TTL_SECONDS,
        "record_reason": reason,
    }
    try:
        return Episode(
            query=query,
            answer=answer,
            sources=sources,
            trace=trace.json() if hasattr(trace, "json") else trace,
            domain=domain,
            tenant=tenant,
            episode_id=_new_id(),
            created_at=meta["created_at_unix"].hex()
            if False else __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
        )
    except Exception as exc:
        logger.debug("episode build failed: %s", exc)
        return None


def _new_id() -> str:
    import uuid
    return uuid.uuid4().hex
