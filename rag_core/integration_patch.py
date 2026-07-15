"""
integration_patch.py — EXAMPLE patched rag_search.py showing how to wire
MemvidMemory + MemvidTraced into the existing auto-rag pipeline.

This is NOT a drop-in replacement for your rag_search.py — it's a
template. Copy the relevant blocks into your real rag_search.py.

Key integration points (search for `# >>> memvid`):
  1. Build a traced memory singleton at module load.
  2. recall() BEFORE running RAG; short-circuit on high-confidence prior.
  3. record() AFTER successful answer (with trace).
  4. Surface trace.from_memory so eval_golden.py / canary can attribute.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

# >>> memvid: import the traced facade + data model
from memvid_memory import Episode, MemvidMemory
from memvid_trace import MemvidTraced

log = logging.getLogger("hermes.rag_search")

# >>> memvid: one traced memory per process/tenant
_memory = MemvidTraced(MemvidMemory.for_tenant("hermes_default"))


# >>> memvid: toggle for short-circuit vs augment-prompt modes
SHORT_CIRCUIT_ON_PRIOR = True   # set False to always augment prompt instead


def search(query: str, topk: int = 5, domain: Optional[str] = None,
           **kwargs) -> Dict[str, Any]:
    """Hermes search entrypoint with memvid memory layer.

    Flow:
      1. recall prior episodes (traced)
      2. if high-confidence prior -> short-circuit (or augment prompt)
      3. else run normal auto-rag pipeline (DCD -> ZVec/Chroma -> MCP -> Verify)
      4. record the new episode (traced)
    """
    trace: Dict[str, Any] = {"stages": []}
    trace["query"] = query
    trace["domain"] = domain

    # ------------------------------------------------------------------
    # 1) RECALL
    # ------------------------------------------------------------------
    priors = _memory.recall(query, domain=domain, trace=trace)

    if priors and priors[0].score >= _memory.recall_threshold:
        top = priors[0]
        trace["from_memory"] = True
        trace["memory_episode_id"] = top.episode_id
        trace["memory_score"] = top.score
        log.info("memory hit q=%r score=%.3f ep=%s",
                 query[:60], top.score, top.episode_id[:8])

        if SHORT_CIRCUIT_ON_PRIOR:
            # Option A: return the verified prior answer directly.
            # Fast path — sub-5ms when memvid native vec search is wired.
            return {
                "answer": top.answer,
                "sources": top.sources,
                "trace": trace,
                "from_memory": True,
                "episode_id": top.episode_id,
            }
        else:
            # Option B: augment the prompt with prior context, still run RAG.
            # Use this if you want fresh sources but informed by history.
            prior_ctx = _memory.recall_as_context(
                query, domain=domain, trace=trace)
            kwargs.setdefault("prompt_prefix", prior_ctx)

    # ------------------------------------------------------------------
    # 2) NORMAL auto-rag PIPELINE  (unchanged — call your real impl)
    # ------------------------------------------------------------------
    result = _search_orig_pipeline(query, topk=topk, domain=domain,
                                   trace=trace, **kwargs)

    # ------------------------------------------------------------------
    # 3) RECORD the new episode
    # ------------------------------------------------------------------
    try:
        _memory.record(Episode(
            query=query,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            trace=result.get("trace") or trace,
            domain=domain,
            tenant="hermes_default",
        ), trace=trace)
    except Exception as e:
        log.warning("memory record failed (non-fatal): %s", e)

    trace["from_memory"] = False
    result["trace"] = trace
    result["from_memory"] = False
    return result


# ----------------------------------------------------------------------
# Placeholder for your EXISTING pipeline. Replace with the real call.
# ----------------------------------------------------------------------
def _search_orig_pipeline(query: str, *, topk: int,
                          domain: Optional[str],
                          trace: Dict[str, Any],
                          **kwargs) -> Dict[str, Any]:
    """STUB — replace with your real rag_core.rag_search.search() body.

    Must return: {"answer": str, "sources": list, "trace": dict}
    """
    # e.g.:
    #   from rag_core.rag_search import search as _orig
    #   return _orig(query, topk=topk, domain=domain, **kwargs)
    return {
        "answer": "",
        "sources": [],
        "trace": trace,
    }
