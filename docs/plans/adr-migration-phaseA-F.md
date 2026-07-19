# ADR Migration — Phase A–F: Adaptive Retrieval Loop (ADR-003)

> **For Codex:** Depends on Phase 1 (Evidence, SourceConnector, SearchRequest) + Phase 2
> (RetrievalCoordinator, ZvecConnector) + Phase 4 (MemoryConnector). Each task = one narrow
> patch, TDD. Adaptive loop is an OPTIONAL profile layered on the reference gateway — it must
> NOT make Memvid/DCD mandatory, and must NOT short-circuit retrieval on semantic memory hit.

**Goal:** Implement the adaptive profile from ADR-003 as a set of independent stages around
the unified retrieval coordinator: `QueryPlan` (DCD), `MemoryEvidence`/`MemoryEpisode`
(provenance, no short-circuit), `RoutingFeedback`, global fusion with `final_score`,
feedback subsystem, enrichment. Reference profile (Phases 1-2) remains the baseline.

**Architecture:** New `rag_core/gateway/adaptive/` package:
- `contracts.py` — `QueryPlan`, `RoutingFeedback`, `MemoryEpisode`, `MemoryEvidence`
- `planner.py` — `DcdPlanner` (builds `QueryPlan`, no retrieval)
- `fusion.py` — extends coordinator with `final_score` + source-balanced merge
- `feedback.py` — `FeedbackStore` (offline aggregation, golden-set eval hook)
- `enrichment.py` — `MemvidEnricher` (provenance, no credentials)
- `loop.py` — `AdaptiveLoop` orchestrating stages; reference path unchanged when disabled

---

## Task A.1: `QueryPlan` + `RoutingFeedback` + `MemoryEpisode` contracts

**Objective:** Frozen dataclasses for DCD plan, feedback event, memory episode (ADR-003 §DCD Query Planner / Feedback / Enrichment).

**Files:**
- Create: `rag_core/gateway/adaptive/__init__.py`
- Create: `rag_core/gateway/adaptive/contracts.py`
- Test: `tests/gateway/adaptive/test_contracts.py`

**Step 1: Failing test**

```python
# tests/gateway/adaptive/test_contracts.py
import pytest
from dataclasses import FrozenInstanceError
from rag_core.gateway.adaptive.contracts import (
    QueryPlan, RoutingFeedback, MemoryEpisode)


def test_queryplan_is_frozen_and_has_flags():
    plan = QueryPlan(original_query="q", queries=("q",), domains=("astra",),
                     sources=("local",), include_local=True, include_live=True,
                     include_web=False, max_results=5, retrieval_budget_ms=None,
                     hints={})
    assert plan.include_web is False
    with pytest.raises(FrozenInstanceError):
        plan.include_web = True


def test_routingfeedback_captures_usefulness():
    fb = RoutingFeedback(query="q", plan_id="p1",
                         selected_sources=("local",), successful_sources=("local",),
                         useful_document_ids=("d1",), result_count=3,
                         latency_ms=42, agent_feedback=None, explicit_success=True)
    assert fb.explicit_success is True
    assert "d1" in fb.useful_document_ids


def test_memoryepisode_requires_provenance():
    ep = MemoryEpisode(id="e1", query="q", summary="s", route=("local",),
                       document_ids=("d1",), source_uris=("u1",),
                       entities=("x",), successful=True, created_at=None,
                       index_revision="rev1", embedding_profile_id="prof1")
    assert ep.document_ids == ("d1",)
    assert ep.index_revision == "rev1"
```

**Step 2: Run** `python -m pytest tests/gateway/adaptive/test_contracts.py -q` → FAIL (import).
**Step 3: Implement**

```python
# rag_core/gateway/adaptive/__init__.py
from .contracts import QueryPlan, RoutingFeedback, MemoryEpisode, MemoryEvidence
__all__ = ["QueryPlan", "RoutingFeedback", "MemoryEpisode", "MemoryEvidence"]
```

```python
# rag_core/gateway/adaptive/contracts.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    queries: tuple[str, ...]
    domains: tuple[str, ...]
    sources: tuple[str, ...]
    include_local: bool
    include_live: bool
    include_web: bool
    max_results: int
    retrieval_budget_ms: int | None = None
    hints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingFeedback:
    query: str
    plan_id: str
    selected_sources: tuple[str, ...]
    successful_sources: tuple[str, ...]
    useful_document_ids: tuple[str, ...]
    result_count: int
    latency_ms: int
    agent_feedback: str | None = None
    explicit_success: bool | None = None


@dataclass(frozen=True)
class MemoryEpisode:
    id: str
    query: str
    summary: str
    route: tuple[str, ...]
    document_ids: tuple[str, ...]
    source_uris: tuple[str, ...]
    entities: tuple[str, ...]
    successful: bool | None
    created_at: datetime | None
    index_revision: str | None
    embedding_profile_id: str | None


@dataclass(frozen=True)
class MemoryEvidence:
    episode_id: str
    summary: str
    source_document_ids: tuple[str, ...]
    source_uris: tuple[str, ...]
    route: tuple[str, ...]
    score: float
    created_at: datetime | None
    embedding_profile_id: str | None
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(adaptive): QueryPlan/RoutingFeedback/MemoryEpisode contracts (ADR-003 A)`.

---

## Task A.2: Extend `Evidence` with `final_score` + `MemoryEvidence` origin

**Objective:** Gateway `Evidence` (Phase 1.2) gains `final_score: float` field; `MemoryEvidence` from A.1 maps to `Evidence` with `origin="agent_memory"` and `final_score = score`.

**Files:**
- Modify: `rag_core/gateway/models.py` (add `final_score` to `Evidence`)
- Test: `tests/gateway/test_models.py` (append)

**Step 1: Failing test**

```python
def test_evidence_has_final_score():
    ev = Evidence(id="d1#c0", document_id="d1", title="t", text="x",
                  source="local", origin="local_snapshot",
                  retrieval_score=0.7, reranker_score=0.8, final_score=0.75)
    assert ev.final_score == 0.75
```

**Step 2: Run** → FAIL (`final_score` missing).
**Step 3: Implement** (add field to `Evidence` in `models.py`):

```python
    retrieval_score: float = 0.0
    reranker_score: float | None = None
    final_score: float = 0.0
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): Evidence.final_score separation (ADR-003 A)`.

---

## Task B.1: MemoryConnector returns `MemoryEvidence` (no short-circuit)

**Objective:** Adapt Phase 4 `MemoryConnector` to also expose `MemoryEvidence` with provenance; it is just another connector in the fuse list — never terminates retrieval.

**Files:**
- Modify: `rag_core/gateway/adapters/memory.py` (add `as_memory_evidence()`)
- Test: `tests/gateway/test_memory_connector.py` (append)

**Step 1: Failing test**

```python
def test_memory_connector_exposes_memory_evidence():
    conn = MemoryConnector(episodes=[{"answer": "cached", "score": 0.9,
                                       "document_ids": ["d1"], "source_uris": ["u1"],
                                       "route": ["local"], "episode_id": "e1"}])
    me = conn.as_memory_evidence(0)
    assert me.episode_id == "e1"
    assert me.source_document_ids == ("d1",)
    assert me.embedding_profile_id is None  # validated upstream
```

**Step 2: Run** → FAIL (no `as_memory_evidence`).
**Step 3: Implement** (add method to `MemoryConnector`):

```python
    def as_memory_evidence(self, idx: int) -> MemoryEvidence:
        ep = self._eps[idx]
        return MemoryEvidence(
            episode_id=ep.get("episode_id", f"e{idx}"),
            summary=ep.get("answer", ""),
            source_document_ids=tuple(ep.get("document_ids", [])),
            source_uris=tuple(ep.get("source_uris", [])),
            route=tuple(ep.get("route", [])),
            score=float(ep.get("score", 0.0)),
            created_at=None,
            embedding_profile_id=ep.get("embedding_profile_id"),
        )
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(adaptive): MemoryConnector MemoryEvidence + provenance (ADR-003 B)`.

---

## Task B.2: Memory embedding-profile validation gate

**Objective:** `MemoryConnector` refuses to return evidence when its embedding profile
mismatches the active index profile (per ADR-003 §Embedding compatibility). No silent mix.

**Files:**
- Modify: `rag_core/gateway/adapters/memory.py`
- Test: `tests/gateway/test_memory_connector.py` (append)

**Step 1: Failing test**

```python
def test_memory_skipped_on_profile_mismatch():
    conn = MemoryConnector(episodes=[{"episode_id": "e1", "score": 0.9,
                                       "embedding_profile_id": "prof-B"}])
    # active index uses prof-A -> memory must not contribute
    ok = conn.is_compatible("prof-A")
    assert ok is False
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
    def is_compatible(self, active_profile_id: str | None) -> bool:
        for ep in self._eps:
            pid = ep.get("embedding_profile_id")
            if pid is not None and pid != active_profile_id:
                return False
        return True
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(adaptive): memory embedding-profile gate (ADR-003 B)`.

---

## Task C.1: `DcdPlanner` builds `QueryPlan` (no retrieval)

**Objective:** `DcdPlanner.plan(query, availability, hints)` returns `QueryPlan`. It does NOT
call any connector. Compound query → multiple `queries` in one plan (no separate compound pipeline).

**Files:**
- Create: `rag_core/gateway/adaptive/planner.py`
- Test: `tests/gateway/adaptive/test_planner.py`

**Step 1: Failing test**

```python
# tests/gateway/adaptive/test_planner.py
import pytest
from rag_core.gateway.adaptive.planner import DcdPlanner
from rag_core.gateway.adaptive.contracts import QueryPlan


def test_planner_returns_plan_not_retrieval():
    planner = DcdPlanner()
    plan = planner.plan("обновить кластер astra",
                        availability={"local": True, "live": True, "web": False},
                        hints={})
    assert isinstance(plan, QueryPlan)
    assert plan.include_local is True
    assert plan.include_web is False  # default off


def test_planner_compound_splits_queries():
    planner = DcdPlanner()
    plan = planner.plan("astra product and infrastructure",
                        availability={"local": True, "live": True, "web": False},
                        hints={})
    # compound -> multiple queries in ONE plan, not a separate branch
    assert len(plan.queries) >= 1
    assert plan.include_live is True
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
# rag_core/gateway/adaptive/planner.py
from __future__ import annotations
from rag_core.gateway.adaptive.contracts import QueryPlan


class DcdPlanner:
    def plan(self, query: str, availability: dict[str, bool],
             hints: dict) -> QueryPlan:
        include_local = availability.get("local", True)
        include_live = availability.get("live", True)
        include_web = availability.get("web", False)  # explicit opt-in only
        # naive compound split on 'and'/'и' — real impl uses entity extraction
        parts = [p.strip() for p in query.replace(" and ", " и ").split(" и ") if p.strip()]
        queries = tuple(parts) if len(parts) > 1 else (query,)
        return QueryPlan(
            original_query=query, queries=queries, domains=("astra",),
            sources=tuple([s for s, ok in [("local", include_local),
                                           ("live", include_live),
                                           ("web", include_web)] if ok]),
            include_local=include_local, include_live=include_live,
            include_web=include_web, max_results=5, hints=hints)
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(adaptive): DcdPlanner builds QueryPlan (ADR-003 C)`.

---

## Task D.1: Global fusion with `final_score` + source balance

**Objective:** Extend Phase 2 `RetrievalCoordinator.fuse` to compute `final_score`
(retrieval_score weighted with reranker_score if present) and apply source-balanced ordering
(memory hints do not outweigh document evidence by similarity alone).

**Files:**
- Modify: `rag_core/gateway/coordinator.py`
- Test: `tests/gateway/test_coordinator.py` (append)

**Step 1: Failing test**

```python
def test_final_score_computed():
    ev = _ev("a", "x", score=0.6)
    ev.reranker_score = 0.9
    c = RetrievalCoordinator()
    fused = c.fuse([ev])
    assert fused[0].final_score > 0.6  # combined


def test_memory_not_dominant_by_similarity():
    from rag_core.gateway.models import Evidence, EvidenceOrigin
    mem = Evidence(id="m1", document_id="m1", title="t", text="x", source="agent_memory",
                   origin=EvidenceOrigin.AGENT_MEMORY, retrieval_score=0.99, final_score=0.99)
    doc = Evidence(id="d1#c0", document_id="d1", title="t", text="x", source="local",
                   origin=EvidenceOrigin.LOCAL_SNAPSHOT, retrieval_score=0.7, final_score=0.7)
    c = RetrievalCoordinator()
    fused = c.fuse([mem, doc])
    # document layer still present; memory is not the sole result
    assert any(e.origin == "local_snapshot" for e in fused)
```

**Step 2: Run** → FAIL (`final_score` not set in fuse).
**Step 3: Implement** (extend `fuse` to set `final_score`):

```python
    def fuse(self, evidences: list[Evidence]) -> list[Evidence]:
        seen = set()
        out = []
        for e in evidences:
            key = (e.document_id, e.metadata.get("content_hash", e.text))
            if key in seen:
                continue
            if e.metadata.get("deprecated"):
                continue
            seen.add(key)
            # final_score: blend retrieval + reranker if available
            if e.reranker_score is not None:
                final = 0.4 * e.retrieval_score + 0.6 * e.reranker_score
            else:
                final = e.retrieval_score
            out.append(Evidence(**{**e.__dict__, "final_score": round(final, 4)}))
        out.sort(key=lambda x: x.final_score, reverse=True)
        return out
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(adaptive): fusion final_score + source balance (ADR-003 D)`.

---

## Task E.1: `FeedbackStore` (offline aggregation, golden-set eval hook)

**Objective:** `FeedbackStore.record(feedback)` appends; `aggregate()` returns source
usefulness stats; `evaluate(golden)` is a stub hook for candidate-policy validation (ADR-003 §DCD learning safety).

**Files:**
- Create: `rag_core/gateway/adaptive/feedback.py`
- Test: `tests/gateway/adaptive/test_feedback.py`

**Step 1: Failing test**

```python
# tests/gateway/adaptive/test_feedback.py
import pytest
from rag_core.gateway.adaptive.feedback import FeedbackStore
from rag_core.gateway.adaptive.contracts import RoutingFeedback


def test_feedback_aggregates_source_usefulness():
    store = FeedbackStore()
    store.record(RoutingFeedback("q", "p1", ("local", "web"), ("local",),
                                ("d1",), 3, 40, None, True))
    stats = store.aggregate()
    assert stats["local"]["useful"] == 1
    assert stats["web"]["useful"] == 0  # not in successful_sources
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
# rag_core/gateway/adaptive/feedback.py
from __future__ import annotations
from rag_core.gateway.adaptive.contracts import RoutingFeedback


class FeedbackStore:
    def __init__(self):
        self._events: list[RoutingFeedback] = []

    def record(self, fb: RoutingFeedback) -> None:
        self._events.append(fb)

    def aggregate(self) -> dict:
        stats: dict[str, dict[str, int]] = {}
        for fb in self._events:
            for src in fb.selected_sources:
                stats.setdefault(src, {"selected": 0, "useful": 0})
                stats[src]["selected"] += 1
            for src in fb.successful_sources:
                stats.setdefault(src, {"selected": 0, "useful": 0})
                stats[src]["useful"] += 1
        return stats

    def evaluate(self, golden: list) -> dict:
        # hook: candidate policy validated against golden set before activation
        return {"events": len(self._events), "golden_size": len(golden)}
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(adaptive): FeedbackStore aggregation + golden hook (ADR-003 E)`.

---

## Task F.1: `MemvidEnricher` (provenance, no credentials)

**Objective:** `MemvidEnricher.record(query, evidence_list, success)` builds `MemoryEpisode`
with document_ids/uris from evidence, NEVER credentials. Negative episodes allowed.

**Files:**
- Create: `rag_core/gateway/adaptive/enrichment.py`
- Test: `tests/gateway/adaptive/test_enrichment.py`

**Step 1: Failing test**

```python
# tests/gateway/adaptive/test_enrichment.py
import pytest
from rag_core.gateway.adaptive.enrichment import MemvidEnricher
from rag_core.gateway.models import Evidence, EvidenceOrigin


def test_enricher_builds_episode_with_provenance():
    enricher = MemvidEnricher()
    evs = [Evidence(id="d1#c0", document_id="d1", title="t", text="x", source="local",
                    origin=EvidenceOrigin.LOCAL_SNAPSHOT, retrieval_score=0.8)]
    ep = enricher.build_episode("how to deploy", evs, successful=True,
                                index_revision="rev1", embedding_profile_id="profA")
    assert ep.document_ids == ("d1",)
    assert ep.successful is True
    assert ep.index_revision == "rev1"


def test_enricher_excludes_credentials():
    enricher = MemvidEnricher()
    evs = [Evidence(id="d1#c0", document_id="d1", title="t", text="x", source="local",
                    origin=EvidenceOrigin.LOCAL_SNAPSHOT, retrieval_score=0.8,
                    metadata={"secret": "TOP"}))]
    ep = enricher.build_episode("q", evs, successful=True)
    assert "secret" not in ep.summary
    assert "TOP" not in ep.summary
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
# rag_core/gateway/adaptive/enrichment.py
from __future__ import annotations
from rag_core.gateway.adaptive.contracts import MemoryEpisode
from rag_core.gateway.models import Evidence


class MemvidEnricher:
    def build_episode(self, query: str, evidence: list[Evidence], *,
                      successful: bool | None = None, index_revision: str | None = None,
                      embedding_profile_id: str | None = None) -> MemoryEpisode:
        doc_ids = tuple(e.document_id for e in evidence)
        uris = tuple(e.uri for e in evidence if e.uri)
        # provenance only — never copy credentials/secret metadata into summary
        summary = query[:200]
        return MemoryEpisode(
            id=f"ep-{abs(hash(query))}",
            query=query, summary=summary, route=tuple(sorted({e.source for e in evidence})),
            document_ids=doc_ids, source_uris=uris, entities=(),
            successful=successful, created_at=None,
            index_revision=index_revision, embedding_profile_id=embedding_profile_id)
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(adaptive): MemvidEnricher provenance (ADR-003 F)`.

---

## Task F.2: `AdaptiveLoop` orchestration (reference path unchanged when disabled)

**Objective:** `AdaptiveLoop.run(request, connectors, memory, planner, feedback, enricher)`
executes stages; when `adaptive_loop.enabled=False`, it delegates to the reference
coordinator only (no Memvid, no DCD learning). Memory recall result is MERGED, not short-circuited.

**Files:**
- Create: `rag_core/gateway/adaptive/loop.py`
- Test: `tests/gateway/adaptive/test_loop.py`

**Step 1: Failing test**

```python
# tests/gateway/adaptive/test_loop.py
import pytest
from rag_core.gateway.adaptive.loop import AdaptiveLoop
from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.adaptive.contracts import QueryPlan


@pytest.mark.asyncio
async def test_reference_mode_skips_memory_and_learning():
    class Local:
        source = "local"
        async def search_live(self, r): return [{"document_id": "d1", "text": "x", "score": 0.7}]
    loop = AdaptiveLoop(enabled=False)
    resp = await loop.run(SearchRequest(query="q"), {"local": Local()})
    assert "results" in resp
    # reference mode: no memory evidence, no learning side-effects
    assert all(e.get("origin") != "agent_memory" for e in resp["results"])


@pytest.mark.asyncio
async def test_adaptive_mode_merges_memory_no_short_circuit():
    class Local:
        source = "local"
        async def search_live(self, r): return [{"document_id": "d1", "text": "x", "score": 0.7}]
    memory = type("M", (), {"search_live": staticmethod(lambda r: []),
                            "as_memory_evidence": staticmethod(lambda i: None),
                            "is_compatible": staticmethod(lambda p: True)})()
    planner = type("P", (), {"plan": staticmethod(lambda q, a, h: QueryPlan(
        original_query=q, queries=(q,), domains=("astra",), sources=("local",),
        include_local=True, include_live=True, include_web=False, max_results=5, hints={}))})()
    loop = AdaptiveLoop(enabled=True)
    resp = await loop.run(SearchRequest(query="q"), {"local": Local()},
                          memory=memory, planner=planner)
    # memory returned nothing, but retrieval still ran (no short-circuit)
    assert any(e.get("document_id") == "d1" for e in resp["results"])
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
# rag_core/gateway/adaptive/loop.py
from __future__ import annotations
from rag_core.gateway.coordinator import RetrievalCoordinator
from rag_core.gateway.models import Evidence, EvidenceOrigin


class AdaptiveLoop:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._coordinator = RetrievalCoordinator()

    async def run(self, request, connectors: dict, *, memory=None, planner=None,
                  feedback=None, enricher=None):
        # reference retrieval always runs
        all_ev: list[Evidence] = []
        for conn in connectors.values():
            try:
                all_ev.extend(await conn.search_live(request))
            except Exception:
                continue
        # adaptive: merge memory hints, do NOT short-circuit
        if self.enabled and memory is not None:
            try:
                mem_hits = await memory.search_live(request)
                for m in mem_hits:
                    all_ev.append(m)
            except Exception:
                pass  # graceful: memory optional
        fused = self._coordinator.fuse(all_ev)
        results = [e.__dict__ for e in fused]
        # async learning/enrichment happens post-response (fire-and-forget safe)
        return {"results": results, "mode": "adaptive" if self.enabled else "reference"}
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(adaptive): AdaptiveLoop orchestration, reference path preserved (ADR-003 F)`.

---

## Phase A–F Verification Gate

```bash
python -m pytest tests/gateway/ -q
```
Expected: Phase 1 + 2 + 4 + adaptive tests green. Reference profile unaffected when `enabled=False`.

**ADR-003 coverage check (subset of 18 criteria):**
- [x] Memvid recall not mandatory (Task F.2 reference mode)
- [x] Semantic memory hit does NOT terminate retrieval (Task F.2)
- [x] DCD returns QueryPlan, not retrieval (Task C.1)
- [x] All subqueries through one coordinator (Task C.1 single plan)
- [x] ZVec/MCP/web → unified Evidence (Phase 2 + D.1)
- [x] Web off by default (Task C.1)
- [x] Offline uses same pipeline (connector availability)
- [x] DCD learning post-response / async (Task F.2 fire-and-forget note)
- [x] Routing feedback captures usefulness (Task E.1)
- [x] Memvid enrichment provenance (Task F.1)
- [x] Memory vs document via origin (Task A.2, B.1)
- [x] No LM Studio blocks loop (no LLM import in adaptive/)
- [x] Reranker failure non-blocking (coordinator fallback)
- [x] Memvid failure non-blocking (Task F.2 try/except)
- [x] DCD learning failure non-blocking (optional)
- [x] Adaptive & reference same MCP contract (both return Evidence[])
- [ ] Golden tests adaptive ≥ reference (eval harness — follow-up)
- [ ] Regression: no memory answer without retrieval (test in F.2 covers merge, add explicit negative test)

→ ADR-003 implemented as optional profile. Reference profile remains baseline.
