# ADR Migration — Phase 2: Agent Gateway MVP

> **For Codex:** Continue from Phase 1 (contracts in `rag_core/gateway/`). Each task = one narrow patch. TDD: write failing test → RED → implement → GREEN → commit. Do NOT touch `rag_async.py` — new gateway lives in `rag_core/gateway/`.

**Goal:** First working MCP-facing retrieval that returns structured `Evidence[]` (no LLM answer generation), with source availability detection and hybrid retrieval via a thin ZVec adapter.

**Architecture:** `rag_core/gateway/server.py` (MCP stdio server using `mcp` SDK if available, else a stdio JSON-RPC loop), `rag_core/gateway/coordinator.py` (RetrievalCoordinator: select connectors → execute → normalize → dedup → filter → rerank → fuse), `rag_core/gateway/adapters/zvec.py` (wraps `rag_core.zvec_adapter.ZvecAdapter.search_hybrid` into `SourceConnector`). Legacy `rag_async.async_rag_search` is NOT called.

**Depends on:** Phase 1 (`Document`, `Evidence`, `SourceConnector`, `SearchRequest`).

---

## Task 2.1: ZVec SourceConnector adapter

**Objective:** Wrap existing `rag_core.zvec_adapter.ZvecAdapter` as a `SourceConnector` returning gateway `Evidence` with `origin=LOCAL_SNAPSHOT`.

**Files:**
- Create: `rag_core/gateway/adapters/__init__.py`
- Create: `rag_core/gateway/adapters/zvec.py`
- Test: `tests/gateway/test_zvec_adapter.py`

**Step 1: Failing test**

```python
# tests/gateway/test_zvec_adapter.py
import pytest
from rag_core.gateway.adapters.zvec import ZvecConnector
from rag_core.gateway.connector import SearchRequest


@pytest.mark.asyncio
async def test_zvec_connector_returns_evidence(monkeypatch):
    # stub the underlying adapter
    class FakeZvec:
        def search_hybrid(self, query, topk=5, domain=None, collection=None):
            return [{
                "id": "doc1#c0", "document_id": "doc1", "title": "T",
                "text": "body", "score": 0.9,
                "source": "local", "uri": None,
            }]
    conn = ZvecConnector(zvec=FakeZvec())
    res = await conn.search_live(SearchRequest(query="q", topk=1))
    assert len(res) == 1
    ev = res[0]
    assert ev.document_id == "doc1"
    assert ev.origin == "local_snapshot"
    assert ev.retrieval_score == 0.9
```

**Step 2: Run** `python -m pytest tests/gateway/test_zvec_adapter.py -q` → FAIL (import).

**Step 3: Implement**

```python
# rag_core/gateway/adapters/__init__.py
from .zvec import ZvecConnector
__all__ = ["ZvecConnector"]
```

```python
# rag_core/gateway/adapters/zvec.py
from __future__ import annotations
from rag_core.gateway.connector import SourceConnector, SearchRequest
from rag_core.gateway.models import Evidence, EvidenceOrigin


class ZvecConnector:
    source = "local_zvec"

    def __init__(self, zvec):
        self._zvec = zvec

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        raw = self._zvec.search_hybrid(
            request.query, topk=request.topk,
            domain=request.domain, collection=request.collection,
        )
        out = []
        for r in raw:
            out.append(Evidence(
                id=r.get("id", f"{r.get('document_id','?')}#c0"),
                document_id=r.get("document_id", r.get("id", "?")),
                title=r.get("title", ""),
                text=r.get("text", ""),
                source="local_zvec",
                uri=r.get("uri"),
                origin=EvidenceOrigin.LOCAL_SNAPSHOT,
                retrieval_score=float(r.get("score", 0.0)),
                metadata=r.get("metadata", {}),
            ))
        return out

    async def fetch(self, ref):
        raise NotImplementedError("fetch not implemented in Phase 2")

    async def sync_changes(self, cursor):
        raise NotImplementedError("sync in Phase 3")

    async def health(self):
        return {"source": self.source, "available": True, "detail": "hybrid ready"}
```

**Step 4: Run** `python -m pytest tests/gateway/test_zvec_adapter.py -q` → PASS.
**Step 5: Commit** `feat(gateway): ZvecConnector adapter (ADR-001 Phase 2)`.

---

## Task 2.2: RetrievalCoordinator — dedup + filters

**Objective:** Coordinator merges connector results, dedups by `document_id+content_hash`, applies version/entity filter (stub: drop `metadata.get("deprecated")`).

**Files:**
- Create: `rag_core/gateway/coordinator.py`
- Test: `tests/gateway/test_coordinator.py`

**Step 1: Failing test**

```python
# tests/gateway/test_coordinator.py
import pytest
from rag_core.gateway.coordinator import RetrievalCoordinator
from rag_core.gateway.models import Evidence, EvidenceOrigin


def _ev(doc_id, text, score=0.5):
    return Evidence(id=f"{doc_id}#c0", document_id=doc_id, title="t",
                    text=text, source="s", origin=EvidenceOrigin.LOCAL_SNAPSHOT,
                    retrieval_score=score, metadata={"content_hash": doc_id})


def test_dedup_by_document_and_hash():
    c = RetrievalCoordinator()
    merged = c.fuse([_ev("a", "x"), _ev("a", "x"), _ev("b", "y")])
    ids = [e.document_id for e in merged]
    assert ids == ["a", "b"]


def test_deprecated_filtered_out():
    ev = _ev("a", "x")
    ev.metadata["deprecated"] = True
    c = RetrievalCoordinator()
    merged = c.fuse([ev])
    assert merged == []
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
# rag_core/gateway/coordinator.py
from __future__ import annotations
from rag_core.gateway.models import Evidence


class RetrievalCoordinator:
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
            out.append(e)
        # deterministic ranking by retrieval_score desc
        out.sort(key=lambda x: x.retrieval_score, reverse=True)
        return out
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): RetrievalCoordinator dedup+filter (ADR-001 Phase 2)`.

---

## Task 2.3: Coordinator source-aware fusion + reranker hook

**Objective:** `fuse` accepts optional `reranker` (RerankerProvider from Phase 1.5); if provided, call `rerank` and set `reranker_score`. Otherwise keep deterministic order.

**Files:**
- Modify: `rag_core/gateway/coordinator.py`
- Test: `tests/gateway/test_coordinator.py` (append)

**Step 1: Failing test**

```python
@pytest.mark.asyncio
async def test_reranker_sets_score():
    class FakeReranker:
        async def rerank(self, query, evidence, limit):
            for i, e in enumerate(evidence):
                e = Evidence(**{**e.__dict__, "reranker_score": 1.0 - i*0.1})
            return evidence
    c = RetrievalCoordinator()
    res = await c.fuse_rerank("q", [_ev("a","x",0.1), _ev("b","y",0.9)],
                              reranker=FakeReranker(), limit=2)
    assert res[0].reranker_score is not None
```

**Step 2: Run** → FAIL (no `fuse_rerank`).
**Step 3: Implement** (add method to `RetrievalCoordinator`):

```python
    async def fuse_rerank(self, query, evidences, reranker=None, limit=10):
        fused = self.fuse(evidences)
        if reranker is None:
            return fused[:limit]
        reranked = await reranker.rerank(query, fused, limit)
        return reranked
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): reranker hook in coordinator (ADR-002 Phase 2)`.

---

## Task 2.4: MCP `search` tool (stdio JSON-RPC)

**Objective:** `rag_core/gateway/server.py` exposes `search` over stdio returning `Evidence[]` JSON + `runtime` block. Use `mcp` SDK if installed; else minimal stdio JSON-RPC loop.

**Files:**
- Create: `rag_core/gateway/server.py`
- Test: `tests/gateway/test_server_search.py`

**Step 1: Failing test** (integration-style, no real zvec needed — inject connector):

```python
# tests/gateway/test_server_search.py
import pytest, json
from rag_core.gateway.server import handle_search
from rag_core.gateway.connector import SearchRequest


@pytest.mark.asyncio
async def test_handle_search_returns_evidence_json():
    fake = type("C", (), {
        "source": "local_zvec",
        "search_live": staticmethod(lambda r: []),
    })()
    req = SearchRequest(query="кластер", topk=3)
    resp = await handle_search(req, connectors={"local_zvec": fake})
    assert "results" in resp
    assert "runtime" in resp
    assert resp["runtime"]["embedding_provider"] in ("none", "unknown")
```

**Step 2: Run** → FAIL.
**Step 3: Implement** (minimal — stdout JSON-RPC is Phase 5; here just the handler):

```python
# rag_core/gateway/server.py
from __future__ import annotations
import time
from rag_core.gateway.coordinator import RetrievalCoordinator
from rag_core.gateway.models import Evidence


async def handle_search(request, connectors: dict, reranker=None):
    coordinator = RetrievalCoordinator()
    all_ev: list[Evidence] = []
    for conn in connectors.values():
        try:
            all_ev.extend(await conn.search_live(request))
        except Exception:
            continue  # graceful degradation per source
    fused = await coordinator.fuse_rerank(
        request.query, all_ev, reranker=reranker, limit=request.topk)
    return {
        "query": request.query,
        "mode": "mixed" if connectors else "empty",
        "results": [e.__dict__ for e in fused],
        "runtime": {
            "retrieval": "hybrid" if connectors else "none",
            "embedding_provider": "unknown",
            "reranker": "disabled" if reranker is None else "enabled",
            "language_model": "none",
            "execution": "cpu",
        },
    }
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): search handler returning structured Evidence (ADR-001 Phase 2)`.

---

## Task 2.5: Source availability detection

**Objective:** `handle_search` marks each source available/unavailable in response; offline source → skipped, not error.

**Files:**
- Modify: `rag_core/gateway/server.py` (add `source_status` tracking)
- Test: `tests/gateway/test_server_search.py` (append)

**Step 1: Failing test**

```python
@pytest.mark.asyncio
async def test_offline_source_skipped_not_error():
    class Down:
        source = "jira"
        async def search_live(self, r):
            raise ConnectionError("network down")
    resp = await handle_search(SearchRequest(query="q"),
                               connectors={"jira": Down()})
    assert "results" in resp
    assert "jira" in resp["runtime"].get("source_status", {})
    assert resp["runtime"]["source_status"]["jira"] == "unavailable"
```

**Step 2: Run** → FAIL.
**Step 3: Implement** (extend `handle_search` to record status):

```python
    source_status = {}
    for name, conn in connectors.items():
        try:
            found = await conn.search_live(request)
            all_ev.extend(found)
            source_status[name] = "available"
        except Exception:
            source_status[name] = "unavailable"
    ...
    resp["runtime"]["source_status"] = source_status
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): per-source availability in search response (ADR-001 Phase 2)`.

---

## Phase 2 Verification Gate

```bash
python -m pytest tests -q
```
Expected: baseline 187 passed + new gateway tests green. No legacy test broken.

**Exit criteria (ADR-001 §Критерии, subset):**
- [ ] `search` returns Evidence + URI + freshness
- [ ] offline mode works (snapshot only)
- [ ] reranker/LLM failure does not block search
- [ ] `rag_async` NOT called by gateway
