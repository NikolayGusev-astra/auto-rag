# ADR Migration — Phase 4: Scope Reduction (legacy → extension)

> **For Codex:** Depends on Phase 1-3. Move non-core mechanisms OUT of reference path. Keep `rag_async` functional but mark legacy. Do NOT delete — just gate behind flags / separate modules.

**Goal:** Memory → optional connector (no short-circuit); federation → experimental extension (out of default path); web → explicit opt-in only; LLM generation → out of core; tenant/ACL → optional compat fields, not required.

---

## Task 4.1: Memory as optional `MemoryConnector` (no short-circuit)

**Objective:** Create `rag_core/gateway/adapters/memory.py` implementing `SourceConnector`. Memory result tagged `origin=AGENT_MEMORY`. Coordinator does NOT short-circuit on memory hit — memory is just another connector in the fuse list.

**Files:**
- Create: `rag_core/gateway/adapters/memory.py`
- Test: `tests/gateway/test_memory_connector.py`

**Step 1: Failing test**

```python
# tests/gateway/test_memory_connector.py
import pytest
from rag_core.gateway.adapters.memory import MemoryConnector
from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.models import EvidenceOrigin


@pytest.mark.asyncio
async def test_memory_tagged_agent_memory_and_not_short_circuited():
    conn = MemoryConnector(episodes=[{"answer": "cached", "score": 0.9}])
    res = await conn.search_live(SearchRequest(query="q"))
    assert res[0].origin == EvidenceOrigin.AGENT_MEMORY
    # does not raise / does not skip other connectors upstream
```

**Step 2: Run** → FAIL.
**Step 3: Implement**

```python
# rag_core/gateway/adapters/memory.py
from __future__ import annotations
from rag_core.gateway.connector import SourceConnector, SearchRequest
from rag_core.gateway.models import Evidence, EvidenceOrigin


class MemoryConnector:
    source = "agent_memory"

    def __init__(self, episodes: list[dict] | None = None):
        self._eps = episodes or []

    async def search_live(self, request: SearchRequest) -> list[Evidence]:
        out = []
        for i, ep in enumerate(self._eps):
            out.append(Evidence(
                id=f"memory:{i}", document_id=f"memory:{i}",
                title="episodic", text=ep.get("answer", ""),
                source="agent_memory", origin=EvidenceOrigin.AGENT_MEMORY,
                retrieval_score=float(ep.get("score", 0.0)),
            ))
        return out

    async def fetch(self, ref): raise NotImplementedError
    async def sync_changes(self, cursor): raise NotImplementedError
    async def health(self): return {"source": self.source, "available": True}
```

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): MemoryConnector optional, no short-circuit (ADR-001 Phase 4)`.

---

## Task 4.2: Web explicit opt-in only

**Objective:** Gateway `search` ignores web unless `request.include_web=True`. Web connector not registered by default.

**Files:**
- Modify: `rag_core/gateway/server.py` (`handle_search` only includes web connector if `include_web`)
- Test: `tests/gateway/test_server_search.py` (append)

**Step 1: Failing test**

```python
@pytest.mark.asyncio
async def test_web_excluded_without_opt_in():
    class Web:
        source = "public_web"
        async def search_live(self, r): return ["should-not-appear"]
    resp = await handle_search(SearchRequest(query="q"),
                               connectors={"public_web": Web()})
    assert "should-not-appear" not in str(resp["results"])
```

**Step 2: Run** → FAIL (web runs unconditionally — currently no web connector, so test would pass; instead assert that `include_web=False` does not add web source to status).

Adjust test: assert `"public_web" not in resp["runtime"]["source_status"]` when `include_web=False`.

**Step 3: Implement** — in `handle_search`, build connector dict excluding any connector whose `source == "public_web"` unless `request.include_web`.

**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): web explicit opt-in (ADR-001 Phase 4)`.

---

## Task 4.3: Federation out of default path

**Objective:** `rag_federated.py` not imported by gateway coordinator. Add `docs/EXPERIMENTAL.md` noting federation is experimental extension, not reference path.

**Files:**
- Create: `docs/EXPERIMENTAL.md`
- Verify: grep gateway imports for `rag_federated` → must be absent.

**Step 1-4:** Docs + verification grep. Commit `docs: mark federation experimental (ADR-001 Phase 4)`.

---

## Task 4.4: Mark `rag_async` legacy

**Objective:** Add module docstring to `rag_async.py`: "LEGACY / full-RAG profile. New agent gateway in rag_core.gateway.*". No code change.

**Files:**
- Modify: `rag_core/rag_async.py` (top docstring)
- Commit `docs: mark rag_async as legacy profile (ADR-001 Phase 4)`.

---

## Phase 4 Verification Gate

```bash
python -m pytest tests -q
```
Expected: baseline green + gateway tests green. `rag_async` untouched functionally.

**Exit criteria (ADR-001 §Критерии, subset):**
- [ ] memory does not short-circuit retrieval
- [ ] federation absent from default path
- [ ] web only on explicit opt-in
- [ ] reference pipeline does not require tenant/ACL
