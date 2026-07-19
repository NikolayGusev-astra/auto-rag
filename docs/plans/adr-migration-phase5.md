# ADR Migration — Phase 5: Decomposition & Agent Integration

> **For Codex:** Final phase. Split monolith, build reference gateway, run agent integration tests.
Depends on Phase 1-4.

**Goal:** Break `rag_async.py` (~865 lines) into `gateway/` submodules; make `rag_core.gateway` the
reference implementation; prove agent integration with ≥2 clients (Hermes + Codex stdio).

---

## Task 5.1: Split `rag_async` retrieval into `gateway/retrieval.py`

**Objective:** Extract pure retrieval+fusion logic from `rag_async.async_rag_search` into
`rag_core/gateway/retrieval.py` as a reusable function `retrieve(request, connectors, reranker)`.
Keep `rag_async` calling it (adapter) so legacy tests stay green.

**Files:**
- Create: `rag_core/gateway/retrieval.py`
- Modify: `rag_core/rag_async.py` (delegate)
- Test: `tests/gateway/test_retrieval.py`

**Step 1: Failing test**

```python
# tests/gateway/test_retrieval.py
import pytest
from rag_core.gateway.retrieval import retrieve
from rag_core.gateway.connector import SearchRequest


@pytest.mark.asyncio
async def test_retrieve_merges_connectors():
    class C:
        source = "s"
        async def search_live(self, r): return [{"document_id":"d","text":"t","score":0.5}]
    res = await retrieve(SearchRequest(query="q"), {"s": C()})
    assert len(res) == 1
```

**Step 2: Run** → FAIL.
**Step 3: Implement** `retrieval.py` (mirror coordinator.fuse + connector loop; convert
dict→Evidence).
**Step 4: Run** → PASS. **Step 5: Commit** `refactor(gateway): extract retrieval from rag_async
(ADR-001 Phase 5)`.

---

## Task 5.2: Full MCP stdio server (JSON-RPC loop)

**Objective:** `rag_core/gateway/server.py` gains a `serve_stdio()` entrypoint that reads JSON-RPC
from stdin, dispatches `search`/`fetch`/`sync`/`sync_status`/`list_sources`/`source_status`, writes
JSON to stdout. Use `mcp` SDK if present, else raw stdio JSON-RPC.

**Files:**
- Modify: `rag_core/gateway/server.py` (add `serve_stdio`, `list_sources`, `source_status`)
- Test: `tests/gateway/test_server_stdio.py` (subprocess or in-process dispatch test)

**Step 1: Failing test** — in-process dispatch:

```python
@pytest.mark.asyncio
async def test_dispatch_search_method():
    from rag_core.gateway.server import dispatch
    msg = {"method":"search","params":{"query":"кластер","topk":3},
           "connectors":{}}
    resp = await dispatch(msg)
    assert resp["results"] == []
    assert "runtime" in resp
```

**Step 2: Run** → FAIL.
**Step 3: Implement** `dispatch(msg)` + `serve_stdio()` (loop over stdin lines).
**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): MCP stdio dispatch loop (ADR-001 Phase
5)`.

---

## Task 5.3: CLI entrypoint `auto-rag-gateway`

**Objective:** Add console script / `python -m rag_core.gateway.server` that starts stdio gateway
with configured connectors (from `rag_config`).

**Files:**
- Modify: `rag_core/gateway/server.py` (`__main__` guard)
- Test: `tests/gateway/test_cli_smoke.py` (subprocess start, send one search, expect JSON)

**Step 1: Failing test** — subprocess smoke:

```python
def test_gateway_stdio_smoke():
    import subprocess, json, sys
    p = subprocess.Popen([sys.executable, "-m", "rag_core.gateway.server"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True)
    p.stdin.write(json.dumps({"method":"search","params":{"query":"x"}})+"\n")
    p.stdin.flush()
    line = p.stdout.readline()
    resp = json.loads(line)
    assert "results" in resp
    p.terminate()
```

**Step 2: Run** → FAIL (no `__main__`).
**Step 3: Implement** `__main__` in `server.py`.
**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): stdio CLI entrypoint (ADR-001 Phase 5)`.

---

## Task 5.4: Agent integration test (Hermes + Codex)

**Objective:** Integration test proving ≥2 agent clients can call `search`. Use mocked stdio
subprocess (no real LM Studio needed). Hermes client = `subprocess` JSON-RPC; Codex client = same
protocol, second fixture.

**Files:**
- Create: `tests/gateway/test_agent_integration.py`

**Step 1: Failing test**

```python
def test_two_agent_clients_search():
    # both Hermes-style and Codex-style clients hit the same stdio server
    for client in ("hermes", "codex"):
        resp = call_gateway(client, {"method":"search",
                                      "params":{"query":"deploy"}})
        assert "results" in resp
```

(`call_gateway` helper spawns subprocess, sends msg, reads 1 line.)

**Step 2: Run** → FAIL.
**Step 3: Implement** helper + test using Task 5.3 entrypoint.
**Step 4: Run** → PASS. **Step 5: Commit** `test(gateway): agent integration (Hermes + Codex)
(ADR-001 Phase 5)`.

---

## Task 5.5: CPU scheduler priority (ADR-002)

**Objective:** `rag_core/gateway/scheduler.py` with priority queue: interactive search > fetch >
sync > rebuild. Embedding sync pauses on interactive request.

**Files:**
- Create: `rag_core/gateway/scheduler.py`
- Test: `tests/gateway/test_scheduler.py`

**Step 1: Failing test**

```python
def test_interactive_priority_over_sync():
    from rag_core.gateway.scheduler import PriorityQueue
    q = PriorityQueue()
    q.put("sync", 3)
    q.put("search", 1)
    assert q.get() == "search"
```

**Step 2: Run** → FAIL.
**Step 3: Implement** `PriorityQueue` (heapq by priority).
**Step 4: Run** → PASS. **Step 5: Commit** `feat(gateway): CPU priority scheduler (ADR-002 Phase
5)`.

---

## Phase 5 Verification Gate

```bash
python -m pytest tests -q
```
Expected: full suite green (baseline + all gateway phases).

**Final ADR-001 acceptance (all 12 criteria):**
- [x] Agent calls MCP `search`
- [x] Evidence + URI + freshness in result
- [x] Live + local merged in one request
- [x] Offline → snapshot only
- [x] Incremental sync add/update/delete
- [x] Failed sync does not corrupt active index
- [x] PAT not in logs/results
- [x] Memory no short-circuit
- [x] Federation absent from default path
- [x] Reference pipeline no tenant/ACL required
- [x] Integration test ≥2 agents
- [x] `rag_async` not mandatory entrypoint for agent mode

→ If all checked: ADR-001 / ADR-002 IMPLEMENTED. Mark ADR status Proposed→Accepted in
`docs/ADR-INDEX.md`.
