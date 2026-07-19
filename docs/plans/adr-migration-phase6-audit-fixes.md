# ADR Migration — Phase 6: Audit Remediation (P0/P1)

> **Context:** Post-implementation audit of `a03b962` found the gateway is a correct
> **scaffold** but not ADR-complete. 2 P0 (blocking) + 6 P1 (major) are real and confirmed
> by direct code inspection. This phase closes them. Each task = one narrow patch, TDD.
> Do NOT push until the full phase is green and re-audited.

**Audit verdict accepted:** ADR-001 ~55-60%, ADR-002 ~70-75%, ADR-003 ~40-45% realized.
The following tasks raise that to functional completeness for the core paths.

**Baseline before this phase:** `258 passed, 4 skipped, 1 xfailed` (tests -q).

---

## P0-1: Non-destructive incremental sync

**Problem:** `SyncEngine.stage_sync` writes ONLY `batch.added`; previous active revision
documents are NOT carried forward; `batch.changed` ignored. Publishing drops A,B,C when
only D is added. Destructive for the local autonomous snapshot (core product feature).

**Files:** `rag_core/gateway/sync/engine.py`, tests appended to `tests/gateway/test_sync.py`

**Task P0-1.1: carry forward previous active documents**
- Step 1 (RED): test — initial A,B,C → incremental add D → `active_documents()` == A,B,C,D.
- Step 2: `stage_sync` seeds new revision with previous active docs + added, applies
  changed (replace by id), excludes deleted (tombstones).
- Step 3 (GREEN): pass.
- Step 4: commit `fix(sync): non-destructive incremental revision (P0-1)`.

**Task P0-1.2: apply changed + tombstones**
- Step 1 (RED): initial A,B,C → change B → active has new B, A,C retained;
  initial A,B,C → delete B → active A,C.
- Step 2: implement changed-replace + deleted-exclude in `stage_sync`.
- Step 3 (GREEN): pass.
- Step 4: commit `fix(sync): handle changed + tombstones in revision (P0-1)`.

**Task P0-1.3: failed publish leaves old revision active**
- Step 1 (RED): publish corrupt staged raises; `active_revision()` still old A,B,C.
- Step 2: `publish` validates BEFORE swap (already does `_validate_revision`); ensure
  manifest swap only after validation + staged dir complete.
- Step 3 (GREEN): pass.
- Step 4: commit `test(sync): failed publish preserves prior revision (P0-1)`.

---

## P0-2: Real MCP transport (protocol surface)

**Problem:** `server.py` speaks a custom line-JSON protocol, not MCP. No `initialize`,
`tools/list`, `tools/call`. Real MCP clients (Claude/Cursor/Codex) cannot register it.

**Decision:** Implement the MCP protocol surface manually (no external SDK dependency,
per ADR-001 "offline-capable" + Phase 5 note). Minimal but compliant:

```
initialize → returns protocolVersion, capabilities, serverInfo
tools/list → lists gateway tools with JSON Schema input schemas
tools/call → dispatches tool, returns content[0].text or structured
ping → {}
notifications/initialized → no-op
notifications/cancelled → no-op
```

**Files:** rewrite `rag_core/gateway/server.py` (keep `dispatch` for tool handlers),
new `rag_core/gateway/mcp_protocol.py` (lifecycle + tool registry),
tests `tests/gateway/test_mcp_protocol.py`.

**Task P0-2.1: MCP lifecycle (initialize/tools/list/ping)**
- Step 1 (RED): client sends `{"jsonrpc":"2.0","id":1,"method":"initialize",...}` →
  response has `protocolVersion`, `capabilities.tools`, `serverInfo`.
  `tools/list` → result.tools contains `search` with `inputSchema`.
- Step 2: implement `handle_mcp(message)` with method routing + JSON-RPC envelope.
- Step 3 (GREEN): pass.
- Step 4: commit `feat(mcp): protocol lifecycle + tools/list (P0-2)`.

**Task P0-2.2: tools/call dispatches to gateway tools**
- Step 1 (RED): `tools/call` with `{name:"search", arguments:{query:"x"}}` →
  result.content[0].text parses to gateway search response.
- Step 2: map tool name → existing `dispatch` handler; wrap output as MCP content.
- Step 3 (GREEN): pass.
- Step 4: commit `feat(mcp): tools/call dispatch (P0-2)`.

**Task P0-2.3: smoke test with two simulated clients**
- Step 1 (RED): a minimal MCP client fixture performs initialize→tools/list→tools/call
  (search) and asserts structured Evidence response. Run twice (different client ids).
- Step 2: ensure idempotent session handling, notification tolerance.
- Step 3 (GREEN): pass.
- Step 4: commit `test(mcp): two-client smoke test (P0-2)`.

---

## P1-1: AdaptiveLoop uses QueryPlan to drive RetrievalCoordinator

**Problem:** `plan` computed but ignored; connectors iterated directly; web bypasses
filter; availability faked; origin corrupted; topk ignored; errors swallowed.

**Files:** `rag_core/gateway/adaptive/loop.py`, `rag_core/gateway/connector.py`
(add `origin` hint or require connector to return Evidence), tests `test_loop.py`.

**Task P1-1.1: connector declares origin; stop inferring from source string**
- Step 1 (RED): a live connector (source="jira") returning dict → Evidence must have
  `origin=LIVE_CORPORATE`, not LOCAL_SNAPSHOT. A web connector (source="web") →
  `origin=PUBLIC_WEB`.
- Step 2: `_as_evidence` requires connector to expose `origin` attribute
  (EvidenceOrigin); if connector returns Evidence already, keep its origin. Remove the
  `source == "agent_memory"` → LOCAL_SNAPSHOT branch.
- Step 3 (GREEN): pass.
- Step 4: commit `fix(adaptive): connector-declared origin, no source-string inference (P1-1)`.

**Task P1-1.2: plan drives coordinator (sources/queries/web/topk)**
- Step 1 (RED): AdaptiveLoop with planner returning `include_web=False` and
  `sources=("local",)` → web connector in `connectors` dict is NOT called; only local.
  Queries from `plan.queries` used (not just original).
- Step 2: loop builds `RetrievalCoordinator(connectors)` once; calls
  `coordinator.search(sub_request)` per `plan.queries` (or single if no decomposition);
  passes `availability` from real `health()` to planner.
- Step 3 (GREEN): pass.
- Step 4: commit `fix(adaptive): QueryPlan drives unified RetrievalCoordinator (P1-1)`.

**Task P1-1.3: real availability via health()**
- Step 1 (RED): connector with `health()` returning available=False → excluded from plan
  sources; not searched.
- Step 2: loop calls `await connector.health()` per connector, builds availability dict,
  passes to `planner.plan(query, availability, hints)`.
- Step 3 (GREEN): pass.
- Step 4: commit `fix(adaptive): availability from health(), not hardcoded True (P1-1)`.

**Task P1-1.4: apply topk / max_results + structured error diagnostics**
- Step 1 (RED): request topk=3 → results length <= 3 even after fuse. Failed connector
  recorded in trace with exception type, not silently swallowed.
- Step 2: after coordinator search, slice to `plan.max_results or request.topk`; collect
  per-source errors into response `trace` (no bare `except: continue`).
- Step 3 (GREEN): pass.
- Step 4: commit `fix(adaptive): topk applied + degraded-source diagnostics (P1-1)`.

---

## P1-2: Unified manifest + publish (single RevisionManifestStore)

**Problem:** Three divergent manifest schemas (IndexManifest non-atomic, SyncEngine
`{active_index,cursor}`, RevisionPublisher `{profile,active_revision}`). Two publish paths.

**File:** new `rag_core/gateway/sync/manifest_store.py` (atomic, versioned); refactor
`SyncEngine`, `RevisionPublisher`, `IndexManifest` to use it. Tests `test_manifest_store.py`.

**Task P1-2.1: versioned atomic RevisionManifestStore**
- Step 1 (RED): store.write(profile, active_revision, cursor) → atomic file;
  store.read() returns same; corrupt write never leaves partial (os.replace).
- Step 2: implement `RevisionManifestStore` with single schema
  `{"schema_version":1,"profile":{...},"active_revision":str,"cursor":str|null}`.
- Step 3 (GREEN): pass.
- Step 4: commit `feat(sync): unified atomic RevisionManifestStore (P1-2)`.

**Task P1-2.2: SyncEngine + RevisionPublisher use the store**
- Step 1 (RED): SyncEngine.publish and RevisionPublisher.publish both go through
  `RevisionManifestStore`; reading active revision from one is visible to the other.
- Step 2: refactor both to delegate; deprecate `IndexManifest.write` non-atomic path
  (keep read-compatible shim or migrate). Remove `atomic_write_json` divergence.
- Step 3 (GREEN): pass.
- Step 4: commit `refactor(sync): unify publish via RevisionManifestStore (P1-2)`.

---

## P1-3: Sync builds actual knowledge index (chunk/embed/index)

**Problem:** SyncEngine writes raw docs.jsonl only; no parse/chunk/embed/lexical/vector.
ReindexPlanner copies dicts without embeddings.

**Scope for this phase:** integrate embedding + lexical index build into the publish path
using Phase 2.5 providers + ZVec adapter. Full ZVec vector publish is environment-dependent
(ZVec installed); guard with capability check, fallback to lexical index.

**File:** `rag_core/gateway/sync/index_builder.py`, extend `engine.py` publish.
Tests `test_index_builder.py` (CPU embeddings via sentence-transformers optional; lexical
always; ZVec skipped if unavailable).

**Task P1-3.1: index builder chunks + lexical-indexes staged docs**
- Step 1 (RED): build_index(docs) → returns lexical index structure + chunk map;
  retrievable offline.
- Step 2: implement chunker + simple lexical inverted index in `index_builder.py`.
- Step 3 (GREEN): pass.
- Step 4: commit `feat(sync): chunk + lexical index builder (P1-3)`.

**Task P1-3.2: embed + validate profile on publish (optional CPU)**
- Step 1 (RED): if EmbeddingProvider available, build_index computes vectors, records
  EmbeddingProfile in manifest; if profile mismatches active → reject (reuse compatibility).
- Step 2: wire provider (CPU/OpenAI from Phase 2.5) into build_index; ZVec publish
  best-effort (skip if backend absent).
- Step 3 (GREEN): pass.
- Step 4: commit `feat(sync): embed + profile validation in index build (P1-3)`.

---

## P1-4: Persistent FeedbackStore + MemvidEnricher integration

**Problem:** FeedbackStore in-memory (lost on restart); evaluate() is a stub.
MemvidEnricher builds episode but never stores; loop drops it.

**File:** `rag_core/gateway/adaptive/feedback_store.py` (JSONL persistence + aggregate +
golden eval hook), `rag_core/gateway/adaptive/enrichment.py` (store backend param),
`adaptive/loop.py` (pass index_revision/embedding_profile_id, persist episode).
Tests `test_feedback_persistence.py`, `test_enrichment_store.py`.

**Task P1-4.1: FeedbackStore persists to JSONL + reloads**
- Step 1 (RED): record event → file written; new store from same path reloads events;
  aggregate reflects persisted events.
- Step 2: add `persist_path` to FeedbackStore; append on record, load on init.
- Step 3 (GREEN): pass.
- Step 4: commit `feat(adaptive): persistent FeedbackStore (P1-4)`.

**Task P1-4.2: evaluate() runs golden-set candidate policy (stub-but-real)**
- Step 1 (RED): evaluate(golden) returns policy delta suggestion, not just counts;
  canary flag field present.
- Step 2: implement minimal aggregate→candidate-policy compare vs golden; return
  `{candidate_policy, canary: bool, events}`.
- Step 3 (GREEN): pass.
- Step 4: commit `feat(adaptive): golden-set eval hook in FeedbackStore (P1-4)`.

**Task P1-4.3: MemvidEnricher stores episode + loop passes provenance**
- Step 1 (RED): enricher with store_backend writes episode JSONL; loop passes
  index_revision + embedding_profile_id + successful; stored episode retrievable.
- Step 2: add `store_backend` to MemvidEnricher (file or in-memory with path);
  loop calls `enricher.build_episode(..., index_revision=..., embedding_profile_id=...)`,
  then `enricher.store(episode)`.
- Step 3 (GREEN): pass.
- Step 4: commit `feat(adaptive): MemvidEnrichment stores episode with provenance (P1-4)`.

---

## Phase 6 Verification Gate

```bash
python -m pytest tests/gateway/ -q
python -m pytest tests -q   # must remain >= 258 passed, 4 skipped, 1 xfailed + new
```

**Closes:** P0-1, P0-2, P1-1, P1-2, P1-3, P1-4 (all 2 P0 + 6 P1 from audit).
**P2 items:** P2-9 (unified core) closed by P1-1.2; P2-10 (topk) by P1-1.4;
P2-11 (error swallow) by P1-1.4; P2-8 (connector factory) — separate follow-up
(config loader) noted but out of this phase's P0/P1 scope.

→ After Phase 6: ADR-001 ~85%, ADR-002 ~85%, ADR-003 ~75%. Re-audit required.
