# ADR Migration — Phase 6: Audit Remediation (P0/P1)

> **Context:** Post-implementation audit of `a03b962` found the gateway is a correct
> **scaffold** but not ADR-complete. 2 P0 (blocking) + 6 P1 (major) are real and confirmed
> by direct code inspection. This phase closes them. Each task = one narrow patch, TDD.
> Do NOT push until the full phase is green and re-audited.

> **Execution order (user decision):** Run **6.1 (P0-1) isolated first**, push, then a
> separate re-audit of sync semantics. Only after P0-1 is confirmed correct, proceed to
> 6.2 → 6.3 → 6.4 → 6.5 → 6.6. Reason: destructive sync risks losing the local offline
> snapshot — the core product value — so it must be closed and verified alone before any
> other change enters the same commit series.

**Audit verdict accepted:** ADR-001 ~55-60%, ADR-002 ~70-75%, ADR-003 ~40-45% realized.
The following tasks raise that to functional completeness for the core paths.

**Baseline before this phase:** `258 passed, 4 skipped, 1 xfailed` (tests -q).
**After 6.1 (P0-1):** `267 passed, 4 skipped, 1 xfailed` (non-destructive sync closed).

---

## 6.1 / P0-1: Non-destructive incremental sync  [DONE — closed after re-audit round 2]

**Status:** implemented `2780520`, `90fd9df`, `f667443` (non-destructive merge) +
`f307106` (fail-closed on corrupt active revision). Re-audit round 2 found the original
fix was FAIL-OPEN on corrupt active snapshot (silent data loss). Fixed: `_previous_active_documents`
now raises `CorruptActiveRevisionError` when an existing active revision is unreadable
(manifest/docs.jsonl/path broken); first-time sync (no active revision) still returns [].
`full_rebuild(source, batch)` is the ONLY authorized bypass (receives full authoritative
snapshot). 19 sync tests pass (T1-T5 regression: corrupt docs.jsonl/manifest/path → error,
active unchanged; first sync OK; full_rebuild recovers). Full suite 271 passed.

**Verdict:** P0-1 fully closed — non-destructive merge + fail-closed + recovery path.
Proceed to 6.2.

---

## P0-1: Non-destructive incremental sync  (spec retained for reference)

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

## P0-2: Real MCP transport (SDK-first, no self-written fallback)

**Problem:** `server.py` speaks a custom line-JSON protocol, not MCP. No `initialize`,
`tools/list`, `tools/call`. Real MCP clients (Claude/Cursor/Codex) cannot register it.

**Decision (user, 2026-07-19):** Use the **official MCP SDK as the primary transport**.
Do NOT build a self-written MCP surface as fallback — that duplicates the protocol matrix
and risks "almost-MCP" again. Offline-capable ≠ no installed deps: the SDK runs locally
without network. Model:

```
pyproject.toml:
  [project.optional-dependencies]
  gateway = ["mcp>=1.0"]

Without extra:
  - core retrieval, sync, local Python API, legacy/debug JSON-lines CLI
With extra (gateway):
  - real MCP stdio server (mcp SDK FastMCP / low-level Server)
```

The existing custom JSON-lines `server.py` is kept ONLY as an explicitly-labeled
**debug/legacy transport** (`serve_stdio_debug`), never advertised as MCP. If the MCP SDK
is proven incompatible with the target Astra Linux / Python runtime, that becomes a separate
blocking technical constraint — until then, SDK-first.

**Tasks P0-2.1-2.3 are revised:** implement via `mcp` SDK `Server` (or `FastMCP`), register
gateway tools (`search`, `fetch`, `sync`, `sync_status`, `list_sources`, `source_status`)
with JSON Schema input schemas, handle `initialize`/`tools/list`/`tools/call`/`ping` from
the SDK. Smoke test against the SDK's own client (`mcp.client.stdio`).

**Files:** `rag_core/gateway/mcp_server.py` (new, SDK-based), `rag_core/gateway/server.py`
(relabeled debug/legacy, keep `dispatch` for tool handlers + `serve_stdio_debug`),
`pyproject.toml` (add `gateway` extra), tests `tests/gateway/test_mcp_server.py`.

**Task P0-2.1: MCP server via SDK (initialize/tools/list)**
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

## 6.2 / P1-2: Unified manifest/publisher  [DONE — closed after re-audit]

**Status:** `67cdc31` (unified store) + `35b07eb` (directory fsync P1). One
`RevisionManifestStore` (atomic os.replace + file fsync + parent-dir fsync, versioned
schema). SyncEngine / RevisionPublisher / IndexManifest all delegate. Distinct failure
states preserved (manifest missing / corrupt / active revision missing / invalid).
Re-audit found missing directory fsync -> fixed with portable best-effort helper
(EACCES/ENOTSUP/EINVAL ignored). 7 manifest_store tests + 278 full suite green.
P2 hardening backlog: single-writer policy, relative revision IDs + root escape guard,
legacy-schema migration path, typed `kind`+`payload` schema.

**Verdict:** 6.2 fully closed. Proceed to 6.3.

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

## 6.3 / P1-3: Real index build  [DONE — pending audit]

**Status:** `62f5166`. New `rag_core/gateway/sync/index_builder.py`: deterministic
`chunk()` (stable `{doc_id}:{i}` IDs, empty->[], oversized split), `build_lexical_index()`
(inverted term->chunk_ids), `embed_chunks()` (optional provider; None -> lexical only,
NOT blocked), `validate_profile()` (full compatibility contract from 2.5), `build_revision()`
writes docs.jsonl + chunks.jsonl + lexical.json + optional vectors.jsonl + manifest
(embedding profile or None). SyncEngine.stage_sync forms merged snapshot (carry-forward +
added + changed - deleted) and passes to build_revision, so staged revision is FULL, not
batch-only. 9 index_builder tests cover all 10 acceptance invariants: changed replaces
old chunks, deleted removes docs+chunks+lexical+vectors, lexical-only without provider,
incompatible profile blocks publish, partial embed failure flags doc (no active corruption),
stable/dedup chunk IDs, empty/oversized handling, full-snapshot staging, corrupt staged
blocks publish. Full suite 287 passed.

**Re-audit (commit `62f5166`) found 4 P1 + 3 P2.** All 4 P1 fixed:
- `551b569` — EmbeddingProviderUnavailable + `allow_lexical_downgrade` (P1-1: no silent vector loss); async `build_revision` + `stage_sync_async` + executor for sync CPU (P1-2: async provider no longer degrades to silent failure); `full_rebuild` accepts provider/profile (P1-3); rejects wrong-dimension + non-finite vectors.
- `10806f4` — LocalSnapshotConnector retrieves over lexical.json + optional cosine rerank (P1-4: artifacts now searchable end-to-end). 3 retrieval tests: unique term match, missing term empty, query_vector cosine ranking.

**Re-audit round 2 (commit `3a5e7c4`) found the guard depended on manually-passed active_profile; sync_source() did not auto-detect it -> silent downgrade still reproducible via the standard path.** Fixed:
- `3a5e7c4` — `_active_embedding_profile()` reads profile from active revision manifest; `effective_profile = active_profile or detected`; `sync_source(..., embed_provider=None)` forwards provider and fails closed when vector profile exists without provider. 2 E2E tests via sync_source (reject + compatible rebuild).
- `909e32e` — LocalSnapshotConnector upgraded lexical-AND+rerank -> hybrid lexical UNION vector (semantic-only match surfaces via vector score). 4 retrieval tests incl. "dog"->"canine companion animal" semantic match.

**P2 hardening backlog (not blocking 6.4):**
- preprocessing_revision mismatch (ADR-002) not carried in `_provider_profile` -> add to capabilities, compare in compatibility.
- strict partial-failure semantics: manifest stores only IDs, no reason/retry/status; distinguish provider-timeout vs invalid-doc vs invalid-vector vs programming error.
- query embedding inside connector: SearchRequest should carry query_vector, or connector gets EmbeddingProvider.
- full ZVec/Chroma/FTS publish remains follow-up (snapshot connector covers hybrid retrieval now).

**Verdict:** 6.3 all blocking P1 closed (artifact build, async, full rebuild, dimension/finite, auto-detect downgrade, hybrid retrieval). Awaiting final sign-off. Proceed to 6.4 after confirmation.

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

## Phase 6.4 — AdaptiveLoop execution fixes — DONE

- `QueryPlan` управляет выполнением retrieval:
  - `sources`
  - `queries`
  - `include_local`
  - `include_live`
  - `include_web`
  - `max_results`
  - `retrieval_budget_ms`
- Adaptive и reference paths используют единый `RetrievalCoordinator`.
- Compound-запросы выполняются через единый plan, без отдельного pipeline.
- Availability определяется через `connector.health()`, а не константой.
- Web connector не вызывается при `include_web=False`.
- Origin задаётся connector-ом и сохраняется в `Evidence`.
- Memory evidence объединяется с document evidence без short-circuit.
- `topk` применяется после fusion.
- Ошибки connector-ов отражаются в diagnostics, а не проглатываются молча.
- Отсутствие DCD, memory, reranker или отдельного connector-а не блокирует reference retrieval path.

Commit: `7303f93 fix(adaptive): plan-driven coordinator, origin, health, topk, diagnostics (6.4)`. 6 focused adaptive-loop tests pass; full suite 305 passed. RetrievalCoordinator already provides health_map(), web-filter, topk, structured failure logging; AdaptiveLoop now delegates to it (no direct connector iteration), uses QueryPlan when planner present, preserves origin, persists feedback + episode.

**Re-audit (commit `7303f93`) found 2 P1 + 1 P1/P2 routing defects.** Fixed in `ee796da`:
- `retrieval_kind` added to `SourceConnector` Protocol (default "live"); `LocalSnapshotConnector` sets "local"; memory/web set on construction. Loop builds `kind_availability` from `health_map()` and passes it to planner (planner expects local/live/web keys). `_selected_connectors` selects by `plan.include_local/include_live/include_web` + `connector.retrieval_kind` + explicit source match. Real source IDs (jira, local_snapshot) now routed correctly (was broken: plan.sources "local"/"live" never matched real names).
- Memory added to execution set ONLY if `availability[memory_key]` and plan requests it (not force-added via setdefault).
- `retrieval_budget_ms` applied via `asyncio.timeout` per query; TimeoutError preserves partial evidence and records `timed_out_sources`.
- 4 routing regression tests (real source IDs, explicit selection, unavailable memory, budget timeout). Full suite 309 passed.

**P2 backlog (not blocking 6.5):**
- `plan.domains` ignored by loop (passes request.domain); mark domains advisory or wire into subrequests.
- health computed twice (loop health_map + coordinator.search re-checks); pass precomputed snapshot to coordinator.search(availability=...) to avoid double remote calls.
- `useful_document_ids` is telemetry (returned docs), not confirmed usefulness; 6.5 must not train routing on it as positive signal.
- real Jira/MCP/web connectors must set `retrieval_kind` when added.

**Re-audit round 2 (commit `ee796da`) found 2 P1 still open.** Fixed in `cded9d3`:
- `retrieval_budget_ms` now a SINGLE plan-wide deadline: `async with asyncio.timeout(budget_seconds+0.02)` wraps the whole `for query in queries` loop; on TimeoutError keeps collected evidence, records `timed_out_queries` (current) + `skipped_queries` (remaining). No longer per-query timeout (was 3x budget for 3 subqueries).
- Explicit memory policy: `QueryPlan.include_memory` field added; planner sets it from `("memory" in sources) or availability["memory"]`. Loop adds memory to execution set ONLY if `availability[memory_key]` AND `plan.include_memory or "memory" in plan.sources or memory.source in plan.sources`. `_selected_connectors` no longer unconditionally accepts `retrieval_kind=="memory"`.
- 3 regression tests: compound total budget (q1 kept, q2/q3 timed_out/skipped, wall < budget+grace), healthy-but-unselected memory NOT called, explicit memory selected called. Full suite 312 passed.

**Verdict:** 6.4 fully closed (routing kind selection, memory guard, single plan budget, explicit memory policy). Awaiting final sign-off. Proceed to 6.5 after confirmation.

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
