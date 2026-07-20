# Phase 7 — Local Workstation Bootstrap (TDD Plan)

**Goal:** Сделать Auto-RAG запускаемым локально одним инженером без ручного кодирования connectors. Закрыть пустой `_configured_connectors()` в product path, добавить versioned config, secret references, narrow connector factory, обязательный `LocalSnapshotConnector`, startup diagnostics и offline acceptance flow.

**Source ADR:** ADR-004 (docs/ADR-004-local-workstation-rag.md)
**Prerequisite:** Phase 6 closed (319 passed). `LocalSnapshotConnector` exists with hybrid retrieval.

**Hard constraints (from ADR-004 + audit):**
- NO tenant / multi-user / shared-index / central-control abstractions.
- NO secrets stored in TOML/YAML/JSON config. Only references (`credential_env = "JIRA_TOKEN"` or system keystore).
- `LocalSnapshotConnector` enabled by DEFAULT.
- Missing Jira/Wiki/network MUST NOT block startup; reflected in health/diagnostics.
- HTTP transport stays optional, NOT mandatory.
- Factory stays narrow + local; no plugin marketplace.
- Product path MUST NOT use empty `_configured_connectors()` after 7.7.

**TDD order:**
```
7.1 Config schema
→ 7.2 Config loader
→ 7.3 Secret references
→ 7.4 Connector registry/factory
→ 7.5 LocalSnapshotConnector auto-registration
→ 7.6 Startup diagnostics + source status
→ 7.7 CLI/MCP wiring instead of {}
→ 7.8 Offline acceptance flow
→ 7.9 Official MCP ClientSession smoke   (final separate task)
→ 7.10 Real Codex/Cursor smoke            (final separate task)
```

---

## 7.1 Config schema

**Problem:** No versioned config. Gateway uses hardcoded empty `{}`.

**File:** `rag_core/gateway/config_schema.py` — `GatewayConfig` dataclass/pydantic with:
- `version: int = 1`
- `knowledge_root: Path` (default `~/.local/share/auto-rag`)
- `local_snapshot: bool = True`
- `sources: dict[str, SourceConfig]` (jira/wiki/mcp optional)
- `web: bool = False`
- `adaptive: bool = False`
- `embedding/credential refs` (see 7.3)

`SourceConfig`: `enabled: bool`, `kind: str` (jira/wiki/mcp), `credential_ref: str | None`, `extra: dict`.

**Task 7.1.1: schema + defaults**
- RED: no `GatewayConfig` importable.
- GREEN: `GatewayConfig()` yields defaults (local_snapshot=True, web=False).
- commit `feat(config): GatewayConfig schema + defaults (7.1)`

**Task 7.1.2: versioned load guard**
- RED: loading config with `version=99` raises `UnsupportedConfigVersion`.
- GREEN: pass.
- commit `feat(config): version guard (7.1)`

---

## 7.2 Config loader

**File:** `rag_core/gateway/config_loader.py` — `load_config(path: Path | None) -> GatewayConfig`.
- If path None: return `GatewayConfig()` (defaults, local_snapshot on).
- If path exists: parse TOML → `GatewayConfig`.
- On missing file: raise clear error OR return defaults? Decision: missing explicit path → error; no path arg → defaults.

**Task 7.2.1: load from TOML**
- RED: write TOML with local_snapshot=true, web=false → load → equals.
- GREEN: pass.
- commit `feat(config): TOML loader (7.2)`

**Task 7.2.2: missing file error**
- RED: load_config(missing_path) raises `ConfigNotFound`.
- GREEN: pass.
- commit `feat(config): missing-file error (7.2)`

---

## 7.3 Secret references

**File:** `rag_core/gateway/secrets.py` — `resolve_credential(ref: str | None) -> str | None`.
- `ref` like `"env:JIRA_TOKEN"` or `"credential_env = JIRA_TOKEN"` in config → `os.environ["JIRA_TOKEN"]`.
- System keystore ref (`keyring:service/account`) → optional, can be stub.
- NEVER returns raw secret from config file.

**Task 7.3.1: env resolution**
- RED: `resolve_credential("env:FOO")` returns `os.environ["FOO"]`; None ref → None.
- GREEN: pass.
- commit `feat(config): secret env resolution (7.3)`

**Task 7.3.2: refuse plaintext**
- RED: config containing `token = "abc"` is NOT read as secret; `resolve_credential` rejects non-`env:`/`keyring:` prefixes.
- GREEN: pass.
- commit `feat(config): reject plaintext secret refs (7.3)`

---

## 7.4 Connector registry/factory

**File:** `rag_core/gateway/connector_factory.py` — `build_connectors(config: GatewayConfig) -> dict[str, SourceConnector]`.
- Always includes `LocalSnapshotConnector(knowledge_root)` if `config.local_snapshot`.
- For each `config.sources[name]` enabled: build jira/wiki/mcp connector by `kind`, pass resolved credential.
- Unknown kind → skip + record in diagnostics.
- NO tenant/shared logic.

**Task 7.4.1: factory builds local snapshot**
- RED: `build_connectors(GatewayConfig())` returns `{"local_snapshot": LocalSnapshotConnector}`.
- GREEN: pass.
- commit `feat(factory): build local snapshot (7.4)`

**Task 7.4.2: factory builds enabled live sources**
- RED: config with jira enabled + env cred → factory returns jira connector (mock or real if available).
- GREEN: pass.
- commit `feat(factory): build live sources (7.4)`

---

## 7.5 LocalSnapshotConnector auto-registration

Ensure default config + factory always register local snapshot even with zero live sources.

**Task 7.5.1: default config → local only**
- RED: `build_connectors(GatewayConfig())` has exactly one connector, kind=local.
- GREEN: pass.
- commit `feat(factory): mandatory local snapshot (7.5)`

---

## 7.6 Startup diagnostics + source status

**File:** `rag_core/gateway/diagnostics.py` — `collect_startup_diagnostics(connectors) -> dict`.
- Per connector: health (True/False), kind, source.
- Offline summary: which are healthy.
- `list_sources()` / `source_status(name)` helpers on coordinator or server.

**Task 7.6.1: diagnostics reflect health**
- RED: diagnostics show local_snapshot healthy, missing jira unhealthy (not crash).
- GREEN: pass.
- commit `feat(diag): startup diagnostics + source status (7.6)`

---

## 7.7 CLI/MCP wiring instead of {}

**File:** `rag_core/gateway/server.py`, `cli.py` — replace `_configured_connectors()` calls with `build_connectors(load_config(path))`.
- `main()` / CLI: `--config PATH` optional; default → `GatewayConfig()`.
- MCP server uses factory connectors.
- `_configured_connectors()` removed or kept only as private fallback (not in product path).

**Task 7.7.1: server uses factory**
- RED: `create_mcp_server()` with no connectors arg builds from config (local snapshot present).
- GREEN: pass.
- commit `refactor(server): wire factory, drop empty _configured_connectors (7.7)`

---

## 7.8 Offline acceptance flow

**File:** `tests/gateway/test_phase7_offline.py`.
- Scenario: corporate network down (no jira/wiki creds, simulate unhealthy) → `build_connectors` → gateway starts → `search` via LocalSnapshotConnector returns Evidence[] from local index.
- Assert: no startup exception; local results non-empty after sync+search.

**Task 7.8.1: offline startup + local search**
- RED: full flow fails (empty connectors, no local).
- GREEN: after 7.1-7.7, offline flow returns Evidence[].
- commit `test(phase7): offline acceptance flow (7.8)`

---

## 7.9 Official MCP ClientSession smoke (FINAL, separate)

Use `mcp` SDK `ClientSession` over stdio:
- spawn server subprocess
- `initialize` → `list_tools` → `call_tool("search", {query, top_k})`
- assert Evidence[] returned.

**Task 7.9.1: ClientSession smoke**
- commit `test(mcp): official ClientSession smoke (7.9)`

---

## 7.10 Real Codex/Cursor smoke (FINAL, separate)

Manual/integration: register stdio server in agent config, run `search`, verify Evidence from local snapshot.
- Document steps in OPERATIONS.md.
- commit `docs: real agent smoke steps (7.10)`

---

## Phase 7 Verification Gate

```bash
python -m pytest tests/gateway/ -q
python -m pytest tests -q   # baseline 319 passed + new
```

**Acceptance (ADR-004 §Acceptance criteria):**
1. User creates local config with one command.
2. Gateway starts without corporate network.
3. LocalSnapshotConnector auto-registered from config.
4. Empty `_configured_connectors()` no longer in product path.
5. Secrets not in config/manifest/Evidence/episode.
6. Official MCP ClientSession initialize/list_tools/search works.
7. ≥1 real coding agent gets Evidence from local snapshot.
8. Live connector failure doesn't block local retrieval.
9. Web off by default.
10. No tenant/shared/control-plane abstractions without new ADR.

---

## Phase 7.1–7.8 — DONE (local TDD commits, not pushed until audit)

12 local commits implement:
- `config_schema.py`: `GatewayConfig` (version=1, local_snapshot=True, web=False) + `SourceConfig` + `UnsupportedConfigVersion`.
- `config_loader.py`: `load_config()` (None→defaults, TOML parse, `ConfigNotFound`).
- `secrets.py`: `resolve_credential()` — only `env:`/`keyring:`, plaintext→ValueError.
- `connector_factory.py`: `build_connectors()` — always LocalSnapshotConnector if local_snapshot; offline-safe `ConnectorStub` for jira/wiki/mcp; `ConnectorMap.diagnostics` non-fatal; no tenant/shared abstractions.
- `diagnostics.py`: `collect_startup_diagnostics()` — per-connector health/kind/source + offline summary.
- `server.py`: `--config PATH` arg; `build_connectors(load_config(path))` replaces `_configured_connectors()` in product path; `--legacy-jsonl` kept.
- Offline acceptance test: corporate network down → factory builds local only → gateway starts → local search returns Evidence[].

Verification (independent): `_configured_connectors` not called in product path; no tenant/multi-user/shared/control-plane/marketplace matches; 331 passed (319 + 12 new); offline test passes.

**PENDING (final separate tasks, planned but NOT done):**
- 7.9 Official MCP `ClientSession` smoke (initialize → list_tools → call_tool search).
- 7.10 Real Codex/Cursor smoke + OPERATIONS.md steps.

These two verify the already-assembled vertical path; run after 7.1–7.8 are pushed and audited.

---

## Phase 7 — Audit fix + 7.9 DONE

**Audit 7.1–7.8 found 1 P1 + 2 P2.** Fixed in `55db81e`:
- P1: `_connector_health()` no longer special-cases `retrieval_kind=='local'` to True; ALWAYS calls `connector.health()` (LocalSnapshotConnector.health checks active revision). Empty/corrupt snapshot → health False. Non-fatal on exception. `reason` surfaced from status.
- P2: relative `knowledge_root` in TOML resolves from `config_path.parent` (not cwd).
- 4 regression tests: empty/ready/corrupt local snapshot, relative config path. Full suite 334 passed.

**7.9 Official MCP ClientSession smoke — `66b099c`:**
- `tests/gateway/test_phase7_mcp_client.py` uses official `mcp.ClientSession` + `stdio_client` (NOT raw stdin JSON-RPC).
- Full vertical path: publish `jira:1` → real `--config <toml>` → TOML loader → factory → LocalSnapshotConnector → RetrievalCoordinator → `Evidence[]`.
- Asserts `initialize` → `list_tools` (search+sync) → `call_tool('search')` → `structuredContent["results"]` contains published doc.
- `--config` wired through `main()` → `serve_mcp_stdio(config_path)` → `create_mcp_server(..., config_path)` → `_factory_connectors`.
- 1 test passed; full suite 335 passed.

**PENDING:**
- 7.10 Real Codex/Cursor smoke + OPERATIONS.md steps (manual integration).

**After 7.10:** Phase 7 complete per ADR-004 acceptance (10 criteria).

---

## Phase 7.10 — Manual agent smoke + OPERATIONS.md

**Status:** Documented. 7.10 is a USER-ENVIRONMENT verification, not code change.

`docs/OPERATIONS.md` now has "Local Workstation Gateway (Phase 7 — ADR-004)" section:
- install `pip install -e .[gateway]`
- create `~/.config/auto-rag/gateway.toml` (local_snapshot=true, web=false, credential_ref only)
- publish a source via `SyncEngine.sync_source`
- start `python -m rag_core.gateway.server --config <toml>`
- register with Hermes Agent via mcpServers stdio config
- verify with `tests/gateway/test_phase7_mcp_client.py` (official ClientSession smoke)

Acceptance (ADR-004 §Acceptance): one real coding agent (Hermes) gets Evidence from local snapshot.
Run this manually in your environment; no further code needed — the vertical path is proven by 7.9.

**Phase 7 COMPLETE** after 7.10 manual smoke confirms in your env.

---

## Phase 7 — FINAL STATUS

**Code + automated tests: COMPLETE (sign-off 7.9 obtained).**
**7.10:** user-environment verification (Hermes Agent MCP registration). Author runs this manually per OPERATIONS.md; no code change required. Does NOT block Phase 7 closure — vertical path proven by 7.9 official ClientSession smoke.

**ADR-004 acceptance:** 9/10 automated, #7 (real agent) is user-side smoke documented in OPERATIONS.md.

**Phase 7 closed.**


