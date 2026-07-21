# auto-rag

Corporate-first knowledge gateway for AI agents. MCP transport, structured Evidence, 8 live connectors — Jira, Confluence, Lodestone, Allowlisted Web, Hub, ZVec, SearXNG, Local Snapshot. Runs locally, degrades offline, 437 tests.

![Architecture](infographic/auto-rag-architecture.png)

## Start Here

| Goal | Entry point |
|---|---|
| Run locally | `python -m rag_core.gateway.server --config gateway.toml` |
| Operations | [Operations Guide](docs/OPERATIONS.md) |
| Architecture | [ADR-006: Stabilization](docs/ADR-006-stabilization-before-expansion.md) |
| Verify | `python -m pytest tests/ -q` → 437 passed |

## Architecture

```
Query → DCD Router (domain classification)
  → Corporate sources (Jira → Confluence → Lodestone → Allowlisted Web)
  → Local sources (Hub → ZVec/Chroma → Local Snapshot)
  → BGE-M3 reranker → canonical dedup + exact boost → Evidence[]
  ↺ memvid episodic memory
```

## Sources

| Connector | Kind | Capabilities |
|---|---|---|
| **Jira** | live-corporate | Exact-key → paginated comments (≤500) + linked issues (≤5) + enrichment diagnostics |
| **Confluence** | live-corporate | Empty-body pages → PDF attachment extraction (pymupdf). `content_status` metadata |
| **Lodestone** | live-corporate | Corporate KB via MCP HTTP |
| **Allowlisted Web** | public-web | SearXNG with domain filter (aldpro.ru, astralinux.ru). Suppressed for SIRIUS-*/INT-* |
| **Hub** | live-corporate | Astra Automation Hub — 51 code-deployment collections |
| **ZVec** | local | In-process HNSW vector search (bge-m3, 1024d). AVX2 required. |
| **SearXNG** | web | Self-hosted meta-search (localhost:8888) |
| **Local Snapshot** | local | Offline-capable published index |
| **Web** | public-web | **DISABLED** — corporate-first policy |

## Config

```toml
# ~/.config/auto-rag/gateway.toml
knowledge_root = "~/.local/share/auto-rag"
local_snapshot = true
web = false
adaptive = true

[sources.jira]
kind = "jira"
enabled = true
credential_ref = "env:JIRA_PAT"

[sources.confluence]
kind = "confluence"
enabled = true
credential_ref = "env:CONFLUENCE_PAT"

[sources.lodestone]
kind = "lodestone"
enabled = true

[sources.allowlisted_web]
kind = "allowlisted-web"
enabled = true

[sources.hub]
kind = "hub"
enabled = true
credential_ref = "env:HUB_TOKEN"

[sources.zvec]
kind = "zvec"
enabled = true

[sources.searxng]
kind = "searxng"
enabled = true
```

Secrets via `credential_ref` (env, keyring, vault) — never plaintext in config.

## MCP Registration (Hermes)

```bash
hermes mcp add auto-rag \
  --command ~/.venv/Scripts/python.exe \
  --args -m rag_core.gateway.server --config ~/.config/auto-rag/gateway.toml \
  --env JIRA_PAT=... \
  --env CONFLUENCE_PAT=... \
  --env HUB_TOKEN=... \
  --env JIRA_BASE_URL=https://jira.corp.example \
  --env CONFLUENCE_BASE_URL=https://wiki.corp.example \
  --env HUB_BASE_URL=https://hub.corp.example \
  --env PYTHONPATH=C:\Users\<user>\projects\auto-rag \
  --env NO_PROXY=127.0.0.1,localhost

hermes mcp test auto-rag   # ✓ Connected
```

## Operations

```bash
# Doctor (read-only health checks)
python -m rag_core.gateway.doctor
python -m rag_core.gateway.doctor --json    # exit codes: 0/1/2/3

# Sync (publish local snapshot)
python -m rag_core.gateway sync --source local_snapshot

# Evaluation
python -m rag_core.eval_golden
python -m rag_core.eval_golden --judge       # + Qwen-2.5 judge

# Pre-commit guard
python scripts/precommit-guard.py            # check only
python scripts/precommit-guard.py --fix      # clean + update .gitignore
```

## Tests

```bash
python -m pytest tests/ -q     # 437 passed, 5 skipped, 1 xfailed
```

Key suites: `test_jira_connector.py` (comments + linked + diagnostics), `test_confluence_connector.py` (PDF extraction + content_status), `test_lodestone_connector.py` (MCP + parsing), `test_allowlisted_web.py` (domain filter + internal skip), `test_canonical_identity.py` (dedup + boost, 10 tests), `test_doctor.py` (profiles, 6 tests), `test_phase7_factory.py` (credential_ref).

## Documents

| Doc | Purpose |
|---|---|
| [ADR-006](docs/ADR-006-stabilization-before-expansion.md) | Architecture decision — stabilization, 10 capabilities, rollout readiness |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | Request flow, security boundaries, component design |
| [OPERATIONS](docs/OPERATIONS.md) | Config, troubleshooting, porting, all connectors |
| [ADR-INDEX](docs/ADR-INDEX.md) | Full ADR history (001–006) |
