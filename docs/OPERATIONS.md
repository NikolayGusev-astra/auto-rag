# Operations Guide

## Local Workstation Gateway (Phase 7 — ADR-004)

Auto-RAG gateway is a local, offline-capable MCP stdio server for ONE engineer.
It serves `search` and `sync` tools to a coding agent (Hermes Agent, Codex, Cursor, Claude).

### Install

```bash
pip install -e .[gateway]   # installs official MCP SDK
```

Core retrieval/sync works without the `gateway` extra; the MCP transport needs it.

### 1. Create a local config

```toml
# ~/.config/auto-rag/gateway.toml
knowledge_root = "~/.local/share/auto-rag"
local_snapshot = true
web = false
adaptive = false

[sources.jira]
kind = "jira"
enabled = false            # offline-safe stub until real connector lands
credential_ref = "env:JIRA_TOKEN"   # NEVER put the token value here

[sources.bitbucket]
kind = "mcp-proxy"
extra = { tool = "bitbucket_search_code", server = "bitbucket" }
```

`LocalSnapshotConnector` is registered automatically. `web` is off by default.
Relative `knowledge_root` resolves from this file's directory.

### 2. Build the local snapshot

The gateway serves whatever the sync engine already published. To publish a source:

```python
from rag_core.gateway.sync.engine import SyncEngine
from rag_core.gateway.models import Document, SyncBatch

engine = SyncEngine("~/.local/share/auto-rag")
await engine.sync_source(my_source)   # my_source.sync_changes() -> SyncBatch
```

After publish, `LocalSnapshotConnector.health()` returns `available=True`.

### 3. Start the gateway

```bash
python -m rag_core.gateway.server --config ~/.config/auto-rag/gateway.toml
```

No corporate network? Live sources show `unhealthy` in diagnostics; local search still works.
Legacy debug transport (newline-delimited JSON) is available with `--legacy-jsonl`.

### 4. Register with Hermes Agent (7.10 manual smoke)

Add to the Hermes MCP config (this is the one real-agent acceptance step from ADR-004):

```json
{
  "mcpServers": {
    "auto-rag": {
      "command": "python",
      "args": ["-m", "rag_core.gateway.server", "--config", "~/.config/auto-rag/gateway.toml"],
      "env": {
        "RAG_EMBED_URL": "http://localhost:1234/v1/embeddings",
        "RAG_EMBED_MODEL": "bge-m3"
      }
    }
  }
}
```

Then in Hermes: the `auto-rag` server exposes `search` and `sync`. A real `search` call
returns `Evidence[]` from the local snapshot. Stdio stderr must not corrupt the JSON-RPC
stream — the server writes diagnostics to stderr only, never stdout.

### 5. Verify

```bash
python -m pytest tests/gateway/test_phase7_mcp_client.py -q   # official ClientSession smoke
```

This launches the server as a subprocess with a real `--config` and a published snapshot,
then drives `initialize` → `list_tools` → `call_tool("search")` via the SDK `ClientSession`.

### Startup diagnostics

```python
from rag_core.gateway.connector_factory import build_connectors
from rag_core.gateway.config_loader import load_config
from rag_core.gateway.diagnostics import collect_startup_diagnostics

diag = collect_startup_diagnostics(build_connectors(load_config(cfg_path)))
# diag["connectors"]["local_snapshot"]["health"]  -> True only if a revision is published
# diag["offline"]["healthy"] / ["unhealthy"]       -> which sources are up
```

A registered-but-empty local snapshot reports `health=False` with a reason — not a false positive.

---

## Prerequisites (legacy full-RAG pipeline)

- Python 3.11+
- LM Studio serving an embedding model at `http://localhost:1234/v1/embeddings`
- AVX2 CPU for ZVec; use the Chroma path when AVX2 is unavailable

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .[dev]
```

## Local Retrieval

Build or refresh the ZVec index:

```bash
python rag_core/indexer.py --clear
python rag_core/indexer.py --incremental
```

Run a query:

```bash
python rag_core/rag_search.py "how do I configure PostgreSQL replication?"
```

## Episodic Memory

Configure an environment file or process environment:

```bash
RAG_MEMVID_ENABLED=true
RAG_MEMVID_MODE=both
RAG_MEMVID_DIR=./memvid_capsules
RAG_MEMVID_TENANT=hermes_default
RAG_MEMVID_EMBED_URL=http://localhost:1234/v1/embeddings
RAG_MEMVID_EMBED_MODEL=bge-m3
RAG_MEMVID_RECALL_THRESHOLD=0.75
```

Inspect a local capsule:

```bash
python rag_core/hermes_memory_cli.py \
  --capsule ./memvid_capsules/memory_hermes_default.mv2 stats
python rag_core/hermes_memory_cli.py \
  --capsule ./memvid_capsules/memory_hermes_default.mv2 search "known query"
```

A successful memory hit appears in the result as:

```text
from_memory=true
trace=memvid.recall(short-circuit, score=<score>)
```

## Verification

Run all unit and integration tests:

```bash
python -m pytest tests/ -q
```

Run the golden set only when LM Studio, the local index and configured external
sources are available:

```bash
python rag_core/eval_golden.py
```

## Troubleshooting

| Symptom | Check |
|---|---|
| `memvid disabled` | Confirm `RAG_MEMVID_ENABLED=true` is loaded by the active process. |
| No memory recall | Confirm LM Studio embeddings, `RAG_MEMVID_ENABLED=true` and the capsule path; `stats` should report `has_vec_index` through the native MV2 backend. |
| Empty local search | Check index path, collection name and embedding endpoint. |
| Federation unavailable | Check the API key, bind address, configured nodes and SSH tunnel state. |
| Web retrieval empty | Check SearXNG; private targets are intentionally blocked by the SSRF guard. |

## Runtime Data

Do not commit these local artifacts:

- `memvid_capsules/` (one `.mv2` capsule contains its native vector index)
- `.pytest_cache/`
- `routing_log.jsonl`
- audit reports and generated review canvases
