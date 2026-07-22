# Compatibility Matrix

| Component | Required | Verified On | Notes |
|-----------|----------|-------------|-------|
| **Python** | ≥ 3.11 | 3.11.9 (Windows), 3.11 (CI/Ubuntu) | 3.12 untested |
| **OS** | Linux, Windows | Windows 10, Ubuntu 22.04 (CI) | macOS untested |
| **Hermes Agent** | Any version with MCP | 2026-07 snapshot | `hermes mcp add auto-rag ...` |
| **Embedding provider** | LM Studio (BGE-M3) or CPU fallback | LM Studio 0.3.x, `intfloat/multilingual-e5-large` | `EMBED_URL`/`EMBED_MODEL` env vars |
| **ZVec** | ≥ 0.5.0 | 0.5.1 | `pip install zvec` or pre-installed |
| **SearXNG** | Any (for allowlisted web) | localhost:8888 | Optional — graceful skip |
| **Jira** | DC 9.x with REST API | 9.12.30 | `JIRA_BASE_URL` + `JIRA_PAT` |
| **Confluence** | DC 7.x+ with REST API | 7.x | `CONFLUENCE_BASE_URL` + `CONFLUENCE_PAT` |
| **Hub** | Astra Automation Hub | hub.corp.example | `HUB_BASE_URL` + `HUB_TOKEN` |
| **Lodestone** | MCP HTTP endpoint | — | `credential_ref` in gateway.toml |

## Extras

| Extra | Purpose | Adds |
|-------|---------|------|
| `[gateway]` | MCP transport | `mcp>=1.0` |
| `[pdf]` | Confluence PDF extraction | `pymupdf>=1.23`, `pdfplumber>=0.10` |
| `[reranker]` | CPU sentence-transformers fallback | `sentence-transformers>=3.0` |
| `[dev]` | Test runner | `pytest>=7.0`, `pytest-asyncio>=0.21` |
| `[web]` | Web retrieval marker | (already in base deps) |

## Install

```bash
pip install -e ".[gateway,pdf,dev]"
pytest -q  # 468 passed
```