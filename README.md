# auto-rag

Production-ready generic RAG pipeline: ZVec (vector search, AVX2 required) / ChromaDB (fallback for no-AVX2 hosts) + MCP (Confluence, Jira, Context7, Lodestone) + Web fallback (SearXNG).

## Quick Start

```bash
# Dependencies
pip install -r requirements.txt

# LM Studio (local embedding + LLM evaluation)
# Load models: bge-m3, qwen2.5-7b-instruct, google/gemma-4-e4b

# Index your wiki/docs
python3 rag_core/indexer.py --clear

# Search
python3 rag_core/rag_search.py "your query"

# Run golden set evaluation
python3 rag_core/eval_golden.py

# For hosts without AVX2 (old Intel Xeon, etc.):
python3 -c "from rag_core.chroma_adapter import ChromaIndexer; ChromaIndexer().index(['~/wiki'])"
python3 -c "from rag_core.chroma_adapter import ChromaSearcher; r=ChromaSearcher().search('your query', topk=5); print(r)"
```

## Architecture

```
Query → DCD Router (domain classification)
  → [low confidence] → MCP fallback chain (Confluence → Jira → Context7 → Lodestone → Web)
  → [high confidence] → ZVec/Chroma vector search → LLM verify → Answer
```

## Pipeline Features

| Stage | Description |
|-------|-------------|
| **DCD Router** | Keyword-based domain/collection classifier (15+ domains, anti-keywords for precision) |
| **ZVec / ChromaDB** | In-process HNSW vector search (bge-m3, 1024d) / ChromaDB fallback for no-AVX2 hosts |
| **LLM Verify** | Local qwen2.5-7b-as-judge for source quality (configurable threshold) |
| **MCP** | Multi-source: Confluence (REST CQL), Jira (REST JQL), Context7 (SSE), Lodestone (SSE) |
| **Web** | SearXNG + Trafilatura full-text extraction |
| **RagTrace** | Structured telemetry: every stage, decision, latency. Saved in eval reports |
| **Canary Deploy** | Compare baseline vs candidate accuracy, auto-rollback if regression >5% |
| **Caching** | 100-query LRU cache, evicts stale entries |
| **RAG v2** | Optional pipeline: LLM decompose → parallel sources → local reranker → LLM fusion |
| **Streaming** | Async generator + SSE for Tauri/web clients |

## Environment

All config via env vars (see `rag_core/rag_config.py` for defaults):

```
# Required
RAG_EMBEDDING_URL=http://localhost:1234/v1/embeddings

# Optional MCP servers
RAG_CONFLUENCE_URL=https://confluence.example.com
RAG_CONFLUENCE_TOKEN=...
RAG_JIRA_URL=https://jira.example.com
RAG_JIRA_TOKEN=...
RAG_LODESTONE_URL=https://lodestone.example.com/mcp/
RAG_LODESTONE_TOKEN=...
RAG_CONTEXT7_URL=https://context7.com/mcp/

# Web fallback
RAG_SEARXNG_URL=http://localhost:8080

# Optional: LLM DCD mode (keyword/llm/hybrid)
RAG_DCD_MODE=keyword

# Optional: local reranker (sentence-transformers)
RAG_LOCAL_RERANKER=true
```

## Evaluation

```bash
# Full golden set
python3 rag_core/eval_golden.py

# Unified eval (ZVec + ChromaDB)
python3 eval_golden_unified.py

# Canary mode
python3 rag_core/canary_deploy.py --baseline-dir ./baseline --backup-dir ./backup
python3 rag_core/canary_deploy.py --quick
```

## ChromaDB (Hosts without AVX2)

ZVec requires AVX2 (2014+ Intel/AMD). For older CPUs (Intel Xeon E5 v2, 2013) use ChromaDB adapter:

```bash
# Populate ChromaDB from wiki (single run)
python3 -c "from rag_core.chroma_adapter import ChromaIndexer; ChromaIndexer().index(['~/wiki'])"

# Search via Chroma
python3 -c "from rag_core.chroma_adapter import ChromaSearcher; r=ChromaSearcher().search('your query', topk=5); print(r)"
```

**Interface:** `ChromaSearcher` matches `ZVecSearcher` — same `search(query, topk, domain)` → same `list[dict]` format. Drop-in replacement.

**Trade-off:** ChromaDB ≈2-3x slower on insert (no batching), but search latency is comparable (~400ms on 1vCPU/891MB VPS). For production, use ZVec where AVX2 available.

Reports saved as `golden_eval_report.json` with LLM-as-judge scoring and full RagTrace per question.
