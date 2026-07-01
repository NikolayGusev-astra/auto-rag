# auto-rag

Production-ready RAG pipeline: ZVec (vector search) + MCP (Confluence, Jira, Context7) + Web fallback.

## Quick Start

```bash
# Dependencies
pip install zvec requests trafilatura

# LM Studio (local embedding + LLM evaluation)
# Load models: bge-m3, qwen2.5-7b-instruct, google/gemma-4-e4b

# Index your wiki/docs
python3 indexer.py --clear
python3 rag_search.py "your query"

# Run golden set evaluation
python3 run_golden.py
```

## Architecture

```
Query → DCD Router (domain classification)
  → [low confidence] → MCP fallback chain (Confluence → Jira → Context7 → Web)
  → [high confidence] → ZVec vector search → LLM verify → Answer

For rusbitech domain:
  ZVec + Lodestone + Confluence + Web → LLM priority chain
  
For devops/software-dev:
  ZVec → Context7 → Web fallback
```

## Pipeline Features

| Stage | Description |
|-------|-------------|
| **DCD Router** | Keyword-based domain/collection classifier (16+ domains, anti-keywords for precision) |
| **ZVec** | In-process HNSW vector search (bge-m3, 1024d), thread-safe singleton, LRU cache |
| **LLM Verify** | Local qwen2.5-7b-as-judge for source quality (0.3 threshold) |
| **MCP** | Multi-source: Confluence (REST CQL), Jira (REST JQL AND), Lodestone (SSE), Context7 (SSE library resolve) |
| **Web** | SearXNG + Trafilatura full-text extraction, per-domain preferred sources |
| **RagTrace** | Structured telemetry: every stage, decision, latency. Saved in eval reports |
| **Canary Deploy** | Compare baseline vs candidate accuracy, auto-rollback if regression >5% |
| **Caching** | 100-query LRU cache, evicts stale entries |

## Environment

All config via env vars (see `rag_config.py` for defaults):

```
# Required
RAG_EMBEDDING_URL=http://localhost:1234/v1/embeddings

# Optional MCP servers
RAG_CONFLUENCE_URL=https://confluence.example.com
RAG_CONFLUENCE_TOKEN=...
RAG_JIRA_URL=https://jira.example.com
RAG_JIRA_TOKEN=...
RAG_CONTEXT7_URL=https://context7.com/mcp/

# Web fallback
RAG_SEARXNG_URL=http://localhost:8888
```

## Evaluation

```bash
# Full golden set
python3 eval_golden.py

# Canary mode
python3 canary_deploy.py --mode baseline
python3 canary_deploy.py --mode compare
```

Reports saved as `golden_eval_report.json` with LLM-as-judge scoring and full RagTrace per question.
