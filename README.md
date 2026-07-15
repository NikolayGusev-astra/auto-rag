# auto-rag

Production-ready generic RAG pipeline: **ZVec** (vector search, AVX2 required) or **ChromaDB** (fallback for no-AVX2 hosts) + MCP (Confluence, Jira, Context7, Lodestone) + Web fallback (SearXNG).

## Which version?

| | ZVec (default) | ChromaDB (fallback) |
|---|---|---|
| **Требования к CPU** | AVX2 (Intel Haswell 2014+, AMD Excavator 2015+) | любой x64 |
| **Скорость индексации** | ~500 docs/min | ~200 docs/min |
| **Поиск** | ~10ms | ~20ms |
| **Установка** | `pip install -r requirements.txt` | уже есть, не требует сборки |
| **Код** | `rag_core/` | `chroma/` |

**Как выбрать:** запусти `python3 -c "import cpuinfo; print(cpuinfo.get_cpu_info().get('flags',[]))" | grep avx2`. Если есть AVX2 — используй ZVec. Нет — иди в `chroma/`.

---

## Quick Start (ZVec)

```bash
pip install -r requirements.txt

# LM Studio: загрузить bge-m3, qwen2.5-7b-instruct

# Индексация wiki
python3 rag_core/indexer.py --clear

# Поиск
python3 rag_core/rag_search.py "твой запрос"

# Golden set evaluation
python3 rag_core/eval_golden.py
```

## Quick Start (ChromaDB)

```bash
cd chroma/
pip install -r ../requirements.txt

# Индексация — рекурсивный чанкинг (рекомендуется)
python3 rag_indexer.py --incremental --chunk-mode recursive

# Поиск — через chroma/rag_search.py
# Калибровка LLM-судьи
python3 calibrate_judge.py

# Canary deploy с ML-метриками
python3 canary_deploy.py --quick
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
| **Recursive Chunking** | `--chunk-mode recursive\|fixed` — параграф-ориентированное разбиение (def: recursive) |
| **ZVec / ChromaDB** | In-process HNSW vector search (bge-m3, 1024d) / ChromaDB fallback |
| **LLM Verify** | Local qwen2.5-7b-as-judge for source quality (calibrated thresholds) |
| **Judge Calibration** | `calibrate_judge.py` — калибровка LLM-судьи на human-verified golden set |
| **Embedding Drift** | `canary_deploy.py` — детекция дрейфа эмбеддинг-модели при деплое |
| **MCP** | Multi-source: Confluence (REST CQL), Jira (REST JQL), Context7 (SSE), Lodestone (SSE) |
| **Web** | SearXNG + Trafilatura full-text extraction |
| **RagTrace** | Structured telemetry: every stage, decision, latency. Saved in eval reports |
| **Canary Deploy** | Compare baseline vs candidate accuracy, auto-rollback if regression >5% |
| **Caching** | 100-query LRU cache, evicts stale entries |
| **RAG v2** | Optional pipeline: LLM decompose → parallel sources → local reranker → LLM fusion |
| **Streaming** | Async generator + SSE for Tauri/web clients |

## Environment

All config via env vars (see `rag_core/rag_config.py` for defaults):

```bash
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

## Evaluation (ZVec)

```bash
# Full golden set
python3 rag_core/eval_golden.py

# Unified eval (ZVec + ChromaDB)
python3 eval_golden_unified.py

# Canary mode
python3 rag_core/canary_deploy.py --baseline-dir ./baseline --backup-dir ./backup
python3 rag_core/canary_deploy.py --quick
```

## Evaluation (ChromaDB)

```bash
cd chroma/

# Judge calibration
python3 calibrate_judge.py

# Canary deploy with embedding drift check
python3 canary_deploy.py --quick
```

## Tests

```bash
# ZVec tests
python3 -m pytest tests/ -v

# ChromaDB tests
python3 -m pytest chroma/tests/ -v
```

Reports saved as `golden_eval_report.json` with LLM-as-judge scoring and full RagTrace per question.
