# auto-rag

Production-ready generic RAG pipeline: **ZVec** (vector search, AVX2 required) or **ChromaDB** (fallback for no-AVX2 hosts) + MCP  + Web fallback

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
  ↺ memvid episodic memory (short-circuit) — see below
```

### memvid Episodic Memory (semantic cache)

`rag_core/memvid_memory.py` adds a **read-through semantic cache** on top
of the generic pipeline. It is dependency-optional: if `memvid-sdk` is
not installed, `_NoopMemvidBackend` is used and the pipeline runs
unchanged.

```
async_rag_search(query)
  ├─ compound split (product + infra → parallel subqueries)
  ├─ memvid.recall(query, domain)        # short-circuit gate
  │     if top_score >= recall_threshold (0.75):
  │        RETURN {answer, sources, from_memory:True}   ← zvec/mcp/web SKIPPED
  ├─ [miss] generic pipeline (zvec ∥ web → entity_match → mcp → federation)
  └─ finally:
        ├─ memvid.record(episode)         # write-through (only if !from_memory)
        └─ dcd_learner.log_routing(...)   # feed DCD training log
```

- **Backend**: wired to the real `memvid-sdk` 2.0.160 (`import
  memvid_sdk`, legacy `memvid` as fallback). SDK `kind="basic"`
  does **not** build a searchable index from `add_memory_cards`
  without a managed embedding backend, so recall uses a **local
  persisted vec index** (`<capsule>.vecidx.jsonl`, cosine-ranked
  against LM Studio bge-m3 embeddings).
- **Embeddings**: LM Studio `:1234` (`text-embedding-baai-bge-m3-568m`,
  1024d). Proxy is force-disabled for localhost (HTTP_PROXY silences loopback).
- **Recall**: semantic match — finds the prior episode on a *paraphrased*
  query, not exact string. e2e: ~3× faster than the full pipeline on cache hit.
- **Degrade**: SDK missing / LM Studio down → noop, pipeline unaffected.



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
| **Episodic Memory (memvid)** | Read-through semantic cache: recall short-circuits (zvec/mcp/web skipped on score≥0.75), record on miss. Local vec index, noop if SDK absent |
| **DCD Learning Loop** | Every routing decision logged (`dcd_learner.log_routing` → `routing_log.jsonl`); `analyze()` audits misroutes, `patch_dcd()` retrains router |
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

# Optional: memvid episodic memory (semantic cache)
# Instal: pip install memvid-sdk==2.0.160
# LM Studio must serve bge-m3 embeddings on :1234 (NO_PROXY for localhost)
RAG_MEMVID_ENABLED=true
RAG_MEMVID_DIR=./memvid_capsules
RAG_MEMVID_EMBED_URL=http://localhost:1234/v1/embeddings
RAG_MEMVID_EMBED_MODEL=text-embedding-baai-bge-m3-568m
RAG_MEMVID_RECALL_THRESHOLD=0.75


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
