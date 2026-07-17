# auto-rag

**Ask questions about your own technical documentation in plain language, and get
answers with sources — entirely on your own machine, with no cloud service
involved.**

## What is this?

You have years of accumulated technical knowledge: a wiki, runbooks, config files,
scattered notes, tickets. Two things are true about it:

1. **`grep` can't find anything.** It matches letters, not meaning. You have to
   already know the exact word someone used three years ago.
2. **ChatGPT can't help.** It has never seen your internal infrastructure, so it
   confidently invents answers — and sending your documentation to a cloud
   provider may not be an option anyway.

`auto-rag` fixes both. You point it at your files. It reads them, then answers
questions like *"how do I add a WireGuard peer to our VPN?"* using **your**
documents, quoting where each answer came from. Everything runs locally: the
language models live in [LM Studio](https://lmstudio.ai) on your own hardware, and
nothing leaves the machine unless you explicitly connect a remote source.

The general technique is called **RAG** (Retrieval-Augmented Generation): find the
relevant passages in your own documents first, then let a language model answer
*from those passages* instead of from memory. That is what stops it making things
up.

## Is this for me?

**Yes, if:**

- You have a real corpus of your own docs — a wiki, runbooks, configs — not a
  handful of notes.
- Your data can't go to the cloud (NDA, compliance, regulated industry, or plain
  preference).
- You want answers with citations you can verify, not confident guesses.
- You have a machine that can run a 7B model at a tolerable speed.

**Probably not, if:**

- You have twenty notes. Just read them.
- Your CPU predates 2014 — the fast backend (ZVec) needs AVX2. There is a
  ChromaDB fallback, but it's slower.
- You don't want to run and maintain a local model server.

## Why not just use...?

| Instead of this | Why it doesn't solve the problem |
|---|---|
| `grep` / IDE search | Matches exact strings. "How do we do releases?" finds nothing. |
| ChatGPT / Claude | Has never seen your private docs, and your docs would have to leave your network. |
| A hosted RAG service | Same problem: your documentation lands on someone else's servers. |
| A weekend RAG script | Getting retrieval to *work* is easy. Knowing whether it still works after you change a model is the hard part — see below. |

## What makes this different from the other 500 RAG repos

Most RAG projects stop at "chop up documents, embed them, search". This one treats
retrieval quality as something you **measure**, not something you hope for:

- **Golden-set evaluation** (`eval_golden.py`) — a fixed set of questions with
  known-good answers. Retrieval accuracy is a number, not a vibe.
- **Canary deployment** (`canary_deploy.py`) — swap the embedding model, and the
  new version is scored against the old one on that golden set. Accuracy drops
  more than 5%? Automatic rollback.
- **Judge calibration** (`calibrate_judge.py`) — the LLM that grades answer quality
  is itself checked against human-verified ratings, so you aren't just measuring
  your measurement.
- **Fails soft, everywhere** — no memory SDK, no AVX2, no Jira token, no web
  search? Each part quietly drops out and the rest keeps working.

## How it works

1. **Index.** It walks your files (`.md`, `.txt`, `.py`, `.yaml`, `.conf`, `.sh`,
   and more), splits them at paragraph boundaries, and converts each piece into a
   vector — a list of numbers where "close together" means "similar in meaning".
2. **Route.** A keyword classifier guesses the topic (`linux-admin`, `networking`,
   `devops`, …) so the search happens in the right section instead of across
   everything.
3. **Remember.** An optional episodic cache checks whether a *semantically*
   similar question was already answered. "How do I set up the VPN?" and
   "wireguard tunnel configuration" count as the same question, so the answer
   comes back instantly.
4. **Search.** Otherwise it searches your local index — and, if configured,
   Confluence, Jira, Context7 or the web as fallbacks.
5. **Verify.** A local model judges whether the retrieved passages are actually
   relevant before you see them.
6. **Learn.** Every routing decision is logged, so `dcd_learner.py` can find where
   the router misfired and retune it.

## Start Here

| Goal | Entry point |
|---|---|
| Run local retrieval | [Quick Start](#quick-start-zvec) |
| Understand the request flow | [Architecture](docs/ARCHITECTURE.md) |
| Configure, operate and troubleshoot | [Operations Guide](docs/OPERATIONS.md) |
| Enable episodic memory | [memvid Episodic Memory](#memvid-episodic-memory-semantic-cache) |
| Verify changes | [Tests](#tests) |

## Highlights

- Local hybrid retrieval: ZVec vector/FTS search with Chroma fallback.
- DCD routing, MCP integrations and Web/Federation fallback paths.
- SSRF-protected web retrieval and authenticated federation endpoints.
- Persistent episodic semantic memory with LM Studio embeddings.
- Structured `RagTrace`, golden-set evaluation and canary tooling.



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

`rag_core/memvid_memory.py` adds a **read-through episodic semantic cache**
on top of the generic pipeline. On a high-confidence hit it returns the prior
answer before ZVec/MCP/Web; on a miss it stores the useful RAG chunk content
and sources for future recall. It is dependency-optional: if `memvid-sdk` is
not installed, `_NoopMemvidBackend` is used and the pipeline runs unchanged.

```
async_rag_search(query)
  ├─ memvid.recall(query, domain)        # semantic short-circuit gate
  │     if top_score >= recall_threshold (0.75):
  │        RETURN {answer, sources, from_memory:True}   ← ZVec/MCP/Web skipped
  ├─ [miss] generic pipeline (ZVec ∥ Web → entity match → MCP → federation)
  └─ memvid.record(episode)              # stores chunk text + sources, only on miss
```

- **Backend**: `memvid-sdk` 2.x (`import memvid_sdk`; legacy `memvid` fallback).
- **Capsule**: one local `<dir>/memory_<tenant>.mv2` file per tenant; the
  native vector index is persisted inside the same file.
- **Semantic index**: episodes are written with precomputed LM Studio vectors via
  the memvid native API. Reopen-safe semantic recall uses the native MV2 index,
  not a JSONL sidecar.
- **Embeddings**: LM Studio OpenAI-compatible `/v1/embeddings` endpoint. Requests
  bypass HTTP proxies for `localhost` so an LLM proxy cannot break local recall.
- **Failure mode**: SDK missing or embedding unavailable means noop/empty recall;
  the normal RAG pipeline remains available.
- **Verification**: use `hermes_memory_cli.py stats` and `search` against the
  capsule; a cache hit returns `from_memory=true` and skips the core pipeline.

### Pipeline Stages

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
# Install: pip install "memvid-sdk>=2.0"
# LM Studio must serve a 1024d embedding model at /v1/embeddings.
RAG_MEMVID_ENABLED=true
RAG_MEMVID_MODE=both                   # off | recall | record | both
RAG_MEMVID_DIR=./memvid_capsules       # local-only; do not commit capsules
RAG_MEMVID_TENANT=hermes_default
RAG_MEMVID_EMBED_URL=http://localhost:1234/v1/embeddings
RAG_MEMVID_EMBED_MODEL=bge-m3
RAG_MEMVID_RECALL_TOPK=5
RAG_MEMVID_RECALL_THRESHOLD=0.75
RAG_MEMVID_TEMPORAL=true

# Verify a configured capsule
python3 rag_core/hermes_memory_cli.py --capsule \
  ./memvid_capsules/memory_hermes_default.mv2 stats
python3 rag_core/hermes_memory_cli.py --capsule \
  ./memvid_capsules/memory_hermes_default.mv2 search "known query"
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
