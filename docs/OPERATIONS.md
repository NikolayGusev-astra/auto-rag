# Operations Guide

## Prerequisites

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
| No memory recall | Confirm LM Studio embeddings, the capsule path and the `.vecidx.jsonl` sidecar. |
| Empty local search | Check index path, collection name and embedding endpoint. |
| Federation unavailable | Check the API key, bind address, configured nodes and SSH tunnel state. |
| Web retrieval empty | Check SearXNG; private targets are intentionally blocked by the SSRF guard. |

## Runtime Data

Do not commit these local artifacts:

- `memvid_capsules/`
- `.pytest_cache/`
- `routing_log.jsonl`
- audit reports and generated review canvases
