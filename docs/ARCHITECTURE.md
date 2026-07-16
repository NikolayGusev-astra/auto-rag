# Architecture

`auto-rag` is a local-first retrieval pipeline with optional remote sources and
an episodic memory layer. The default deployment uses LM Studio for embeddings
and generation, ZVec for local retrieval, and MCP/Web/Federation as fallbacks.

## Request Flow

```text
Query
  -> DCD router (domain, collection, confidence)
  -> memvid recall
       -> high-confidence hit: return prior answer
       -> miss: continue
  -> ZVec + Web in parallel
  -> entity/score gate
  -> MCP fallback
  -> Web fallback
  -> Federation fallback
  -> result chunks + RagTrace
  -> LRU cache + routing log + memvid record
```

## Components

| Component | Responsibility | Main files |
|---|---|---|
| DCD | Classifies query domain and collection. | `rag_core/dcd_router.py` |
| Core RAG | Orchestrates retrieval, fallback policy, cache and tracing. | `rag_core/rag_async.py` |
| Local search | ZVec hybrid vector/FTS retrieval; Chroma fallback on non-AVX2 hosts. | `rag_core/zvec_adapter.py`, `rag_core/unified_searcher.py` |
| MCP | Queries Jira, Confluence, Context7 and other configured sources. | `rag_core/rag_mcp_client.py` |
| Web | SearXNG discovery plus guarded full-text fetch. | `rag_core/rag_async.py` |
| Federation | Queries other RAG nodes through HTTP or SSH tunnels. | `rag_core/rag_federated.py` |
| Episodic memory | Records useful chunks and semantically recalls prior answers. | `rag_core/memvid_memory.py` |
| Observability | Structured stage timings and routing decisions. | `rag_core/rag_trace.py`, `rag_core/routing_log.jsonl` |

## Episodic Memory

Memvid is opt-in and local-only. A tenant has one capsule file and a persisted
sidecar vector index:

```text
memory_<tenant>.mv2
memory_<tenant>.mv2.vecidx.jsonl
```

On record, `auto-rag` stores the query, useful chunk text, source metadata and
`RagTrace`. On recall, it embeds the query with LM Studio and cosine-ranks prior
episodes. A hit at or above `RAG_MEMVID_RECALL_THRESHOLD` short-circuits the
core retrieval pipeline.

The memory layer fails open: unavailable SDK, capsule or embedding endpoint
returns an empty recall and leaves the normal RAG pipeline intact.

## Storage and Security Boundaries

- Capsules and local vector indexes are runtime data; do not commit them.
- Web full-text retrieval rejects loopback, private, link-local and carrier-grade
  NAT addresses before fetching URLs.
- Federation binds to `127.0.0.1` when no API key is configured.
- JQL queries escape quotes before URL encoding.

## Indexers

`indexer.py` and `zvec_incremental_indexer.py` target different deployments but
share pure helpers through `rag_core/index_common.py`. Their path selection,
state files, chunking policy and collection schema remain intentionally local to
each indexer.
