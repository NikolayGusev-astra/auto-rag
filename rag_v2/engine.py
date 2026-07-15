"""
RAG v2 Engine — главный async pipeline.

Архитектура:
  1. DCD классификация (keyword/llm/hybrid — configurable)
  2. LLM декомпозиция запроса → подзапросы
  3. Параллельный поиск во всех источниках (ZVec hybrid FTS+Vec)
  4. Reranker (local cross-encoder or bge-reranker)
  5. LLM fusion → итоговый ответ
  6. RagTrace
  7. LRU cache
"""

import asyncio
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from typing import Any

import aiohttp

_rag_core = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rag_core'))
if _rag_core not in sys.path:
    sys.path.insert(0, _rag_core)

from rag_config import (
    MCP_SERVERS, HYBRID_SEARCH_ENABLED, HYBRID_SEARCH_RECALL_TOPK,
    HYBRID_SEARCH_RRF_CONSTANT, DCD_MODE,
    LOCAL_RERANKER_ENABLED, LOCAL_RERANKER_MODEL, LOCAL_RERANKER_DEVICE,
)

# V2 modules
from .decomposer import decompose
from .mcp import AsyncMCPClient
from .fuser import fuse

# Reuse trace from v1
from rag_trace import RagTrace


# ── LRU Cache ────────────────────────────────────────────────────────
_CACHE = OrderedDict()
_CACHE_MAX = 100


def _cache_key(query: str, domain: str) -> str:
    return hashlib.md5(f"{query}|{domain}".encode()).hexdigest()


def _cache_get(query: str, domain: str) -> dict | None:
    key = _cache_key(query, domain)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    return None


def _cache_put(query: str, domain: str, result: dict) -> None:
    key = _cache_key(query, domain)
    _CACHE[key] = result
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


# ── DCD dispatch based on config ────────────────────────────────────
def _classify(query: str) -> dict:
    """Dispatch to configured DCD router."""
    if DCD_MODE == "llm":
        from dcd_router_llm import classify_llm
        return classify_llm(query)
    elif DCD_MODE == "hybrid":
        from dcd_router_llm import classify_hybrid
        return classify_hybrid(query)
    else:
        # keyword (default)
        from dcd_router import classify
        return classify(query)


# ── ZVec search with hybrid support ──────────────────────────────────
async def _zvec_search(query: str, domain: str = None, topk: int = 10) -> list[dict]:
    """Search ZVec with hybrid FTS+Vector if enabled."""
    from zvec_adapter import ZVecSearcher

    searcher = ZVecSearcher()

    if HYBRID_SEARCH_ENABLED:
        # Run hybrid in thread pool (Zvec is sync/C-based)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: searcher.search_hybrid(
                query,
                topk=topk,
                domain=domain,
                recall_topk=HYBRID_SEARCH_RECALL_TOPK,
                rrf_constant=HYBRID_SEARCH_RRF_CONSTANT,
            ),
        )
    else:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: searcher.search(query, topk=topk, domain=domain),
        )

    return results


async def rag_v2_search(query: str) -> dict:
    """Main RAG v2 search."""
    trace = RagTrace(query, "v2", "v2")
    t0 = time.time()

    # 1. DCD
    trace.begin("dcd")
    try:
        dcd_result = _classify(query)
    except Exception as e:
        dcd_result = {"domain": "unknown", "collection": "general", "confidence": 0.0, "fallback": True}
    trace.end("dcd", domain=dcd_result.get("domain", ""),
              collection=dcd_result.get("collection", ""),
              confidence=dcd_result.get("confidence", 0),
              router=dcd_result.get("router", "keyword"))

    # Check cache
    cached = _cache_get(query, dcd_result.get("domain", ""))
    if cached is not None:
        trace.event("cache", hit=True)
        cached["from_cache"] = True
        cached["trace"] = trace.json()
        return cached

    async with aiohttp.ClientSession() as session:
        # 2. Query decomposition
        trace.begin("decompose")
        subqueries = []
        try:
            subqueries = await decompose(query, session)
            if not subqueries:
                subqueries = [{"query": query, "source": "auto", "reason": "fallback"}]
        except Exception as e:
            subqueries = [{"query": query, "source": "auto", "reason": f"decompose error: {e}"}]
        finally:
            trace.end("decompose", subqueries=len(subqueries))

        # 3. Determine available sources
        available_sources = list(MCP_SERVERS.keys())
        available_sources.insert(0, "zvec")  # Zvec always first
        mcp = AsyncMCPClient(session, timeout=20)

        # 4. Parallel search across all sources
        trace.begin("parallel_search")
        all_chunks: dict[str, list[dict]] = {}
        try:
            tasks = []

            for sq in subqueries:
                sq_query = sq["query"]
                target = sq.get("source", "auto")

                if target == "auto":
                    # Search all available sources
                    for src in available_sources:
                        tasks.append(_search_source(mcp, src, sq_query, all_chunks, dcd_result))
                elif target == "zvec":
                    tasks.append(_search_source(mcp, "zvec", sq_query, all_chunks, dcd_result))
                elif target in available_sources:
                    tasks.append(_search_source(mcp, target, sq_query, all_chunks, dcd_result))
                else:
                    # Unknown target — search ZVec + web
                    tasks.append(_search_source(mcp, "zvec", sq_query, all_chunks, dcd_result))

            if tasks:
                await asyncio.gather(*tasks)

            total_chunks = sum(len(c) for c in all_chunks.values())
            sources_with_data = [s for s, c in all_chunks.items() if c]
        finally:
            trace.end("parallel_search", sources=len(available_sources),
                      with_data=len(sources_with_data), total_chunks=total_chunks)

        # 5. Fuser — rerank + LLM fusion
        trace.begin("fuse")
        result = {}
        try:
            result = await fuse(query, all_chunks, session)
        finally:
            trace.end("fuse", source=result.get("source", "empty"),
                      fusion=result.get("fusion_needed", False),
                      sources_used=result.get("sources_used", []))

    # 6. Final result
    elapsed = int((time.time() - t0) * 1000)
    trace.event("total", duration_ms=elapsed)
    result["trace"] = trace.json()
    result["dcd_domain"] = dcd_result.get("domain", "")
    result["dcd_collection"] = dcd_result.get("collection", "")
    result["dcd_router"] = dcd_result.get("router", "keyword")
    result["latency_s"] = round(time.time() - t0, 2)

    # Cache result
    _cache_put(query, dcd_result.get("domain", ""), result)

    return result


async def _search_source(
    mcp: AsyncMCPClient, source: str, query: str,
    all_chunks: dict[str, list[dict]], dcd_result: dict
):
    """Search a single source. Results merged into all_chunks."""
    try:
        if source == "zvec":
            chunks = await _zvec_search(query, domain=dcd_result.get("collection"))
            # Ensure text field for fuser compatibility
            for c in chunks:
                if "text" not in c and "content" in c:
                    c["text"] = c["content"]
        else:
            chunks = await mcp.query(source, query, max_results=3)

        if chunks:
            if source not in all_chunks:
                all_chunks[source] = []
            all_chunks[source].extend(chunks)
    except Exception:
        pass


# ── CLI ─────────────────────────────────────────────────────────────
def main():
    import asyncio
    if len(sys.argv) < 2:
        print("Usage: python -m rag_v2.engine <query>")
        return

    query = " ".join(sys.argv[1:])
    result = asyncio.run(rag_v2_search(query))

    print(f"\nSource: {result.get('source', 'empty')}")
    print(f"DCD: {result.get('dcd_domain', '')}/{result.get('dcd_collection', '')} (router: {result.get('dcd_router', 'keyword')})")
    print(f"Fusion: {result.get('fusion_needed', False)}")
    print(f"Sources used: {result.get('sources_used', [])}")
    print(f"Latency: {result.get('latency_s', 0):.2f}s")
    if result.get('from_cache'):
        print("(from cache)")
    print(f"\nAnswer:\n{result.get('answer', '')[:500]}")
    print(f"\nChunks: {len(result.get('chunks', []))}")

    trace = result.get("trace", "")
    if trace and isinstance(trace, dict):
        print(f"\nTrace: {json.dumps(trace.get('stages', []), ensure_ascii=False, indent=2)[:500]}")


if __name__ == "__main__":
    main()
