"""
RAG v2 Engine — главный async pipeline.

Архитектура:
  1. DCD классификация (domain)
  2. LLM декомпозиция запроса → подзапросы
  3. Параллельный поиск во всех источниках
  4. Bge-reranker → топ-10
  5. LLM fusion → итоговый ответ
  6. RagTrace
"""

import asyncio
import json
import os
import sys
import time
from typing import Any

import aiohttp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dcd_router import classify
from rag_config import MCP_SERVERS

# V2 modules
from .decomposer import decompose
from .mcp import AsyncMCPClient
from .fuser import fuse

# Reuse trace from v1
from rag_trace import RagTrace


async def rag_v2_search(query: str) -> dict:
    """Main RAG v2 search."""
    trace = RagTrace(query, "v2", "v2")
    t0 = time.time()
    
    # 1. DCD
    dcd_result = classify(query)
    trace.event("dcd", domain=dcd_result.get("domain", ""),
                collection=dcd_result.get("collection", ""),
                confidence=dcd_result.get("confidence", 0))
    
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
        
        # 3. Узнать какие MCP серверы доступны
        available_sources = list(MCP_SERVERS.keys())
        mcp = AsyncMCPClient(session, timeout=20)
        
        # 4. Параллельный поиск во всех источниках
        trace.begin("parallel_search")
        try:
            all_chunks: dict[str, list[dict]] = {}
            tasks = []
            
            # Для каждого подзапроса — во все доступные источники
            for sq in subqueries:
                sq_query = sq["query"]
                target = sq.get("source", "auto")
                
                # Если указан конкретный источник — только в него
                if target != "auto" and target in available_sources:
                    tasks.append(_search_source(mcp, target, sq_query, all_chunks))
                else:
                    # Во все источники
                    for src in available_sources:
                        tasks.append(_search_source(mcp, src, sq_query, all_chunks))
            
            if tasks:
                await asyncio.gather(*tasks)
        
            # Статистика
            total_chunks = sum(len(c) for c in all_chunks.values())
            sources_with_data = [s for s, c in all_chunks.items() if c]
        finally:
            trace.end("parallel_search", sources=len(available_sources),
                      with_data=len(sources_with_data), total_chunks=total_chunks)
        
        # 5. Fuser — реранк + LLM склейка
        trace.begin("fuse")
        try:
            result = await fuse(query, all_chunks, session)
        finally:
            trace.end("fuse", source=result.get("source", "empty"),
                      fusion=result.get("fusion_needed", False),
                      sources_used=result.get("sources_used", []))
    
    # 6. Итог
    elapsed = int((time.time() - t0) * 1000)
    trace.event("total", duration_ms=elapsed)
    result["trace"] = trace.json()
    result["dcd_domain"] = dcd_result.get("domain", "")
    result["dcd_collection"] = dcd_result.get("collection", "")
    result["latency_s"] = round(time.time() - t0, 2)
    return result


async def _search_source(
    mcp: AsyncMCPClient, source: str, query: str,
    all_chunks: dict[str, list[dict]]
):
    """Поиск в одном источнике. Результат добавляется в all_chunks."""
    try:
        chunks = await mcp.query(source, query, max_results=3)
        if chunks:
            if source not in all_chunks:
                all_chunks[source] = []
            all_chunks[source].extend(chunks)
    except Exception:
        pass


# ── CLI ──

def main():
    import asyncio
    if len(sys.argv) < 2:
        print("Usage: python -m rag_v2.engine <query>")
        return
    
    query = " ".join(sys.argv[1:])
    result = asyncio.run(rag_v2_search(query))
    
    print(f"\nSource: {result.get('source', 'empty')}")
    print(f"DCD: {result.get('dcd_domain')}/{result.get('dcd_collection')}")
    print(f"Fusion: {result.get('fusion_needed', False)}")
    print(f"Sources used: {result.get('sources_used', [])}")
    print(f"Latency: {result.get('latency_s', 0):.2f}s")
    print(f"\nAnswer:\n{result.get('answer', '')[:500]}")
    print(f"\nChunks: {len(result.get('chunks', []))}")
    
    trace = result.get("trace", "")
    if trace and isinstance(trace, dict):
        print(f"\nTrace: {json.dumps(trace.get('stages', []), ensure_ascii=False, indent=2)[:500]}")


if __name__ == "__main__":
    main()