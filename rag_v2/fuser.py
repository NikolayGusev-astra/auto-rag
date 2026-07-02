"""
Multi-Source Fuser — собирает чанки из всех источников, реранк, LLM fusion.

Стратегия:
  1. Все чанки от всех источников → общий пул
  2. Reranker: local cross-encoder (default) or bge-reranker (fallback)
  3. LLM (qwen2.5-7b) оценивает: достаточен ли один источник или нужна комбинация
  4. Если составной запрос → склейка ответов из разных источников
"""

import json
import sys
import os
from typing import Any

import aiohttp

_rag_core = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rag_core'))
if _rag_core not in sys.path:
    sys.path.insert(0, _rag_core)
from rag_config import (
    LM_STUDIO_CHAT_URL, EMBEDDING_URL,
    LOCAL_RERANKER_ENABLED, LOCAL_RERANKER_MODEL, LOCAL_RERANKER_DEVICE,
)

_LLM_URL = LM_STUDIO_CHAT_URL.rstrip('/')
_RERANK_URL = f"{EMBEDDING_URL.rstrip('/')}"

FUSION_PROMPT = """Ты — система оценки качества RAG ответов.

Тебе даны чанки из нескольких источников. Оцени:
1. Какие чанки релевантны запросу (score 0.0-1.0)
2. Достаточен ли один источник или нужна комбинация
3. Если комбинация → итоговый ответ на основе всех источников

Запрос: {query}

Чанки:
{chunks}

Ответь ТОЛЬКО JSON:
{{
  "relevant_chunks": [индексы чанков (0-based)],
  "fusion_needed": true/false,
  "primary_source": "название источника с лучшим ответом",
  "answer": "краткий ответ на основе всех релевантных чанков" (если fusion_needed=false — только из primary_source)
}}"""


async def fuse(
    query: str,
    all_chunks: dict[str, list[dict]],  # {source: [chunks]}
    session: aiohttp.ClientSession,
) -> dict:
    """
    Собрать все чанки, реранк, LLM fusion.
    Возвращает: {source, chunks, answer, fusion_needed, sources_used}
    """
    # 1. Собрать все чанки в общий пул с источником
    pool = []
    for src, chunks in all_chunks.items():
        for c in chunks:
            if c.get("text", "").strip():
                pool.append({**c, "_src": src})

    if not pool:
        return {"source": "empty", "chunks": [], "answer": "",
                "fusion_needed": False, "sources_used": []}

    # 2. Rerank (local cross-encoder preferred, bge fallback)
    if len(pool) > 5:
        reranked = await _rerank(query, pool, session)
        top_k = reranked[:10]
    else:
        top_k = pool

    # 3. LLM fusion
    result = await _llm_fuse(query, top_k, session)
    if result.get("answer"):
        return {
            "source": result.get("primary_source", "fusion"),
            "chunks": top_k,
            "answer": result["answer"],
            "fusion_needed": result.get("fusion_needed", False),
            "sources_used": list(set(c["_src"] for c in top_k)),
        }

    # Fallback: просто топ-1 чанк
    best = top_k[0]
    return {
        "source": best["_src"],
        "chunks": top_k[:3],
        "answer": best["text"][:500],
        "fusion_needed": False,
        "sources_used": [best["_src"]],
    }


async def _rerank(
    query: str, chunks: list[dict], session: aiohttp.ClientSession
) -> list[dict]:
    """Rerank chunks. Uses local cross-encoder if enabled, else bge-reranker."""
    if LOCAL_RERANKER_ENABLED:
        return await _rerank_local(query, chunks)

    # Fallback: bge-reranker-v2-m3 via LM Studio
    return await _rerank_bge(query, chunks, session)


async def _rerank_local(query: str, chunks: list[dict]) -> list[dict]:
    """Local cross-encoder reranker (ms-marco-MiniLM-L6-v2).

    Runs in thread pool to avoid blocking event loop.
    No LM Studio round-trip needed.
    """
    import asyncio

    def _do_rerank():
        try:
            from local_reranker import LocalReranker
            reranker = LocalReranker(
                query=query,
                rerank_field="text",
                model_name=LOCAL_RERANKER_MODEL,
                device=LOCAL_RERANKER_DEVICE,
            )
            results = reranker.rerank(chunks, topn=10)
            return results
        except ImportError:
            # sentence-transformers not installed — skip rerank
            return chunks
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Local reranker failed: %s", e)
            return chunks

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do_rerank)


async def _rerank_bge(
    query: str, chunks: list[dict], session: aiohttp.ClientSession
) -> list[dict]:
    """Bge-reranker-v2-m3 через LM Studio."""
    try:
        texts = [c["text"][:500] for c in chunks]
        payload = {
            "model": "text-embedding-bge-reranker-v2-m3",
            "query": query,
            "documents": texts,
        }
        async with session.post(
            _RERANK_URL, json=payload,
            timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            data = await resp.json()
            scores = [r.get("relevance_score", 0) for r in data.get("data", [])]
    except Exception:
        scores = [c.get("score", 0.5) for c in chunks]

    # Отсортировать по score
    indexed = list(enumerate(chunks))
    if scores:
        indexed.sort(key=lambda x: scores[x[0]], reverse=True)
    else:
        indexed.sort(key=lambda x: x[1].get("score", 0.5), reverse=True)

    return [c for _, c in indexed]


async def _llm_fuse(
    query: str, chunks: list[dict], session: aiohttp.ClientSession
) -> dict:
    """LLM оценивает чанки и решает нужна ли склейка."""
    # Подготовить чанки для промпта
    chunk_texts = []
    for i, c in enumerate(chunks):
        src = c.get("_src", c.get("source", "?"))
        txt = c["text"][:400]
        chunk_texts.append(f"[{i}] [{src}] {txt}")

    prompt = FUSION_PROMPT.format(
        query=query,
        chunks="\n\n".join(chunk_texts),
    )

    payload = {
        "model": "qwen2.5-7b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 800,
    }

    try:
        async with session.post(
            _LLM_URL, json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            data = await resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            return json.loads(text)
    except Exception:
        return {"fusion_needed": False, "primary_source": "", "answer": chunks[0]["text"][:500] if chunks else ""}
