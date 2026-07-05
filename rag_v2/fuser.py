"""
Multi-Source Fuser — собирает чанки из всех источников, реранк, LLM fusion.

Стратегия:
  1. Все чанки от всех источников → общий пул
  2. Bge-reranker (bge-reranker-v2-m3) → топ-10
  3. LLM (qwen2.5-7b) оценивает: достаточен ли один источник или нужна комбинация
  4. Если составной запрос → склейка ответов из разных источников
"""

import json
import sys
import os
from typing import Any

import aiohttp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rag_config import LM_STUDIO_CHAT_URL
from reranker_service import RerankerService, rerank_chunks

_LLM_URL = LM_STUDIO_CHAT_URL.rstrip('/')

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
    
    # 2. Bge-reranker (опционально, если >5 чанков)
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
    """Rerank через RerankerService."""
    if not chunks:
        return []
    try:
        return RerankerService.get().rerank_chunks(query, chunks, top_k=10)
    except Exception:
        return chunks


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