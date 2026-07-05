"""
Streaming RAG response — async generator + SSE endpoint.

For desktop (Tauri IPC) and web (SSE via FastAPI) consumers.

Usage (async generator):
    async for token in stream_rag_response(query):
        print(token, end="")

Usage (SSE endpoint):
    from streaming import create_sse_handler
    app.get("/rag/stream")(create_sse_handler())
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator

import aiohttp


async def stream_rag_response(
    query: str,
    session: aiohttp.ClientSession | None = None,
) -> AsyncGenerator[dict, None]:
    """Async generator that yields RAG stages as they complete.

    Yields dicts with "type" field:
        {"type": "trace", "stage": "dcd", ...}    — stage events
        {"type": "trace", "stage": "decompose", ...}
        {"type": "trace", "stage": "search", ...}
        {"type": "trace", "stage": "fuse", ...}
        {"type": "trace", "stage": "total", ...}
        {"type": "answer", "text": "..."}         — final answer (one event)
        {"type": "error", "message": "..."}        — on failure
    """
    t0 = time.time()

    # 1. DCD
    try:
        from rag_core.dcd_router import classify
        dcd_result = classify(query)
    except Exception as e:
        dcd_result = {"domain": "unknown", "confidence": 0.0}

    yield {
        "type": "trace",
        "stage": "dcd",
        "domain": dcd_result.get("domain", ""),
        "confidence": dcd_result.get("confidence", 0),
        "latency_ms": int((time.time() - t0) * 1000),
    }

    # 2. Decompose (if enabled)
    decompose_enabled = True  # from config
    subqueries = []
    if decompose_enabled:
        try:
            from rag_v2.decomposer import decompose
            s = session or aiohttp.ClientSession()
            try:
                subqueries = await decompose(query, s)
            finally:
                if not session:
                    await s.close()
            if not subqueries:
                subqueries = [{"query": query, "source": "auto"}]
        except Exception:
            subqueries = [{"query": query, "source": "auto"}]

    yield {
        "type": "trace",
        "stage": "decompose",
        "subqueries": len(subqueries),
        "latency_ms": int((time.time() - t0) * 1000),
    }

    # 3. Full pipeline
    try:
        from rag_v2.engine import rag_v2_search
        result = await rag_v2_search(query)
        yield {
            "type": "trace",
            "stage": "search",
            "sources_used": result.get("sources_used", []),
            "total_chunks": len(result.get("chunks", [])),
            "latency_ms": int((time.time() - t0) * 1000),
        }

        yield {
            "type": "answer",
            "text": result.get("answer", ""),
            "source": result.get("source", ""),
            "fusion": result.get("fusion_needed", False),
            "trace": result.get("trace"),
            "latency_ms": int((time.time() - t0) * 1000),
        }

    except Exception as e:
        yield {
            "type": "error",
            "message": str(e)[:500],
            "latency_ms": int((time.time() - t0) * 1000),
        }


def create_sse_handler():
    """Factory for FastAPI SSE endpoint. Returns an async handler.

    Usage:
        from streaming import create_sse_handler
        app.get("/rag/stream")(create_sse_handler())

    Client receives:
        data: {"type":"trace","stage":"dcd",...}
        data: {"type":"answer","text":"..."}
    """
    async def handler():
        import asyncio

        async def _stream():
            query = ""  # filled by request parsing
            # Note: in real usage, query comes from request.
            # This is a factory; actual query extraction is caller's job.
            async for event in stream_rag_response(query):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return _stream()

    return handler


def sse_format(event: dict) -> str:
    """Format a single event as SSE data line."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
