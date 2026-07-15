#!/usr/bin/env python3
"""
Federated RAG endpoint — FastAPI server that wraps rag_async for external queries.
Run on any remote node on port 8000 (or any port).

Auth: X-API-Key header, value from RAG_FEDERATED_API_KEY env var.
Если env var не задан — auth отключен (только для trusted LAN/localhost).
"""
import os
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Header
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dcd_router import classify, DOMAIN_KEYWORDS
from rag_async import async_rag_search

app = FastAPI(title="Federated RAG Endpoint", version="1.1.0")

REQUIRE_AUTH = os.getenv("RAG_FEDERATED_API_KEY", "") != ""


def get_bind_host() -> str:
    """Без API-ключа биндимся только на localhost (иначе удалённый узел
    без аутентификации открыт всей сети — security HIGH #2)."""
    require_auth = os.getenv("RAG_FEDERATED_API_KEY", "") != ""
    return "127.0.0.1" if not require_auth else "0.0.0.0"


def _check_auth(x_api_key: Optional[str]) -> None:
    if not REQUIRE_AUTH:
        return
    expected = os.getenv("RAG_FEDERATED_API_KEY", "")
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _empty_result_hint(dcd_result: dict, query: str) -> dict:
    """Generate hints for Hermes agent when no chunks found."""
    domain = dcd_result.get("domain", "unknown")
    hint = "No results found"
    suggested_actions = ["try rephrasing query", "check if data is indexed"]

    if domain != "unknown":
        primary_source = dcd_result.get("primary_source")
        if primary_source:
            hint = f"No results in primary source '{primary_source}', try fallback sources or LLM fallback"
            suggested_actions.append(f"search in fallback sources: {', '.join(dcd_result.get('fallback_sources', []))}")
            suggested_actions.append("enable LLM fallback for this domain")

    return {
        "hint": hint,
        "suggested_actions": suggested_actions,
        "dcd_info": dcd_result,
    }


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


class SearchResponse(BaseModel):
    chunks: list
    source: str
    chunks_count: int
    trace: str
    dcd_info: dict = {}
    hint: str = ""
    suggested_actions: list = []


@app.get("/health")
async def health(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    _check_auth(x_api_key)
    from lm_studio_monitor import get_lm_studio
    lm_status = get_lm_studio().get_status()

    return {
        "status": "ok" if lm_status["available"] else "degraded",
        "service": "federated-rag",
        "auth_required": REQUIRE_AUTH,
        "lm_studio": {
            "available": lm_status["available"],
            "loaded_models": lm_status["loaded_models"],
            "vram": lm_status.get("vram"),
        },
    }


@app.get("/domains")
async def domains(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Возвращает домены + статус LM Studio на этом узле."""
    _check_auth(x_api_key)
    from lm_studio_monitor import get_lm_studio
    lm_status = get_lm_studio().get_status()

    loaded_models = lm_status.get("loaded_models", [])

    capabilities = {
        "embedding": any(
            os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-baai-bge-m3-568m") in m
            or "bge-m3" in m.lower()
            for m in loaded_models
        ),
        "reranker": any("rerank" in m.lower() for m in loaded_models),
        "llm_classify": any(
            os.getenv("RAG_CLASSIFY_MODEL", "qwen2.5-7b-instruct") in m for m in loaded_models
        ),
        "llm_verify": any(
            os.getenv("RAG_LLM_VERIFY_MODEL", "qwen2.5-7b-instruct") in m for m in loaded_models
        ),
    }

    return {
        "domains": list(DOMAIN_KEYWORDS.keys()),
        "doc_count": _get_doc_count(),
        "lm_studio": {
            "available": lm_status["available"],
            "loaded_models": loaded_models,
        },
        "capabilities": capabilities,
    }


@app.post("/warmup")
async def warmup(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Pre-load все модели в LM Studio."""
    _check_auth(x_api_key)
    from lm_studio_monitor import get_lm_studio
    results = get_lm_studio().warmup_all()
    return {"warmup_results": results}


@app.post("/rag/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    _check_auth(x_api_key)
    try:
        dcd_result = classify(request.query)
        result = await async_rag_search(request.query, dcd_result)
        chunks = result.get("chunks", [])[: request.max_results]

        empty_hint = {}
        if not chunks:
            empty_hint = _empty_result_hint(dcd_result, request.query)

        return SearchResponse(
            chunks=chunks,
            source=result.get("source", "empty"),
            chunks_count=len(chunks),
            trace=result.get("trace", "?"),
            dcd_info=empty_hint.get("dcd_info", dcd_result),
            hint=empty_hint.get("hint", ""),
            suggested_actions=empty_hint.get("suggested_actions", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/search")
async def search_get(
    q: str = Query(..., description="Query string"),
    max_results: int = Query(5, description="Maximum results to return"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    _check_auth(x_api_key)
    try:
        dcd_result = classify(q)
        result = await async_rag_search(q, dcd_result)
        chunks = result.get("chunks", [])[:max_results]

        empty_hint = {}
        if not chunks:
            empty_hint = _empty_result_hint(dcd_result, q)

        return SearchResponse(
            chunks=chunks,
            source=result.get("source", "empty"),
            chunks_count=len(chunks),
            trace=result.get("trace", "?"),
            dcd_info=empty_hint.get("dcd_info", dcd_result),
            hint=empty_hint.get("hint", ""),
            suggested_actions=empty_hint.get("suggested_actions", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_doc_count() -> int:
    try:
        from rag_config import ZVEC_PATH, ZVEC_COLLECTION
        import os
        index_path = os.path.join(ZVEC_PATH, ZVEC_COLLECTION)
        if os.path.isdir(index_path):
            return len([f for f in os.listdir(index_path) if f.endswith(".json") or f.endswith(".bin")])
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("RAG_FEDERATED_PORT", "8000"))
    # Без API-ключа биндимся только на localhost (иначе удалённый узел
    # без аутентификации открыт всей сети — security HIGH #2).
    uvicorn.run(app, host=get_bind_host(), port=port)
