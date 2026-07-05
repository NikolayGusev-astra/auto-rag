#!/usr/bin/env python3
"""
Federated RAG endpoint — FastAPI server that wraps rag_async for external queries.
Run on any remote node on port 8000 (or any port).

Auth: X-API-Key header, value from RAG_FEDERATED_API_KEY env var.
Если env var не задан — auth отключен (только для trusted LAN/localhost).
"""
import os
import sys
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dcd_router import classify, DOMAIN_KEYWORDS
from rag_async import async_rag_search

app = FastAPI(title="Federated RAG Endpoint", version="1.1.0")

API_KEY = os.getenv("RAG_FEDERATED_API_KEY", "")
REQUIRE_AUTH = bool(API_KEY)


def _check_auth(x_api_key: str | None):
    if REQUIRE_AUTH and x_api_key != API_KEY:
        raise HTTPException(401, "Invalid or missing X-API-Key")


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


class SearchResponse(BaseModel):
    chunks: list
    source: str
    chunks_count: int
    trace: str


@app.get("/health")
async def health():
    """Health check — без auth, для k8s liveness/readiness."""
    return {"status": "ok", "service": "federated-rag", "auth_required": REQUIRE_AUTH}


@app.get("/domains")
async def domains(x_api_key: str | None = Header(None, alias="X-API-Key")):
    """Возвращает список доменов, которые этот инстанс умеет искать."""
    _check_auth(x_api_key)
    return {
        "domains": list(DOMAIN_KEYWORDS.keys()),
    }


@app.post("/rag/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    _check_auth(x_api_key)
    try:
        dcd_result = classify(request.query)
        result = await async_rag_search(request.query, dcd_result)
        chunks = result.get('chunks', [])[:request.max_results]
        return SearchResponse(
            chunks=chunks,
            source=result.get('source', 'empty'),
            chunks_count=len(chunks),
            trace=result.get('trace', '?'),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rag/search")
async def search_get(
    q: str = Query(..., description="Query string"),
    max_results: int = Query(5, ge=1, le=20),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    _check_auth(x_api_key)
    try:
        dcd_result = classify(q)
        result = await async_rag_search(q, dcd_result)
        chunks = result.get('chunks', [])[:max_results]
        return SearchResponse(
            chunks=chunks,
            source=result.get('source', 'empty'),
            chunks_count=len(chunks),
            trace=result.get('trace', '?'),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("RAG_FEDERATED_PORT", "8000"))
    host = os.getenv("RAG_FEDERATED_HOST", "127.0.0.1")
    print(f"Starting federated RAG endpoint on {host}:{port}")
    print(f"Auth required: {REQUIRE_AUTH}")
    uvicorn.run(app, host=host, port=port)
