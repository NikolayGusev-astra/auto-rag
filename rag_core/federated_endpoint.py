#!/usr/bin/env python3
"""
Federated RAG endpoint — FastAPI server that wraps rag_async for external queries.
Run on any remote node on port 8000 (or any port).
"""
import sys
import os
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Add rag_core to path for sibling imports (dcd_router, rag_async)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dcd_router import classify
from rag_async import async_rag_search

app = FastAPI(title="Federated RAG Endpoint", version="1.0.0")

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
    return {"status": "ok", "service": "federated-rag"}

@app.post("/rag/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search endpoint for federated RAG queries."""
    try:
        # Classify query
        dcd_result = classify(request.query)
        
        # Run async search
        result = await async_rag_search(request.query, dcd_result)
        
        # Limit chunks
        chunks = result.get('chunks', [])[:request.max_results]
        
        return SearchResponse(
            chunks=chunks,
            source=result.get('source', 'empty'),
            chunks_count=len(chunks),
            trace=result.get('trace', '?')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rag/search")
async def search_get(
    q: str = Query(..., description="Query string"),
    max_results: int = Query(5, description="Maximum results to return")
):
    """GET endpoint for simple queries."""
    try:
        dcd_result = classify(q)
        result = await async_rag_search(q, dcd_result)
        chunks = result.get('chunks', [])[:max_results]
        
        return SearchResponse(
            chunks=chunks,
            source=result.get('source', 'empty'),
            chunks_count=len(chunks),
            trace=result.get('trace', '?')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("RAG_FEDERATED_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)