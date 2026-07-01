#!/usr/bin/env python3
"""
ZVec Daemon — FastAPI сервер, держит ZVec открытым, 
обходит LOCK баг Windows.
"""
import asyncio
import json
import os
import sys
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_config import ZVEC_PATH, ZVEC_COLLECTION, EMBEDDING_URL, EMBEDDING_MODEL

app = FastAPI(title="ZVec RAG Daemon")

# Глобальный держатель коллекции (открывается 1 раз при старте)
_zvec_collection = None
_zvec_lock = asyncio.Lock()

class SearchRequest(BaseModel):
    query: str
    topk: int = 5

class SearchResult(BaseModel):
    text: str
    score: float
    source: str

@app.on_event("startup")
async def startup():
    global _zvec_collection
    import zvec
    from zvec import Query as ZQ
    zpath = os.path.join(os.path.expanduser(ZVEC_PATH), ZVEC_COLLECTION)
    # LOCK workaround — создаём пустой lock при старте
    lock = os.path.join(zpath, "LOCK")
    try:
        with open(lock, 'w') as f: f.write('')
    except: pass
    _zvec_collection = {
        'coll': zvec.open(zpath),
        'zpath': zpath,
    }
    # Проверка: пробуем query
    try:
        test = _zvec_collection['coll'].query(
            queries=[ZQ(field_name='embedding', vector=[0.0]*1024)],
            topk=1
        )
        print(f"ZVec daemon: opened {zpath}, {len(test) if test else 0} docs")
    except Exception as e:
        print(f"ZVec daemon WARNING: {e}")

@app.get("/health")
async def health():
    return {"status": "ok", "collection": ZVEC_COLLECTION}

@app.post("/embed", response_model=list[float])
async def embed(text: str):
    """Embed text via LM Studio."""
    import requests
    try:
        resp = requests.post(EMBEDDING_URL, json={
            'model': EMBEDDING_MODEL,
            'input': [f'Instruct: Given a wiki search query, retrieve relevant wiki passages\nQuery: {text[:2000]}']
        }, timeout=30)
        return resp.json()['data'][0]['embedding']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=list[SearchResult])
async def search(req: SearchRequest):
    """Search ZVec collection."""
    import requests as _req
    from zvec import Query as ZQ
    
    async with _zvec_lock:
        if _zvec_collection is None:
            raise HTTPException(status_code=503, detail="ZVec not initialized")
        
        # Get embedding
        emb = _req.post(EMBEDDING_URL, json={
            'model': EMBEDDING_MODEL,
            'input': [f'Instruct: Given a wiki search query, retrieve relevant wiki passages\nQuery: {req.query[:2000]}']
        }, timeout=30).json()['data'][0]['embedding']
        
        # Search
        vq = ZQ(field_name='embedding', vector=emb)
        doclist = _zvec_collection['coll'].query(queries=[vq], topk=req.topk)
        
        results = []
        for d in doclist:
            txt = d.fields.get('text', '') or d.fields.get('content', '')
            if txt:
                results.append(SearchResult(
                    text=txt[:2000],
                    score=d.score,
                    source='zvec/wiki'
                ))
        return results

if __name__ == "__main__":
    port = int(os.getenv("ZVEC_DAEMON_PORT", "8765"))
    print(f"Starting ZVec daemon on :{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")