#!/usr/bin/env python3
"""
ZVec FastAPI сервер — обход LOCK бага через долгоживущий процесс.
Запуск: python3 zvec_server.py (порт 8765)

Клиент вместо zvec.open() шлёт HTTP:
  curl http://localhost:8765/search?q=Ford+Explorer&topk=5
  curl http://localhost:8765/search -d '{"query":"...","topk":5}' -H 'Content-Type: application/json'
"""
import os, sys, json, subprocess, re, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_config import ZVEC_PATH, ZVEC_WIKI_COLLECTION, EMBEDDING_DIM, EMBEDDING_URL, EMBEDDING_MODEL, ensure_zvec_lock

try:
    from fastapi import FastAPI, Query
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("Установи: pip install fastapi uvicorn")
    sys.exit(1)

app = FastAPI(title="ZVec RAG Server")

# ── Singleton: коллекция открывается один раз ─────────────────────
_coll = None

def get_coll():
    global _coll
    if _coll is not None:
        return _coll
    import zvec
    zvec.init()
    coll_path = os.path.join(ZVEC_PATH, ZVEC_WIKI_COLLECTION)
    ensure_zvec_lock(coll_path)
    _coll = zvec.open(coll_path)
    print(f"ZVec opened: {_coll.stats.doc_count} docs", flush=True)
    return _coll

def embed(text):
    try:
        payload = json.dumps({"model": EMBEDDING_MODEL, "input": [text]})
        r = subprocess.run(["curl","-s","--max-time","10",EMBEDDING_URL,"-d",payload,
                          "-H","Content-Type: application/json"], capture_output=True, text=True, timeout=15)
        if r.returncode == 0 and r.stdout:
            return json.loads(r.stdout)["data"][0]["embedding"]
    except: pass
    return [0.0] * EMBEDDING_DIM

@app.get("/search")
@app.post("/search")
async def search(query: str = Query(None), q: str = Query(None), topk: int = 5):
    q = query or q
    if not q:
        return JSONResponse({"error": "empty query"}, status_code=400)

    emb = embed(q)
    if sum(abs(v) for v in emb) == 0:
        return {"results": [], "source": "embed_failed"}

    coll = get_coll()
    from zvec import Query as ZQ
    try:
        results = coll.query(queries=[ZQ(field_name="embedding", vector=emb)], topk=topk,
                            output_fields=["source","heading","category","content","title"])
        return {
            "query": q,
            "results": [{
                "source": r.fields.get("source","") if r.fields else "",
                "title": r.fields.get("title","") if r.fields else "",
                "score": r.score,
                "content": (r.fields.get("content","") if r.fields else "")[:300],
            } for r in results],
            "total": len(results),
            "source": "zvec"
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/stats")
async def stats():
    coll = get_coll()
    s = coll.stats
    return {"doc_count": s.doc_count, "index_completeness": s.index_completeness, "collection": ZVEC_WIKI_COLLECTION}

@app.get("/health")
async def health():
    return {"status": "ok", "collection": ZVEC_WIKI_COLLECTION}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
