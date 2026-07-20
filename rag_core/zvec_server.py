#!/usr/bin/env python3
"""FastAPI server for ZVec — loads collection once, keeps it in memory.

Usage:
  python zvec_server.py          # default port 8678
  python zvec_server.py --port 8765

Endpoints:
  GET /health                    → {"status": "ok", "collection": "wiki"}
  GET /search?q=ALD+Pro+IP       → {"chunks": [...], "max_score": 0.45, "latency_seconds": 0.5}
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# ── Config (matches rag_config.py) ────────────────────────────────────
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-baai-bge-m3-568m")
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://localhost:1234/v1/embeddings")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))
ZVEC_PATH = os.environ.get("ZVEC_PATH", os.path.expanduser("~/.cache/zvec"))
ZVEC_COLLECTION = os.environ.get("ZVEC_COLLECTION", "wiki")

app_state = {"zvec": None, "started": None, "port": 8678}


def _embed_via_lmstudio(text: str) -> list[float] | None:
    """LM Studio embedding API. Returns None on failure, never fake zeros."""
    payload = json.dumps({
        "model": EMBEDDING_MODEL,
        "input": [text[:2000]],
    }).encode("utf-8")
    req = urllib.request.Request(
        EMBEDDING_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read().decode())
        return data["data"][0]["embedding"]
    except Exception:
        return None


def _load_zvec():
    """Load ZVec collection once at startup."""
    import zvec

    zpath = os.path.join(ZVEC_PATH, ZVEC_COLLECTION)
    if not os.path.isdir(zpath):
        raise FileNotFoundError(f"ZVec collection not found: {zpath}")

    print(f"[zvec-server] Loading ZVec from {zpath} ...")
    t0 = time.time()
    coll = zvec.open(zpath)
    print(f"[zvec-server] ZVec loaded in {time.time()-t0:.1f}s")
    return coll


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ZVec on startup, clean up on shutdown."""
    try:
        coll = _load_zvec()
        app_state["zvec"] = coll
        app_state["started"] = time.time()
        print(f"[zvec-server] Ready on port {app_state.get('port', 8678)}")
    except Exception as e:
        print(f"[zvec-server] FAILED: {e}", file=sys.stderr)
        app_state["error"] = str(e)
    yield
    app_state["zvec"] = None


app = FastAPI(title="ZVec Search Server", version="1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    if app_state.get("error"):
        return JSONResponse({"status": "error", "message": app_state["error"]}, status_code=503)
    if app_state["zvec"] is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    uptime = time.time() - (app_state.get("started") or time.time())
    return {
        "status": "ok",
        "collection": ZVEC_COLLECTION,
        "embedding": EMBEDDING_MODEL,
        "uptime_seconds": round(uptime, 1),
    }


# ZVec category filters (from rag_config)
ZVEC_CATEGORY_FILTERS = {
    "wiki": "category = 'wiki' OR category = 'llm-wiki'",
    "skills": "category = 'skill'",
    "sessions": "category = 'session'",
}

@app.get("/search")
async def search(
    q: str = Query(..., description="Search query text"),
    topk: int = Query(5, ge=1, le=20, description="Number of results"),
    category: str = Query("wiki", description="Collection category filter"),
):
    coll = app_state["zvec"]
    if coll is None:
        return JSONResponse({"error": "Not ready"}, status_code=503)

    t0 = time.time()
    from zvec import Query as ZQ

    emb = _embed_via_lmstudio(q)
    if emb is None:
        return JSONResponse(
            {"error": "embedding unavailable", "chunks": [], "max_score": 0},
            status_code=503,
        )
    filter_expr = ZVEC_CATEGORY_FILTERS.get(category, ZVEC_CATEGORY_FILTERS["wiki"])
    try:
        doclist = coll.query(
            queries=[ZQ(field_name="embedding", vector=emb)],
            topk=topk,
            filter=filter_expr,
            output_fields=["source", "heading", "content", "title", "category"],
        )
    except Exception:
        doclist = coll.query(
            queries=[ZQ(field_name="embedding", vector=emb)], topk=topk
        )

    chunks = []
    for d in doclist:
        txt = d.fields.get("text", "") or d.fields.get("content", "")
        if txt:
            chunks.append({
                "id": str(getattr(d, "id", "") or (d.fields or {}).get("id", "")),
                "text": txt[:500],
                "score": d.score,
                "source": (d.fields or {}).get("source", "zvec/wiki"),
            })

    max_score = max((c["score"] for c in chunks), default=0)
    latency = round(time.time() - t0, 3)

    return {
        "chunks": chunks,
        "max_score": max_score,
        "latency_seconds": latency,
        "query": q[:100],
    }


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8678, help="Listen port")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    args = parser.parse_args()
    port = args.port
    app_state["port"] = port
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
