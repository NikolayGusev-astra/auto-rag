#!/usr/bin/env python3
"""
RAG Indexer — ZVec wiki collection builder for Windows.
Usage: python zvec_incremental_indexer.py [--incremental] [--clear]

Адаптировано из auto-rag (NikolayGusev-astra/auto-rag):
- Инкрементальная индексация (file hash tracking)
- Heading-based chunking (по заголовкам ##/###)
- Структурированная схема (source, heading, category, content, title, tags)
- Batch embedding + insert
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_config import EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDING_URL
from rag_config import EXCLUDE_DIRS, ZVEC_PATH, ZVEC_COLLECTION

# ── Config ────────────────────────────────────────────────────────
COLL_PATH = os.path.join(ZVEC_PATH, ZVEC_COLLECTION)
WIKI_PATHS = [
    os.path.expanduser("~/llm-wiki"),
    os.path.expanduser("~/llm-wiki/raw"),
    os.path.expanduser("~/wiki"),          # <-- rusbitech docs: products, customers, cross
]
CHUNK_SIZE = 2000
STATE_FILE = os.path.join(ZVEC_PATH, ".index_state.json")
BATCH_SIZE = 8

SUPPORTED_EXT = {".md", ".txt", ".rst", ".py", ".yaml", ".yml",
                 ".json", ".toml", ".cfg", ".ini", ".conf", ".sh", ".env"}


def should_exclude(path: str) -> bool:
    for pat in EXCLUDE_DIRS:
        if pat in path:
            return True
    return False


def collect_files() -> list[str]:
    files = []
    for base in WIKI_PATHS:
        if not os.path.isdir(base):
            continue
        for root, dirs, fnames in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for fn in fnames:
                fp = os.path.join(root, fn)
                ext = os.path.splitext(fn)[1].lower()
                if ext in SUPPORTED_EXT and not should_exclude(fp):
                    files.append(fp)
    return sorted(files)


def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ── Frontmatter + Chunking ────────────────────────────────────────

def parse_frontmatter(text: str) -> tuple[dict, str]:
    text = text.lstrip('\ufeff')
    if not text.startswith('---'):
        return {}, text
    end = text.find('---', 3)
    if end == -1:
        return {}, text
    meta = {}
    try:
        import yaml
        meta = yaml.safe_load(text[3:end].strip()) or {}
    except Exception:
        pass
    return (meta if isinstance(meta, dict) else {}), text[end + 3:].strip()


def chunk_text(text: str, heading: str = "Overview") -> list[dict]:
    chunks = []
    lines = text.split('\n')
    current_heading = heading
    current_lines = []

    def flush():
        nonlocal current_lines
        if not current_lines:
            return
        content = '\n'.join(current_lines).strip()
        if len(content) < 20:
            current_lines = []
            return
        if len(content) > CHUNK_SIZE:
            parts = content.split('\n\n')
            buf = ""
            for p in parts:
                if len(buf) + len(p) > CHUNK_SIZE and buf:
                    chunks.append({"heading": current_heading, "text": buf})
                    buf = p
                else:
                    buf = (buf + '\n\n' + p) if buf else p
            if buf:
                chunks.append({"heading": current_heading, "text": buf})
        else:
            chunks.append({"heading": current_heading, "text": content})
        current_lines = []

    for line in lines:
        if line.startswith('# ') or line.startswith('## ') or line.startswith('### '):
            flush()
            current_heading = line.lstrip('#').strip()
        else:
            current_lines.append(line)
    flush()
    return chunks


# ── Embedding ─────────────────────────────────────────────────────

def _embed_batch(texts: list[str]) -> list[list[float]] | None:
    """Embedding via requests (LM Studio)."""
    import requests as req
    if not texts:
        return None
    try:
        r = req.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        }, timeout=120)
        if r.status_code == 200:
            return [d["embedding"] for d in r.json()["data"]]
    except Exception:
        pass
    return None


# ── Schema ────────────────────────────────────────────────────────

def get_schema():
    from zvec import (CollectionSchema, DataType, FieldSchema,
                      FtsIndexParam, HnswIndexParam, InvertIndexParam,
                      MetricType, VectorSchema)
    return CollectionSchema(
        name=ZVEC_COLLECTION,
        fields=[
            FieldSchema("source", DataType.STRING, nullable=False,
                        index_param=InvertIndexParam()),
            FieldSchema("heading", DataType.STRING, nullable=True),
            FieldSchema("category", DataType.STRING, nullable=False,
                        index_param=InvertIndexParam()),
            FieldSchema("node", DataType.STRING, nullable=False),
            FieldSchema("content_hash", DataType.STRING, nullable=True),
            FieldSchema("char_count", DataType.INT32, nullable=True),
            FieldSchema("title", DataType.STRING, nullable=True),
            FieldSchema("tags", DataType.STRING, nullable=True),
            FieldSchema("content", DataType.STRING, nullable=False,
                        index_param=FtsIndexParam(
                            tokenizer_name="standard",
                            filters=["lowercase"])),
        ],
        vectors=[
            VectorSchema("embedding", DataType.VECTOR_FP32,
                         dimension=EMBEDDING_DIM,
                         index_param=HnswIndexParam(
                             metric_type=MetricType.COSINE)),
        ],
    )


# ── Main ──────────────────────────────────────────────────────────

def index(incremental: bool = False, clear: bool = False):
    import zvec
    from zvec import CollectionOption, Doc

    # ZVec LOCK workaround
    lock_path = os.path.join(COLL_PATH, "LOCK")
    try:
        with open(lock_path, 'w') as f:
            f.write("")
    except OSError:
        pass

    # Open / create collection
    if clear and os.path.exists(COLL_PATH):
        import shutil
        shutil.rmtree(COLL_PATH, ignore_errors=True)
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        print("  🗑 Collection cleared")
        coll = None
    elif os.path.exists(COLL_PATH):
        try:
            coll = zvec.open(COLL_PATH)
            print(f"  📂 Opened existing: {coll.stats}")
        except Exception:
            import shutil
            shutil.rmtree(COLL_PATH, ignore_errors=True)
            coll = zvec.create_and_open(
                COLL_PATH, get_schema(),
                CollectionOption(read_only=False, enable_mmap=True))
            print(f"  ✅ Created fresh: {coll.stats}")
    else:
        coll = zvec.create_and_open(
            COLL_PATH, get_schema(),
            CollectionOption(read_only=False, enable_mmap=True))
        print(f"  ✅ Created new: {coll.stats}")

    if coll is None:
        coll = zvec.create_and_open(
            COLL_PATH, get_schema(),
            CollectionOption(read_only=False, enable_mmap=True))

    # Collect files
    files = collect_files()
    print(f"  📄 Found {len(files)} files")

    state = load_state() if incremental else {}
    to_index: list[str] = []
    if incremental:
        current = set()
        for fp in files:
            rel = os.path.relpath(fp, os.path.commonpath(WIKI_PATHS))
            fh = file_hash(fp)
            current.add(rel)
            if state.get(rel) != fh:
                to_index.append(fp)
        print(f"  📝 To index: {len(to_index)}")
    else:
        to_index = files
        print(f"  📝 Full reindex: {len(to_index)} files")

    if not to_index:
        print("  ✅ Nothing to index")
        return

    # Process files → chunks
    all_chunks: list[dict] = []
    file_to_rel: dict[str, str] = {}

    for fp in to_index:
        try:
            with open(fp, 'r', errors='replace') as f:
                text = f.read()
        except Exception:
            continue
        meta, body = parse_frontmatter(text)
        rel = os.path.relpath(fp, os.path.commonpath(WIKI_PATHS))
        file_to_rel[fp] = rel

        title = meta.get("title", os.path.splitext(os.path.basename(fp))[0])
        tags = meta.get("tags", [])
        if isinstance(tags, list):
            tags = ", ".join(tags)
        elif not isinstance(tags, str):
            tags = ""

        category = rel.split(os.sep)[0] if os.sep in rel else "wiki"
        chunks = chunk_text(body, heading=title)
        for c in chunks:
            all_chunks.append({
                "source": rel,
                "heading": c["heading"],
                "category": category,
                "node": ZVEC_COLLECTION,
                "content_hash": hashlib.md5(
                    c["text"].encode()).hexdigest()[:8],
                "char_count": len(c["text"]),
                "title": title,
                "tags": tags,
                "content": c["text"][:32000],
            })

    print(f"  📦 Total chunks: {len(all_chunks)}")

    # Embed + insert in batches
    total_docs = 0
    emb_errors = 0
    texts_batch: list[str] = []
    chunks_batch: list[dict] = []

    for i, chunk in enumerate(all_chunks):
        texts_batch.append(chunk["content"])
        chunks_batch.append(chunk)

        if len(texts_batch) >= BATCH_SIZE or i == len(all_chunks) - 1:
            embs = _embed_batch(texts_batch)
            if embs is None:
                embs = [None] * len(texts_batch)
                emb_errors += len(texts_batch)
                print(f"  ⚠ Embed batch failed, zeros used")
            elif len(embs) < len(texts_batch):
                embs = embs + [None] * (len(texts_batch) - len(embs))
                emb_errors += len(texts_batch) - len(embs)

            docs = []
            for j, c in enumerate(chunks_batch):
                d = Doc(
                    id=_safe_id(c['source'], c['content']),
                    score=1.0,
                    fields={
                        "source": c["source"],
                        "heading": c["heading"],
                        "category": c["category"],
                        "node": c["node"],
                        "content_hash": c["content_hash"],
                        "char_count": c["char_count"],
                        "title": c["title"],
                        "tags": c["tags"],
                        "content": c["content"],
                    },
                )
                if embs and j < len(embs) and embs[j] is not None:
                    d.vectors = {"embedding": embs[j]}
                docs.append(d)

            try:
                coll.insert(docs)
                total_docs += len(docs)
            except Exception as e:
                print(f"  ⚠ Insert error: {e}")

            if i > 0 and (i % (BATCH_SIZE * 10) == 0
                          or i == len(all_chunks) - 1):
                pct = (i + 1) * 100 // len(all_chunks)
                print(f"  📊 {pct}% ({total_docs} chunks indexed)")
                coll.flush()

            texts_batch = []
            chunks_batch = []

    # Update state
    if incremental:
        new_state = dict(state)
        for fp in to_index:
            rel = file_to_rel.get(fp)
            if rel:
                new_state[rel] = file_hash(fp)
        save_state(new_state)

    s = coll.stats
    print(f"\n  ✅ Done: {s.doc_count} docs (embed errors: {emb_errors})")
    coll.flush()


import re as _re


def _safe_id(source: str, content: str) -> str:
    """Zvec-safe doc ID: max 64 chars, only alphanumeric and underscore."""
    raw = f"{source}#{hashlib.md5(content.encode()).hexdigest()[:12]}"
    safe = _re.sub(r'[^a-zA-Z0-9]', '_', raw)
    if safe[0] == '_':
        safe = 'doc' + safe
    if len(safe) > 64:
        suffix = hashlib.md5(safe.encode()).hexdigest()[:12]
        safe = safe[:51] + suffix
    return safe[:64]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="ZVec incremental wiki indexer")
    p.add_argument("--incremental", action="store_true",
                   help="Only index changed files")
    p.add_argument("--clear", action="store_true",
                   help="Clear collection and full reindex")
    args = p.parse_args()
    index(incremental=args.incremental, clear=args.clear)