#!/usr/bin/env python3
"""
RAG v2 Indexer — ZVec wiki collection builder for Autolycus.
Usage: python3 indexer.py [--incremental] [--clear]
"""
import os, sys, json, hashlib, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_config import ZVEC_PATH, ZVEC_COLLECTION, EMBEDDING_DIM
from rag_config import EMBEDDING_URL, EMBEDDING_MODEL

# ── Config ────────────────────────────────────────────────────────
COLL_PATH = os.path.join(ZVEC_PATH, ZVEC_COLLECTION)
WIKI_PATHS = [
    os.path.expanduser("~/wiki"),
    os.path.expanduser("~/llm-wiki"),
    os.path.expanduser("~/skills"),
]
CHUNK_SIZE = 2000
STATE_FILE = os.path.join(ZVEC_PATH, ".index_state.v2.json")
BATCH_SIZE = 8  # smaller = less stuck on failed batch

EXCLUDE_PATTERNS = [
    ".email_cache", "queries", "session-notes", "wiki/log",
    "raw/search_", "raw/import-digest", "raw/product/meeting",
    "concepts/202605", "concepts/202606",
    "skills/references", "skills/scripts", "skills/assets",
    "node_modules", ".git", "__pycache__",
]

# ── Files ─────────────────────────────────────────────────────────
SUPPORTED_EXT = {".md", ".txt", ".rst", ".py", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".conf", ".sh", ".env"}

def should_exclude(path):
    for pat in EXCLUDE_PATTERNS:
        if pat in path: return True
    return False

def collect_files():
    files = []
    for base in WIKI_PATHS:
        if not os.path.isdir(base): continue
        for root, dirs, fnames in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for fn in fnames:
                fp = os.path.join(root, fn)
                ext = os.path.splitext(fn)[1].lower()
                if ext in SUPPORTED_EXT and not should_exclude(fp):
                    files.append(fp)
    return sorted(files)

def file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''): h.update(chunk)
    return h.hexdigest()

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f: return json.load(f)
        except: pass
    return {}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f: json.dump(state, f)

# ── Chunking ──────────────────────────────────────────────────────
def parse_frontmatter(text):
    text = text.lstrip('\ufeff')
    if not text.startswith('---'): return {}, text
    end = text.find('---', 3)
    if end == -1: return {}, text
    meta = {}
    try:
        import yaml
        meta = yaml.safe_load(text[3:end].strip()) or {}
    except: pass
    return (meta if isinstance(meta, dict) else {}), text[end + 3:].strip()

def chunk_text(text, heading="Overview"):
    chunks = []
    lines = text.split('\n')
    current_heading = heading
    current_lines = []

    def flush():
        nonlocal current_lines
        if not current_lines: return
        content = '\n'.join(current_lines).strip()
        if not content:
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
                if len(buf) > CHUNK_SIZE:
                    for i in range(0, len(buf), CHUNK_SIZE):
                        chunks.append({"heading": current_heading, "text": buf[i:i+CHUNK_SIZE]})
                else:
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
def _embed_via_curl(texts):
    """Embedding via subprocess curl (requests.connect fails on localhost:1234)."""
    import subprocess, json
    if not texts: return None
    try:
        payload = json.dumps({"model": EMBEDDING_MODEL, "input": texts})
        r = subprocess.run(
            ["curl", "-s", "--max-time", "10", EMBEDDING_URL, "-d", payload, "-H", "Content-Type: application/json"],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode == 0 and r.stdout:
            data = json.loads(r.stdout)
            return [d["embedding"] for d in data["data"]]
    except: pass
    return None

def get_embeddings_batch(texts):
    return _embed_via_curl(texts)

def get_embedding_single(text):
    r = _embed_via_curl([text])
    return r[0] if r else None

# ── Schema ────────────────────────────────────────────────────────
def get_schema():
    from zvec import CollectionSchema, FieldSchema, VectorSchema, DataType
    from zvec import FtsIndexParam, HnswIndexParam, InvertIndexParam, MetricType
    return CollectionSchema(
        name="wiki",
        fields=[
            FieldSchema("source", DataType.STRING, nullable=False, index_param=InvertIndexParam()),
            FieldSchema("heading", DataType.STRING, nullable=True),
            FieldSchema("category", DataType.STRING, nullable=False, index_param=InvertIndexParam()),
            FieldSchema("node", DataType.STRING, nullable=False),
            FieldSchema("content_hash", DataType.STRING, nullable=True),
            FieldSchema("char_count", DataType.INT32, nullable=True),
            FieldSchema("title", DataType.STRING, nullable=True),
            FieldSchema("tags", DataType.STRING, nullable=True),
            FieldSchema("content", DataType.STRING, nullable=False,
                       index_param=FtsIndexParam(tokenizer_name="standard", filters=["lowercase"])),
        ],
        vectors=[
            VectorSchema("embedding", DataType.VECTOR_FP32, dimension=EMBEDDING_DIM,
                        index_param=HnswIndexParam(metric_type=MetricType.COSINE)),
        ],
    )

def ensure_zvec_lock(path: str) -> str:
    """ZVec 0.5.1 LOCK workaround."""
    import os as _os
    lock_path = _os.path.join(path, "LOCK")
    if _os.path.exists(path) and not _os.path.exists(lock_path):
        try:
            fd = _os.open(lock_path, _os.O_CREAT | _os.O_WRONLY, 0o644)
            _os.close(fd)
        except OSError:
            pass
    return lock_path


# ── Main ──────────────────────────────────────────────────────────
def index(incremental=False, clear=False):
    import zvec
    zvec.init()

    # Open/create collection
    ensure_zvec_lock(COLL_PATH)
    if clear and os.path.exists(COLL_PATH):
        import shutil
        shutil.rmtree(COLL_PATH, ignore_errors=True)
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        print("  🗑 Collection cleared")
        coll = None
    elif os.path.exists(COLL_PATH):
        try:
            from zvec import CollectionOption
            coll = zvec.open(COLL_PATH)
            print(f"  📂 Opened existing: {coll.stats}")
        except Exception:
            from zvec import CollectionOption
            import shutil
            shutil.rmtree(COLL_PATH, ignore_errors=True)
            coll = zvec.create_and_open(COLL_PATH, get_schema(), CollectionOption(read_only=False, enable_mmap=True))
            print(f"  ✅ Created fresh: {coll.stats}")
    else:
        from zvec import CollectionOption
        coll = zvec.create_and_open(COLL_PATH, get_schema(), CollectionOption(read_only=False, enable_mmap=True))
        print(f"  ✅ Created new: {coll.stats}")

    if coll is None:
        from zvec import CollectionOption
        coll = zvec.create_and_open(COLL_PATH, get_schema(), CollectionOption(read_only=False, enable_mmap=True))

    # Collect files
    files = collect_files()
    print(f"  📄 Found {len(files)} files")

    state = load_state() if incremental else {}
    to_index = []
    if incremental:
        tracked = set(state.keys())
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

    # Process files
    from zvec import Doc
    all_chunks = []
    file_to_rel = {}

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
        if isinstance(tags, list): tags = ", ".join(tags)
        elif isinstance(tags, str): tags = tags
        else: tags = ""

        category = rel.split("/")[0] if "/" in rel else "wiki"
        chunks = chunk_text(body, heading=title)
        for c in chunks:
            all_chunks.append({
                "source": rel,
                "heading": c["heading"],
                "category": category,
                "node": "work",
                "content_hash": hashlib.md5(c["text"].encode()).hexdigest()[:8],
                "char_count": len(c["text"]),
                "title": title,
                "tags": tags,
                "content": c["text"][:32000],
            })

    print(f"  📦 Total chunks: {len(all_chunks)}")

    # Embed + insert in batches
    total_docs = 0
    emb_errors = 0
    texts_batch = []
    chunks_batch = []

    for i, chunk in enumerate(all_chunks):
        texts_batch.append(chunk["content"])
        chunks_batch.append(chunk)

        if len(texts_batch) >= BATCH_SIZE or i == len(all_chunks) - 1:
            embs = get_embeddings_batch(texts_batch)
            if embs is None:
                embs = []
                for t in texts_batch:
                    e = get_embedding_single(t)
                    embs.append(e)
                failed = sum(1 for e in embs if e is None)
                if failed:
                    print(f"  ⚠ {failed}/{len(texts_batch)} single failed, zeros used")

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
                if embs[j] is not None:
                    d.vectors = {"embedding": embs[j]}
                docs.append(d)

            try:
                coll.insert(docs)
                total_docs += len(docs)
            except Exception as e:
                print(f"  ⚠ Insert error: {e}")

            if i > 0 and (i % (BATCH_SIZE * 10) == 0 or i == len(all_chunks) - 1):
                pct = (i + 1) * 100 // len(all_chunks)
                print(f"  📊 {pct}% ({total_docs} chunks indexed)")
                coll.flush()

            texts_batch = []
            chunks_batch = []

    # Update state
    new_state = dict(state)
    for fp in to_index:
        rel = file_to_rel.get(fp)
        if rel:
            new_state[rel] = file_hash(fp)
    save_state(new_state)

    s = coll.stats
    print(f"\n  ✅ Done: {s.doc_count} docs (embed errors: {emb_errors})")
    coll.flush()


import re
def _safe_id(source, content):
    """Zvec-safe doc ID: max 64 chars, only alphanumeric and underscore."""
    raw = f"{source}#{hashlib.md5(content.encode()).hexdigest()[:12]}"
    safe = re.sub(r'[^a-zA-Z0-9]', '_', raw)
    if safe[0] == '_':
        safe = 'doc' + safe
    # Zvec rejects IDs longer than 64 chars
    if len(safe) > 64:
        # Keep first 48 chars + 12-char hash
        suffix = hashlib.md5(safe.encode()).hexdigest()[:12]
        safe = safe[:51] + suffix
    return safe[:64]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--incremental", action="store_true")
    p.add_argument("--clear", action="store_true")
    args = p.parse_args()
    index(incremental=args.incremental, clear=args.clear)
