#!/usr/bin/env python3
"""
ChromaDB Indexer — индексирует wiki/*.md в векторную БД.
Использует text-embedding-multilingual-e5-large-instruct через API.

CRUD-стабильная индексация:
- Детерминистичные ID (хеш от source + heading) — не зависят от порядка файлов
- Delete-before-upsert для incremental — старые чанки заменяются корректно
- Очистка чанков удалённых файлов
- Content-based dedup — пропуск точных дубликатов

Usage:
    python3 rag_indexer.py              # полная индексация
    python3 rag_indexer.py --incremental # только новые/изменённые + cleanup
    python3 rag_indexer.py --stats       # статистика
    python3 rag_indexer.py --clear       # очистить
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml
import chromadb
from chromadb.config import Settings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_config import *


# ═══════════════════════════════════════════════════════════════════════════
# Хелперы
# ═══════════════════════════════════════════════════════════════════════════

def chunk_id(source: str, heading: str, seq: int = 0) -> str:
    """Детерминистичный ID чанка: хеш от source + heading + seq.
    
    При повторной индексации того же файла ID не меняется,
    не зависит от порядка файлов или глобального счётчика.
    """
    raw = f"{source}::{heading}::{seq}"
    return hashlib.md5(raw.encode(errors='replace')).hexdigest()[:16]


def content_hash(text: str) -> str:
    """Хеш содержимого чанка для content-based dedup."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Извлекает YAML frontmatter и body."""
    text = text.lstrip('\ufeff')
    if not text.startswith('---'):
        return {}, text
    end = text.find('---', 3)
    if end == -1:
        return {}, text
    try:
        meta = yaml.safe_load(text[3:end].strip()) or {}
    except Exception:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), text[end + 3:].strip()


# ── Chunking strategies ───────────────────────────────────────────
def recursive_chunk_text(text, heading="Overview", chunk_size=2000, min_chunk=200):
    """Recursive chunking: paragraphs → sentences → fixed-size.
    Preserves semantic boundaries better than fixed-size splitting."""
    chunks = []
    lines = text.split('\n')
    current_heading = heading
    current_lines = []

    def flush():
        nonlocal current_lines
        if not current_lines: return
        content = '\n'.join(current_lines).strip()
        if not content or len(content) < 20:
            current_lines = []
            return
        if len(content) <= chunk_size:
            chunks.append({"text": content, "heading": current_heading})
            current_lines = []
            return
        paragraphs = content.split('\n\n')
        buf = ""
        for p in paragraphs:
            if len(buf) + len(p) > chunk_size and buf:
                if len(buf) >= min_chunk:
                    chunks.append({"text": buf, "heading": current_heading})
                buf = p
            else:
                buf = (buf + '\n\n' + p) if buf else p
        if buf:
            if len(buf) >= min_chunk:
                chunks.append({"text": buf, "heading": current_heading})
            else:
                if chunks:
                    chunks[-1]["text"] += '\n\n' + buf
                else:
                    chunks.append({"text": buf, "heading": current_heading})
        current_lines = []

    for line in lines:
        h_match = re.match(r'^(#{1,4})\s+(.+)$', line)
        if h_match:
            flush()
            current_heading = h_match.group(2).strip()
        else:
            current_lines.append(line)
    flush()
    return chunks


def fixed_chunk_text(text, heading="Overview", chunk_size=2000):
    """Original fixed-size chunking with paragraph-aware splitting."""
    chunks = []
    lines = text.split('\n')
    current_heading = heading
    current_lines = []

    def flush():
        nonlocal current_lines
        if not current_lines: return
        content = '\n'.join(current_lines).strip()
        if not content or len(content) < 20:
            current_lines = []
            return
        if len(content) > chunk_size:
            parts = content.split('\n\n')
            buf = ""
            for p in parts:
                if len(buf) + len(p) > chunk_size and buf:
                    chunks.append({"text": buf, "heading": current_heading})
                    buf = p
                else:
                    buf = (buf + '\n\n' + p) if buf else p
            if buf:
                if len(buf) > chunk_size:
                    for i in range(0, len(buf), chunk_size):
                        chunks.append({"text": buf[i:i+chunk_size], "heading": current_heading})
                else:
                    chunks.append({"text": buf, "heading": current_heading})
        else:
            chunks.append({"text": content, "heading": current_heading})
        current_lines = []

    for line in lines:
        _h = re.match(r'^(#{1,4})\s+(.+)$', line)
        if _h:
            flush()
            current_heading = _h.group(2).strip()
        else:
            current_lines.append(line)
    flush()
    return chunks


def auto_category(source: str) -> str:
    """Auto-detect category from path."""
    rules = [
        (r"\.email_cache/", "email"),
        (r"\.tg_history/", "tg"),
        (r"concepts/", "concept"),
        (r"entities/", "entity"),
        (r"raw/", "raw"),
        (r"verified/", "verified"),
        (r"sessions/", "session"),
        (r"llm-wiki/", "llm-wiki"),
    ]
    for pattern, category in rules:
        if re.search(pattern, source):
            return category
    parts = source.split("/")
    if len(parts) > 1:
        return parts[0]
    return "wiki"


# DCD: Domain mapping from file path → domain name
# Синхронизировано с DCD.md и dcd_router.py
DOMAIN_PATH_MAP = [
    # software-dev
    ("autolycus-desktop", "software-dev"),
    ("autolycus-desktop-dev", "software-dev"),
    ("tauri", "software-dev"),
    # devops
    ("adr-", "devops"),
    ("autolycus-architecture", "devops"),
    ("autolycus-full-structure", "devops"),
    ("autolycus-release", "devops"),
    ("infrastructure", "devops"),
    ("xray", "devops"),
    ("3x-ui", "devops"),
    ("openvpn", "devops"),
    ("merge-plan", "devops"),
    ("merge-refactor", "devops"),
    ("sbl", "devops"),
    ("salt-ipa", "devops"),
    ("kanban", "devops"),
    ("llm-provider", "devops"),
    ("telegram-gateway-development", "devops"),
    ("website-architecture", "devops"),
    ("policy-gui", "devops"),
    # publishing
    ("autolycus-article", "publishing"),
    ("hermes-article-writer", "publishing"),
    # research
    ("llm-wiki", "research"),
    ("dcd-paper", "research"),
    ("osint", "research"),
    ("oss-forensics", "research"),
    ("diagnos", "research"),
    # creative
    ("diagram", "creative"),
    # integrations
    ("onto", "integrations"),
    ("carpc-ford", "integrations"),
    # personal
    ("ford-club", "personal"),
    ("ford-explorer", "personal"),
    ("sibionics", "personal"),
    ("how-to-work", "personal"),
    # analysis
    ("prism", "analysis"),
    ("premortem", "analysis"),
    ("phase3", "analysis"),
    ("phase4", "analysis"),
    ("posthoc", "analysis"),
    # security
    ("sbl-service-map", "security"),
    # automation
    ("memoryagent", "automation"),
    # agent-core
    ("rtk-ck", "agent-core"),
    ("recovered", "agent-core"),
]


def auto_domain(source: str) -> str:
    """Auto-detect DCD domain from file path."""
    source_lower = source.lower()
    for pattern, domain in DOMAIN_PATH_MAP:
        if pattern in source_lower:
            return domain
    return "general"


def _get_ftype(fp: str) -> str:
    """Human-friendly file type label for metadata."""
    from file_readers import get_file_category
    return get_file_category(fp)


def chunk_markdown(text: str, meta: dict, filepath: str, chunk_mode: str = "recursive") -> list[dict]:
    """Чанкинг markdown по заголовкам ## и ###.

    Каждый чанк получает мета-поле chunk_content_hash для content dedup.
    chunk_mode: 'recursive' (paragraph-boundary aware) or 'fixed' (size-based).
    """
    if not text.strip():
        return []

    # ── Recursive mode (default) ──
    if chunk_mode == "recursive":
        raw_chunks = recursive_chunk_text(text, heading="Overview", chunk_size=CHUNK_SIZE)
        result = []
        for c in raw_chunks:
            result.append({
                "text": c["text"],
                "metadata": {
                    "source": filepath,
                    "heading": c["heading"],
                    "char_count": len(c["text"]),
                    "node": os.environ.get("RAG_NODE_NAME", "autolycus"),
                    "category": auto_category(filepath),
                    "domain": meta.get("domain", auto_domain(filepath)),
                }
            })
        return result

    # ── Fixed mode (legacy) ──
    return _chunk_markdown_fixed(text, meta, filepath)


def chunk_text(text, heading="Overview", mode="recursive"):
    """Dispatch to chunking strategy. Wrapper for test compatibility."""
    if mode == "recursive":
        return recursive_chunk_text(text, heading, chunk_size=CHUNK_SIZE)
    return fixed_chunk_text(text, heading, chunk_size=CHUNK_SIZE)


def _chunk_markdown_fixed(text, meta, filepath):
    """Legacy fixed-size chunking logic."""
    if not text.strip():
        return []
    lines = text.split('\n')
    chunks = []
    current_heading = "Overview"
    current_lines = []

    def flush():
        nonlocal current_lines
        if not current_lines:
            return
        content = '\n'.join(current_lines).strip()
        if len(content) < 20:
            current_lines = []
            return
        chunks.append({
            "text": content,
            "metadata": {
                "source": filepath,
                "heading": current_heading,
                "char_count": len(content),
                "node": os.environ.get("RAG_NODE_NAME", "autolycus"),
                "category": auto_category(filepath),
                "domain": meta.get("domain", auto_domain(filepath)),
            }
        })
        current_lines = []

    for line in lines:
        h_match = re.match(r'^(#{1,4})\s+(.+)$', line)
        if h_match:
            flush()
            current_heading = h_match.group(2).strip()
        else:
            current_lines.append(line)
    flush()

    if not chunks and text.strip():
        chunks.append({
            "text": text.strip()[:CHUNK_SIZE],
            "metadata": {
                "source": filepath,
                "heading": "Overview",
                "char_count": min(len(text.strip()), CHUNK_SIZE),
                "domain": meta.get("domain", auto_domain(filepath)),
            }
        })

    # Smart split длинных чанков
    final = []
    for chunk in chunks:
        cc = chunk["metadata"]["char_count"]
        if cc <= CHUNK_SIZE:
            final.append(chunk)
        else:
            text = chunk["text"]
            parts = text.split('\n\n')
            if len(parts) > 1:
                buf, blen = [], 0
                for p in parts:
                    to_add = []
                    if len(p) > CHUNK_SIZE:
                        pos = 0
                        while pos < len(p):
                            end = min(pos + CHUNK_SIZE, len(p))
                            if end < len(p):
                                ls = p.rfind(' ', pos, end)
                                if ls > pos:
                                    end = ls
                            to_add.append(p[pos:end].strip())
                            pos = end + 1 if end < len(p) and p[end] == ' ' else end
                    else:
                        to_add = [p]
                    for subp in to_add:
                        if blen + len(subp) > CHUNK_SIZE and buf:
                            final.append({
                                "text": '\n\n'.join(buf),
                                "metadata": {**chunk["metadata"], "char_count": sum(len(x) for x in buf), "domain": chunk["metadata"].get("domain", auto_domain(filepath))}
                            })
                            buf, blen = [subp], len(subp)
                        else:
                            buf.append(subp)
                            blen += len(subp)
                if buf:
                    final.append({
                        "text": '\n\n'.join(buf),
                        "metadata": {**chunk["metadata"], "char_count": blen, "domain": chunk["metadata"].get("domain", auto_domain(filepath))}
                    })
            else:
                pos = 0
                while pos < len(text):
                    end = min(pos + CHUNK_SIZE, len(text))
                    if end < len(text):
                        last_space = text.rfind(' ', pos, end)
                        if last_space > pos:
                            end = last_space
                    final.append({
                        "text": text[pos:end].strip(),
                        "metadata": {**chunk["metadata"], "char_count": end - pos, "domain": auto_domain(filepath)}
                    })
                    pos = end + 1 if end < len(text) and text[end] == ' ' else end
    
    # Добавляем content_hash в каждый чанк
    for c in final:
        c["metadata"]["content_hash"] = content_hash(c["text"])
        c["content_hash"] = c["metadata"]["content_hash"]
    
    return final


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embedding через API с правильным префиксом e5-instruct."""
    prefixed = [
        f"Instruct: Given a wiki search query, retrieve relevant wiki passages\nQuery: {t[:MAX_CHARS_PER_INPUT]}"
        for t in texts
    ]
    try:
        resp = requests.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL,
            "input": prefixed
        }, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        data["data"].sort(key=lambda x: x["index"])
        return [item["embedding"] for item in data["data"]]
    except requests.Timeout:
        print(f"  ⚠ Embedding timeout, retrying with smaller batch...", file=sys.stderr)
        results = []
        for t in texts:
            try:
                r = requests.post(EMBEDDING_URL, json={
                    "model": EMBEDDING_MODEL,
                    "input": [f"Instruct: Given a wiki search query, retrieve relevant wiki passages\nQuery: {t[:MAX_CHARS_PER_INPUT]}"]
                }, timeout=120)
                r.raise_for_status()
                results.append(r.json()["data"][0]["embedding"])
            except Exception as e:
                print(f"  ⚠ Single embedding failed: {e}", file=sys.stderr)
                results.append([0.0] * EMBEDDING_DIM)
            time.sleep(0.1)
        return results
    except Exception as e:
        print(f"  ⚠ Embedding error: {e}", file=sys.stderr)
        raise


def file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# State management
# ═══════════════════════════════════════════════════════════════════════════

STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.rag_state.json')


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state: dict):
    with open(STATE_PATH + '.tmp', 'w') as f:
        json.dump(state, f)
    os.replace(STATE_PATH + '.tmp', STATE_PATH)


# ═══════════════════════════════════════════════════════════════════════════
# File collection
# ═══════════════════════════════════════════════════════════════════════════

def collect_md_files(paths: list[str]) -> list[str]:
    """Сбор файлов для индексации. Поддерживает .md .txt .yml .pdf .docx .tf .py и др."""
    import fnmatch
    import re
    from file_readers import SUPPORTED_EXT
    files = []
    _auto_finding_re = re.compile(r'^\d{8}-\d{6}-')
    for base in paths:
        if not os.path.isdir(base):
            continue
        for root, dirs, fnames in os.walk(base):
            dirs[:] = [
                d for d in dirs
                if not any(
                    fnmatch.fnmatch(d, pat) or fnmatch.fnmatch(os.path.relpath(os.path.join(root, d), base), pat)
                    for pat in EXCLUDE_DIRS
                )
            ]
            for fn in fnames:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in SUPPORTED_EXT:
                    continue
                if _auto_finding_re.match(fn):
                    continue
                relpath = os.path.relpath(os.path.join(root, fn), base)
                if EXCLUDE_EMAIL_CACHE and ".email_cache" in Path(relpath).parts:
                    continue
                files.append(os.path.join(root, fn))
    return sorted(files)


# ═══════════════════════════════════════════════════════════════════════════
# Batch flush
# ═══════════════════════════════════════════════════════════════════════════

def _flush_batch(collection, texts: list, ids: list, metadatas: list, dedup_set: set = None):
    """Flush batch с content-based dedup."""
    if not texts:
        return
    
    # Content-based dedup: пропускаем чанки с уже виденным content_hash
    if dedup_set is not None:
        filtered_texts, filtered_ids, filtered_md = [], [], []
        dedup_count = 0
        for i in range(len(texts)):
            h = metadatas[i].get("content_hash", "")
            if h and h in dedup_set:
                dedup_count += 1
                continue
            filtered_texts.append(texts[i])
            filtered_ids.append(ids[i])
            filtered_md.append(metadatas[i])
        texts, ids, metadatas = filtered_texts, filtered_ids, filtered_md
        
        if dedup_count > 0:
            print(f"   ⏭ Skipped {dedup_count} duplicate chunks (content dedup)", flush=True)
        
        if not texts:
            texts.clear()
            ids.clear()
            metadatas.clear()
            return
        
        # Record new hashes
        for m in metadatas:
            h = m.get("content_hash", "")
            if h:
                dedup_set.add(h)

    try:
        embs = embed_texts(texts)
        collection.upsert(embeddings=embs, documents=texts, ids=ids, metadatas=metadatas)
        print(f"   📥 Batch {len(texts)} chunks → Chroma (total: {collection.count()})", flush=True)
    except Exception as e:
        print(f"   ⚠ Batch failed ({len(texts)} chunks): {e}", flush=True)
        for i in range(len(texts)):
            try:
                emb = embed_texts([texts[i]])
                collection.upsert(embeddings=emb, documents=[texts[i]], ids=[ids[i]], metadatas=[metadatas[i]])
            except Exception as e2:
                print(f"   ⚠ Skip chunk {ids[i]}: {e2}", flush=True)
    texts.clear()
    ids.clear()
    metadatas.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Main index function (CRUD)
# ═══════════════════════════════════════════════════════════════════════════

def index_all(incremental: bool = False, source_filter: str | None = None, chunk_mode: str = "recursive"):
    # ═══ Стабильный CWD для relpath — не зависит от того, откуда вызван скрипт ═══
    os.chdir('/root')

    print(f"📚 Wiki RAG Indexer", flush=True)
    print(f"   Search paths: {WIKI_PATHS}", flush=True)
    print(f"   Embedding: {EMBEDDING_MODEL} ({EMBEDDING_DIM}d) @ {EMBEDDING_URL}", flush=True)
    print(f"   Chroma: {CHROMA_PATH}/{COLLECTION_NAME}", flush=True)
    mode = "source rebuild" if source_filter else ("incremental (CRUD)" if incremental else "full rebuild")
    print(f"   Mode: {mode}", flush=True)

    files = collect_md_files(WIKI_PATHS)
    current_rel_set = set(os.path.relpath(fp) for fp in files)
    print(f"   Found {len(files)} .md files", flush=True)

    state = load_state()
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )

    if not incremental and not source_filter:
        # Полная перестройка
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        state = {}
    else:
        try:
            collection = client.get_collection(COLLECTION_NAME)
            print(f"   Existing: {collection.count()} chunks", flush=True)
        except Exception:
            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            state = {}

    # ── Шаг 2: Cleanup удалённых файлов ─────────────────────────────────
    if incremental and not source_filter:
        deleted = [rel for rel in state if rel not in current_rel_set]
        if deleted:
            print(f"   🗑 Cleanup {len(deleted)} deleted files:", flush=True)
            for rel in deleted[:10]:
                try:
                    collection.delete(where={"source": rel})
                    print(f"      Deleted chunks for: {rel}", flush=True)
                except Exception as e:
                    print(f"      ⚠ Failed to delete {rel}: {e}", flush=True)
                    pass
            if len(deleted) > 10:
                print(f"      ... and {len(deleted) - 10} more", flush=True)
            # Удаляем из state
            for rel in deleted:
                del state[rel]

    # ── Определяем файлы для индексации ─────────────────────────────────
    todo = []
    for fp in files:
        rel = os.path.relpath(fp)
        fh = file_hash(fp)
        if source_filter and rel != source_filter:
            continue
        if incremental and state.get(rel) == fh:
            continue
        todo.append((fp, rel, fh))

    if not todo:
        print(f"   ✅ All files up to date ({len(files)} total)", flush=True)
        save_state(state)
        return

    print(f"   Files to index: {len(todo)}", flush=True)

    BATCH_SIZE = 512
    
    # Глобальный set content_hash для dedup в пределах одной сессии индексации
    global_dedup = set()
    batch_texts, batch_ids, batch_metadatas = [], [], []

    for fp, rel, fh in todo:
        # ── Шаг 2 (для incremental): delete-before-upsert ──────────────
        if incremental:
            if batch_texts:
                _flush_batch(collection, batch_texts, batch_ids, batch_metadatas)
            try:
                # Удаляем старые чанки этого файла
                old_count = 0
                old_meta = collection.get(where={"source": rel})
                if old_meta and old_meta["ids"]:
                    old_count = len(old_meta["ids"])
                    collection.delete(where={"source": rel})
                if old_count > 0:
                    print(f"   🗑 Removed {old_count} old chunks for: {rel}", flush=True)
            except Exception as e:
                print(f"   ⚠ Could not delete old chunks for {rel}: {e}", flush=True)
                pass

        # Читаем и чанкуем
        try:
            from file_readers import read_file_text
            raw = read_file_text(fp)
            if raw is None:
                print(f"   ⏭ Skip unsupported/unreadable: {rel}", flush=True)
                state[rel] = fh
                continue
        except Exception as e:
            print(f"   ⚠ Read error {rel}: {e}", flush=True)
            state[rel] = fh
            continue

        md, body = parse_frontmatter(raw)
        if not body.strip():
            state[rel] = fh
            continue

        chunks = chunk_markdown(body, md, rel, chunk_mode=chunk_mode)
        if not chunks:
            state[rel] = fh
            continue

        title = md.get("title", "")
        ptype = md.get("type", "")
        tags = ",".join(md.get("tags", [])) if isinstance(md.get("tags"), list) else str(md.get("tags", ""))

        for chunk_idx, c in enumerate(chunks):
            # ── Шаг 3: Content-based dedup ────────────────────────────
            h = c.get("content_hash", "")
            if h and h in global_dedup:
                print(f"   ⏭ Skipped duplicate chunk: {rel} › {c['metadata']['heading']}", flush=True)
                continue
            if h:
                global_dedup.add(h)
            
            # Чанк-индекс как seq для детерминистичного ID
            cid = chunk_id(rel, c["metadata"]["heading"], chunk_idx)
            
            batch_texts.append(c["text"])
            batch_ids.append(cid)
            batch_metadatas.append({
                "source": rel,
                "heading": c["metadata"]["heading"],
                "title": title,
                "type": ptype,
                "tags": tags,
                "char_count": c["metadata"]["char_count"],
                "content_hash": h,
                "node": c["metadata"].get("node", os.environ.get("RAG_NODE_NAME", "autolycus")),
                "category": c["metadata"].get("category", auto_category(rel)),
                "domain": c["metadata"].get("domain", auto_domain(rel)),
                "fext": os.path.splitext(fp)[1].lower(),
                "ftype": _get_ftype(fp),
            })

            if len(batch_texts) >= BATCH_SIZE:
                _flush_batch(collection, batch_texts, batch_ids, batch_metadatas)

        if batch_texts:
            _flush_batch(collection, batch_texts, batch_ids, batch_metadatas)

        state[rel] = fh

    save_state(state)
    stats = show_stats(collection=collection)
    print(f"✅ Done! {len(todo)} files, {collection.count()} total chunks", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════════

def show_stats(collection=None):
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        coll = collection or client.get_collection(COLLECTION_NAME)
        count = coll.count()
        print(f"📊 Collection '{COLLECTION_NAME}': {count} chunks", flush=True)
        if count > 0:
            result = coll.peek()
            sources = set(m.get("source", "") for m in result["metadatas"])
            print(f"   Sources: {list(sources)[:5]}", flush=True)
    except Exception as e:
        print(f"⚠ Collection not found: {e}", flush=True)
    state = load_state()
    if state:
        print(f"   Tracked files: {len(state)}", flush=True)
    return count


def clear_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑 Collection '{COLLECTION_NAME}' deleted")
    except Exception as e:
        print(f"⚠ {e}")
    save_state({})
    print("📝 State reset")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaDB Indexer для wiki (CRUD)")
    parser.add_argument("--incremental", action="store_true", help="Только изменённые файлы + cleanup удалённых")
    parser.add_argument("--stats", action="store_true", help="Статистика")
    parser.add_argument("--clear", action="store_true", help="Очистить")
    parser.add_argument("--source", type=str, help="Индексировать только один файл/директорию (относительный путь)")
    parser.add_argument("--chunk-mode", choices=["fixed", "recursive"], default="recursive",
                        help="Стратегия чанкинга: recursive (по границам параграфов) или fixed (по размеру)")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.clear:
        clear_collection()
    else:
        index_all(incremental=args.incremental, source_filter=args.source, chunk_mode=args.chunk_mode)