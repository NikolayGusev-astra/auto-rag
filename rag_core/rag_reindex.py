#!/usr/bin/env python3
"""ZVec index freshness checker — cron job.

Запуск:
  python rag_reindex.py                     # проверить + отчёт
  python rag_reindex.py --reindex           # проверить и переиндексировать если stale
  python rag_reindex.py --cron              # тихий режим для cron (вывод только если нужен reindex)

Формат вывода (JSON):
  {
    "collections": {
      "wiki": {
        "path": "...",
        "exists": true,
        "doc_count": 1234,
        "last_modified": "2026-07-01T12:00:00",
        "age_hours": 5.2,
        "stale": false,
        "index_size_mb": 45.2
      }
    },
    "reindex_needed": false,
    "summary": "2/3 collections fresh"
  }
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import zvec
    ZVEC_OK = True
except ImportError:
    ZVEC_OK = False

ZVEC_BASE = os.path.expanduser("~/.cache/zvec")
COLLECTIONS = ["wiki", "sessions", "work"]
STALE_HOURS = 24  # считаем stale если не обновлялось >24ч


def _get_collection_info(name: str) -> dict:
    """Get collection metadata."""
    path = os.path.join(ZVEC_BASE, name)
    info = {"path": path, "exists": False, "doc_count": 0,
            "last_modified": None, "age_hours": None, "stale": True,
            "index_size_mb": 0.0}

    if not os.path.isdir(path):
        info["stale_reason"] = "collection directory not found"
        return info
    info["exists"] = True

    # Size
    total_bytes = 0
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                total_bytes += os.path.getsize(fp)
            except OSError:
                pass
    info["index_size_mb"] = round(total_bytes / (1024 * 1024), 1)

    # Last modified (newest file)
    newest = 0.0
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                mtime = os.path.getmtime(fp)
                if mtime > newest:
                    newest = mtime
            except OSError:
                pass

    if newest > 0:
        info["last_modified"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(newest))
        age = (time.time() - newest) / 3600
        info["age_hours"] = round(age, 1)
        info["stale"] = age > STALE_HOURS
        if info["stale"]:
            info["stale_reason"] = f"last modified {info['age_hours']}h ago (threshold {STALE_HOURS}h)"
    else:
        info["stale_reason"] = "no files found in collection"

    # Doc count (try to open with zvec)
    if ZVEC_OK:
        try:
            lock = path + "/LOCK"
            try:
                with open(lock, "w") as f:
                    f.write("")
            except OSError:
                pass
            coll = zvec.open(path)
            # zvec doesn't expose total count, approximate via file count
            info["doc_count"] = len(os.listdir(path)) - 1  # -1 for LOCK
        except Exception as e:
            info["open_error"] = str(e)[:200]

    return info


def check_freshness() -> dict:
    """Check all collections."""
    result = {"collections": {}, "reindex_needed": False, "issues": []}

    for name in COLLECTIONS:
        info = _get_collection_info(name)
        result["collections"][name] = info
        if info["stale"]:
            result["reindex_needed"] = True
            result["issues"].append(f"{name}: {info.get('stale_reason', 'unknown reason')}" if info.get("stale_reason") else name)

    fresh = sum(1 for c in result["collections"].values() if c.get("exists") and not c.get("stale"))
    total = sum(1 for c in result["collections"].values() if c.get("exists"))
    result["summary"] = f"{fresh}/{total} collections fresh"
    if result["issues"]:
        result["summary"] += f", {len(result['issues'])} issue(s): " + "; ".join(result["issues"])
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ZVec freshness checker")
    parser.add_argument("--reindex", action="store_true", help="reindex stale collections")
    parser.add_argument("--cron", action="store_true", help="quiet mode for cron (output only if stale)")
    args = parser.parse_args()

    report = check_freshness()

    if args.cron:
        if report["reindex_needed"]:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            sys.exit(0)  # silent exit — nothing to report
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.reindex and report["reindex_needed"]:
        print("\n⚠ Reindex requested but not implemented — run zvec_indexer.py manually")
        print("  python zvec_wiki_indexer.py")
        print("  python zvec_work_indexer.py")
        print("  python zvec_session_indexer.py")