#!/usr/bin/env python3
"""
hermes_memory_cli.py — management CLI for Hermes memvid capsules.

Commands:
  stats       Show capsule stats (frames, size, domains, time range)
  inspect     List recent episodes (optionally filtered by domain/score/time)
  search      Free-text search over a capsule
  compact     Compact/rewrite a capsule (dedup + drop low-value frames)
  branch      Branch a capsule at a given frame/time (time-travel)
  rewind      Restore capsule to a previous state (.bak)
  export      Export episodes to JSONL
  purge       Delete frames matching a filter (DANGEROUS — backs up first)

All commands work on the capsule resolved from env (RAG_MEMVID_DIR,
RAG_MEMVID_TENANT) unless --capsule is given.

Usage:
    python3 hermes_memory_cli.py stats
    python3 hermes_memory_cli.py inspect --domain astra --limit 20
    python3 hermes_memory_cli.py search "сброс пароля" --topk 5
    python3 hermes_memory_cli.py compact --min-score 0.5
    python3 hermes_memory_cli.py branch --at-frame 1234 --to memory_branch.mv2
    python3 hermes_memory_cli.py rewind --tenant hermes_default
    python3 hermes_memory_cli.py export --out episodes.jsonl
    python3 hermes_memory_cli.py purge --before 2024-12-01 --yes
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# reuse the memory facade + data model
from memvid_memory import (Episode, MemvidConfig, MemvidMemory,
                           _NoopMemvidBackend, _attr)


# ---------------------------------------------------------------------------
# capsule resolution
# ---------------------------------------------------------------------------
def _resolve_capsule(args) -> Path:
    if getattr(args, "capsule", None):
        p = Path(args.capsule)
    else:
        cfg = MemvidConfig.from_env()
        p = cfg.capsule_path
    if not p.exists():
        sys.exit(f"capsule not found: {p}")
    return p


def _open_backend(path: Path) -> Any:
    cfg = MemvidConfig.from_env()
    cfg.enabled = True
    # point capsule path at requested file
    cfg.dir = path.parent
    cfg.tenant = path.stem.replace("memory_", "")
    mem = MemvidMemory(cfg)
    if isinstance(mem._backend, _NoopMemvidBackend):
        sys.exit("memvid backend unavailable (SDK missing or init failed)")
    return mem


def _iter_frames(backend) -> List[Any]:
    """Best-effort frame listing across memvid-sdk API shapes."""
    for name in ("list_frames", "frames", "iter_frames", "all_frames"):
        fn = getattr(backend._mem, name, None) if hasattr(backend, "_mem") \
            else None
        if fn:
            try:
                return list(fn())
            except Exception:
                continue
    return []


def _frame_text(f) -> str:
    return _attr(f, "text") or (f.get("text") if isinstance(f, dict) else "")


def _frame_episode(f) -> Optional[Episode]:
    txt = _frame_text(f)
    if not txt:
        return None
    try:
        ep = Episode.from_payload(txt)
        ep.score = float(_attr(f, "score") or 0.0)
        ep.frame_id = _attr(f, "id") or _attr(f, "frame_id")
        return ep
    except Exception:
        return None


def _native_episodes(backend) -> List[Episode]:
    """Recover full Episode payloads from native MV2 search frames.

    Native `frame()` exposes structural metadata but not document text. The
    lexical index can retrieve the frame by its title; its hit text begins with
    the original JSON payload followed by SDK-added metadata.
    """
    mem = getattr(backend, "_mem", None)
    if mem is None or not hasattr(mem, "timeline"):
        return []
    out: List[Episode] = []
    decoder = json.JSONDecoder()
    try:
        timeline = mem.timeline(limit=100000)
    except Exception:
        return []
    for entry in timeline:
        frame_id = _attr(entry, "frame_id")
        uri = _attr(entry, "uri")
        try:
            frame = mem.frame(uri) if uri else {}
            title = _attr(frame, "title") or ""
            if not title:
                continue
            result = mem.find(title, k=20, mode="lex")
            hits = result.get("hits", []) if isinstance(result, dict) else []
            hit = next((h for h in hits if _attr(h, "frame_id") == frame_id), None)
            if hit is None:
                continue
            payload, _ = decoder.raw_decode(_frame_text(hit))
            ep = Episode.from_payload(json.dumps(payload, ensure_ascii=False))
            ep.frame_id = frame_id
            ep.score = float(_attr(hit, "score") or 0.0)
            out.append(ep)
        except Exception:
            continue
    return out


def _episodes(backend) -> List[Episode]:
    """Use native MV2 introspection first, then legacy frame fallbacks."""
    native = _native_episodes(backend)
    if native:
        return native
    frames = _iter_frames(backend)
    return [e for e in (_frame_episode(f) for f in frames) if e]


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_stats(args):
    p = _resolve_capsule(args)
    mem = _open_backend(p)
    eps = _episodes(mem._backend)
    size = p.stat().st_size
    native_stats = {}
    try:
        native_stats = mem._backend._mem.stats()
    except Exception:
        pass
    frame_count = native_stats.get("frame_count", len(eps)) if isinstance(native_stats, dict) else len(eps)
    domains = Counter(e.domain or "-" for e in eps)
    feedback = Counter(e.feedback or "-" for e in eps)
    times = [e.created_at for e in eps if e.created_at]
    times.sort()
    print(f"capsule       : {p}")
    print(f"size          : {size:,} bytes ({size/1024:.1f} KB)")
    print(f"frames        : {frame_count}")
    if isinstance(native_stats, dict):
        print(f"native vec    : {native_stats.get('has_vec_index', False)} "
              f"({native_stats.get('effective_vec_dimension', 0)}d, "
              f"{native_stats.get('vec_index_bytes', 0):,} bytes)")
    print(f"parsed episodes: {len(eps)}")
    if times:
        print(f"time range    : {times[0]} .. {times[-1]}")
    print(f"domains       : {dict(domains)}")
    print(f"feedback      : {dict(feedback)}")
    if eps:
        scores = [e.score for e in eps]
        print(f"score min/max : {min(scores):.3f} / {max(scores):.3f}")
    mem.close()


def cmd_inspect(args):
    p = _resolve_capsule(args)
    mem = _open_backend(p)
    eps = _episodes(mem._backend)
    # filters
    if args.domain:
        eps = [e for e in eps if e.domain == args.domain]
    if args.min_score is not None:
        eps = [e for e in eps if e.score >= args.min_score]
    if args.since:
        eps = [e for e in eps if e.created_at >= args.since]
    # newest first
    eps.sort(key=lambda e: e.created_at, reverse=True)
    eps = eps[:args.limit]
    print(f"showing {len(eps)} episodes (newest first)")
    for e in eps:
        print(f"\n[{e.created_at}] id={e.episode_id[:8]} "
              f"domain={e.domain} score={e.score:.3f}")
        print(f"  Q: {e.query[:120]}")
        print(f"  A: {e.answer[:160]}{'...' if len(e.answer)>160 else ''}")
        if e.sources:
            print(f"  sources: {len(e.sources)}")
    mem.close()


def cmd_search(args):
    p = _resolve_capsule(args)
    mem = _open_backend(p)
    eps = mem.recall(args.query, top_k=args.topk, when=args.when)
    if not eps:
        print("no hits")
        mem.close()
        return
    for i, e in enumerate(eps, 1):
        print(f"\n#{i} score={e.score:.3f}  {e.created_at[:10]}  "
              f"domain={e.domain}  id={e.episode_id[:8]}")
        print(f"  Q: {e.query}")
        print(f"  A: {e.answer[:240]}")
    mem.close()


def cmd_compact(args):
    """Rewrite a capsule keeping only high-value frames.

    Strategy: read all episodes, drop those with score < --min-score and
    duplicate (query,answer) pairs (keep newest), then write a NEW
    capsule. The original is preserved as .bak.
    """
    p = _resolve_capsule(args)
    mem = _open_backend(p)
    eps = _episodes(mem._backend)
    # dedup by (query, answer[:80]) keeping newest
    seen: Dict[str, Episode] = {}
    for e in eps:
        if args.min_score is not None and e.score < args.min_score:
            continue
        key = (e.query or "").strip().lower() + "\x00" + (e.answer or "")[:80].lower()
        prev = seen.get(key)
        if prev is None or e.created_at > prev.created_at:
            seen[key] = e
    keep = list(seen.values())
    kept_n = len(keep)
    dropped = len(eps) - kept_n
    print(f"compact: {len(eps)} -> {kept_n} (drop {dropped})")
    if args.dry_run:
        print("--dry-run: not writing")
        mem.close()
        return
    # backup
    bak = p.with_suffix(p.suffix + ".bak")
    shutil.copy2(p, bak)
    print(f"backup -> {bak}")
    # write fresh capsule
    out = Path(args.out) if args.out else p
    if out.exists() and out != p:
        out.unlink()
    cfg = MemvidConfig.from_env()
    cfg.dir = out.parent
    cfg.tenant = out.stem.replace("memory_", "")
    cfg.enabled = True
    fresh = MemvidMemory(cfg)
    for e in keep:
        e.frame_id = None
        e.score = 0.0
        fresh.record(e)
    fresh.close()
    mem.close()
    print(f"compacted capsule -> {out}")


def cmd_branch(args):
    """Create a branch capsule containing only frames up to --at-frame or
    --before. Time-travel snapshot."""
    p = _resolve_capsule(args)
    mem = _open_backend(p)
    eps = _episodes(mem._backend)
    if args.at_frame is not None:
        eps = [e for e in eps
               if (e.frame_id is None) or str(e.frame_id) <= str(args.at_frame)]
    if args.before:
        eps = [e for e in eps if e.created_at < args.before]
    out = Path(args.to)
    if out.exists():
        if not args.force:
            sys.exit(f"output exists: {out} (use --force to overwrite)")
        out.unlink()
    cfg = MemvidConfig.from_env()
    cfg.dir = out.parent
    cfg.tenant = out.stem.replace("memory_", "")
    cfg.enabled = True
    fresh = MemvidMemory(cfg)
    for e in eps:
        e.frame_id = None
        e.score = 0.0
        fresh.record(e)
    fresh.close()
    mem.close()
    print(f"branched {len(eps)} episodes -> {out}")


def cmd_rewind(args):
    """Restore capsule from latest .bak (or a specific one)."""
    p = Path(args.capsule) if args.capsule else \
        MemvidConfig.from_env().capsule_path
    if args.backup:
        bak = Path(args.backup)
    else:
        bak = p.with_suffix(p.suffix + ".bak")
    if not bak.exists():
        sys.exit(f"backup not found: {bak}")
    if p.exists() and not args.force:
        # park current as .bak.2
        park = p.with_suffix(p.suffix + ".bak.2")
        shutil.copy2(p, park)
        print(f"current capsule parked -> {park}")
    shutil.copy2(bak, p)
    print(f"rewound: {bak} -> {p}")


def cmd_export(args):
    p = _resolve_capsule(args)
    mem = _open_backend(p)
    eps = _episodes(mem._backend)
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        for e in eps:
            fh.write(json.dumps({
                "episode_id": e.episode_id,
                "created_at": e.created_at,
                "domain": e.domain,
                "tenant": e.tenant,
                "query": e.query,
                "answer": e.answer,
                "sources": e.sources,
                "trace": e.trace,
                "feedback": e.feedback,
                "score": e.score,
                "frame_id": e.frame_id,
            }, ensure_ascii=False) + "\n")
    mem.close()
    print(f"exported {len(eps)} episodes -> {out}")


def cmd_purge(args):
    """Delete frames matching filter. ALWAYS backs up first."""
    p = _resolve_capsule(args)
    if not args.yes:
        sys.exit("purge requires --yes (and is irreversible after .bak rotation)")
    mem = _open_backend(p)
    eps = _episodes(mem._backend)
    keep = []
    for e in eps:
        if args.before and e.created_at >= args.before:
            keep.append(e); continue
        if args.domain and e.domain == args.domain:
            continue  # purge this
        if args.min_score is not None and e.score < args.min_score:
            continue  # purge this
        keep.append(e)
    bak = p.with_suffix(p.suffix + ".bak")
    shutil.copy2(p, bak)
    print(f"backup -> {bak}")
    cfg = MemvidConfig.from_env()
    cfg.dir = p.parent
    cfg.tenant = p.stem.replace("memory_", "")
    cfg.enabled = True
    fresh = MemvidMemory(cfg)
    for e in keep:
        e.frame_id = None; e.score = 0.0
        fresh.record(e)
    fresh.close()
    mem.close()
    print(f"purged {len(eps) - len(keep)} frames; kept {len(keep)} -> {p}")


# ---------------------------------------------------------------------------
# arg parser
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        prog="hermes-memory",
        description="Manage Hermes memvid capsules")
    ap.add_argument("--capsule", type=Path, default=None,
                    help="override capsule path (default: env)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="capsule stats").set_defaults(fn=cmd_stats)

    p_ins = sub.add_parser("inspect", help="list recent episodes")
    p_ins.add_argument("--domain", default=None)
    p_ins.add_argument("--min-score", type=float, default=None)
    p_ins.add_argument("--since", default=None, help="ISO8601 lower bound")
    p_ins.add_argument("--limit", type=int, default=20)
    p_ins.set_defaults(fn=cmd_inspect)

    p_se = sub.add_parser("search", help="free-text search")
    p_se.add_argument("query")
    p_se.add_argument("--topk", type=int, default=5)
    p_se.add_argument("--when", default=None, help="temporal filter")
    p_se.set_defaults(fn=cmd_search)

    p_co = sub.add_parser("compact", help="dedup + drop low-value frames")
    p_co.add_argument("--min-score", type=float, default=0.5)
    p_co.add_argument("--out", default=None, help="output capsule path")
    p_co.add_argument("--dry-run", action="store_true")
    p_co.set_defaults(fn=cmd_compact)

    p_br = sub.add_parser("branch", help="time-travel branch")
    p_br.add_argument("--at-frame", default=None)
    p_br.add_argument("--before", default=None, help="ISO8601 upper bound")
    p_br.add_argument("--to", required=True, help="output capsule")
    p_br.add_argument("--force", action="store_true")
    p_br.set_defaults(fn=cmd_branch)

    p_rw = sub.add_parser("rewind", help="restore from .bak")
    p_rw.add_argument("--backup", default=None)
    p_rw.add_argument("--force", action="store_true")
    p_rw.set_defaults(fn=cmd_rewind)

    p_ex = sub.add_parser("export", help="export episodes to JSONL")
    p_ex.add_argument("--out", required=True)
    p_ex.set_defaults(fn=cmd_export)

    p_pu = sub.add_parser("purge", help="delete frames (DANGEROUS)")
    p_pu.add_argument("--before", default=None)
    p_pu.add_argument("--domain", default=None)
    p_pu.add_argument("--min-score", type=float, default=None)
    p_pu.add_argument("--yes", action="store_true")
    p_pu.set_defaults(fn=cmd_purge)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
