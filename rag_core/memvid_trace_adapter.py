"""memvid_trace_adapter.py — MemvidTracedAdapter wrapping MemvidTraced.

Accepts RagTrace objects (from rag_trace.py) OR plain dicts as the ``trace``
argument to ``recall`` / ``record``.

Exposes the same public API as MemvidTraced:
    recall(query, *, domain, top_k, when, trace)
    record(episode, *, trace)
    recall_as_context(query, *, domain, top_k, max_chars, trace)
    active
    recall_threshold
    close
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from memvid_memory import Episode
from memvid_trace import MemvidTraced

log = logging.getLogger("hermes.memvid.trace.adapter")


def _is_ragtrace(trace: Any) -> bool:
    """Return True if *trace* is a RagTrace object (not a plain dict)."""
    return hasattr(trace, "event") and callable(trace.event) and hasattr(trace, "stages")


class MemvidTracedAdapter:
    """Wrapper around MemvidTraced that adapts the ``trace`` argument.

    * If ``trace`` is a ``RagTrace`` object — measure elapsed time in the
      adapter and push an event via ``trace.event(name, duration_ms=..., ...)``.
    * If ``trace`` is a plain ``dict`` — delegate to the inner
      ``MemvidTraced`` which uses the existing ``_push_stage`` / latency_ms
      path.
    * If ``trace`` is ``None`` — skip tracing entirely (pass through to the
      inner with ``trace=None``, no event recorded).
    """

    def __init__(self, inner: MemvidTraced):
        self._inner = inner

    # -- pass-through properties --------------------------------------------
    @property
    def active(self) -> bool:
        return self._inner.active

    @property
    def recall_threshold(self) -> float:
        return self._inner.recall_threshold

    def close(self):
        return self._inner.close()

    # -- traced recall ------------------------------------------------------
    def recall(self, query: str, *, domain: Optional[str] = None,
               top_k: Optional[int] = None, when: Optional[str] = None,
               trace: Any = None) -> List[Episode]:
        if _is_ragtrace(trace):
            t0 = time.perf_counter()
            err: Optional[str] = None
            eps: List[Episode] = []
            try:
                eps = self._inner._inner.recall(  # bypass timing wrapper
                    query, domain=domain, top_k=top_k, when=when)
            except Exception as e:
                err = repr(e)
                log.warning("adapter recall error: %s", e)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            top_score = eps[0].score if eps else 0.0
            trace.event("memvid.recall",
                        duration_ms=round(dt_ms, 2),
                        hits=len(eps),
                        top_score=round(top_score, 4),
                        above_threshold=bool(
                            eps and top_score >= self._inner.recall_threshold),
                        error=err)
            return eps
        # dict or None path — delegate to inner MemvidTraced
        return self._inner.recall(query, domain=domain, top_k=top_k,
                                  when=when, trace=trace)

    # -- traced record ------------------------------------------------------
    def record(self, episode: Episode, *,
               trace: Any = None) -> bool:
        if _is_ragtrace(trace):
            t0 = time.perf_counter()
            ok = False
            err: Optional[str] = None
            try:
                ok = self._inner._inner.record(episode)  # bypass timing wrapper
            except Exception as e:
                err = repr(e)
                log.warning("adapter record error: %s", e)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            trace.event("memvid.record",
                        duration_ms=round(dt_ms, 2),
                        episode_id=episode.episode_id,
                        domain=episode.domain,
                        commit=ok,
                        error=err)
            return ok
        # dict or None path — delegate to inner MemvidTraced
        return self._inner.record(episode, trace=trace)

    # -- traced context helper ----------------------------------------------
    def recall_as_context(self, query: str, *, domain: Optional[str] = None,
                          top_k: Optional[int] = None, max_chars: int = 1200,
                          trace: Any = None) -> str:
        """Recall prior episodes and format them as a prompt prefix.

        Internally calls ``self.recall`` so the trace-adapter logic applies.
        """
        eps = self.recall(query, domain=domain, top_k=top_k, trace=trace)
        if not eps:
            return ""
        lines = ["[PRIOR EPISODES — what Hermes already answered before]"]
        total = 0
        used = 0
        for i, ep in enumerate(eps, 1):
            if ep.score < self._inner.recall_threshold:
                continue
            block = (f"#{i} (score={ep.score:.2f}, {ep.created_at[:10]}, "
                     f"domain={ep.domain or '-'})\n"
                     f"Q: {ep.query}\nA: {ep.answer}")
            if total + len(block) > max_chars:
                break
            lines.append(block)
            total += len(block)
            used += 1
        # annotate the previously-pushed recall stage with usage info
        if _is_ragtrace(trace):
            # push an additional annotation via a second event point
            if trace.stages:
                last = trace.stages[-1]
                last["used_in_prompt"] = used
        elif trace and isinstance(trace, dict) and trace.get("stages"):
            last = trace["stages"][-1]
            last["used_in_prompt"] = used
        return "\n\n".join(lines) + "\n\n" if used else ""