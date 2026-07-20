from __future__ import annotations

import json
from pathlib import Path

from rag_core.gateway.adaptive.contracts import QueryPlan


STRONG_AFFINITY_THRESHOLD = 0.5


class DcdPlanner:
    def __init__(self, routing_path: Path | None = None) -> None:
        self._routing_table = _load_routing_table(routing_path or _routing_path())

    def plan(
        self, query: str, availability: dict[str, bool], hints: dict
    ) -> QueryPlan:
        include_local = availability.get("local", True)
        include_live = availability.get("live", True)
        include_web = availability.get("web", False)
        parts = [part.strip() for part in query.replace(" and ", " и ").split(" и ") if part.strip()]
        queries = tuple(parts) if len(parts) > 1 else (query,)
        sources = tuple(
            source
            for source, available in (("local", include_local), ("live", include_live), ("web", include_web))
            if available
        )
        sources = _boost_affine_sources(query, sources, self._routing_table)
        include_memory = ("memory" in sources) or availability.get("memory", False) is not False
        route = _route_for_query(query, self._routing_table)
        return QueryPlan(
            original_query=query,
            queries=queries,
            domains=(str(route["space"]),) if route else ("astra",),
            sources=sources,
            include_local=include_local,
            include_live=include_live,
            include_web=include_web,
            max_results=5,
            include_memory=include_memory,
            include_docs=bool(route and route.get("doc_root")),
            hints=hints,
        )


def _routing_path() -> Path:
    return Path.home() / ".config" / "auto-rag" / "routing.json"


def _load_routing_table(path: Path) -> dict[str, dict[str, object]]:
    try:
        contents = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return contents if isinstance(contents, dict) else {}


def _route_for_query(query: str, routing_table: dict[str, dict[str, object]]) -> dict[str, object] | None:
    query_lower = query.casefold()
    for slug, route in routing_table.items():
        name = str(route.get("name") or slug).casefold()
        if name in query_lower or slug.replace("-", " ") in query_lower:
            return route
    return None


def _boost_affine_sources(
    query: str, sources: tuple[str, ...], routing_table: dict[str, dict[str, object]]
) -> tuple[str, ...]:
    affinities = routing_table.get("source_affinity")
    if not isinstance(affinities, dict):
        return sources
    scores: dict[str, float] = {}
    for keyword in _tokenize(query):
        source_scores = affinities.get(keyword)
        if not isinstance(source_scores, dict):
            continue
        for source in sources:
            score = source_scores.get(source)
            if isinstance(score, int | float) and score > STRONG_AFFINITY_THRESHOLD:
                scores[source] = max(scores.get(source, float("-inf")), float(score))
    return tuple(sorted(sources, key=lambda source: scores.get(source, 0.0), reverse=True))


def _tokenize(query: str) -> set[str]:
    import re

    return set(re.findall(r"\w+", query.casefold(), flags=re.UNICODE))
