from __future__ import annotations

from rag_core.gateway.adaptive.contracts import QueryPlan


class DcdPlanner:
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
        include_memory = ("memory" in sources) or availability.get("memory", False) is not False
        return QueryPlan(
            original_query=query,
            queries=queries,
            domains=("astra",),
            sources=sources,
            include_local=include_local,
            include_live=include_live,
            include_web=include_web,
            max_results=5,
            include_memory=include_memory,
            hints=hints,
        )
