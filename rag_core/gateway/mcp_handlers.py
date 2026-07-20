from __future__ import annotations

import time
from dataclasses import asdict
from typing import Mapping

from rag_core.gateway.adaptive.enrichment import MemvidEnricher
from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.coordinator import RetrievalCoordinator


async def handle_search(
    request: SearchRequest,
    connectors: Mapping[str, SourceConnector],
    *,
    enricher: MemvidEnricher | None = None,
    active_revision_path: str | None = None,
    embedding_profile_id: str | None = None,
) -> dict[str, object]:
    active_connectors = {
        name: connector
        for name, connector in connectors.items()
        if request.include_web
        or getattr(connector, "source", "").lower() not in {"web", "public_web"}
    }
    coordinator = RetrievalCoordinator(active_connectors)
    t0 = time.perf_counter()
    results = await coordinator.search(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    # Persist memory episode from search results
    if enricher is not None and results:
        episode = enricher.build_episode(
            request.query,
            results,
            successful=True,
            index_revision=active_revision_path,
            embedding_profile_id=embedding_profile_id,
        )
        enricher.persist_episode(episode)

    return {
        "results": [asdict(result) for result in results],
        "trace": {
            "query": request.query,
            "connector_count": len(active_connectors),
            "result_count": len(results),
            "elapsed_ms": elapsed_ms,
        },
        "runtime": {
            "source_status": {
                connector.source: "enabled" for connector in active_connectors.values()
            },
            "episode_persisted": enricher is not None and len(results) > 0,
        },
    }
