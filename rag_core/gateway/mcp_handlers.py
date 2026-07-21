from __future__ import annotations

import time
from dataclasses import asdict
from typing import Mapping

from rag_core.gateway.adaptive.enrichment import MemvidEnricher
from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.connectors.memvid_connector import MemvidConnector
from rag_core.gateway.coordinator import RetrievalCoordinator


async def handle_search(
    request: SearchRequest,
    connectors: Mapping[str, SourceConnector],
    *,
    enricher: MemvidEnricher | None = None,
    reranker: object | None = None,
    active_revision_path: str | None = None,
    embedding_profile_id: str | None = None,
) -> dict[str, object]:
    active_connectors = {
        name: connector
        for name, connector in connectors.items()
        if request.include_web
        or getattr(connector, "source", "").lower() not in {"web", "public_web"}
    }
    if enricher is not None and not any(
        getattr(connector, "source", "").lower() == "memvid"
        for connector in active_connectors.values()
    ):
        active_connectors = {"memvid": MemvidConnector(enricher), **active_connectors}
    retrieval_config = getattr(connectors, "retrieval_config", {})
    coordinator = RetrievalCoordinator(
        active_connectors,
        reranker=reranker,
        exact_id_boost=retrieval_config.get("exact_id_boost", 1.0),
        exact_slug_title_boost=retrieval_config.get("exact_slug_title_boost", 0.7),
    )
    t0 = time.perf_counter()
    results = await coordinator.search(request)
    elapsed_ms = coordinator.last_latency["search"]["duration_ms"]

    # Persist memory episode from search results
    if enricher is not None and results and any(result.source != "memvid" for result in results):
        episode = enricher.build_episode(
            request.query,
            results,
            successful=True,
            index_revision=active_revision_path,
            embedding_profile_id=embedding_profile_id,
        )
        enricher.persist_episode(episode)

        # Auto-learn DCD routing from new episode
        try:
            from pathlib import Path as _Path
            from rag_core.gateway.adaptive.dcd_learner import DcdLearner
            routing_path = _Path.home() / ".config" / "auto-rag" / "routing.json"
            learner = DcdLearner(enricher.path, routing_path)
            learner.learn()
        except Exception:
            pass  # DCD learn is best-effort, never blocks retrieval

    # Usage logging (ADR-006 Step 5)
    _log_usage(request, results, elapsed_ms, enricher is not None)

    return {
        "results": [asdict(result) for result in results],
        "trace": {
            "query": request.query,
            "connector_count": len(active_connectors),
            "result_count": len(results),
            "elapsed_ms": elapsed_ms,
            "latency": coordinator.last_latency,
            "reranker": {
                "enabled": reranker is not None,
                "provider": type(reranker).__name__ if reranker is not None else None,
            },
            "deduplication": coordinator.last_deduplication,
        },
        "runtime": {
            "source_status": {
                connector.source: "enabled" for connector in active_connectors.values()
            },
            "episode_persisted": (
                enricher is not None
                and len(results) > 0
                and any(result.source != "memvid" for result in results)
            ),
            "usage_logged": False,
        },
    }


def _log_usage(request, results, elapsed_ms, enricher_enabled):
    try:
        from rag_core.gateway.usage_log import log_usage
        log_usage(request.query, results, elapsed_ms)
    except Exception:
        pass
