from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.coordinator import RetrievalCoordinator


async def handle_search(
    request: SearchRequest, connectors: Mapping[str, SourceConnector]
) -> dict[str, object]:
    active_connectors = {
        name: connector
        for name, connector in connectors.items()
        if request.include_web
        or getattr(connector, "source", "").lower() not in {"web", "public_web"}
    }
    results = await RetrievalCoordinator(active_connectors).search(request)
    return {
        "results": [asdict(result) for result in results],
        "trace": {
            "query": request.query,
            "connector_count": len(active_connectors),
            "result_count": len(results),
        },
        "runtime": {
            "source_status": {
                connector.source: "enabled" for connector in active_connectors.values()
            }
        },
    }
