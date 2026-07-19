from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

from rag_core.gateway.connector import SearchRequest, SourceConnector
from rag_core.gateway.coordinator import RetrievalCoordinator


async def handle_search(
    request: SearchRequest, connectors: Mapping[str, SourceConnector]
) -> dict[str, object]:
    results = await RetrievalCoordinator(connectors).search(request)
    return {
        "results": [asdict(result) for result in results],
        "trace": {
            "query": request.query,
            "connector_count": len(connectors),
            "result_count": len(results),
        },
    }
