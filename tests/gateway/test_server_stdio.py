import pytest


@pytest.mark.asyncio
async def test_dispatch_search_method_returns_json_shape():
    from rag_core.gateway.server import dispatch

    response = await dispatch(
        {"method": "search", "params": {"query": "cluster", "topk": 3}, "connectors": {}}
    )

    assert response["results"] == []
    assert "runtime" in response
