import pytest

from rag_core import rag_federated


@pytest.mark.asyncio
async def test_federation_is_experimental_and_warns_when_used(monkeypatch):
    class EmptyClient:
        configs = {}

    async def get_empty_client():
        return EmptyClient()

    monkeypatch.setattr(rag_federated, "get_federated_client", get_empty_client)

    assert rag_federated.FEDERATION_EXPERIMENTAL is True
    with pytest.warns(RuntimeWarning, match="experimental"):
        assert await rag_federated.query_federated_servers("q") == {}
