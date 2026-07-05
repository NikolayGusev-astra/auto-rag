import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from rag_federated import FederatedRAGClient, _ServerHealth


class TestCircuitBreaker:
    @pytest.fixture
    def client(self):
        c = FederatedRAGClient({})
        c._health = {"bad": _ServerHealth()}
        return c

    async def test_circuit_opens_after_3_failures(self, client):
        async def failing_query(*a, **kw):
            raise RuntimeError("simulated")

        client._do_query = failing_query
        client.configs = {"bad": MagicMock()}

        for i in range(3):
            await client.query("bad", "test", 3)

        health = client._health["bad"]
        assert health.consecutive_failures == 3
        assert not health.is_healthy, "Circuit should be open after 3 failures"

    async def test_circuit_skips_unhealthy_server(self, client):
        health = client._health["bad"]
        health.cooldown_until = float('inf')

        result = await client.query("bad", "test", 3)
        assert "cooldown" in result[0]["text"].lower()
