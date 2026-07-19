import pytest
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, MagicMock
from rag_core import federated_endpoint as endpoint
from rag_core import rag_federated as federation
from rag_core.rag_federated import FederatedRAGClient, _ServerHealth


class TestCircuitBreaker:
    @pytest.fixture
    def client(self):
        c = FederatedRAGClient({})
        c._health = {"bad": _ServerHealth()}
        c.configs = {"bad": MagicMock()}
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


@pytest.mark.asyncio
async def test_federation_rejects_requests_beyond_hop_limit():
    """S9: a ping-pong request cannot exceed three federation hops."""
    with pytest.raises(endpoint.HTTPException) as exc:
        await endpoint.search(
            endpoint.SearchRequest(query="test"), x_federation_hop="4"
        )

    assert exc.value.status_code == 400
    assert "hop limit" in exc.value.detail.lower()


def test_query_helper_uses_a_fresh_session_for_each_asyncio_run(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"chunks": [{"text": "result"}]}).encode())

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    previous_client = federation._CLIENT
    federation._CLIENT = FederatedRAGClient([
        federation.FederatedServerConfig(
            name="test",
            host="127.0.0.1",
            remote_port=server.server_port,
            use_ssh=False,
            use_tls=False,
        )
    ])
    try:
        first = asyncio.run(federation.query_federated_servers("first"))
        second = asyncio.run(federation.query_federated_servers("second"))
    finally:
        federation._CLIENT = previous_client
        server.shutdown()
        thread.join()

    assert first["test"][0]["text"] == "result"
    assert second["test"][0]["text"] == "result"


@pytest.mark.asyncio
async def test_query_helper_returns_and_logs_federation_error(caplog, monkeypatch):
    client = MagicMock(configs={"test": MagicMock()})
    client.query_all = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(federation, "get_federated_client", AsyncMock(return_value=client))

    result = await federation.query_federated_servers("test")

    assert result["federation"][0]["is_error"] is True
    assert "boom" in result["federation"][0]["text"]
    assert "Federated query failed" in caplog.text