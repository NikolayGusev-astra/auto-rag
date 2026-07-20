import importlib.util
import json
import os
import subprocess
import sys

import pytest
import requests


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mcp") is None,
    reason="MCP SDK is unavailable; install auto-rag[gateway] to run MCP transport tests",
)


def _mcp_messages(*messages: dict) -> list[dict]:
    completed = subprocess.run(
        [sys.executable, "-m", "rag_core.gateway.server"],
        input="".join(json.dumps(message) + "\n" for message in messages),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line]


def _initialize_messages() -> tuple[dict, dict]:
    return (
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
    )


def _require_embedding_runtime() -> None:
    url = os.getenv(
        "EMBED_URL", os.getenv("RAG_EMBED_URL", "http://localhost:1234/v1/embeddings")
    )
    model = os.getenv("EMBED_MODEL", os.getenv("RAG_EMBED_MODEL", "bge-m3"))
    try:
        response = requests.post(url, json={"model": model, "input": ["smoke test"]}, timeout=1)
        response.raise_for_status()
    except requests.RequestException as error:
        pytest.skip(f"LM Studio embeddings are unavailable: {error}")


def test_mcp_server_imports_when_sdk_is_available():
    import rag_core.gateway.server as server

    assert server.create_mcp_server().name == "auto-rag-gateway"


def test_mcp_initialize_advertises_server_capabilities():
    initialize, _ = _initialize_messages()

    response = _mcp_messages(initialize)[0]

    assert response["id"] == 1
    assert response["result"]["serverInfo"]["name"] == "auto-rag-gateway"
    assert "tools" in response["result"]["capabilities"]


def test_mcp_lists_search_and_sync_tools():
    initialize, initialized = _initialize_messages()

    responses = _mcp_messages(
        initialize,
        initialized,
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )

    tool_names = {tool["name"] for tool in responses[-1]["result"]["tools"]}
    assert {"search", "sync"} <= tool_names


def test_mcp_search_tool_returns_evidence_shape():
    _require_embedding_runtime()
    initialize, initialized = _initialize_messages()

    responses = _mcp_messages(
        initialize,
        initialized,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "deploy", "top_k": 2}},
        },
    )

    result = responses[-1]["result"]
    assert result["structuredContent"]["results"] == []
    assert "runtime" in result["structuredContent"]
