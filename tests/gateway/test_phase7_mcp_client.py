import json
import sys

import pytest

from rag_core.gateway.connectors.snapshot import LocalSnapshotConnector
from rag_core.gateway.models import Document, SyncBatch
from rag_core.gateway.sync.engine import SyncEngine


class _PublishedSnapshotSource:
    source = "local_snapshot"

    def __init__(self, batch: SyncBatch) -> None:
        self._batch = batch

    async def sync_changes(self, cursor: str | None) -> SyncBatch:
        return self._batch


@pytest.mark.asyncio
async def test_official_mcp_client_session_searches_published_local_snapshot(tmp_path):
    """Exercise the SDK client through stdio and a real gateway TOML file."""
    mcp = pytest.importorskip("mcp", reason="official MCP SDK is not installed")
    try:
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError:
        pytest.skip("installed MCP SDK does not provide the stdio client API")

    engine = SyncEngine(tmp_path)
    source = _PublishedSnapshotSource(
        SyncBatch(
            added=[
                Document(
                    id="jira:1",
                    source="jira",
                    source_instance="project",
                    title="Deployment",
                    text="deploy to kubernetes",
                    content_hash="jira:1",
                )
            ]
        )
    )
    await engine.sync_source(source)
    assert (await LocalSnapshotConnector(engine, "local_snapshot").health())["available"] is True

    cfg_path = tmp_path / "gateway.toml"
    cfg_path.write_text(
        "\n".join(
            (
                f"knowledge_root = {json.dumps(tmp_path.resolve().as_posix())}",
                "local_snapshot = true",
                "web = false",
                "",
            )
        ),
        encoding="utf-8",
    )

    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "rag_core.gateway.server", "--config", str(cfg_path)],
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with mcp.ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} >= {"search", "sync"}

            response = await session.call_tool("search", {"query": "deploy", "top_k": 3})

    results = response.structuredContent["results"]
    assert results
    assert any(
        result["document_id"] == "jira:1" or "deploy to kubernetes" in result["text"]
        for result in results
    )
