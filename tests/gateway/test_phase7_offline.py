import pytest

from rag_core.gateway.config_schema import GatewayConfig, SourceConfig
from rag_core.gateway.connector import SearchRequest
from rag_core.gateway.connector_factory import build_connectors
from rag_core.gateway.mcp_handlers import handle_search
from rag_core.gateway.models import Document, SyncBatch


@pytest.mark.asyncio
async def test_offline_startup_keeps_local_snapshot_search_available(tmp_path):
    connectors = build_connectors(
        GatewayConfig(
            knowledge_root=tmp_path,
            sources={"jira": SourceConfig(name="jira", kind="jira", credential_ref="env:MISSING_JIRA_TOKEN")},
        )
    )
    local = connectors["local_snapshot"]
    revision = local._engine.stage_sync(
        "local_snapshot",
        SyncBatch(added=[Document("local:1", "local", "workstation", "Offline guide", "offline bootstrap works")]),
    )
    local._engine.publish("local_snapshot", revision)

    response = await handle_search(SearchRequest(query="bootstrap"), connectors)

    assert response["results"]
    assert response["results"][0]["source"] == "local_snapshot"
