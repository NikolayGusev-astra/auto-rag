import json
from unittest.mock import AsyncMock, patch

import pytest

from rag_core.gateway.adaptive.planner import DcdPlanner
from rag_core.gateway.adaptive.source_discovery import SourceDiscovery
from rag_core.gateway.config_schema import GatewayConfig, SourceConfig


@pytest.mark.asyncio
async def test_discover_parses_product_pages_and_verifies_doc_tree(tmp_path, monkeypatch):
    product_page = tmp_path / "rusbitech" / "products" / "astra-linux.md"
    product_page.parent.mkdir(parents=True)
    product_page.write_text(
        "---\nproduct: Astra Linux\nversion: 1.8\nspace: AL\ndoc_root: 123456\n---\n",
        encoding="utf-8",
    )
    routing_path = tmp_path / "routing.json"
    monkeypatch.setattr(
        "rag_core.gateway.adaptive.source_discovery._routing_path", lambda: routing_path
    )
    config = GatewayConfig(
        sources={
            "docs": SourceConfig(
                name="docs",
                kind="confluence",
                credential_ref="env:CONFLUENCE_TOKEN",
                extra={"base_url": "https://wiki.example.test"},
            )
        }
    )

    with patch(
        "rag_core.gateway.adaptive.source_discovery.resolve_credential", return_value="secret"
    ), patch(
        "rag_core.gateway.adaptive.source_discovery.ConfluenceConnector.child_pages",
        new=AsyncMock(return_value=[{"id": "10", "title": "PMI"}, {"id": "11", "title": "RA"}]),
    ):
        routing = await SourceDiscovery(tmp_path, config).discover()

    assert routing == {
        "astra-linux": {
            "name": "Astra Linux",
            "version": "1.8",
            "space": "AL",
            "doc_root": "123456",
            "pmi_page": "10",
            "ra_page": "11",
        }
    }
    assert json.loads(routing_path.read_text(encoding="utf-8")) == routing


def test_planner_loads_saved_routing_and_enables_docs(tmp_path):
    routing_path = tmp_path / "routing.json"
    routing_path.write_text(
        json.dumps({"astra-linux": {"name": "Astra Linux", "space": "AL", "doc_root": "123456"}}),
        encoding="utf-8",
    )

    plan = DcdPlanner(routing_path=routing_path).plan(
        "How do I update Astra Linux?", {"local": True, "live": True}, {}
    )

    assert plan.domains == ("AL",)
    assert plan.include_docs is True
