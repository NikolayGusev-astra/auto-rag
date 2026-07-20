def test_factory_builds_local_snapshot_from_default_config():
    from rag_core.gateway.config_schema import GatewayConfig
    from rag_core.gateway.connector_factory import build_connectors
    from rag_core.gateway.connectors.snapshot import LocalSnapshotConnector

    connectors = build_connectors(GatewayConfig())

    assert isinstance(connectors["local_snapshot"], LocalSnapshotConnector)


def test_factory_builds_enabled_live_source_without_network(monkeypatch):
    from rag_core.gateway.config_schema import GatewayConfig, SourceConfig
    from rag_core.gateway.connector_factory import build_connectors
    from rag_core.gateway.connectors.jira_connector import JiraConnector

    monkeypatch.setenv("JIRA_TOKEN", "from-env")
    monkeypatch.setenv("JIRA_BASE_URL", "https://jira.example.test")
    connectors = build_connectors(
        GatewayConfig(sources={"jira": SourceConfig(name="jira", kind="jira", credential_ref="env:JIRA_TOKEN")})
    )

    assert isinstance(connectors["jira"], JiraConnector)
    assert connectors["jira"]._token == "from-env"


def test_factory_builds_confluence_from_configured_base_url(monkeypatch):
    from rag_core.gateway.config_schema import GatewayConfig, SourceConfig
    from rag_core.gateway.connector_factory import build_connectors
    from rag_core.gateway.connectors.confluence_connector import ConfluenceConnector

    monkeypatch.setenv("CONFLUENCE_TOKEN", "from-env")
    connectors = build_connectors(
        GatewayConfig(
            sources={
                "docs": SourceConfig(
                    name="docs",
                    kind="confluence",
                    credential_ref="env:CONFLUENCE_TOKEN",
                    extra={"base_url": "https://wiki.example.test"},
                )
            }
        )
    )

    assert isinstance(connectors["docs"], ConfluenceConnector)
    assert connectors["docs"].source == "docs"


def test_default_config_registers_only_mandatory_local_snapshot():
    from rag_core.gateway.config_schema import GatewayConfig
    from rag_core.gateway.connector_factory import build_connectors

    connectors = build_connectors(GatewayConfig())

    assert list(connectors) == ["local_snapshot"]
    assert connectors["local_snapshot"].retrieval_kind == "local"
