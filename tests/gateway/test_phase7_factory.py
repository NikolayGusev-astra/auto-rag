def test_factory_builds_local_snapshot_from_default_config():
    from rag_core.gateway.config_schema import GatewayConfig
    from rag_core.gateway.connector_factory import build_connectors
    from rag_core.gateway.connectors.snapshot import LocalSnapshotConnector

    connectors = build_connectors(GatewayConfig())

    assert isinstance(connectors["local_snapshot"], LocalSnapshotConnector)


def test_factory_builds_enabled_live_source_without_network(monkeypatch):
    from rag_core.gateway.config_schema import GatewayConfig, SourceConfig
    from rag_core.gateway.connector_factory import ConnectorStub, build_connectors

    monkeypatch.setenv("JIRA_TOKEN", "from-env")
    connectors = build_connectors(
        GatewayConfig(sources={"jira": SourceConfig(name="jira", kind="jira", credential_ref="env:JIRA_TOKEN")})
    )

    assert isinstance(connectors["jira"], ConnectorStub)
    assert connectors["jira"].credential == "from-env"
