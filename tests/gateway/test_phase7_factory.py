def test_factory_builds_local_snapshot_from_default_config():
    from rag_core.gateway.config_schema import GatewayConfig
    from rag_core.gateway.connector_factory import build_connectors
    from rag_core.gateway.connectors.snapshot import LocalSnapshotConnector

    connectors = build_connectors(GatewayConfig())

    assert isinstance(connectors["local_snapshot"], LocalSnapshotConnector)
