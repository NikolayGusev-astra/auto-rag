def test_startup_diagnostics_keeps_local_healthy_when_live_source_is_offline():
    from rag_core.gateway.config_schema import GatewayConfig, SourceConfig
    from rag_core.gateway.connector_factory import build_connectors
    from rag_core.gateway.diagnostics import collect_startup_diagnostics

    connectors = build_connectors(
        GatewayConfig(sources={"jira": SourceConfig(name="jira", kind="jira", credential_ref="env:MISSING_TOKEN")})
    )

    diagnostics = collect_startup_diagnostics(connectors)

    assert diagnostics["connectors"]["local_snapshot"]["health"] is True
    assert diagnostics["connectors"]["jira"]["health"] is False
    assert diagnostics["offline"]["healthy"] == ["local_snapshot"]
