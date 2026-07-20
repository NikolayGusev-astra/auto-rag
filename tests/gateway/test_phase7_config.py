from pathlib import Path


def test_gateway_config_defaults_enable_local_snapshot():
    from rag_core.gateway.config_schema import GatewayConfig

    config = GatewayConfig()

    assert config.version == 1
    assert config.knowledge_root == Path.home() / ".local" / "share" / "auto-rag"
    assert config.local_snapshot is True
    assert config.web is False
    assert config.adaptive is False
    assert config.sources == {}
