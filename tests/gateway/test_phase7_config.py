from pathlib import Path

import pytest


def test_gateway_config_defaults_enable_local_snapshot():
    from rag_core.gateway.config_schema import GatewayConfig

    config = GatewayConfig()

    assert config.version == 1
    assert config.knowledge_root == Path.home() / ".local" / "share" / "auto-rag"
    assert config.local_snapshot is True
    assert config.web is False
    assert config.adaptive is False
    assert config.sources == {}


def test_gateway_config_rejects_unsupported_version():
    from rag_core.gateway.config_schema import GatewayConfig, UnsupportedConfigVersion

    with pytest.raises(UnsupportedConfigVersion):
        GatewayConfig(version=99)


def test_load_config_reads_toml(tmp_path):
    from rag_core.gateway.config_loader import load_config

    path = tmp_path / "gateway.toml"
    path.write_text(
        'version = 1\nknowledge_root = "./knowledge"\nlocal_snapshot = true\nweb = false\n'
        '\n[sources.jira]\nkind = "jira"\nenabled = true\ncredential_ref = "env:JIRA_TOKEN"\n',
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.knowledge_root == (path.parent / "knowledge").resolve()
    assert config.local_snapshot is True
    assert config.sources["jira"].credential_ref == "env:JIRA_TOKEN"


def test_load_config_reads_exact_match_boosts(tmp_path):
    from rag_core.gateway.config_loader import load_config

    path = tmp_path / "gateway.toml"
    path.write_text(
        "[retrieval]\nexact_id_boost = 1.0\nexact_slug_title_boost = 0.7\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.exact_id_boost == 1.0
    assert config.exact_slug_title_boost == 0.7


def test_load_config_rejects_missing_explicit_file(tmp_path):
    from rag_core.gateway.config_loader import ConfigNotFound, load_config

    with pytest.raises(ConfigNotFound):
        load_config(tmp_path / "missing.toml")
