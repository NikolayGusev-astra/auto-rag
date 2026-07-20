from rag_core.gateway.config_schema import GatewayConfig
from rag_core.gateway.connector_factory import build_connectors
from rag_core.gateway.connectors.snapshot import LocalSnapshotConnector
from rag_core.gateway.diagnostics import collect_startup_diagnostics
from rag_core.gateway.models import Document, SyncBatch
from rag_core.gateway.sync.engine import SyncEngine, _await_synchronously


def _publish_snapshot(root):
    engine = SyncEngine(root)
    revision = engine.stage_sync(
        "local_snapshot",
        SyncBatch(
            added=[Document("doc-1", "local_snapshot", "default", "Document", "snapshot text")]
        ),
    )
    engine.publish("local_snapshot", revision)
    return engine


def test_startup_diagnostics_marks_empty_local_snapshot_unavailable(tmp_path):
    connectors = build_connectors(GatewayConfig(knowledge_root=tmp_path))

    diagnostics = collect_startup_diagnostics(connectors)

    assert diagnostics["connectors"]["local_snapshot"]["health"] is False


def test_published_local_snapshot_reports_healthy(tmp_path):
    engine = _publish_snapshot(tmp_path)
    connector = LocalSnapshotConnector(engine, "local_snapshot")

    assert _await_synchronously(connector.health())["available"] is True
    diagnostics = collect_startup_diagnostics({"local_snapshot": connector})

    assert diagnostics["connectors"]["local_snapshot"]["health"] is True


def test_corrupt_local_snapshot_manifest_is_unhealthy_and_non_fatal(tmp_path):
    manifest = tmp_path / "local_snapshot" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{garbage", encoding="utf-8")
    connectors = build_connectors(GatewayConfig(knowledge_root=tmp_path))

    diagnostics = collect_startup_diagnostics(connectors)

    assert diagnostics["connectors"]["local_snapshot"]["health"] is False


def test_load_config_resolves_relative_knowledge_root_from_config_directory(tmp_path):
    from rag_core.gateway.config_loader import load_config

    config_path = tmp_path / "config" / "gateway.toml"
    config_path.parent.mkdir()
    config_path.write_text("knowledge_root = 'data'\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.knowledge_root == (config_path.parent / "data").resolve()
