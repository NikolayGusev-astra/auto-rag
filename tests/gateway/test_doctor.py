import json
import subprocess
import sys

from rag_core.gateway.models import Document, SyncBatch
from rag_core.gateway.sync.engine import SyncEngine


def _publish_snapshot(root):
    engine = SyncEngine(root)
    revision = engine.stage_sync(
        "local_snapshot",
        SyncBatch(added=[Document("doc-1", "local_snapshot", "default", "Document", "text")]),
    )
    engine.publish("local_snapshot", revision)


def _run_doctor(config, *extra):
    return subprocess.run(
        [sys.executable, "-m", "rag_core.cli", "doctor", "--config", str(config), *extra],
        capture_output=True,
        text=True,
        timeout=5,
    )


def _config(path, root, sources=""):
    path.write_text(
        f'knowledge_root = "{root.as_posix()}"\nlocal_snapshot = true\n{sources}',
        encoding="utf-8",
    )


def test_doctor_minimal_offline(tmp_path):
    root = tmp_path / "knowledge"
    _publish_snapshot(root)
    config = tmp_path / "gateway.toml"
    _config(config, root)

    completed = _run_doctor(config)

    assert completed.returncode == 0
    assert "LOCAL SNAPSHOT OK" in completed.stdout


def test_doctor_config_error(tmp_path):
    completed = _run_doctor(tmp_path / "missing.toml")

    assert completed.returncode == 1
    assert "CONFIG ERROR" in completed.stdout


def test_doctor_snapshot_missing(tmp_path):
    config = tmp_path / "gateway.toml"
    _config(config, tmp_path / "knowledge")

    completed = _run_doctor(config)

    assert completed.returncode == 2
    assert "LOCAL SNAPSHOT UNAVAILABLE" in completed.stdout


def test_doctor_jira_unavailable(tmp_path):
    root = tmp_path / "knowledge"
    _publish_snapshot(root)
    config = tmp_path / "gateway.toml"
    _config(config, root, '\n[sources.jira]\nkind = "jira"\ncredential_ref = "env:MISSING_JIRA_TOKEN"\n')

    completed = _run_doctor(config)

    assert completed.returncode == 3
    assert "JIRA UNAVAILABLE" in completed.stdout


def test_doctor_json_output(tmp_path):
    root = tmp_path / "knowledge"
    _publish_snapshot(root)
    config = tmp_path / "gateway.toml"
    _config(config, root)

    completed = _run_doctor(config, "--json")

    assert completed.returncode == 0
    assert json.loads(completed.stdout)["exit_code"] == 0


def test_doctor_readonly(tmp_path):
    root = tmp_path / "knowledge"
    _publish_snapshot(root)
    config = tmp_path / "gateway.toml"
    _config(config, root)
    before = {path.relative_to(tmp_path): path.stat().st_mtime_ns for path in tmp_path.rglob("*")}

    completed = _run_doctor(config)
    after = {path.relative_to(tmp_path): path.stat().st_mtime_ns for path in tmp_path.rglob("*")}

    assert completed.returncode == 0
    assert after == before
