import os

from rag_core.gateway.models import Document, SyncBatch
from rag_core.gateway.sync.engine import SyncEngine


def test_sync_writes_staged_not_active(tmp_path):
    engine = SyncEngine(root=tmp_path)
    docs = [
        Document(
            id="jira:1",
            source="jira",
            source_instance="p",
            title="t",
            text="x",
            content_hash="h1",
        )
    ]
    batch = SyncBatch(added=docs, cursor="c1")

    revision = engine.stage_sync("jira", batch)

    assert os.path.isdir(revision.path)
    assert engine.active_revision("jira") is None
