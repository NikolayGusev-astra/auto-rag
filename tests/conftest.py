import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _has_mod(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _embedding_available():
    url = os.environ.get("RAG_EMBEDDING_URL", "") or os.environ.get("RAG_MEMVID_EMBED_URL", "")
    model = os.environ.get("RAG_EMBEDDING_MODEL", "") or os.environ.get("RAG_MEMVID_EMBED_MODEL", "")
    return bool(url) and bool(model)


skip_if_no_chromadb = pytest.mark.skipif(
    not _has_mod("chromadb"), reason="chromadb not installed"
)
skip_if_no_embedding = pytest.mark.skipif(
    not _embedding_available(), reason="no embedding service configured"
)


@pytest.fixture
def sample_query():
    return "настройка postgresql streaming replication"


@pytest.fixture
def malicious_query():
    return 'foo" OR summary!="'
