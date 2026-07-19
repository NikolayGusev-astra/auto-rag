from rag_core import rag_async


def test_rag_async_is_marked_as_legacy_pipeline():
    assert rag_async.LEGACY_PIPELINE is True
