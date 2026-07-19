from rag_core.gateway.connector import SourceConnector, SearchRequest


def test_searchrequest_has_query_and_defaults():
    req = SearchRequest(query="кластер", topk=5)
    assert req.query == "кластер"
    assert req.topk == 5
    assert req.include_web is False


def test_sourceconnector_is_protocol():
    import inspect
    assert hasattr(SourceConnector, "search_live")
    assert hasattr(SourceConnector, "fetch")
    assert hasattr(SourceConnector, "sync_changes")
    assert hasattr(SourceConnector, "health")
