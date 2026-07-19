import pytest

from rag_core.gateway.model_runtime.policy import CloudPolicy, guard_cloud_call


def test_default_policy_disabled_blocks_all():
    assert CloudPolicy.default() == CloudPolicy.DISABLED


def test_guard_blocks_document_when_policy_query_only():
    with pytest.raises(PermissionError):
        guard_cloud_call(CloudPolicy.QUERY_ONLY, sends_document=True)


def test_guard_allows_query_when_query_only():
    guard_cloud_call(CloudPolicy.QUERY_ONLY, sends_document=False, sends_query=True)


def test_guard_blocks_when_disabled():
    with pytest.raises(PermissionError):
        guard_cloud_call(CloudPolicy.DISABLED, sends_query=True)
