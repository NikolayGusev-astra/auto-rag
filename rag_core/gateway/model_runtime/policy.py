from __future__ import annotations

from enum import Enum


class CloudPolicy(str, Enum):
    DISABLED = "disabled"
    QUERY_ONLY = "query_only"
    SELECTED_EVIDENCE = "selected_evidence"
    FULL = "full"

    @classmethod
    def default(cls) -> "CloudPolicy":
        return cls.DISABLED


def guard_cloud_call(
    policy: CloudPolicy,
    *,
    sends_query: bool = False,
    sends_document: bool = False,
) -> None:
    """Reject cloud requests that exceed the selected data-sharing policy."""
    if policy == CloudPolicy.DISABLED:
        raise PermissionError("Cloud provider disabled by default policy")
    if sends_document and policy not in (CloudPolicy.SELECTED_EVIDENCE, CloudPolicy.FULL):
        raise PermissionError(
            f"Policy {policy.value} forbids sending document content to cloud"
        )
