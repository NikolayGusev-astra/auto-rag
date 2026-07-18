"""Query execution context — carries tenant/ACL/index identity through the pipeline.

P0 fix (audit): the previous cache key was `query|domain|max_results` only.
In a multi-tenant server mode that lets one user's cached result be returned
to a different principal. We now thread tenant + ACL + index revision into the
cache key and into every retrieval result so cross-request leakage is impossible.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryContext:
    """Identity + scope for a single RAG query.

    All fields except `query` are optional; when absent they default to the
    process-wide safe values (single-tenant local mode). In server mode the
    caller MUST populate `tenant_id` and `principal_acl_hash`.
    """

    query: str
    domain: str = ""
    collection: str = ""
    max_results: int = 5
    tenant_id: str = field(default_factory=lambda: os.environ.get("RAG_TENANT_ID", "default"))
    principal_acl_hash: str = field(
        default_factory=lambda: os.environ.get("RAG_ACL_HASH", "none")
    )
    index_revision: str = field(
        default_factory=lambda: os.environ.get("RAG_INDEX_REVISION", "unknown")
    )
    config_revision: str = field(
        default_factory=lambda: os.environ.get("RAG_CONFIG_REVISION", "unknown")
    )
    model_revision: str = field(
        default_factory=lambda: os.environ.get("RAG_MODEL_REVISION", "unknown")
    )

    def cache_key(self) -> str:
        """Stable cache key including tenant + ACL + index/config/model revs."""
        raw = "|".join(
            [
                self.query,
                self.domain,
                self.collection,
                str(self.max_results),
                self.tenant_id,
                self.principal_acl_hash,
                self.index_revision,
                self.config_revision,
                self.model_revision,
            ]
        )
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def to_dcd(self) -> dict[str, Any]:
        """Build the legacy dcd_result dict used by the orchestrator."""
        return {
            "domain": self.domain,
            "collection": self.collection,
            "confidence": 0,
            "fallback": False,
        }
