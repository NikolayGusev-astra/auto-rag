from __future__ import annotations

from rag_core.gateway.model_providers import EmbeddingProfile


def check_compatible(
    provider_profile: EmbeddingProfile,
    index_profile: EmbeddingProfile,
) -> tuple[bool, str]:
    """Return whether the provider fulfils the complete index contract."""
    if provider_profile.model_id != index_profile.model_id:
        return False, (
            f"Embedding model mismatch: provider={provider_profile.model_id!r} "
            f"index={index_profile.model_id!r}. Different models are not "
            "interchangeable even at equal dimension."
        )
    if provider_profile.model_revision != index_profile.model_revision:
        return False, "Model revision differs"
    if provider_profile.dimension != index_profile.dimension:
        return False, "Embedding dimension differs"
    if provider_profile.normalized != index_profile.normalized:
        return False, "Normalization policy differs"
    if provider_profile.distance_metric != index_profile.distance_metric:
        return False, "Distance metric differs"
    if provider_profile.preprocessing_revision != index_profile.preprocessing_revision:
        return False, "Preprocessing contract differs"
    return True, "compatible"
