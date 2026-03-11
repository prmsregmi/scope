"""Embedding providers for text vectorization."""

from typing import Literal, Optional

from .base import EmbeddingProvider
from .disk_cache import DiskEmbeddingCache


def get_embedding_provider(
    provider_type: Literal["sentence-transformers", "jina"],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    disk_cache: Optional[DiskEmbeddingCache] = None,
    **kwargs
) -> EmbeddingProvider:
    """Factory function to get an embedding provider.

    Args:
        provider_type: Type of embedding provider
        model: Model name for the provider
        api_key: API key (for Jina)
        disk_cache: Optional disk cache for persistent embedding storage
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured embedding provider instance
    """
    if provider_type == "sentence-transformers":
        from .sentence_transformer import SentenceTransformerProvider

        return SentenceTransformerProvider(
            model_name=model or "all-MiniLM-L12-v2",
            disk_cache=disk_cache,
            **kwargs
        )

    elif provider_type == "jina":
        from .jina import JinaEmbeddingProvider

        parallel_requests = kwargs.pop("parallel_requests", False)
        max_workers = kwargs.pop("max_workers", 5)

        return JinaEmbeddingProvider(
            api_key=api_key,
            model=model or "jina-embeddings-v3",
            parallel_requests=parallel_requests,
            max_workers=max_workers,
            disk_cache=disk_cache,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider_type}. "
            "Must be 'sentence-transformers' or 'jina'"
        )


__all__ = ["DiskEmbeddingCache", "EmbeddingProvider", "get_embedding_provider"]
