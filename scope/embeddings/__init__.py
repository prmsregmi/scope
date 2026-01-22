"""Embedding providers for text vectorization."""

from typing import Literal, Optional

from .base import EmbeddingProvider


def get_embedding_provider(
    provider_type: Literal["sentence-transformers", "jina"],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> EmbeddingProvider:
    """Factory function to get an embedding provider.

    Args:
        provider_type: Type of embedding provider
        model: Model name for the provider
        api_key: API key (for Jina)
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured embedding provider instance

    Raises:
        ValueError: If provider_type is unknown
        ImportError: If required dependencies are not installed
    """
    if provider_type == "sentence-transformers":
        from .sentence_transformer import SentenceTransformerProvider

        return SentenceTransformerProvider(
            model_name=model or "all-MiniLM-L12-v2",
            **kwargs
        )

    elif provider_type == "jina":
        from .jina import JinaEmbeddingProvider

        # Extract parallel_requests and max_workers if provided
        parallel_requests = kwargs.pop("parallel_requests", False)
        max_workers = kwargs.pop("max_workers", 5)

        return JinaEmbeddingProvider(
            api_key=api_key,
            model=model or "jina-embeddings-v3",
            parallel_requests=parallel_requests,
            max_workers=max_workers,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider_type}. "
            "Must be 'sentence-transformers' or 'jina'"
        )


__all__ = ["EmbeddingProvider", "get_embedding_provider"]
