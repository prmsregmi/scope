"""SentenceTransformer embedding provider."""

from typing import Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .base import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """SentenceTransformer-based embedding provider."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L12-v2",
        disk_cache: Optional["DiskEmbeddingCache"] = None,
    ) -> None:
        super().__init__(disk_cache=disk_cache)

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: uv sync --extra sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _encode_impl(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def _encode_batch_impl(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
