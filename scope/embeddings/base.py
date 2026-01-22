"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingProvider(ABC):
    """Abstract base class for text embedding providers."""

    def __init__(self) -> None:
        """Initialize the embedding provider."""
        self._cache: dict[str, np.ndarray] = {}

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Input text string

        Returns:
            Embedding vector as numpy array
        """
        pass

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts with caching.

        Args:
            texts: List of input text strings

        Returns:
            Array of embedding vectors
        """
        embeddings = []
        for text in texts:
            if text in self._cache:
                embeddings.append(self._cache[text])
            else:
                embedding = self.encode(text)
                self._cache[text] = embedding
                embeddings.append(embedding)
        return np.array(embeddings)

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0-1)
        """
        emb1 = self.encode(text1.lower())
        emb2 = self.encode(text2.lower())
        return float(cosine_similarity([emb1], [emb2])[0][0])

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def cache_size(self) -> int:
        """Get the current cache size.

        Returns:
            Number of cached embeddings
        """
        return len(self._cache)
