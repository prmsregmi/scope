"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingProvider(ABC):
    """Abstract base class for text embedding providers."""

    def __init__(self, disk_cache: Optional["DiskEmbeddingCache"] = None) -> None:
        self._cache: dict[str, np.ndarray] = {}
        self.disk_cache = disk_cache

    def encode(self, text: str) -> np.ndarray:
        """Generate embedding for a single text with multi-level caching.

        Checks: memory cache → disk cache → compute via _encode_impl.
        """
        if text in self._cache:
            return self._cache[text]

        if self.disk_cache is not None:
            cached = self.disk_cache.get(text)
            if cached is not None:
                self._cache[text] = cached
                return cached

        embedding = self._encode_impl(text)
        self._cache[text] = embedding

        if self.disk_cache is not None:
            self.disk_cache.put(text, embedding)

        return embedding

    @abstractmethod
    def _encode_impl(self, text: str) -> np.ndarray:
        """Provider-specific single-text encoding (no caching)."""
        pass

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Generate embeddings for multiple texts with multi-level caching.

        Resolves from memory cache and disk cache first, then calls
        _encode_batch_impl for the remainder.
        """
        results: list[tuple[int, np.ndarray]] = []
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            if text in self._cache:
                results.append((i, self._cache[text]))
                continue
            if self.disk_cache is not None:
                cached = self.disk_cache.get(text)
                if cached is not None:
                    self._cache[text] = cached
                    results.append((i, cached))
                    continue
            uncached_texts.append(text)
            uncached_indices.append(i)

        if uncached_texts:
            new_embeddings = self._encode_batch_impl(uncached_texts, batch_size)
            for text, idx, emb in zip(uncached_texts, uncached_indices, new_embeddings):
                self._cache[text] = emb
                if self.disk_cache is not None:
                    self.disk_cache.put(text, emb)
                results.append((idx, emb))

        results.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in results])

    @abstractmethod
    def _encode_batch_impl(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Provider-specific batch encoding (no caching)."""
        pass

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        emb1 = self.encode(text1.lower())
        emb2 = self.encode(text2.lower())
        return float(cosine_similarity([emb1], [emb2])[0][0])

    def clear_cache(self) -> None:
        """Clear the in-memory embedding cache."""
        self._cache.clear()

    def cache_size(self) -> int:
        return len(self._cache)
