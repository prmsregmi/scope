"""SentenceTransformer embedding provider."""

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .base import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """SentenceTransformer-based embedding provider."""

    def __init__(self, model_name: str = "all-MiniLM-L12-v2") -> None:
        """Initialize SentenceTransformer provider.

        Args:
            model_name: Name of the SentenceTransformer model to use

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        super().__init__()

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: uv sync --extra sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Input text string

        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        if text in self._cache:
            return self._cache[text]

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Cache the result
        self._cache[text] = embedding

        return embedding

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts with batching.

        Args:
            texts: List of input text strings
            batch_size: Batch size for encoding

        Returns:
            Array of embedding vectors
        """
        # Separate cached and uncached texts
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text in self._cache:
                embeddings.append((i, self._cache[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Encode uncached texts in batch
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                self._cache[text] = embedding

            # Add to embeddings list with correct indices
            for i, embedding in zip(uncached_indices, new_embeddings):
                embeddings.append((i, embedding))

        # Sort by original order and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
