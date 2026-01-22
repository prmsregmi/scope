"""Jina AI embedding provider."""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np

try:
    import httpx
except ImportError:
    httpx = None

from .base import EmbeddingProvider


class JinaEmbeddingProvider(EmbeddingProvider):
    """Jina AI embedding provider using their API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jina-embeddings-v3",
        task: str = "text-matching",
        dimensions: Optional[int] = 384,
        base_url: str = "https://api.jina.ai/v1/embeddings",
        max_retries: int = 3,
        timeout: float = 30.0,
        parallel_requests: bool = False,
        max_workers: int = 5,
    ) -> None:
        """Initialize Jina embedding provider.

        Args:
            api_key: Jina API key (or use JINA_API_KEY env var)
            model: Jina model name (default: jina-embeddings-v3)
            task: Task type (default: text-matching)
            dimensions: Output embedding dimensions (default: 384 to match SentenceTransformers)
            base_url: API endpoint URL
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            parallel_requests: Enable parallel API requests for batches (default: False)
            max_workers: Maximum number of parallel workers (default: 5)

        Raises:
            ImportError: If httpx is not installed
            ValueError: If API key is not provided
        """
        super().__init__()

        if httpx is None:
            raise ImportError(
                "httpx is not installed. Install it with: uv sync --extra jina"
            )

        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Jina API key required. Set JINA_API_KEY environment variable "
                "or pass api_key parameter"
            )

        self.model = model
        self.task = task
        self.dimensions = dimensions
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.parallel_requests = parallel_requests
        self.max_workers = max_workers

    def _make_request(self, texts: list[str]) -> dict:
        """Make API request to Jina.

        Args:
            texts: List of texts to embed

        Returns:
            API response dictionary

        Raises:
            httpx.HTTPError: If request fails after all retries
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "input": texts,
            "task": self.task,
        }

        # Add dimensions if specified
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        self.base_url,
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    return response.json()

            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    raise

            except httpx.RequestError as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    raise

        if last_exception:
            raise last_exception

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

        # Make API request
        result = self._make_request([text])
        embedding = np.array(result["data"][0]["embedding"])

        # Cache the result
        self._cache[text] = embedding

        return embedding

    def _make_request_batch(self, batch_data: tuple) -> tuple:
        """Make a single batch request (for parallel processing).

        Args:
            batch_data: Tuple of (batch_texts, batch_indices, batch_start_index)

        Returns:
            Tuple of (batch_indices, embeddings)
        """
        batch_texts, batch_indices, _ = batch_data
        result = self._make_request(batch_texts)

        batch_embeddings = []
        for text, data in zip(batch_texts, result["data"]):
            embedding = np.array(data["embedding"])
            self._cache[text] = embedding
            batch_embeddings.append(embedding)

        return batch_indices, batch_embeddings

    def encode_batch(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """Generate embeddings for multiple texts with batching.

        Args:
            texts: List of input text strings
            batch_size: Maximum number of texts per API request

        Returns:
            Array of embedding vectors
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Separate cached and uncached texts
        for i, text in enumerate(texts):
            if text in self._cache:
                embeddings.append((i, self._cache[text]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if not uncached_texts:
            # All cached, return immediately
            embeddings.sort(key=lambda x: x[0])
            return np.array([emb for _, emb in embeddings])

        # Process uncached texts in batches
        if self.parallel_requests and len(uncached_texts) > batch_size:
            # Parallel processing for multiple batches
            batches = []
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]
                batch_indices = uncached_indices[i:i + batch_size]
                batches.append((batch, batch_indices, i))

            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._make_request_batch, batch_data)
                          for batch_data in batches]

                for future in as_completed(futures):
                    batch_indices, batch_embeddings = future.result()
                    for idx, emb in zip(batch_indices, batch_embeddings):
                        embeddings.append((idx, emb))
        else:
            # Sequential processing
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i + batch_size]
                batch_indices = uncached_indices[i:i + batch_size]

                # Make API request
                result = self._make_request(batch)

                # Cache and store results
                for j, (text, data) in enumerate(zip(batch, result["data"])):
                    embedding = np.array(data["embedding"])
                    self._cache[text] = embedding
                    embeddings.append((batch_indices[j], embedding))

        # Sort by original order and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
