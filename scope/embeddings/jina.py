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
        disk_cache: Optional["DiskEmbeddingCache"] = None,
    ) -> None:
        super().__init__(disk_cache=disk_cache)

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
                if e.response.status_code == 429:
                    wait_time = 2 ** attempt
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

    def _encode_impl(self, text: str) -> np.ndarray:
        result = self._make_request([text])
        return np.array(result["data"][0]["embedding"])

    def _encode_batch_impl(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        embeddings: list[tuple[int, np.ndarray]] = []

        if self.parallel_requests and len(texts) > batch_size:
            batches = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_indices = list(range(i, i + len(batch)))
                batches.append((batch, batch_indices))

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._make_request, batch): (batch, indices)
                    for batch, indices in batches
                }
                for future in as_completed(futures):
                    batch, indices = futures[future]
                    result = future.result()
                    for idx, data in zip(indices, result["data"]):
                        embeddings.append((idx, np.array(data["embedding"])))
        else:
            offset = 0
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                result = self._make_request(batch)
                for j, data in enumerate(result["data"]):
                    embeddings.append((offset + j, np.array(data["embedding"])))
                offset += len(batch)

        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
