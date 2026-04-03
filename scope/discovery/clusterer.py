"""Unsupervised topic discovery via UMAP dimensionality reduction and clustering."""

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from scope.embeddings import EmbeddingProvider


@dataclass
class ClusterResult:
    """Results from unsupervised clustering."""

    users: list[str]
    hour_indices: list[int]
    texts: list[str]
    labels: np.ndarray
    probabilities: np.ndarray
    embeddings: np.ndarray
    reduced_embeddings: np.ndarray
    n_clusters: int
    noise_count: int


class TopicDiscoverer:
    """Discover topics from chat data using embedding clustering."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        algorithm: str = "hdbscan",
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.0,
        hdbscan_min_cluster_size: int = 5,
        hdbscan_min_samples: int = 3,
        kmeans_n_clusters: int | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.algorithm = algorithm
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.hdbscan_min_samples = hdbscan_min_samples
        self.kmeans_n_clusters = kmeans_n_clusters

    def fit(
        self,
        user_hourly_original: dict[str, list[str]],
    ) -> ClusterResult:
        """Run the full discovery pipeline: embed, reduce, cluster.

        Args:
            user_hourly_original: {user: [original_text_per_hour]}

        Returns:
            ClusterResult with per-item cluster assignments
        """
        users, hour_indices, texts = self._collect_texts(user_hourly_original)

        if not texts:
            tqdm.write("[discovery] no non-empty hour-texts found — nothing to cluster")
            return ClusterResult(
                users=[], hour_indices=[], texts=[], labels=np.array([], dtype=int),
                probabilities=np.array([]), embeddings=np.empty((0, 0)),
                reduced_embeddings=np.empty((0, 0)), n_clusters=0, noise_count=0,
            )

        tqdm.write(f"[discovery] {len(texts)} non-empty hour-texts from {len(set(users))} users")

        embeddings = self._embed(texts)
        reduced = self._reduce(embeddings)
        labels, probabilities = self._cluster(reduced)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int(np.sum(labels == -1))

        tqdm.write(
            f"[discovery] {n_clusters} clusters found, "
            f"{noise_count} noise points ({100 * noise_count / len(labels):.1f}%)"
        )

        return ClusterResult(
            users=users,
            hour_indices=hour_indices,
            texts=texts,
            labels=labels,
            probabilities=probabilities,
            embeddings=embeddings,
            reduced_embeddings=reduced,
            n_clusters=n_clusters,
            noise_count=noise_count,
        )

    def _collect_texts(
        self,
        user_hourly_original: dict[str, list[str]],
    ) -> tuple[list[str], list[int], list[str]]:
        """Collect all non-empty (user, hour_idx, text) triples."""
        users = []
        hour_indices = []
        texts = []
        for user, hourly_texts in user_hourly_original.items():
            for h, text in enumerate(hourly_texts):
                if text and text.strip():
                    users.append(user)
                    hour_indices.append(h)
                    texts.append(text)
        return users, hour_indices, texts

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Batch-encode texts using the embedding provider (with disk cache)."""
        tqdm.write(f"[discovery] embedding {len(texts)} texts...")
        return self.embedding_provider.encode_batch(texts, batch_size=64)

    def _reduce(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensionality with UMAP."""
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError(
                "umap-learn is required for topic discovery. "
                "Install with: uv sync --extra unsupervised"
            )

        tqdm.write(
            f"[discovery] UMAP reducing {embeddings.shape[1]}d -> {self.umap_n_components}d "
            f"(n_neighbors={self.umap_n_neighbors})"
        )
        reducer = UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)

    def _cluster(self, reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cluster reduced embeddings. Returns (labels, probabilities)."""
        if self.algorithm == "hdbscan":
            return self._cluster_hdbscan(reduced)
        elif self.algorithm == "kmeans":
            return self._cluster_kmeans(reduced)
        else:
            raise ValueError(f"Unknown clustering algorithm: {self.algorithm}")

    def _cluster_hdbscan(self, reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.cluster import HDBSCAN

        tqdm.write(
            f"[discovery] HDBSCAN clustering "
            f"(min_cluster_size={self.hdbscan_min_cluster_size}, "
            f"min_samples={self.hdbscan_min_samples})"
        )
        clusterer = HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            store_centers="centroid",
        )
        clusterer.fit(reduced)
        return clusterer.labels_, clusterer.probabilities_

    def _cluster_kmeans(self, reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.cluster import KMeans

        n_clusters = self.kmeans_n_clusters
        if n_clusters is None:
            raise ValueError("--kmeans-k is required when using kmeans algorithm")

        tqdm.write(f"[discovery] KMeans clustering (k={n_clusters})")
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusterer.fit(reduced)

        distances = clusterer.transform(reduced)
        min_distances = distances.min(axis=1)
        max_dist = min_distances.max()
        probabilities = 1.0 - (min_distances / max_dist) if max_dist > 0 else np.ones(len(reduced))

        return clusterer.labels_, probabilities
