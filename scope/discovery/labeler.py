"""Cluster labeling via c-TF-IDF and optional mapping to predefined topics."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from scope.discovery.clusterer import ClusterResult
from scope.embeddings import EmbeddingProvider


class ClusterLabeler:
    """Label discovered clusters with representative keywords."""

    def __init__(self, top_n: int = 5) -> None:
        self.top_n = top_n

    def label_clusters(self, cluster_result: ClusterResult) -> dict[int, str]:
        """Extract representative keywords per cluster using c-TF-IDF.

        Each cluster's texts are concatenated into one "document", then TF-IDF
        identifies the most distinctive terms per cluster.

        Returns:
            {cluster_id: "kw1, kw2, kw3, ..."}
        """
        unique_labels = sorted(set(cluster_result.labels))
        unique_labels = [l for l in unique_labels if l != -1]

        if not unique_labels:
            return {}

        cluster_docs = []
        label_order = []
        for label in unique_labels:
            mask = cluster_result.labels == label
            texts = [cluster_result.texts[i] for i in range(len(mask)) if mask[i]]
            cluster_docs.append(" ".join(texts))
            label_order.append(label)

        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            min_df=1,
            max_df=0.95,
        )
        tfidf_matrix = vectorizer.fit_transform(cluster_docs)
        feature_names = vectorizer.get_feature_names_out()

        labels = {}
        for i, label in enumerate(label_order):
            row = tfidf_matrix[i].toarray().flatten()
            top_indices = row.argsort()[-self.top_n:][::-1]
            keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            labels[label] = ", ".join(keywords) if keywords else f"cluster_{label}"

        return labels

    def map_to_predefined(
        self,
        cluster_result: ClusterResult,
        predefined_topics: list[str],
        embedding_provider: EmbeddingProvider,
        similarity_threshold: float = 0.5,
    ) -> dict[int, str | None]:
        """Map discovered clusters to predefined topic names via centroid similarity.

        Returns:
            {cluster_id: "PredefinedTopicName" or None if no match above threshold}
        """
        from sklearn.metrics.pairwise import cosine_similarity

        unique_labels = sorted(set(cluster_result.labels))
        unique_labels = [l for l in unique_labels if l != -1]

        if not unique_labels:
            return {}

        # Compute cluster centroids in full embedding space
        centroids = []
        label_order = []
        for label in unique_labels:
            mask = cluster_result.labels == label
            cluster_embs = cluster_result.embeddings[mask]
            centroids.append(cluster_embs.mean(axis=0))
            label_order.append(label)
        centroids = np.array(centroids)

        # Embed predefined topic names
        topic_embs = embedding_provider.encode_batch(predefined_topics)

        # Cosine similarity: (n_clusters, n_topics)
        sim_matrix = cosine_similarity(centroids, topic_embs)

        mapping = {}
        for i, label in enumerate(label_order):
            max_idx = sim_matrix[i].argmax()
            max_sim = sim_matrix[i, max_idx]
            if max_sim >= similarity_threshold:
                mapping[label] = predefined_topics[max_idx]
            else:
                mapping[label] = None

        return mapping
