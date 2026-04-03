"""Data structures for evaluation metrics."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerformanceMetrics:
    """Performance-related metrics."""

    execution_time: float  # Total execution time in seconds
    preprocessing_time: float = 0.0  # Time spent on preprocessing
    embedding_time: float = 0.0  # Time spent generating embeddings
    analysis_time: float = 0.0  # Time spent on probability calculation and block finding

    memory_peak_mb: float = 0.0  # Peak memory usage in MB
    memory_average_mb: float = 0.0  # Average memory usage in MB

    # Embedding provider specific
    api_calls: int = 0  # Number of API calls (for Jina)
    cache_hits: int = 0  # Number of cache hits
    cache_misses: int = 0  # Number of cache misses
    cache_hit_rate: float = 0.0  # Cache hit rate (0-1)

    # Data metrics
    total_messages: int = 0
    total_users: int = 0
    total_hours: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_time_seconds": round(self.execution_time, 2),
            "preprocessing_time_seconds": round(self.preprocessing_time, 2),
            "embedding_time_seconds": round(self.embedding_time, 2),
            "analysis_time_seconds": round(self.analysis_time, 2),
            "memory_peak_mb": round(self.memory_peak_mb, 2),
            "memory_average_mb": round(self.memory_average_mb, 2),
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "total_messages": self.total_messages,
            "total_users": self.total_users,
            "total_hours": self.total_hours,
            "messages_per_second": round(self.total_messages / self.execution_time, 2) if self.execution_time > 0 else 0,
        }


@dataclass
class QualityMetrics:
    """Quality-related metrics."""

    num_segments: int  # Total number of segments detected
    avg_segment_length: float  # Average number of messages per segment
    total_messages_captured: int  # Total messages in all segments
    coverage: float  # Percentage of messages captured (0-1)

    avg_probability: float  # Average probability score across all segments
    min_probability: float  # Minimum probability score
    max_probability: float  # Maximum probability score

    # Topic distribution
    topic_distribution: dict[str, int] = field(default_factory=dict)  # Topic -> count
    topics_detected: int = 0  # Number of unique topics detected

    # Segment duration statistics
    avg_segment_duration_hours: float = 0.0  # Average duration in hours
    min_segment_duration_hours: float = 0.0
    max_segment_duration_hours: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_segments": self.num_segments,
            "avg_segment_length": round(self.avg_segment_length, 2),
            "total_messages_captured": self.total_messages_captured,
            "coverage_percentage": round(self.coverage * 100, 2),
            "avg_probability": round(self.avg_probability, 4),
            "min_probability": round(self.min_probability, 4),
            "max_probability": round(self.max_probability, 4),
            "topics_detected": self.topics_detected,
            "topic_distribution": self.topic_distribution,
            "avg_segment_duration_hours": round(self.avg_segment_duration_hours, 2),
            "min_segment_duration_hours": round(self.min_segment_duration_hours, 2),
            "max_segment_duration_hours": round(self.max_segment_duration_hours, 2),
        }


@dataclass
class AccuracyMetrics:
    """Accuracy metrics comparing predictions to ground truth labels."""

    total_samples: int  # Total number of labeled samples
    correct_predictions: int  # Number of correct predictions
    accuracy: float  # Overall accuracy (0-1)

    # Classification metrics
    precision: float = 0.0  # Weighted precision
    recall: float = 0.0  # Weighted recall
    f1_score: float = 0.0  # Weighted F1 score

    # Per-topic metrics
    per_topic_accuracy: dict[str, float] = field(default_factory=dict)  # Topic -> accuracy
    per_topic_precision: dict[str, float] = field(default_factory=dict)  # Topic -> precision
    per_topic_recall: dict[str, float] = field(default_factory=dict)  # Topic -> recall
    per_topic_f1: dict[str, float] = field(default_factory=dict)  # Topic -> F1
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)  # True -> Predicted -> count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_samples": self.total_samples,
            "correct_predictions": self.correct_predictions,
            "accuracy": round(self.accuracy, 4),
            "accuracy_percentage": round(self.accuracy * 100, 2),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "per_topic_accuracy": {k: round(v, 4) for k, v in self.per_topic_accuracy.items()},
            "per_topic_precision": {k: round(v, 4) for k, v in self.per_topic_precision.items()},
            "per_topic_recall": {k: round(v, 4) for k, v in self.per_topic_recall.items()},
            "per_topic_f1": {k: round(v, 4) for k, v in self.per_topic_f1.items()},
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass
class ClusteringMetrics:
    """Metrics for unsupervised topic discovery."""

    n_clusters: int
    noise_ratio: float
    silhouette_score: float

    # External metrics (require ground truth labels)
    nmi: float | None = None
    ari: float | None = None
    v_measure: float | None = None

    # Cluster purity (require ground truth)
    mean_purity: float | None = None
    high_purity_count: int | None = None
    per_cluster_purity: dict[str, float] = field(default_factory=dict)

    # Cluster size distribution
    cluster_sizes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "n_clusters": self.n_clusters,
            "noise_ratio": round(self.noise_ratio, 4),
            "silhouette_score": round(self.silhouette_score, 4),
            "cluster_sizes": self.cluster_sizes,
        }
        if self.nmi is not None:
            result["nmi"] = round(self.nmi, 4)
        if self.ari is not None:
            result["ari"] = round(self.ari, 4)
        if self.v_measure is not None:
            result["v_measure"] = round(self.v_measure, 4)
        if self.mean_purity is not None:
            result["mean_purity"] = round(self.mean_purity, 4)
            result["high_purity_count"] = self.high_purity_count
            result["per_cluster_purity"] = {
                k: round(v, 4) for k, v in self.per_cluster_purity.items()
            }
        return result


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics for a SCOPE run."""

    run_name: str  # Name/identifier for this run
    config: dict[str, Any]  # Configuration used for this run

    performance: PerformanceMetrics
    quality: QualityMetrics

    timestamp: str = ""  # Timestamp of the evaluation
    accuracy: AccuracyMetrics | None = None  # Optional accuracy metrics
    clustering: ClusteringMetrics | None = None  # Optional clustering metrics

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "config": self.config,
            "performance": self.performance.to_dict(),
            "quality": self.quality.to_dict(),
        }
        if self.accuracy:
            result["accuracy"] = self.accuracy.to_dict()
        if self.clustering:
            result["clustering"] = self.clustering.to_dict()
        return result

    def summary_str(self) -> str:
        """Generate a human-readable summary string."""
        lines = [
            "=" * 70,
            f"SCOPE Evaluation: {self.run_name}",
            "=" * 70,
            "",
            "CONFIGURATION:",
            f"  Embedding Provider: {self.config.get('embedding_provider', 'N/A')}",
            f"  Embedding Model: {self.config.get('embedding_model', 'N/A')}",
            f"  Probability Threshold: {self.config.get('probability_threshold', 'N/A')}",
            f"  Dataset: {self.config.get('dataset_path', 'N/A')}",
            "",
            "PERFORMANCE:",
            f"  Total Execution Time: {self.performance.execution_time:.2f}s",
            f"  - Preprocessing: {self.performance.preprocessing_time:.2f}s",
            f"  - Embedding Generation: {self.performance.embedding_time:.2f}s",
            f"  - Analysis: {self.performance.analysis_time:.2f}s",
            f"  Peak Memory: {self.performance.memory_peak_mb:.2f} MB",
            f"  Messages/Second: {self.performance.total_messages / self.performance.execution_time:.2f}" if self.performance.execution_time > 0 else "  Messages/Second: N/A",
            "",
            "CACHE STATISTICS:",
            f"  Cache Hit Rate: {self.performance.cache_hit_rate * 100:.2f}%",
            f"  Cache Hits: {self.performance.cache_hits}",
            f"  Cache Misses: {self.performance.cache_misses}",
            f"  API Calls: {self.performance.api_calls}",
            "",
            "QUALITY:",
            f"  Segments Detected: {self.quality.num_segments}",
            f"  Coverage: {self.quality.coverage * 100:.2f}% ({self.quality.total_messages_captured}/{self.performance.total_messages} messages)",
            f"  Avg Segment Length: {self.quality.avg_segment_length:.2f} messages",
            f"  Avg Segment Duration: {self.quality.avg_segment_duration_hours:.2f} hours",
            f"  Avg Probability: {self.quality.avg_probability:.4f}",
            f"  Probability Range: [{self.quality.min_probability:.4f}, {self.quality.max_probability:.4f}]",
            f"  Topics Detected: {self.quality.topics_detected}",
            "",
            "TOPIC DISTRIBUTION:",
        ]

        # Sort topics by count (descending)
        sorted_topics = sorted(
            self.quality.topic_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for topic, count in sorted_topics:
            lines.append(f"  {topic}: {count} segments")

        # Add accuracy section if available
        if self.accuracy:
            lines.append("")
            lines.append("ACCURACY (model-level — independent of pipeline params):")
            lines.append(f"  Overall Accuracy: {self.accuracy.accuracy * 100:.2f}% ({self.accuracy.correct_predictions}/{self.accuracy.total_samples})")
            lines.append(f"  Precision: {self.accuracy.precision:.4f}")
            lines.append(f"  Recall: {self.accuracy.recall:.4f}")
            lines.append(f"  F1 Score: {self.accuracy.f1_score:.4f}")

            if self.accuracy.per_topic_f1:
                lines.append("")
                lines.append("  Per-Topic Metrics (Top 10 by F1):")
                sorted_topics = sorted(
                    self.accuracy.per_topic_f1.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                for topic, f1 in sorted_topics:
                    acc = self.accuracy.per_topic_accuracy.get(topic, 0)
                    prec = self.accuracy.per_topic_precision.get(topic, 0)
                    rec = self.accuracy.per_topic_recall.get(topic, 0)
                    lines.append(f"    {topic}: Acc={acc*100:.1f}% P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

        if self.clustering:
            lines.append("")
            lines.append("CLUSTERING:")
            lines.append(f"  Clusters Found: {self.clustering.n_clusters}")
            lines.append(f"  Noise Ratio: {self.clustering.noise_ratio * 100:.1f}%")
            lines.append(f"  Silhouette Score: {self.clustering.silhouette_score:.4f}")
            if self.clustering.nmi is not None:
                lines.append(f"  NMI: {self.clustering.nmi:.4f}")
            if self.clustering.ari is not None:
                lines.append(f"  ARI: {self.clustering.ari:.4f}")
            if self.clustering.v_measure is not None:
                lines.append(f"  V-Measure: {self.clustering.v_measure:.4f}")
            if self.clustering.mean_purity is not None:
                lines.append(f"  Mean Purity: {self.clustering.mean_purity:.4f}")
                lines.append(f"  High-Purity Clusters (>=0.7): {self.clustering.high_purity_count}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
