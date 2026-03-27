"""Configuration management for SCOPE."""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default topics from the original notebook
DEFAULT_TOPICS = [
    "News",
    "Research",
    "Technology",
    "Travel",
    "Personal",
    "Education",
    "Career",
    "Health",
    "Sports",
    "Vacation",
    "Movie",
    "Entertainment",
    "Book",
    "Event",
    "Food",
    "Politics",
    "Finance",
    "Relationships",
    "Religion",
    "Immigration",
    "Fantasy",
]


@dataclass
class ScopeConfig:
    """Configuration for SCOPE analysis."""

    dataset_path: str
    output_path: str = "results/scope_results.csv"
    topics: list[str] = field(default_factory=lambda: DEFAULT_TOPICS.copy())
    probability_threshold: float = 0.07
    embedding_provider: str = "sentence-transformers"
    embedding_model: Optional[str] = None
    jina_api_key: Optional[str] = None
    jina_parallel_requests: bool = True
    jina_max_workers: int = 10
    prefilter_sim_threshold: float = 0.0
    use_keybert: bool = True
    keybert_model: str = "all-MiniLM-L12-v2"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    enable_spell_check: bool = False
    enable_lemmatization: bool = True
    verbose: bool = True
    include_summary: bool = True
    # Disk embedding cache
    cache_dir: str = ".scope_cache"
    enable_disk_cache: bool = True
    # Calculation mode: auto, st_baseline, jina_mixed, hybrid, etc.
    calculation_mode: str = "auto"
    # Unsupervised topic discovery
    discover_topics: bool = False
    clustering_algorithm: str = "hdbscan"
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    kmeans_n_clusters: Optional[int] = None
    umap_n_components: int = 5
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.0
    cluster_label_top_n: int = 5
    map_to_predefined: bool = False
    map_similarity_threshold: float = 0.5
    # Evaluation
    enable_evaluation: bool = True
    labeled_test_data: Optional[str] = None
    # PostgreSQL vector storage
    use_postgres: bool = False
    postgres_host: Optional[str] = None
    postgres_port: Optional[int] = None
    postgres_dbname: Optional[str] = None
    postgres_user: Optional[str] = None
    postgres_password: Optional[str] = None

    @property
    def accuracy_fingerprint(self) -> str:
        """Fingerprint capturing every parameter that affects accuracy.

        Two runs with identical fingerprints will always produce identical
        accuracy regardless of dataset_path, date range, threshold, or
        prefilter settings.
        """
        return (
            f"{self.embedding_provider}|{self.embedding_model}|"
            f"{self.use_keybert}|{self.keybert_model}|{','.join(sorted(self.topics))}|"
            f"{self.enable_spell_check}|{self.enable_lemmatization}|"
            f"{self.calculation_mode}"
        )

    def __post_init__(self) -> None:
        """Validate and set defaults after initialization."""
        # Set default embedding model if not specified
        if self.embedding_model is None:
            if self.embedding_provider == "sentence-transformers":
                self.embedding_model = "all-MiniLM-L12-v2"
            elif self.embedding_provider == "jina":
                self.embedding_model = "jina-embeddings-v3"

        # Get Jina API key from environment if not provided
        if self.embedding_provider == "jina" and self.jina_api_key is None:
            self.jina_api_key = os.getenv("JINA_API_KEY")
            if not self.jina_api_key:
                raise ValueError(
                    "Jina API key required. Set JINA_API_KEY environment variable or pass --jina-api-key"
                )

        # Validate probability threshold
        if not 0.0 < self.probability_threshold < 1.0:
            raise ValueError("Probability threshold must be between 0 and 1")

        # Validate embedding provider
        if self.embedding_provider not in ["sentence-transformers", "jina"]:
            raise ValueError(
                f"Unknown embedding provider: {self.embedding_provider}. "
                "Must be 'sentence-transformers' or 'jina'"
            )

        # Validate clustering algorithm
        valid_clustering = ["hdbscan", "kmeans"]
        if self.clustering_algorithm not in valid_clustering:
            raise ValueError(
                f"Unknown clustering algorithm: {self.clustering_algorithm}. "
                f"Must be one of {valid_clustering}"
            )

        # Validate calculation mode
        valid_modes = ["auto", "st_baseline", "jina_mixed", "jina_bag_of_words", "jina_full_text", "hybrid"]
        if self.calculation_mode not in valid_modes:
            raise ValueError(
                f"Unknown calculation mode: {self.calculation_mode}. "
                f"Must be one of {valid_modes}"
            )

        # Hybrid mode requires Jina API key for full-text embeddings
        if self.calculation_mode == "hybrid" and not self.jina_api_key:
            self.jina_api_key = os.getenv("JINA_API_KEY")
            if not self.jina_api_key:
                raise ValueError(
                    "Hybrid mode requires Jina API key for full-text embeddings. "
                    "Set JINA_API_KEY environment variable or pass --jina-api-key"
                )

    @classmethod
    def from_env(cls, dataset_path: Optional[str] = None) -> "ScopeConfig":
        """Create configuration from environment variables and defaults.

        Args:
            dataset_path: Path to the dataset file (uses SCOPE_DATASET_PATH env var if not provided)

        Returns:
            ScopeConfig with values from environment variables or defaults
        """
        # Use provided path or fall back to env var or default
        if dataset_path is None:
            dataset_path = os.getenv("SCOPE_DATASET_PATH", "data/Conversation.csv")

        # Get PostgreSQL configuration from environment
        # Use system user as default for postgres_user
        default_pg_user = os.getenv("USER", "postgres")

        return cls(
            dataset_path=dataset_path,
            embedding_provider=os.getenv("SCOPE_EMBEDDING_PROVIDER", "sentence-transformers"),
            embedding_model=os.getenv("SCOPE_EMBEDDING_MODEL"),
            probability_threshold=float(os.getenv("SCOPE_PROBABILITY_THRESHOLD", "0.07")),
            keybert_model=os.getenv("SCOPE_KEYBERT_MODEL", "all-MiniLM-L12-v2"),
            jina_api_key=os.getenv("JINA_API_KEY"),
            enable_spell_check=os.getenv("SCOPE_SPELL_CHECK", "false").lower() == "true",
            enable_lemmatization=os.getenv("SCOPE_LEMMATIZE", "true").lower() == "true",
            jina_parallel_requests=os.getenv("JINA_PARALLEL_REQUESTS", "true").lower() == "true",
            jina_max_workers=int(os.getenv("JINA_MAX_WORKERS", "10")),
            prefilter_sim_threshold=float(os.getenv("SCOPE_PREFILTER_SIM_THRESHOLD", "0.0")),
            output_path=os.getenv("SCOPE_OUTPUT_PATH", "results/scope_results.csv"),
            # Evaluation configuration
            enable_evaluation=os.getenv("SCOPE_ENABLE_EVALUATION", "true").lower() == "true",
            labeled_test_data=os.getenv("SCOPE_LABELED_TEST_DATA", "data/labeled_test_data.csv"),
            # PostgreSQL configuration
            use_postgres=os.getenv("SCOPE_USE_POSTGRES", "false").lower() == "true",
            postgres_host=os.getenv("DATABASE_HOST"),
            postgres_port=int(os.getenv("DATABASE_PORT", "5432")),
            postgres_dbname=os.getenv("DATABASE_NAME"),
            postgres_user=os.getenv("DATABASE_USER", default_pg_user),
            postgres_password=os.getenv("DATABASE_PASSWORD", ""),
        )

    def merge_with_args(self, **kwargs) -> None:
        """Merge configuration with command-line arguments (CLI args take precedence)."""
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

        # Re-derive model default when provider changed but model wasn't explicitly set
        if "embedding_provider" in kwargs and kwargs.get("embedding_model") is None:
            if self.embedding_provider == "sentence-transformers":
                self.embedding_model = "all-MiniLM-L12-v2"
            elif self.embedding_provider == "jina":
                self.embedding_model = "jina-embeddings-v3"

        # Re-validate Jina API key after provider may have changed
        if self.embedding_provider == "jina" and not self.jina_api_key:
            self.jina_api_key = os.getenv("JINA_API_KEY")
            if not self.jina_api_key:
                raise ValueError(
                    "Jina API key required. Set JINA_API_KEY environment variable or pass --jina-api-key"
                )
