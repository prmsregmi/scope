"""Probability calculation using KeyBERT + Cosine Similarity."""

from typing import Optional

from scope.embeddings import EmbeddingProvider

from .keybert_similarity import KeyBERTSimilarityCalculator


class ProbabilityCalculator:
    """Calculate and cache topic probabilities using KeyBERT + Cosine Similarity.

    This replaces the LDA-based approach with the Hybrid Cosine-KeyBERT method.
    """

    def __init__(
        self,
        topics: list[str],
        embedding_provider: EmbeddingProvider,
        keybert_model: str = "all-MiniLM-L12-v2",
        calculation_mode: str = "jina_mixed",
        vector_store: Optional["VectorStore"] = None,
        use_keybert: bool = True,
    ) -> None:
        """Initialize probability calculator.

        Args:
            topics: List of topic names
            embedding_provider: Provider for embeddings (Jina or SentenceTransformers)
            keybert_model: Model name for KeyBERT keyword extraction
            calculation_mode: Calculation mode (st_baseline, jina_mixed, jina_bag_of_words, jina_full_text)
            vector_store: Optional PostgreSQL vector store for embedding caching
            use_keybert: Whether to use KeyBERT keyword extraction (False = direct cosine only)
        """
        self.keybert_calc = KeyBERTSimilarityCalculator(
            topics=topics,
            embedding_provider=embedding_provider,
            keybert_model=keybert_model,
            calculation_mode=calculation_mode,
            vector_store=vector_store,
            use_keybert=use_keybert,
        )

    def calculate_probability(
        self,
        original_text: str,
        cleaned_word_list: list[str],
    ) -> list[float]:
        """Calculate topic probabilities.

        Args:
            original_text: Original text with full context (for embeddings)
            cleaned_word_list: Preprocessed word list (for frequency counting)

        Returns:
            List of probabilities for each topic
        """
        return self.keybert_calc.calculate_probability(
            original_text=original_text,
            cleaned_word_list=cleaned_word_list,
        )

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.keybert_calc.clear_cache()

    def cache_size(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache sizes
        """
        return self.keybert_calc.cache_size()

    def predict_topic(self, text: str, cleaned_words: Optional[list[str]] = None) -> str:
        """Predict the most likely topic for given text.

        Args:
            text: Original text to classify
            cleaned_words: Optional preprocessed word list (if not provided, uses empty list)

        Returns:
            Predicted topic name
        """
        # If no cleaned words provided, use empty list (relies on original text for embeddings)
        if cleaned_words is None:
            cleaned_words = []

        # Calculate probabilities for all topics
        probabilities = self.calculate_probability(text, cleaned_words)

        # Get topic with highest probability
        topics = self.keybert_calc.topics
        max_idx = probabilities.index(max(probabilities))

        return topics[max_idx]
