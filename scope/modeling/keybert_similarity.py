"""KeyBERT-based keyword extraction and similarity calculation."""

import logging
import math
import time
from typing import Optional

import numpy as np
from keybert import KeyBERT

from scope.embeddings import EmbeddingProvider


class KeyBERTSimilarityCalculator:
    """Calculate topic probabilities using KeyBERT keyword extraction and cosine similarity.

    This implements the Hybrid Cosine-KeyBERT approach from the paper, with the critical
    improvement of using original (non-preprocessed) text for embeddings.

    Supports 4 calculation modes:
    1. 'st_baseline' - SentenceTransformers (original approach)
    2. 'jina_mixed' - KeyBERT relevance (ST), similarity (JINA) - current default
    3. 'jina_bag_of_words' - Both scores from JINA, document = bag of words
    4. 'jina_full_text' - Both scores from JINA, document = original text
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
        """Initialize KeyBERT similarity calculator.

        Args:
            topics: List of topic names to compare against
            embedding_provider: Provider for generating embeddings (can be Jina or SentenceTransformers)
            keybert_model: Model name for KeyBERT keyword extraction (uses SentenceTransformers)
            calculation_mode: Mode for probability calculation
                - 'st_baseline': Use ST for everything (original)
                - 'jina_mixed': Use KeyBERT relevance (ST) + similarity (JINA)
                - 'jina_bag_of_words': Use JINA for both, document = bag of words
                - 'jina_full_text': Use JINA for both, document = original text
            vector_store: Optional PostgreSQL vector store for embedding caching and fast similarity
        """
        self.topics = topics
        self.embedding_provider = embedding_provider
        self.calculation_mode = calculation_mode
        self.vector_store = vector_store
        self.use_keybert = use_keybert

        # Initialize KeyBERT for keyword extraction (skip if disabled)
        if self.use_keybert:
            self.keybert = KeyBERT(keybert_model)
        else:
            self.keybert = None

        # Cache for embeddings (in-memory, used when vector_store is not available)
        self._embedding_cache: dict[str, list[float]] = {}

        # Cache for topic embeddings (computed once)
        self._topic_embeddings: Optional[list[list[float]]] = None

        # Cache for probability calculations
        self._probability_cache: dict[tuple, list[float]] = {}

        # Timing instrumentation
        self._time_keybert_extract: float = 0.0
        self._time_similarity: float = 0.0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Pre-compute topic embeddings
        self._compute_topic_embeddings()

    def _compute_topic_embeddings(self) -> None:
        """Pre-compute embeddings for all topics."""
        self._topic_embeddings = []

        # If vector store is available, check if topics are already stored
        if self.vector_store:
            stored_topics = self.vector_store.get_topic_embeddings()

            # Check if all topics are already in the database
            if all(topic in stored_topics for topic in self.topics):
                # Use stored embeddings
                for topic in self.topics:
                    self._topic_embeddings.append(stored_topics[topic].tolist())
                return

            # Otherwise, compute and store new embeddings
            topic_embeddings_dict = {}
            for topic in self.topics:
                embedding = self.embedding_provider.encode(topic)
                self._topic_embeddings.append(embedding)
                # Convert to numpy for storage
                topic_embeddings_dict[topic] = np.array(embedding)

            # Store in PostgreSQL
            self.vector_store.store_topic_embeddings(topic_embeddings_dict)
        else:
            # No vector store, compute normally
            for topic in self.topics:
                embedding = self.embedding_provider.encode(topic)
                self._topic_embeddings.append(embedding)

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text with caching.

        Checks PostgreSQL first if available, then in-memory cache, then computes.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check in-memory cache first (fastest)
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        # Check PostgreSQL if available
        if self.vector_store:
            stored = self.vector_store.get_keyword_embeddings([text])
            if stored.get(text) is not None:
                embedding = stored[text].tolist()
                self._embedding_cache[text] = embedding
                return embedding

        # Compute embedding if not cached
        embedding = self.embedding_provider.encode(text)
        self._embedding_cache[text] = embedding

        # Store in PostgreSQL if available
        if self.vector_store:
            self.vector_store.store_keyword_embeddings({text: np.array(embedding)})

        return embedding

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        import numpy as np

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def calculate_probability(
        self,
        original_text: str,
        cleaned_word_list: list[str],
    ) -> list[float]:
        """Calculate topic probabilities using KeyBERT + Cosine similarity.

        This is the core implementation of the Hybrid Cosine-KeyBERT approach.

        Args:
            original_text: Original text with full context (NOT preprocessed)
            cleaned_word_list: Preprocessed word list (for frequency counting)

        Returns:
            List of probabilities for each topic (sums to 1.0)
        """
        # Create cache key (include mode to avoid cache collision)
        mode_key = self.calculation_mode if self.use_keybert else "no_keybert"
        cache_key = (mode_key, original_text, tuple(cleaned_word_list))
        if cache_key in self._probability_cache:
            self._cache_hits += 1
            return self._probability_cache[cache_key]
        self._cache_misses += 1

        # Direct cosine similarity mode (no KeyBERT)
        if not self.use_keybert:
            return self._calculate_direct_cosine(original_text, cache_key)

        # Extract keywords from ORIGINAL text (not preprocessed)
        # This preserves context for better keyword extraction
        try:
            t0 = time.perf_counter()
            keyword_ranks = self.keybert.extract_keywords(
                original_text,
                top_n=min(len(cleaned_word_list), 20),  # Extract up to 20 keywords
                stop_words='english',
            )
            self._time_keybert_extract += time.perf_counter() - t0
        except Exception:
            # If KeyBERT fails (e.g., empty text), return uniform distribution
            uniform_prob = 1.0 / len(self.topics)
            result = [uniform_prob] * len(self.topics)
            self._probability_cache[cache_key] = result
            return result

        # If no keywords extracted, return uniform distribution
        if not keyword_ranks:
            uniform_prob = 1.0 / len(self.topics)
            result = [uniform_prob] * len(self.topics)
            self._probability_cache[cache_key] = result
            return result

        # Calculate document embedding if needed (for JINA modes 3 & 4)
        document_embedding = None
        if self.calculation_mode in ['jina_bag_of_words', 'jina_full_text']:
            if self.calculation_mode == 'jina_bag_of_words':
                # Use bag of words (cleaned text joined)
                document_text = ' '.join(cleaned_word_list)
            else:  # jina_full_text
                # Use original full text
                document_text = original_text

            document_embedding = self._get_embedding(document_text)

        # Calculate similarity scores for each topic
        t1 = time.perf_counter()
        prob_list = [0.0 for _ in range(len(self.topics))]

        # Extract keyword texts for batch processing
        keywords_list = [keyword for keyword, _ in keyword_ranks]

        # Ensure all keyword embeddings are computed/cached
        for keyword in keywords_list:
            self._get_embedding(keyword)

        # Use PostgreSQL for batch similarity calculation if available
        if self.vector_store:
            try:
                # Get all keyword-topic similarities in one query
                similarities = self.vector_store.calculate_similarities(
                    keywords_list,
                    self.topics
                )

                for keyword, keybert_relevance in keyword_ranks:
                    # Calculate word frequency from cleaned word list
                    word_frequency = cleaned_word_list.count(keyword.lower())

                    # Determine keyword-document relevance based on mode
                    if self.calculation_mode in ['st_baseline', 'jina_mixed']:
                        # Use KeyBERT's relevance score
                        keyword_doc_relevance = keybert_relevance
                    else:  # jina_bag_of_words or jina_full_text
                        # Calculate relevance in JINA embedding space
                        keyword_embedding = self._get_embedding(keyword)
                        keyword_doc_relevance = self._cosine_similarity(
                            keyword_embedding,
                            document_embedding
                        )

                    # Use pre-computed similarities from PostgreSQL
                    for topic_idx, topic in enumerate(self.topics):
                        keyword_topic_similarity = similarities.get((keyword, topic), 0.0)

                        # Weight by relevance and frequency
                        weighted_score = keyword_topic_similarity * keyword_doc_relevance * word_frequency
                        prob_list[topic_idx] += weighted_score

            except Exception as e:
                # Fall back to in-memory calculation if PostgreSQL fails
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"PostgreSQL similarity calculation failed, using in-memory: {e}")
                prob_list = self._calculate_similarity_in_memory(
                    keyword_ranks,
                    cleaned_word_list,
                    document_embedding
                )
        else:
            # No vector store, use in-memory calculation
            prob_list = self._calculate_similarity_in_memory(
                keyword_ranks,
                cleaned_word_list,
                document_embedding
            )

        self._time_similarity += time.perf_counter() - t1

        # Apply softmax normalization to convert to probabilities
        prob_list = self._softmax(prob_list)

        # Cache result
        self._probability_cache[cache_key] = prob_list

        return prob_list

    def _calculate_direct_cosine(
        self,
        original_text: str,
        cache_key: tuple,
    ) -> list[float]:
        """Calculate topic probabilities using direct cosine similarity (no KeyBERT).

        Embeds the original text and computes cosine similarity against each topic embedding,
        then applies softmax normalization.
        """
        t1 = time.perf_counter()
        text_embedding = self._get_embedding(original_text)
        scores = [
            self._cosine_similarity(text_embedding, self._topic_embeddings[i])
            for i in range(len(self.topics))
        ]
        self._time_similarity += time.perf_counter() - t1
        result = self._softmax(scores)
        self._probability_cache[cache_key] = result
        return result

    def _calculate_similarity_in_memory(
        self,
        keyword_ranks: list[tuple[str, float]],
        cleaned_word_list: list[str],
        document_embedding: Optional[list[float]],
    ) -> list[float]:
        """Calculate similarities in-memory (original method).

        Args:
            keyword_ranks: List of (keyword, relevance_score) tuples
            cleaned_word_list: Cleaned words for frequency counting
            document_embedding: Document embedding (for modes 3 & 4)

        Returns:
            List of weighted scores for each topic
        """
        prob_list = [0.0 for _ in range(len(self.topics))]

        for keyword, keybert_relevance in keyword_ranks:
            # Get embedding for this keyword
            keyword_embedding = self._get_embedding(keyword)

            # Calculate word frequency from cleaned word list
            word_frequency = cleaned_word_list.count(keyword.lower())

            # Determine keyword-document relevance based on mode
            if self.calculation_mode in ['st_baseline', 'jina_mixed']:
                # Use KeyBERT's relevance score
                keyword_doc_relevance = keybert_relevance
            else:  # jina_bag_of_words or jina_full_text
                # Calculate relevance in JINA embedding space
                keyword_doc_relevance = self._cosine_similarity(
                    keyword_embedding,
                    document_embedding
                )

            # Calculate cosine similarity with each topic
            for topic_idx in range(len(self.topics)):
                topic_embedding = self._topic_embeddings[topic_idx]

                # Cosine similarity between keyword and topic
                keyword_topic_similarity = self._cosine_similarity(
                    keyword_embedding,
                    topic_embedding
                )

                # Weight by relevance and frequency
                weighted_score = keyword_topic_similarity * keyword_doc_relevance * word_frequency
                prob_list[topic_idx] += weighted_score

        return prob_list

    def _softmax(self, values: list[float]) -> list[float]:
        """Apply softmax normalization.

        Args:
            values: Raw values

        Returns:
            Normalized probabilities summing to 1.0
        """
        import numpy as np

        # Convert to numpy array
        arr = np.array(values)

        # Subtract max for numerical stability
        arr = arr - np.max(arr)

        # Calculate softmax
        exp_values = np.exp(arr)
        total = np.sum(exp_values)

        if total == 0:
            # Return uniform distribution if all values are very small
            return [1.0 / len(values)] * len(values)

        softmax_values = exp_values / total

        return softmax_values.tolist()

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._embedding_cache.clear()
        self._probability_cache.clear()

    def cache_size(self) -> dict[str, int]:
        """Get cache sizes.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "embedding_cache": len(self._embedding_cache),
            "probability_cache": len(self._probability_cache),
        }

    def log_timing_stats(self) -> None:
        """Log aggregate timing statistics for KeyBERT scoring."""
        logger = logging.getLogger(__name__)
        total = self._time_keybert_extract + self._time_similarity
        logger.info(
            f"[KeyBERT timing] extract_keywords: {self._time_keybert_extract:.2f}s | "
            f"similarity: {self._time_similarity:.2f}s | "
            f"total: {total:.2f}s | "
            f"cache hits: {self._cache_hits}, misses: {self._cache_misses}"
        )
