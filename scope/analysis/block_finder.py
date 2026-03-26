"""Contiguous block detection algorithm."""

import copy
from typing import Optional

import numpy as np
from tqdm import tqdm

from scope.modeling.probability import ProbabilityCalculator


class ContiguousBlockFinder:
    """Find contiguous hourly blocks exceeding probability threshold."""

    def __init__(
        self,
        probability_threshold: float,
        probability_calculator: ProbabilityCalculator,
        prefilter_sim_threshold: float = 0.0,
    ) -> None:
        """Initialize block finder.

        Args:
            probability_threshold: Minimum probability for a block to be considered relevant
            probability_calculator: Calculator for topic probabilities
            prefilter_sim_threshold: Cosine similarity threshold for the HNSW-style
                embedding pre-filter. 0.0 disables it. When > 0, all unique non-empty
                hour texts are batch-encoded once upfront, then a full (hours × topics)
                cosine similarity matrix is computed via a single numpy matrix multiply.
                Hours whose similarity to a given topic falls below this threshold skip
                the expensive KeyBERT probability calculation entirely.
                Recommended range: 0.10–0.20. Higher = faster but may miss borderline blocks.
        """
        self.probability_threshold = probability_threshold
        self.prob_calc = probability_calculator
        self.prefilter_sim_threshold = prefilter_sim_threshold
        # Pre-computed similarity matrix populated by _build_prefilter_index
        # Shape: (num_unique_hour_texts, num_topics)
        self._sim_matrix: Optional[np.ndarray] = None
        # Mapping: original_text → row index in _sim_matrix
        self._text_to_row: dict[str, int] = {}
        # Stats
        self.prefilter_skipped_hours = 0
        self.prefilter_skipped_extensions = 0
        self.prefilter_total_hours = 0
        self.prefilter_total_extensions = 0

    def _build_prefilter_index(
        self,
        user_hourly_data: dict[str, list[list[str]]],
        user_hourly_original: dict[str, list[str]],
        num_topics: int,
    ) -> None:
        """Batch-encode all unique non-empty hour texts and compute a cosine
        similarity matrix against all topic embeddings.

        This is done ONCE before block-finding begins. The result is a
        (num_unique_texts × num_topics) matrix that gives every (hour, topic)
        similarity with a single numpy matrix multiply — the vectorized
        equivalent of HNSW approximate nearest-neighbor search over the topic set.

        During block-finding, prefilter checks are O(1) table lookups with no
        additional inference calls.
        """
        provider = self.prob_calc.keybert_calc.embedding_provider

        # Collect all unique non-empty hour original texts
        unique_texts = []
        for user, hourly_texts in user_hourly_data.items():
            orig = user_hourly_original[user]
            for h in range(len(hourly_texts)):
                if hourly_texts[h]:
                    text = orig[h] if orig[h] else ""
                    if text and text not in self._text_to_row:
                        self._text_to_row[text] = len(unique_texts)
                        unique_texts.append(text)

        if not unique_texts:
            return

        tqdm.write(f"[prefilter] batch-encoding {len(unique_texts)} unique hour texts...")

        # Single batched inference call — much faster than N individual encode() calls
        hour_embs = provider.encode_batch(unique_texts, batch_size=64)  # (H, D)

        # Normalize rows to unit vectors for cosine similarity
        norms = np.linalg.norm(hour_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        hour_embs_norm = hour_embs / norms  # (H, D)

        # In hybrid mode, use primary (Jina) topic embeddings for prefilter similarity
        primary = self.prob_calc.keybert_calc._topic_embeddings_primary
        topic_emb_source = primary if primary is not None else self.prob_calc.keybert_calc._topic_embeddings
        topic_embs = np.array(topic_emb_source)  # (T, D)
        t_norms = np.linalg.norm(topic_embs, axis=1, keepdims=True)
        t_norms = np.where(t_norms == 0, 1.0, t_norms)
        topic_embs_norm = topic_embs / t_norms  # (T, D)

        # Single matrix multiply: (H, D) @ (D, T) → (H, T) cosine similarities
        self._sim_matrix = hour_embs_norm @ topic_embs_norm.T  # (H, T)

        tqdm.write(
            f"[prefilter] similarity matrix built: {self._sim_matrix.shape} | "
            f"threshold={self.prefilter_sim_threshold:.2f}"
        )

    def _sim_lookup(self, text: str, topic_idx: int) -> float:
        """O(1) similarity lookup from the pre-computed matrix."""
        row = self._text_to_row.get(text)
        if row is None or self._sim_matrix is None:
            return 1.0  # Unknown text: don't filter
        return float(self._sim_matrix[row, topic_idx])

    def _vector_to_word_list(
        self,
        hour_indices: list[int],
        hourly_texts: list[list[str]],
    ) -> list[str]:
        """Convert hour indices to combined word list.

        Args:
            hour_indices: List of hour indices
            hourly_texts: List of word lists for each hour

        Returns:
            Combined word list
        """
        word_list = []
        for idx in hour_indices:
            word_list.extend(hourly_texts[idx])
        return word_list

    def _vector_to_original_text(
        self,
        hour_indices: list[int],
        hourly_original_texts: list[str],
    ) -> str:
        """Convert hour indices to combined original text.

        Args:
            hour_indices: List of hour indices
            hourly_original_texts: List of original text strings for each hour

        Returns:
            Combined original text
        """
        texts = []
        for idx in hour_indices:
            if hourly_original_texts[idx]:
                texts.append(hourly_original_texts[idx])
        return " ".join(texts)

    def find_blocks_for_user(
        self,
        hourly_texts: list[list[str]],
        hourly_original_texts: list[str],
        target_topic_idx: int,
    ) -> list[list[int]]:
        """Find contiguous blocks for a single user and topic using greedy algorithm.

        When prefilter_sim_threshold > 0, a fast embedding similarity check gates each
        expensive KeyBERT probability call. Two gates are applied:
          1. Initial PP gate: skip hours whose direct cosine similarity to the topic
             is below the threshold (avoids KeyBERT on clearly irrelevant hours).
          2. Extension gate: reject block extension attempts whose combined-text
             cosine similarity falls below the threshold (avoids KeyBERT on growing
             multi-hour combined texts that are very unlikely to pass).

        Args:
            hourly_texts: List of cleaned word lists, one per hour
            hourly_original_texts: List of original text strings, one per hour
            target_topic_idx: Index of the target topic

        Returns:
            List of hour index sequences representing contiguous blocks
        """
        topic_blocks = []
        use_prefilter = self.prefilter_sim_threshold > 0.0

        # Identify hours with non-empty text
        Set = []
        PP = []

        for h in range(len(hourly_texts)):
            if len(hourly_texts[h]) == 0:
                continue

            Set.append(h)

            original_text = self._vector_to_original_text([h], hourly_original_texts)
            word_list = self._vector_to_word_list([h], hourly_texts)

            # Gate 1: O(1) similarity table lookup — skip full KeyBERT for low-sim hours
            if use_prefilter:
                self.prefilter_total_hours += 1
                if self._sim_lookup(original_text, target_topic_idx) < self.prefilter_sim_threshold:
                    self.prefilter_skipped_hours += 1
                    continue  # Skip full KeyBERT for this hour

            prob = self.prob_calc.calculate_probability(original_text, word_list)[
                target_topic_idx
            ]

            if prob >= self.probability_threshold:
                PP.append(h)

        # Greedy algorithm to build contiguous blocks
        curr = []

        for h in Set:
            if h in PP:
                if not curr:
                    curr.append(h)
                else:
                    temp = copy.deepcopy(curr)
                    temp.append(h)

                    combined_original = self._vector_to_original_text(temp, hourly_original_texts)
                    combined_words = self._vector_to_word_list(temp, hourly_texts)

                    # Gate 2: for combined-block texts (not in precomputed index),
                    # use the min per-hour similarity as a proxy to avoid a full encode.
                    # If even the best constituent hour is below threshold, don't extend.
                    if use_prefilter:
                        self.prefilter_total_extensions += 1
                        min_sim = min(
                            self._sim_lookup(
                                self._vector_to_original_text([hi], hourly_original_texts),
                                target_topic_idx,
                            )
                            for hi in temp
                        )
                        if min_sim < self.prefilter_sim_threshold:
                            self.prefilter_skipped_extensions += 1
                            topic_blocks.append(copy.deepcopy(curr))
                            curr = [h]
                            continue

                    prob = self.prob_calc.calculate_probability(combined_original, combined_words)[
                        target_topic_idx
                    ]

                    if prob >= self.probability_threshold:
                        curr = copy.deepcopy(temp)
                    else:
                        topic_blocks.append(copy.deepcopy(curr))
                        curr = [h]

            elif curr:
                topic_blocks.append(copy.deepcopy(curr))
                curr = []

        if curr:
            topic_blocks.append(copy.deepcopy(curr))

        return topic_blocks

    def find_all_blocks(
        self,
        user_hourly_data: dict[str, list[list[str]]],
        user_hourly_original: dict[str, list[str]],
        topics: list[str],
    ) -> dict[str, dict[str, list[list[int]]]]:
        """Find blocks for all users and topics.

        Args:
            user_hourly_data: Dictionary mapping user to hourly cleaned word lists
            user_hourly_original: Dictionary mapping user to hourly original texts
            topics: List of topic names

        Returns:
            Nested dictionary: {user: {topic: [block1, block2, ...]}}
        """
        # If pre-filter is enabled, batch-encode all unique hour texts once upfront
        # and build the (hours × topics) cosine similarity matrix before the main loop.
        if self.prefilter_sim_threshold > 0.0:
            self._build_prefilter_index(user_hourly_data, user_hourly_original, len(topics))

        results = {}
        total_operations = len(user_hourly_data) * len(topics)

        with tqdm(total=total_operations, desc="Finding contiguous blocks", unit="user-topic") as pbar:
            for user, hourly_texts in user_hourly_data.items():
                hourly_original_texts = user_hourly_original[user]
                user_results = {}

                for topic_idx, topic_name in enumerate(topics):
                    blocks = self.find_blocks_for_user(
                        hourly_texts,
                        hourly_original_texts,
                        topic_idx,
                    )
                    user_results[topic_name] = blocks
                    pbar.update(1)

                results[user] = user_results

        if self.prefilter_sim_threshold > 0.0:
            hr_skip_pct = 100 * self.prefilter_skipped_hours / max(self.prefilter_total_hours, 1)
            ext_skip_pct = 100 * self.prefilter_skipped_extensions / max(self.prefilter_total_extensions, 1)
            tqdm.write(
                f"[prefilter] sim_threshold={self.prefilter_sim_threshold:.2f} | "
                f"hours skipped: {self.prefilter_skipped_hours}/{self.prefilter_total_hours} ({hr_skip_pct:.1f}%) | "
                f"extensions skipped: {self.prefilter_skipped_extensions}/{self.prefilter_total_extensions} ({ext_skip_pct:.1f}%)"
            )

        return results
