"""Contiguous block detection algorithm."""

import copy
from typing import Optional

from tqdm import tqdm

from scope.modeling.probability import ProbabilityCalculator


class ContiguousBlockFinder:
    """Find contiguous hourly blocks exceeding probability threshold."""

    def __init__(
        self,
        probability_threshold: float,
        probability_calculator: ProbabilityCalculator,
    ) -> None:
        """Initialize block finder.

        Args:
            probability_threshold: Minimum probability for a block to be considered relevant
            probability_calculator: Calculator for topic probabilities
        """
        self.probability_threshold = probability_threshold
        self.prob_calc = probability_calculator

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

        Args:
            hourly_texts: List of cleaned word lists, one per hour
            hourly_original_texts: List of original text strings, one per hour
            target_topic_idx: Index of the target topic

        Returns:
            List of hour index sequences representing contiguous blocks
        """
        topic_blocks = []

        # Identify hours with non-empty text
        Set = []
        PP = []

        for h in range(len(hourly_texts)):
            if len(hourly_texts[h]) == 0:
                continue

            Set.append(h)

            # Check if this hour meets threshold
            original_text = self._vector_to_original_text([h], hourly_original_texts)
            word_list = self._vector_to_word_list([h], hourly_texts)
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
                    # Start new block
                    curr.append(h)
                else:
                    # Try to extend current block
                    temp = copy.deepcopy(curr)
                    temp.append(h)

                    # Check if extended block still meets threshold
                    original_text = self._vector_to_original_text(temp, hourly_original_texts)
                    word_list = self._vector_to_word_list(temp, hourly_texts)
                    prob = self.prob_calc.calculate_probability(original_text, word_list)[
                        target_topic_idx
                    ]

                    if prob >= self.probability_threshold:
                        # Extend block
                        curr = copy.deepcopy(temp)
                    else:
                        # Close current block and start new one
                        topic_blocks.append(copy.deepcopy(curr))
                        curr = [h]

            elif curr:
                # Close current block when we hit a non-qualifying hour
                topic_blocks.append(copy.deepcopy(curr))
                curr = []

        # Don't forget the last block
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

        return results
