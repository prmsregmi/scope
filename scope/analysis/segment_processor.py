"""Process and format detected segments."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

if TYPE_CHECKING:
    from scope.discovery.block_finder import ClusterBlockFinder
    from scope.discovery.clusterer import ClusterResult
    from scope.modeling.probability import ProbabilityCalculator


class SegmentProcessor:
    """Process hour-based segments into formatted output."""

    def __init__(self, date_list: list[str]) -> None:
        """Initialize segment processor.

        Args:
            date_list: List of dates in 'YYYY-MM-DD' format
        """
        self.date_list = date_list

    def process_segments(
        self,
        user_blocks: dict[str, dict[str, list[list[int]]]],
        user_hourly_data: dict[str, list[list[str]]],
        user_hourly_original: dict[str, list[str]],
        user_full_messages: dict[str, list[list[Any]]],
        prob_calc: ProbabilityCalculator,
    ) -> list[dict]:
        """Process all user blocks into output format.

        Args:
            user_blocks: User/topic/blocks structure from BlockFinder
            user_hourly_data: Hourly cleaned word lists per user
            user_hourly_original: Hourly original text per user
            user_full_messages: Full message data per user (with timestamps)
            prob_calc: Probability calculator for final probabilities

        Returns:
            List of segment dictionaries ready for output
        """
        all_segments = []

        # Count total blocks for progress bar
        total_blocks = sum(
            len(blocks)
            for topic_blocks in user_blocks.values()
            for blocks in topic_blocks.values()
        )

        with tqdm(total=total_blocks, desc="Processing segments", unit="segment") as pbar:
            for user, topic_blocks in user_blocks.items():
                for topic, blocks in topic_blocks.items():
                    for block in blocks:
                        pbar.update(1)
                        if not block:
                            continue

                        # Get time range
                        start_date, start_time, end_date, end_time, duration = (
                            self._calculate_time_range(
                                block, user, user_full_messages
                            )
                        )

                        # Calculate probability for this segment
                        original_text = self._get_original_text_for_block(
                            block, user, user_hourly_original
                        )
                        word_list = self._get_word_list_for_block(
                            block, user, user_hourly_data
                        )
                        probabilities = prob_calc.calculate_probability(
                            original_text, word_list
                        )
                        topic_idx = list(topic_blocks.keys()).index(topic)
                        probability = probabilities[topic_idx]

                        # Aggregate chat messages
                        chat_summary = self._aggregate_messages(
                            block, user, user_full_messages
                        )

                        # Create segment record
                        segment = {
                            "User": user,
                            "Start Date": start_date,
                            "Start Time": start_time,
                            "End Date": end_date,
                            "End Time": end_time,
                            "Time Duration": str(duration),
                            "Topic": topic,
                            "Probability": str(probability),
                            "Chat Summary": chat_summary,
                        }

                        all_segments.append(segment)

        return all_segments

    def _calculate_time_range(
        self,
        block: list[int],
        user: str,
        user_full_messages: dict[str, list[list[Any]]],
    ) -> tuple[str, str, str, str, timedelta]:
        """Calculate start/end times and duration for a block.

        Args:
            block: List of hour indices
            user: User identifier
            user_full_messages: Full message data

        Returns:
            Tuple of (start_date, start_time, end_date, end_time, duration)
        """
        first_hour = block[0]
        last_hour = block[-1]

        # Get start time (earliest message in first hour)
        start_date = self.date_list[first_hour // 24]
        start_time = self._get_earliest_time(
            first_hour, user, user_full_messages
        )

        # Get end time (latest message in last hour)
        end_date = self.date_list[last_hour // 24]
        end_time = self._get_latest_time(
            last_hour, user, user_full_messages
        )

        # Calculate duration
        start_datetime = datetime.strptime(
            f"{start_date} {start_time}", "%Y-%m-%d %H:%M:%S"
        )
        end_datetime = datetime.strptime(
            f"{end_date} {end_time}", "%Y-%m-%d %H:%M:%S"
        )
        duration = end_datetime - start_datetime

        return start_date, start_time, end_date, end_time, duration

    def _get_earliest_time(
        self,
        hour_idx: int,
        user: str,
        user_full_messages: dict[str, list[list[Any]]],
    ) -> str:
        """Get earliest time in an hour for a user.

        Args:
            hour_idx: Hour index
            user: User identifier
            user_full_messages: Full message data

        Returns:
            Time string in 'HH:MM:SS' format
        """
        messages = user_full_messages[user][hour_idx]
        target_date = self.date_list[hour_idx // 24]
        target_hour = hour_idx % 24

        earliest = "23:59:59"

        for msg in messages:
            msg_date, msg_time = msg[2], msg[3]
            msg_hour = datetime.strptime(msg_time, "%H:%M:%S").hour

            if msg_date == target_date and msg_hour == target_hour:
                if datetime.strptime(msg_time, "%H:%M:%S") < datetime.strptime(
                    earliest, "%H:%M:%S"
                ):
                    earliest = msg_time

        return earliest

    def _get_latest_time(
        self,
        hour_idx: int,
        user: str,
        user_full_messages: dict[str, list[list[Any]]],
    ) -> str:
        """Get latest time in an hour for a user.

        Args:
            hour_idx: Hour index
            user: User identifier
            user_full_messages: Full message data

        Returns:
            Time string in 'HH:MM:SS' format
        """
        messages = user_full_messages[user][hour_idx]
        target_date = self.date_list[hour_idx // 24]
        target_hour = hour_idx % 24

        latest = "00:00:00"

        for msg in messages:
            msg_date, msg_time = msg[2], msg[3]
            msg_hour = datetime.strptime(msg_time, "%H:%M:%S").hour

            if msg_date == target_date and msg_hour == target_hour:
                if datetime.strptime(msg_time, "%H:%M:%S") > datetime.strptime(
                    latest, "%H:%M:%S"
                ):
                    latest = msg_time

        return latest

    def _get_word_list_for_block(
        self,
        block: list[int],
        user: str,
        user_hourly_data: dict[str, list[list[str]]],
    ) -> list[str]:
        """Get combined word list for a block.

        Args:
            block: List of hour indices
            user: User identifier
            user_hourly_data: Hourly text data

        Returns:
            Combined word list
        """
        word_list = []
        for hour_idx in block:
            word_list.extend(user_hourly_data[user][hour_idx])
        return word_list

    def _get_original_text_for_block(
        self,
        block: list[int],
        user: str,
        user_hourly_original: dict[str, list[str]],
    ) -> str:
        """Get combined original text for a block.

        Args:
            block: List of hour indices
            user: User identifier
            user_hourly_original: Hourly original text data

        Returns:
            Combined original text
        """
        texts = []
        for hour_idx in block:
            if user_hourly_original[user][hour_idx]:
                texts.append(user_hourly_original[user][hour_idx])
        return " ".join(texts)

    def _aggregate_messages(
        self,
        block: list[int],
        user: str,
        user_full_messages: dict[str, list[list[Any]]],
    ) -> str:
        """Aggregate all messages in a block into a summary string.

        Args:
            block: List of hour indices
            user: User identifier
            user_full_messages: Full message data

        Returns:
            Concatenated chat messages
        """
        chats = []

        for hour_idx in block:
            messages = user_full_messages[user][hour_idx]

            for msg in messages:
                chatroom, sender, date, time, text, prompt = msg
                chat_line = f"{user} :: {chatroom} [{date} {time}]: {text}"
                chats.append(chat_line)

        return "\n".join(chats)

    def process_cluster_segments(
        self,
        user_blocks: dict[str, dict[str, list[list[int]]]],
        user_full_messages: dict[str, list[list[Any]]],
        cluster_result: ClusterResult,
        cluster_block_finder: ClusterBlockFinder,
    ) -> list[dict]:
        """Process cluster-based blocks into output format.

        Same output schema as process_segments(), but uses cluster membership
        probability instead of ProbabilityCalculator.
        """
        all_segments = []

        total_blocks = sum(
            len(blocks)
            for topic_blocks in user_blocks.values()
            for blocks in topic_blocks.values()
        )

        with tqdm(total=total_blocks, desc="Processing segments", unit="segment") as pbar:
            for user, topic_blocks in user_blocks.items():
                for topic, blocks in topic_blocks.items():
                    for block in blocks:
                        pbar.update(1)
                        if not block:
                            continue

                        start_date, start_time, end_date, end_time, duration = (
                            self._calculate_time_range(
                                block, user, user_full_messages
                            )
                        )

                        probability = cluster_block_finder.get_block_probabilities(
                            cluster_result, user, block
                        )

                        chat_summary = self._aggregate_messages(
                            block, user, user_full_messages
                        )

                        segment = {
                            "User": user,
                            "Start Date": start_date,
                            "Start Time": start_time,
                            "End Date": end_date,
                            "End Time": end_time,
                            "Time Duration": str(duration),
                            "Topic": topic,
                            "Probability": str(probability),
                            "Chat Summary": chat_summary,
                        }

                        all_segments.append(segment)

        return all_segments
