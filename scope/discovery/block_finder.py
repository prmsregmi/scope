"""Convert cluster assignments into contiguous hour blocks."""

from collections import defaultdict

import numpy as np

from scope.discovery.clusterer import ClusterResult


class ClusterBlockFinder:
    """Find contiguous hour blocks from cluster assignments.

    Converts per-hour cluster labels into the same {user: {topic: [[hours]]}}
    structure that the supervised ContiguousBlockFinder produces.
    """

    def __init__(self) -> None:
        # Lookup built after find_all_blocks: {(user, hour_idx): probability}
        self._prob_lookup: dict[tuple[str, int], float] = {}

    def find_all_blocks(
        self,
        cluster_result: ClusterResult,
        cluster_labels: dict[int, str],
    ) -> dict[str, dict[str, list[list[int]]]]:
        """Build contiguous blocks from cluster assignments.

        For each user, scans hours left-to-right. Extends the current block
        while the cluster label matches. Starts a new block on label change.
        Noise points (label == -1) are skipped.

        Args:
            cluster_result: Output from TopicDiscoverer.fit()
            cluster_labels: {cluster_id: label_string} from ClusterLabeler

        Returns:
            {user: {topic_label: [[hour_indices], ...]}}
        """
        # Build probability lookup for O(1) access
        self._prob_lookup = {}
        for i in range(len(cluster_result.users)):
            self._prob_lookup[(cluster_result.users[i], cluster_result.hour_indices[i])] = (
                float(cluster_result.probabilities[i])
            )

        # Group (hour_idx, cluster_label) by user
        user_hours: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i in range(len(cluster_result.users)):
            label = int(cluster_result.labels[i])
            if label == -1:
                continue
            user_hours[cluster_result.users[i]].append(
                (cluster_result.hour_indices[i], label)
            )

        results: dict[str, dict[str, list[list[int]]]] = {}

        for user, hour_label_pairs in user_hours.items():
            # Sort by hour index
            hour_label_pairs.sort(key=lambda x: x[0])

            # Group into contiguous runs of the same label
            topic_blocks: dict[str, list[list[int]]] = defaultdict(list)
            current_block: list[int] = []
            current_label: int | None = None

            for hour_idx, label in hour_label_pairs:
                label_str = cluster_labels.get(label, f"cluster_{label}")

                if current_label is None:
                    current_label = label
                    current_block = [hour_idx]
                elif label == current_label and hour_idx == current_block[-1] + 1:
                    current_block.append(hour_idx)
                else:
                    # Flush current block
                    prev_label_str = cluster_labels.get(current_label, f"cluster_{current_label}")
                    topic_blocks[prev_label_str].append(current_block)
                    current_label = label
                    current_block = [hour_idx]

            # Flush last block
            if current_block and current_label is not None:
                label_str = cluster_labels.get(current_label, f"cluster_{current_label}")
                topic_blocks[label_str].append(current_block)

            results[user] = dict(topic_blocks)

        return results

    def get_block_probabilities(
        self,
        cluster_result: ClusterResult,
        user: str,
        block: list[int],
    ) -> float:
        """Get average cluster membership probability for a block's hours.

        Uses O(1) lookups from the pre-built probability index.
        """
        probs = [
            self._prob_lookup[(user, h)]
            for h in block
            if (user, h) in self._prob_lookup
        ]
        return float(np.mean(probs)) if probs else 0.0
