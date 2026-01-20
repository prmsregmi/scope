"""Result output and formatting."""

from pathlib import Path
from typing import Optional

import pandas as pd


class ResultWriter:
    """Write analysis results to files."""

    def __init__(self, output_path: str) -> None:
        """Initialize result writer.

        Args:
            output_path: Path to output CSV file
        """
        self.output_path = Path(output_path)

    def write(self, segments: list[dict]) -> None:
        """Write segments to CSV file.

        Args:
            segments: List of segment dictionaries
        """
        if not segments:
            raise ValueError("No segments to write")

        # Create DataFrame and write to CSV
        df = pd.DataFrame(segments)

        # Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to CSV
        df.to_csv(self.output_path, index=False)

    def write_summary(
        self,
        segments: list[dict],
        wall_time: float,
        threshold: float,
    ) -> None:
        """Write summary statistics to a separate file.

        Args:
            segments: List of segment dictionaries
            wall_time: Total processing time in seconds
            threshold: Probability threshold used
        """
        if not segments:
            return

        df = pd.DataFrame(segments)

        # Calculate statistics
        num_segments = len(segments)

        # Calculate average length (number of messages)
        total_messages = sum(seg["Chat Summary"].count("\n") + 1 for seg in segments)
        avg_length = total_messages / max(1, num_segments)

        # Calculate average relevance score
        avg_relevance = sum(float(seg["Probability"]) for seg in segments) / max(
            1, num_segments
        )

        # Calculate topic distribution
        topic_counts = df["Topic"].value_counts().to_dict()

        # Create summary text
        summary = []
        summary.append("=" * 60)
        summary.append("SCOPE Analysis Summary")
        summary.append("=" * 60)
        summary.append("")
        summary.append(f"Probability Threshold: {threshold}")
        summary.append(f"Wall Time: {wall_time:.2f} seconds")
        summary.append("")
        summary.append(f"Number of Extracted Segments: {num_segments}")
        summary.append(f"Average Length of Segments: {avg_length:.2f} messages")
        summary.append(f"Total Messages Captured: {total_messages}")
        summary.append(f"Average Topic Relevance Score: {avg_relevance:.4f}")
        summary.append("")
        summary.append("Topic Distribution:")
        for topic, count in sorted(
            topic_counts.items(), key=lambda x: x[1], reverse=True
        ):
            summary.append(f"  {topic}: {count} segments")
        summary.append("")
        summary.append("=" * 60)

        # Write summary to file
        summary_path = self.output_path.with_suffix(".summary.txt")
        with open(summary_path, "w") as f:
            f.write("\n".join(summary))

        return summary_path
