"""SCOPE evaluation framework."""

import json
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from scope.analysis import ContiguousBlockFinder, SegmentProcessor
from scope.config import ScopeConfig
from scope.embeddings import get_embedding_provider
from scope.io import DatasetLoader
from scope.modeling import ProbabilityCalculator
from scope.preprocessing import TextCleaner
from scope.utils import get_logger, setup_logging

from .metrics import EvaluationMetrics, PerformanceMetrics, QualityMetrics


class ScopeEvaluator:
    """Evaluator for SCOPE analysis with comprehensive metrics collection."""

    def __init__(self, output_dir: str = "results/evaluation"):
        """Initialize evaluator.

        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

    def evaluate(
        self,
        config: ScopeConfig,
        run_name: Optional[str] = None,
    ) -> EvaluationMetrics:
        """Run SCOPE analysis and collect evaluation metrics.

        Args:
            config: SCOPE configuration
            run_name: Optional name for this evaluation run

        Returns:
            Evaluation metrics
        """
        # Setup logging
        setup_logging(verbose=config.verbose)

        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{config.embedding_provider}_{timestamp}"

        self.logger.info(f"Starting evaluation: {run_name}")

        # Start memory tracking
        tracemalloc.start()

        # Track times
        start_time = time.time()
        times = {
            "preprocessing": 0.0,
            "embedding": 0.0,
            "analysis": 0.0,
        }

        try:
            # 1. Load dataset
            self.logger.info("Loading dataset...")
            loader = DatasetLoader(config.dataset_path)
            df = loader.load(config.start_date, config.end_date)

            # Get date range
            if config.start_date and config.end_date:
                start_date = config.start_date
                end_date = config.end_date
            else:
                start_date, end_date = loader.get_date_range(df)

            # Generate date list
            from datetime import datetime as dt, timedelta
            date_list = []
            current = dt.strptime(start_date, "%Y-%m-%d")
            end = dt.strptime(end_date, "%Y-%m-%d")
            while current <= end:
                date_list.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)

            # 2. Preprocess text
            self.logger.info("Preprocessing...")
            preprocess_start = time.time()

            text_cleaner = TextCleaner(
                enable_spell_check=config.enable_spell_check,
                enable_lemmatization=config.enable_lemmatization,
            )

            user_hourly_texts, user_hourly_original, user_hourly_messages = (
                self._organize_data_by_user_and_hour(df, text_cleaner, date_list)
            )

            times["preprocessing"] = time.time() - preprocess_start

            # 3. Get embedding provider
            self.logger.info(f"Initializing embedding provider: {config.embedding_provider}")
            embedding_start = time.time()

            embedding_provider = get_embedding_provider(
                config.embedding_provider,
                model=config.embedding_model,
                api_key=config.jina_api_key,
            )

            # 4. Initialize Probability Calculator
            # Check if calculation_mode is set (for mode comparison tests)
            calculation_mode = getattr(config, '_calculation_mode', 'jina_mixed')

            prob_calc = ProbabilityCalculator(
                topics=config.topics,
                embedding_provider=embedding_provider,
                keybert_model=config.keybert_model,
                calculation_mode=calculation_mode,
            )

            times["embedding"] = time.time() - embedding_start

            # 5. Find contiguous blocks and process segments
            self.logger.info("Running analysis...")
            analysis_start = time.time()

            block_finder = ContiguousBlockFinder(
                config.probability_threshold,
                prob_calc,
            )

            user_blocks = block_finder.find_all_blocks(
                user_hourly_texts,
                user_hourly_original,
                config.topics,
            )

            processor = SegmentProcessor(date_list)
            segments = processor.process_segments(
                user_blocks,
                user_hourly_texts,
                user_hourly_original,
                user_hourly_messages,
                prob_calc,
            )

            times["analysis"] = time.time() - analysis_start

            # 6. Collect metrics
            total_time = time.time() - start_time

            # Memory stats
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Performance metrics
            cache_stats = embedding_provider.cache_size()
            cache_hits = cache_stats if isinstance(cache_stats, int) else 0
            cache_total = cache_hits  # Simplified for now

            performance = PerformanceMetrics(
                execution_time=total_time,
                preprocessing_time=times["preprocessing"],
                embedding_time=times["embedding"],
                analysis_time=times["analysis"],
                memory_peak_mb=peak / 1024 / 1024,
                memory_average_mb=current / 1024 / 1024,
                cache_hits=cache_hits,
                cache_misses=0,  # Would need to track this in embedding provider
                cache_hit_rate=1.0 if cache_total > 0 else 0.0,
                total_messages=len(df),
                total_users=len(df["Sender"].unique()),
                total_hours=len(date_list) * 24,
            )

            # Quality metrics
            quality = self._calculate_quality_metrics(segments, len(df), date_list)

            # Create evaluation metrics
            metrics = EvaluationMetrics(
                run_name=run_name,
                config={
                    "embedding_provider": config.embedding_provider,
                    "embedding_model": config.embedding_model,
                    "probability_threshold": config.probability_threshold,
                    "dataset_path": config.dataset_path,
                    "keybert_model": config.keybert_model,
                    "start_date": start_date,
                    "end_date": end_date,
                    "num_topics": len(config.topics),
                },
                performance=performance,
                quality=quality,
                timestamp=datetime.now().isoformat(),
            )

            # Save results
            self._save_results(metrics, segments)

            self.logger.info(f"Evaluation complete: {run_name}")

            return metrics

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            tracemalloc.stop()
            raise

    def _organize_data_by_user_and_hour(
        self,
        df: pd.DataFrame,
        text_cleaner: TextCleaner,
        date_list: list[str],
    ) -> tuple[dict, dict, dict]:
        """Organize dataset by user and hour (same as in CLI)."""
        users = df["Sender"].unique().tolist()
        num_hours = 24 * len(date_list)

        user_hourly_texts = {}
        user_hourly_original = {}
        user_hourly_messages = {}

        for user in users:
            user_hourly_texts[user] = [[] for _ in range(num_hours)]
            user_hourly_original[user] = ["" for _ in range(num_hours)]
            user_hourly_messages[user] = [[] for _ in range(num_hours)]

        for _, row in df.iterrows():
            user = row["Sender"]
            timestamp = row["Timestamp"]
            text = str(row["Text"])

            date_str = timestamp.strftime("%Y-%m-%d")
            if date_str not in date_list:
                continue

            day_idx = date_list.index(date_str)
            hour = timestamp.hour
            hour_idx = day_idx * 24 + hour

            cleaned_words = text_cleaner.clean(text)
            user_hourly_texts[user][hour_idx].extend(cleaned_words)

            if user_hourly_original[user][hour_idx]:
                user_hourly_original[user][hour_idx] += " " + text
            else:
                user_hourly_original[user][hour_idx] = text

            message_info = [
                row["Chatroom"],
                user,
                date_str,
                timestamp.strftime("%H:%M:%S"),
                text,
                row.get("Prompt", ""),
            ]
            user_hourly_messages[user][hour_idx].append(message_info)

        return user_hourly_texts, user_hourly_original, user_hourly_messages

    def _calculate_quality_metrics(
        self,
        segments: list[dict[str, Any]],
        total_messages: int,
        date_list: list[str],
    ) -> QualityMetrics:
        """Calculate quality metrics from segments."""
        if not segments:
            return QualityMetrics(
                num_segments=0,
                avg_segment_length=0.0,
                total_messages_captured=0,
                coverage=0.0,
                avg_probability=0.0,
                min_probability=0.0,
                max_probability=0.0,
                topic_distribution={},
                topics_detected=0,
            )

        # Calculate segment statistics
        total_messages_captured = sum(
            len(seg["Chat Summary"].split("\n")) for seg in segments
        )
        avg_segment_length = total_messages_captured / len(segments)

        # Probability statistics
        probabilities = [float(seg["Probability"]) for seg in segments]
        avg_probability = sum(probabilities) / len(probabilities)
        min_probability = min(probabilities)
        max_probability = max(probabilities)

        # Topic distribution
        topic_distribution = {}
        for seg in segments:
            topic = seg["Topic"]
            topic_distribution[topic] = topic_distribution.get(topic, 0) + 1

        # Segment duration statistics
        durations = []
        for seg in segments:
            from datetime import datetime as dt
            start = dt.strptime(f"{seg['Start Date']} {seg['Start Time']}", "%Y-%m-%d %H:%M:%S")
            end = dt.strptime(f"{seg['End Date']} {seg['End Time']}", "%Y-%m-%d %H:%M:%S")
            duration = (end - start).total_seconds() / 3600  # hours
            durations.append(duration)

        return QualityMetrics(
            num_segments=len(segments),
            avg_segment_length=avg_segment_length,
            total_messages_captured=total_messages_captured,
            coverage=total_messages_captured / total_messages if total_messages > 0 else 0.0,
            avg_probability=avg_probability,
            min_probability=min_probability,
            max_probability=max_probability,
            topic_distribution=topic_distribution,
            topics_detected=len(topic_distribution),
            avg_segment_duration_hours=sum(durations) / len(durations) if durations else 0.0,
            min_segment_duration_hours=min(durations) if durations else 0.0,
            max_segment_duration_hours=max(durations) if durations else 0.0,
        )

    def _save_results(self, metrics: EvaluationMetrics, segments: list[dict]) -> None:
        """Save evaluation results to files."""
        # Create run directory
        run_dir = self.output_dir / metrics.run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        self.logger.info(f"Saved metrics to {metrics_path}")

        # Save summary as text
        summary_path = run_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write(metrics.summary_str())

        self.logger.info(f"Saved summary to {summary_path}")

        # Save segments as CSV
        if segments:
            segments_df = pd.DataFrame(segments)
            segments_path = run_dir / "segments.csv"
            segments_df.to_csv(segments_path, index=False)
            self.logger.info(f"Saved segments to {segments_path}")

    def compare_runs(self, run_names: list[str], output_file: str = "comparison.txt") -> str:
        """Compare multiple evaluation runs.

        Args:
            run_names: List of run names to compare
            output_file: Output file for comparison report

        Returns:
            Comparison report as string
        """
        metrics_list = []

        for run_name in run_names:
            metrics_path = self.output_dir / run_name / "metrics.json"
            if not metrics_path.exists():
                self.logger.warning(f"Metrics not found for run: {run_name}")
                continue

            with open(metrics_path) as f:
                metrics_dict = json.load(f)
                metrics_list.append(metrics_dict)

        if not metrics_list:
            return "No metrics found to compare"

        # Generate comparison report
        lines = [
            "=" * 100,
            "SCOPE EVALUATION COMPARISON",
            "=" * 100,
            "",
        ]

        # Comparison table header
        lines.append(f"{'Metric':<40} " + " ".join(f"{m['run_name']:<20}" for m in metrics_list))
        lines.append("-" * 100)

        # Performance metrics
        lines.append("\nPERFORMANCE:")
        perf_metrics = [
            ("Execution Time (s)", "performance.execution_time_seconds"),
            ("Messages/Second", "performance.messages_per_second"),
            ("Peak Memory (MB)", "performance.memory_peak_mb"),
            ("Cache Hit Rate (%)", "performance.cache_hit_rate"),
        ]

        for label, key in perf_metrics:
            values = []
            for m in metrics_list:
                keys = key.split(".")
                value = m
                for k in keys:
                    value = value.get(k, 0)
                if "rate" in key.lower():
                    value = value * 100
                values.append(f"{value:<20.2f}")

            lines.append(f"{label:<40} " + " ".join(values))

        # Quality metrics
        lines.append("\nQUALITY:")
        quality_metrics = [
            ("Segments Detected", "quality.num_segments"),
            ("Coverage (%)", "quality.coverage_percentage"),
            ("Avg Segment Length", "quality.avg_segment_length"),
            ("Avg Probability", "quality.avg_probability"),
            ("Topics Detected", "quality.topics_detected"),
        ]

        for label, key in quality_metrics:
            values = []
            for m in metrics_list:
                keys = key.split(".")
                value = m
                for k in keys:
                    value = value.get(k, 0)
                values.append(f"{value:<20.2f}")

            lines.append(f"{label:<40} " + " ".join(values))

        lines.append("")
        lines.append("=" * 100)

        report = "\n".join(lines)

        # Save report
        output_path = self.output_dir / output_file
        with open(output_path, "w") as f:
            f.write(report)

        self.logger.info(f"Saved comparison report to {output_path}")

        return report
