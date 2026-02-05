"""SCOPE evaluation framework."""

import json
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

from scope.analysis import ContiguousBlockFinder, SegmentProcessor
from scope.config import ScopeConfig
from scope.embeddings import get_embedding_provider
from scope.io import DatasetLoader
from scope.modeling import ProbabilityCalculator
from scope.preprocessing import TextCleaner
from scope.utils import get_logger, setup_logging

from .metrics import AccuracyMetrics, EvaluationMetrics, PerformanceMetrics, QualityMetrics


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
        labeled_data_path: Optional[str] = None,
    ) -> EvaluationMetrics:
        """Run SCOPE analysis and collect evaluation metrics.

        Args:
            config: SCOPE configuration
            run_name: Optional name for this evaluation run
            labeled_data_path: Optional path to labeled test data CSV for accuracy calculation

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

            # Accuracy metrics (if labeled data provided)
            accuracy = None
            if labeled_data_path:
                self.logger.info("Calculating accuracy metrics on test data...")
                accuracy = self._calculate_accuracy_metrics(labeled_data_path, prob_calc, text_cleaner)

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
                accuracy=accuracy,
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

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing messages", unit="msg"):
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

    def _calculate_accuracy_metrics(
        self,
        labeled_data_path: str,
        prob_calc: "ProbabilityCalculator",
        text_cleaner: "TextCleaner",
    ) -> AccuracyMetrics | None:
        """Calculate accuracy metrics by running current config on test data.

        Args:
            labeled_data_path: Path to CSV file with columns: Chat Summary, Human Label
            prob_calc: ProbabilityCalculator with current config (model, threshold, etc.)
            text_cleaner: TextCleaner with current preprocessing settings

        Returns:
            AccuracyMetrics or None if file doesn't exist
        """
        import os
        if not os.path.exists(labeled_data_path):
            self.logger.warning(f"Labeled test data not found: {labeled_data_path}")
            return None

        try:
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix as sk_confusion_matrix

            df = pd.read_csv(labeled_data_path)

            # Validate required columns
            required_cols = ["Chat Summary", "Human Label"]
            if not all(col in df.columns for col in required_cols):
                self.logger.warning(f"Labeled data missing required columns: {required_cols}")
                return None

            self.logger.info(f"Running predictions on {len(df)} test samples...")

            # Generate predictions using current config
            predictions = []
            ground_truth = []

            for idx, row in df.iterrows():
                text = str(row["Chat Summary"])
                true_label = row["Human Label"]

                # Clean text using current config's preprocessing
                cleaned_words = text_cleaner.clean(text)

                # Predict using current config's model/settings
                predicted_topic = prob_calc.predict_topic(text, cleaned_words)

                predictions.append(predicted_topic)
                ground_truth.append(true_label)

                if (idx + 1) % 50 == 0:
                    self.logger.debug(f"  Processed {idx + 1}/{len(df)} test samples")

            # Calculate overall metrics
            total_samples = len(ground_truth)
            correct_predictions = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0

            # Get unique labels (union of predictions and ground truth)
            all_labels = sorted(set(predictions + ground_truth))

            # Calculate weighted precision, recall, F1
            precision = precision_score(ground_truth, predictions, labels=all_labels, average='weighted', zero_division=0)
            recall = recall_score(ground_truth, predictions, labels=all_labels, average='weighted', zero_division=0)
            f1 = f1_score(ground_truth, predictions, labels=all_labels, average='weighted', zero_division=0)

            # Calculate per-topic metrics
            per_topic_accuracy = {}
            per_topic_precision = {}
            per_topic_recall = {}
            per_topic_f1 = {}

            for label in all_labels:
                # Accuracy
                label_mask = [g == label for g in ground_truth]
                if sum(label_mask) > 0:
                    label_correct = sum(1 for i, m in enumerate(label_mask) if m and predictions[i] == ground_truth[i])
                    per_topic_accuracy[label] = label_correct / sum(label_mask)
                else:
                    per_topic_accuracy[label] = 0.0

                # Precision, Recall, F1
                try:
                    prec = precision_score(ground_truth, predictions, labels=[label], average='micro', zero_division=0)
                    rec = recall_score(ground_truth, predictions, labels=[label], average='micro', zero_division=0)
                    f1_val = f1_score(ground_truth, predictions, labels=[label], average='micro', zero_division=0)

                    per_topic_precision[label] = prec
                    per_topic_recall[label] = rec
                    per_topic_f1[label] = f1_val
                except:
                    per_topic_precision[label] = 0.0
                    per_topic_recall[label] = 0.0
                    per_topic_f1[label] = 0.0

            # Build confusion matrix
            conf_matrix = {}
            sk_conf = sk_confusion_matrix(ground_truth, predictions, labels=all_labels)

            for i, true_label in enumerate(all_labels):
                conf_matrix[true_label] = {}
                for j, pred_label in enumerate(all_labels):
                    count = int(sk_conf[i][j])
                    if count > 0:
                        conf_matrix[true_label][pred_label] = count

            self.logger.info(f"Accuracy: {accuracy * 100:.2f}% ({correct_predictions}/{total_samples})")

            return AccuracyMetrics(
                total_samples=total_samples,
                correct_predictions=correct_predictions,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                per_topic_accuracy=per_topic_accuracy,
                per_topic_precision=per_topic_precision,
                per_topic_recall=per_topic_recall,
                per_topic_f1=per_topic_f1,
                confusion_matrix=conf_matrix,
            )

        except ImportError:
            self.logger.error("scikit-learn required for accuracy metrics. Install with: pip install scikit-learn")
            return None
        except Exception as e:
            self.logger.error(f"Error calculating accuracy metrics: {e}")
            if hasattr(self, 'verbose') or True:  # Show traceback for debugging
                import traceback
                traceback.print_exc()
            return None

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
