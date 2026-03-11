"""Command-line interface for SCOPE."""

import argparse
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

from scope.analysis import ContiguousBlockFinder, SegmentProcessor
from scope.config import ScopeConfig
from scope.embeddings import DiskEmbeddingCache, get_embedding_provider
from scope.evaluation import ScopeEvaluator
from scope.io import DatasetLoader, ResultWriter
from scope.modeling import ProbabilityCalculator
from scope.preprocessing import TextCleaner
from scope.utils import get_logger, setup_logging


def _resolve_calculation_mode(config: ScopeConfig) -> str:
    """Derive concrete calculation_mode from config when set to 'auto'."""
    if config.calculation_mode != "auto":
        return config.calculation_mode
    if config.embedding_provider == "jina":
        return "jina_mixed"
    return "st_baseline"


def _create_disk_cache(config: ScopeConfig, model_name: str) -> DiskEmbeddingCache | None:
    if not config.enable_disk_cache:
        return None
    return DiskEmbeddingCache(cache_dir=config.cache_dir, model_name=model_name)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="SCOPE - Topic modeling CLI for finding contiguous conversation blocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset path (optional - defaults to SCOPE_DATASET_PATH env var or data/Conversation.csv)
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=None,
        help="Path to input CSV file (default: SCOPE_DATASET_PATH env var or data/Conversation.csv)",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        default="results/scope_results.csv",
        help="Output CSV path (default: results/scope_results.csv)",
    )

    # Core options
    parser.add_argument(
        "-t",
        "--threshold",
        dest="probability_threshold",
        type=float,
        help="Probability threshold (default: 0.07)",
    )

    parser.add_argument(
        "--topics",
        help="Comma-separated topic list (overrides defaults)",
    )

    # Embedding options
    parser.add_argument(
        "-e",
        "--embedding",
        dest="embedding_provider",
        choices=["sentence-transformers", "jina"],
        help="Embedding provider (default: sentence-transformers)",
    )

    parser.add_argument(
        "--embedding-model",
        help="Model name for the embedding provider",
    )

    parser.add_argument(
        "--jina-api-key",
        help="Jina API key (or use JINA_API_KEY env var)",
    )

    # KeyBERT options
    parser.add_argument(
        "--no-keybert",
        action="store_true",
        dest="no_keybert",
        help="Disable KeyBERT keyword extraction; use direct cosine similarity only",
    )

    parser.add_argument(
        "--keybert-model",
        help="Model name for KeyBERT keyword extraction (default: all-MiniLM-L12-v2)",
    )

    # Date filtering
    parser.add_argument(
        "--start-date",
        help="Start date YYYY-MM-DD (inclusive)",
    )

    parser.add_argument(
        "--end-date",
        help="End date YYYY-MM-DD (inclusive)",
    )

    # Preprocessing options
    parser.add_argument(
        "--spell-check",
        action="store_true",
        dest="enable_spell_check",
        help="Enable spell checking (slow, disabled by default)",
    )

    parser.add_argument(
        "--no-lemmatize",
        action="store_false",
        dest="enable_lemmatization",
        help="Disable lemmatization (enabled by default)",
    )

    # Output options
    parser.add_argument(
        "--no-summary",
        action="store_false",
        dest="include_summary",
        help="Don't generate summary statistics",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        dest="quiet",
        help="Minimal output (disables verbose)",
    )

    # PostgreSQL options
    parser.add_argument(
        "--use-postgres",
        action="store_true",
        help="Use PostgreSQL vector store for embeddings (requires pgvector)",
    )

    parser.add_argument(
        "--postgres-host",
        help="PostgreSQL host (default: localhost)",
    )

    parser.add_argument(
        "--postgres-port",
        type=int,
        help="PostgreSQL port (default: 5432)",
    )

    parser.add_argument(
        "--postgres-db",
        dest="postgres_dbname",
        help="PostgreSQL database name (default: scope)",
    )

    parser.add_argument(
        "--postgres-user",
        help="PostgreSQL user (default: postgres)",
    )

    parser.add_argument(
        "--postgres-password",
        help="PostgreSQL password",
    )

    # Disk cache options
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Directory for disk embedding cache (default: .scope_cache)",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable disk embedding cache",
    )

    # Calculation mode
    parser.add_argument(
        "--mode",
        dest="calculation_mode",
        choices=["auto", "st_baseline", "jina_mixed", "jina_bag_of_words", "jina_full_text", "hybrid"],
        default=None,
        help="Calculation mode (default: auto — derives from embedding provider)",
    )

    # Pre-filter option
    parser.add_argument(
        "--prefilter",
        dest="prefilter_sim_threshold",
        type=float,
        default=None,
        metavar="SIM",
        help="Enable embedding pre-filter for block finding. "
             "Skips expensive KeyBERT calls on hour/block texts whose direct "
             "cosine similarity to the topic is below SIM (e.g. 0.15). "
             "Speeds up analysis at the cost of possibly missing borderline blocks.",
    )

    parser.add_argument(
        "--max-workers",
        dest="jina_max_workers",
        type=int,
        default=None,
        help="Maximum parallel workers for Jina API requests (default: 10)",
    )

    # Evaluation options
    parser.add_argument(
        "--no-evaluation",
        action="store_true",
        help="Disable evaluation mode (no performance/accuracy metrics)",
    )

    parser.add_argument(
        "--run-name",
        help="Name for this evaluation run (default: auto-generated from config)",
    )

    parser.add_argument(
        "--compare-runs",
        nargs="+",
        metavar="RUN_NAME",
        help="Compare multiple evaluation runs (e.g., --compare-runs run1 run2 run3)",
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version="SCOPE 0.1.0",
    )

    return parser.parse_args()


def organize_data_by_user_and_hour(
    df: pd.DataFrame,
    text_cleaner: TextCleaner,
    date_list: list[str],
    logger,
) -> tuple[dict[str, list[list[str]]], dict[str, list[str]], dict[str, list[list[Any]]]]:
    """Organize dataset by user and hour.

    Args:
        df: DataFrame with conversation data
        text_cleaner: Text cleaning instance
        date_list: List of dates in date range
        logger: Logger instance

    Returns:
        Tuple of (hourly_cleaned_texts, hourly_original_texts, hourly_messages) dictionaries
    """
    logger.info("Organizing data by user and hour...")

    # Get unique users
    users = df["Sender"].unique().tolist()
    num_hours = 24 * len(date_list)

    # Initialize data structures
    user_hourly_texts = {}
    user_hourly_original = {}
    user_hourly_messages = {}

    for user in users:
        user_hourly_texts[user] = [[] for _ in range(num_hours)]
        user_hourly_original[user] = ["" for _ in range(num_hours)]
        user_hourly_messages[user] = [[] for _ in range(num_hours)]

    # Process each message
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing messages", unit="msg"):
        user = row["Sender"]
        timestamp = row["Timestamp"]
        text = str(row["Text"])

        # Calculate hour index
        date_str = timestamp.strftime("%Y-%m-%d")
        if date_str not in date_list:
            continue

        day_idx = date_list.index(date_str)
        hour = timestamp.hour
        hour_idx = day_idx * 24 + hour

        # Clean text for frequency counting
        cleaned_words = text_cleaner.clean(text)

        # Store cleaned text
        user_hourly_texts[user][hour_idx].extend(cleaned_words)

        # Store original text (for embeddings) - append to existing text in this hour
        if user_hourly_original[user][hour_idx]:
            user_hourly_original[user][hour_idx] += " " + text
        else:
            user_hourly_original[user][hour_idx] = text

        # Store full message info: [Chatroom, Sender, Date, Time, Text, Prompt]
        message_info = [
            row["Chatroom"],
            user,
            date_str,
            timestamp.strftime("%H:%M:%S"),
            text,
            row.get("Prompt", ""),
        ]
        user_hourly_messages[user][hour_idx].append(message_info)

    logger.info(f"Organized data for {len(users)} users across {num_hours} hours")

    return user_hourly_texts, user_hourly_original, user_hourly_messages


def run_analysis(config: ScopeConfig) -> None:
    """Run the full SCOPE analysis pipeline.

    Args:
        config: SCOPE configuration
    """
    # Setup logging
    logger = setup_logging(config.verbose)

    if not config.verbose and not hasattr(config, "quiet"):
        logger.info("Starting SCOPE analysis...")

    start_time = time.time()

    # Initialize PostgreSQL vector store if enabled
    vector_store = None
    if config.use_postgres:
        try:
            from scope.database import DatabaseConfig, VectorStore

            # Build database config from ScopeConfig
            db_config = DatabaseConfig(
                host=config.postgres_host or "localhost",
                port=config.postgres_port or 5432,
                dbname=config.postgres_dbname or "scope",
                user=config.postgres_user or "postgres",
                password=config.postgres_password or "",
            )

            logger.info(
                f"Initializing PostgreSQL vector store at {db_config.host}:{db_config.port}"
            )
            vector_store = VectorStore(db_config)
            vector_store.connect()
            vector_store.initialize_schema()

            stats = vector_store.get_stats()
            logger.info(
                f"Connected to PostgreSQL: {stats['topic_count']} topics, "
                f"{stats['keyword_count']} keywords cached"
            )

        except ImportError:
            logger.error(
                "PostgreSQL support not installed. Install with: pip install 'scope[postgres]'"
            )
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            sys.exit(1)

    try:
        # 1. Load dataset
        logger.info(f"Loading dataset from {config.dataset_path}")
        loader = DatasetLoader(config.dataset_path)
        df = loader.load(config.start_date, config.end_date)
        logger.info(f"Loaded {len(df)} messages")

        # Get date range
        if config.start_date and config.end_date:
            start_date = config.start_date
            end_date = config.end_date
        else:
            start_date, end_date = loader.get_date_range(df)
            logger.info(f"Date range: {start_date} to {end_date}")

        # Generate date list
        date_list = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            date_list.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)

        logger.info(f"Processing {len(date_list)} days")

        # 2. Preprocess text
        logger.info("Preprocessing text...")
        text_cleaner = TextCleaner(
            enable_spell_check=config.enable_spell_check,
            enable_lemmatization=config.enable_lemmatization,
        )

        # Organize data by user and hour (now includes original text)
        user_hourly_texts, user_hourly_original, user_hourly_messages = (
            organize_data_by_user_and_hour(df, text_cleaner, date_list, logger)
        )

        # 3. Resolve calculation mode and create disk cache
        calc_mode = _resolve_calculation_mode(config)
        logger.info(f"Calculation mode: {calc_mode}")

        # Determine primary model name for disk cache key
        primary_model = config.embedding_model or (
            "jina-embeddings-v3" if config.embedding_provider == "jina" else "all-MiniLM-L12-v2"
        )
        disk_cache = _create_disk_cache(config, primary_model)
        if disk_cache:
            logger.info(f"Disk cache: {disk_cache.stats()['path']} ({disk_cache.stats()['size']} cached)")

        # 3b. Get embedding provider
        logger.info(f"Initializing embedding provider: {config.embedding_provider}")
        provider_kwargs = {
            "model": config.embedding_model,
            "api_key": config.jina_api_key,
            "disk_cache": disk_cache,
        }

        if config.embedding_provider == "jina":
            provider_kwargs["parallel_requests"] = config.jina_parallel_requests
            provider_kwargs["max_workers"] = config.jina_max_workers
            if config.jina_parallel_requests:
                logger.info(f"Using parallel requests with {config.jina_max_workers} workers")

        embedding_provider = get_embedding_provider(
            config.embedding_provider,
            **provider_kwargs
        )

        # 3c. Create keyword provider for hybrid mode (ST for keywords, Jina for full-text)
        keyword_provider = None
        if calc_mode == "hybrid":
            st_cache = _create_disk_cache(config, config.keybert_model)
            keyword_provider = get_embedding_provider(
                "sentence-transformers",
                model=config.keybert_model,
                disk_cache=st_cache,
            )
            logger.info(f"Hybrid mode: Jina for full-text, ST ({config.keybert_model}) for keywords")

        # 4. Initialize Probability Calculator (KeyBERT + Cosine Similarity)
        logger.info("Setting up KeyBERT + Cosine Similarity probability calculator...")
        prob_calc = ProbabilityCalculator(
            topics=config.topics,
            embedding_provider=embedding_provider,
            keybert_model=config.keybert_model,
            calculation_mode=calc_mode,
            vector_store=vector_store,
            use_keybert=config.use_keybert,
            keyword_provider=keyword_provider,
        )

        # 5. Find contiguous blocks
        logger.info(
            f"Finding contiguous blocks (threshold: {config.probability_threshold})..."
        )
        block_finder = ContiguousBlockFinder(
            config.probability_threshold,
            prob_calc,
            prefilter_sim_threshold=config.prefilter_sim_threshold,
        )

        user_blocks = block_finder.find_all_blocks(
            user_hourly_texts,
            user_hourly_original,
            config.topics,
        )

        # 6. Process segments
        logger.info("Processing segments...")
        processor = SegmentProcessor(date_list)

        segments = processor.process_segments(
            user_blocks,
            user_hourly_texts,
            user_hourly_original,
            user_hourly_messages,
            prob_calc,
        )

        logger.info(f"Found {len(segments)} segments")
        prob_calc.keybert_calc.log_timing_stats()

        # 8. Write results
        logger.info(f"Writing results to {config.output_path}")
        writer = ResultWriter(config.output_path)
        writer.write(segments)

        # Write summary if requested
        if config.include_summary:
            wall_time = time.time() - start_time
            summary_path = writer.write_summary(
                segments,
                wall_time,
                config.probability_threshold,
            )
            logger.info(f"Summary written to {summary_path}")

        # Flush disk caches
        if disk_cache:
            disk_cache.flush()
            logger.info(f"Disk cache stats: {disk_cache.stats()}")
        if keyword_provider and keyword_provider.disk_cache:
            keyword_provider.disk_cache.flush()

        wall_time = time.time() - start_time
        logger.info(f"Analysis complete in {wall_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if config.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup: close PostgreSQL connection if it was opened
        if vector_store is not None:
            try:
                vector_store.close()
                logger.debug("Closed PostgreSQL connection")
            except Exception:
                pass


def run_evaluation(config: ScopeConfig, run_name: str, labeled_data_path: Optional[str] = None) -> None:
    """Run SCOPE in evaluation mode.

    Args:
        config: SCOPE configuration
        run_name: Name for this evaluation run
        labeled_data_path: Optional path to labeled test data for accuracy
    """
    logger = setup_logging(config.verbose)

    evaluator = ScopeEvaluator(output_dir="results/evaluation")

    logger.info(f"Running evaluation: {run_name}")

    metrics = evaluator.evaluate(
        config=config,
        run_name=run_name,
        labeled_data_path=labeled_data_path,
    )

    # Print summary
    print("\n" + metrics.summary_str())
    print(f"\nResults saved to: results/evaluation/{run_name}/")


def compare_evaluation_runs(run_names: list[str]) -> None:
    """Compare multiple evaluation runs.

    Args:
        run_names: List of run names to compare
    """
    evaluator = ScopeEvaluator(output_dir="results/evaluation")

    print(f"\nComparing {len(run_names)} evaluation runs...")

    report = evaluator.compare_runs(run_names, output_file="comparison.txt")

    print("\n" + report)
    print("\nComparison saved to: results/evaluation/comparison.txt")


def main() -> None:
    """Main entry point for CLI."""
    args = parse_arguments()

    # Handle comparison mode
    if args.compare_runs:
        compare_evaluation_runs(args.compare_runs)
        return

    # Build configuration from environment variables and defaults
    config = ScopeConfig.from_env(dataset_path=args.dataset_path)

    # Override with command-line arguments (CLI args take precedence over .env)
    cli_args = {
        "output_path": args.output_path,
        "probability_threshold": args.probability_threshold,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "jina_api_key": args.jina_api_key,
        "keybert_model": args.keybert_model,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "enable_spell_check": args.enable_spell_check if args.enable_spell_check else None,
        "enable_lemmatization": args.enable_lemmatization if hasattr(args, "enable_lemmatization") else None,
        "verbose": False if args.quiet else None,
        "include_summary": args.include_summary if hasattr(args, "include_summary") else None,
        # Evaluation - disable if --no-evaluation flag is set
        "enable_evaluation": False if args.no_evaluation else None,
        "prefilter_sim_threshold": args.prefilter_sim_threshold,
        "use_keybert": False if args.no_keybert else None,
        "jina_max_workers": args.jina_max_workers,
        "cache_dir": args.cache_dir,
        "enable_disk_cache": False if args.no_cache else None,
        "calculation_mode": args.calculation_mode,
        # For boolean flags (store_true), only override if explicitly True
        # Otherwise, keep the env var value to avoid overriding True with False
        "use_postgres": args.use_postgres if args.use_postgres else None,
        "postgres_host": args.postgres_host if hasattr(args, "postgres_host") else None,
        "postgres_port": args.postgres_port if hasattr(args, "postgres_port") else None,
        "postgres_dbname": args.postgres_dbname if hasattr(args, "postgres_dbname") else None,
        "postgres_user": args.postgres_user if hasattr(args, "postgres_user") else None,
        "postgres_password": args.postgres_password if hasattr(args, "postgres_password") else None,
    }

    # Parse topics if provided
    if args.topics:
        cli_args["topics"] = [t.strip() for t in args.topics.split(",")]

    # Merge CLI args (only non-None values override)
    config.merge_with_args(**cli_args)

    # Handle evaluation mode (enabled by default)
    if config.enable_evaluation:
        # Generate run name if not provided
        run_name = args.run_name
        if not run_name:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{config.embedding_provider}_t{config.probability_threshold}_{timestamp}"

        run_evaluation(config, run_name, config.labeled_test_data)
    else:
        # Run normal analysis (no evaluation)
        run_analysis(config)


if __name__ == "__main__":
    main()
