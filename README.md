# SCOPE

**S**egmented **CO**ntiguous **P**robability **E**xtraction

A CLI tool for topic modeling that identifies contiguous time blocks in conversation data related to specific topics using Hybrid Cosine-KeyBERT semantic similarity.

## Overview

SCOPE analyzes conversational data to find temporal segments where users are discussing particular subjects. It uses a Hybrid Cosine-KeyBERT approach combining keyword extraction, semantic embeddings, and cosine similarity to calculate topic relevance probabilities, then identifies contiguous hourly blocks that exceed a specified threshold.

## Features

- **Hybrid Cosine-KeyBERT**: Advanced semantic similarity using KeyBERT keyword extraction and cosine similarity
- **Dual Text Processing**: Uses original text for embeddings (preserving context) and cleaned text for frequency weighting
- **Contiguous Block Detection**: Greedy algorithm to find temporal segments of related conversation
- **Flexible Embedding Support**: Choose between SentenceTransformers (local) or Jina AI (API-based) embeddings
- **PostgreSQL Vector Storage**: Optional pgvector integration for scalable embedding storage
- **Comprehensive Preprocessing**: Text cleaning, stop word removal, spell checking, and lemmatization
- **Configurable**: Environment variables (.env) or command-line arguments
- **Smart Caching**: Caches embeddings and probability calculations for performance

## Installation

### Requirements

- Python 3.12+
- uv (for package management)

### Install with SentenceTransformers (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
cd SCOPE

# Install with sentence-transformers (local embeddings)
uv sync --extra sentence-transformers

# Download required spaCy model
uv run python -m spacy download en_core_web_sm
```

### Install with Jina AI Support

```bash
# Install with Jina API support
uv sync --extra jina

# Download spaCy model
uv run python -m spacy download en_core_web_sm
```

### Install with PostgreSQL Vector Storage

```bash
# Install with PostgreSQL support
uv sync --extra postgres

# Requires PostgreSQL with pgvector extension
# See: https://github.com/pgvector/pgvector
```

### Install Everything

```bash
# Install all optional dependencies (sentence-transformers, jina, postgres, dev tools)
uv sync --extra all

# Download spaCy model
uv run python -m spacy download en_core_web_sm
```

## Quick Start

### Basic Usage

```bash
# Analyze a conversation dataset
scope path/to/conversations.csv

# Specify output file
scope conversations.csv -o results.csv

# Set custom probability threshold
scope conversations.csv -t 0.08

# Use verbose output
scope conversations.csv -v
```

### Using Jina AI Embeddings

```bash
# Set your API key
export JINA_API_KEY=your_api_key_here

# Run analysis with Jina embeddings
scope conversations.csv -e jina
```

### Date Range Filtering

```bash
# Analyze specific date range
scope conversations.csv \
  --start-date 2018-05-01 \
  --end-date 2018-05-31 \
  -o may_results.csv
```

## Configuration

SCOPE can be configured through environment variables (.env file) or command-line arguments. CLI arguments always take precedence over .env values.

### Environment Variables

Create a `.env` file in your project directory:

```bash
# API Keys
JINA_API_KEY=your_api_key_here

# Embedding Configuration
SCOPE_EMBEDDING_PROVIDER=sentence-transformers  # or jina
SCOPE_EMBEDDING_MODEL=all-MiniLM-L12-v2
SCOPE_KEYBERT_MODEL=all-MiniLM-L12-v2

# Analysis Settings
SCOPE_PROBABILITY_THRESHOLD=0.07
SCOPE_OUTPUT_PATH=results/scope_results.csv

# Preprocessing
SCOPE_SPELL_CHECK=true
SCOPE_LEMMATIZE=true
```

See `.env.example` for a complete list of available configuration options.

## CLI Reference

```
scope <dataset_path> [OPTIONS]

Required:
  dataset_path              Path to input CSV file

Output Options:
  -o, --output PATH        Output CSV path (default: results/scope_results.csv)
  --no-summary             Don't generate summary statistics

Configuration:
  -t, --threshold FLOAT    Probability threshold (default: 0.07)
  --topics TEXT            Comma-separated topic list

Embedding Options:
  -e, --embedding TYPE     Embedding provider (sentence-transformers|jina)
  --embedding-model TEXT   Model name for embedding provider
  --jina-api-key TEXT      Jina API key

KeyBERT Options:
  --keybert-model TEXT     Model for keyword extraction (default: all-MiniLM-L12-v2)

Date Filtering:
  --start-date DATE        Start date YYYY-MM-DD (inclusive)
  --end-date DATE          End date YYYY-MM-DD (inclusive)

Preprocessing:
  --no-spell-check         Disable spell checking
  --no-lemmatize           Disable lemmatization

Other:
  -v, --verbose            Verbose output
  --quiet                  Minimal output
  -h, --help               Show help message
  --version                Show version
```

## Input Data Format

Your CSV file must contain these columns:

- **Chatroom**: Identifier for the discussion channel/room
- **Sender**: User identifier
- **Timestamp**: Date and time in `YYYY-MM-DD HH:MM:SS` format
- **Text**: Message content
- **Prompt** (optional): Associated prompt or context

Example:

```csv
Chatroom,Sender,Timestamp,Text,Prompt
Room1,User123,2018-05-01 09:30:00,"Discussion about machine learning",""
Room1,User456,2018-05-01 09:31:15,"Yes neural networks are fascinating",""
```

## Output Format

SCOPE generates a CSV file with these columns:

- **User**: User identifier
- **Start Date**: Beginning date of segment
- **Start Time**: Beginning time of segment
- **End Date**: Ending date of segment
- **End Time**: Ending time of segment
- **Time Duration**: Total duration of segment
- **Topic**: Identified topic name
- **Probability**: Topic relevance score
- **Chat Summary**: Full text of all messages in segment

### Summary Statistics

When `--include-summary` is enabled (default), a `.summary.txt` file is generated with:

- Number of extracted segments
- Average segment length
- Total messages captured
- Processing time
- Average topic relevance score
- Topic distribution

## Examples

### Example 1: Basic Analysis

```bash
scope "Chit-Chat Dataset/Conversation.csv"
```

### Example 2: Custom Topics and Threshold

```bash
scope conversations.csv \
  --topics "Politics,Technology,Sports,Education" \
  --threshold 0.10 \
  -o filtered_results.csv
```

### Example 3: Date Range with Jina

```bash
export JINA_API_KEY=your_key_here
scope conversations.csv \
  -e jina \
  --start-date 2018-05-01 \
  --end-date 2018-05-31 \
  -o may_jina_results.csv \
  -v
```

### Example 4: Using Environment Variables

```bash
# Set up .env file with your preferences
cat > .env << EOF
SCOPE_EMBEDDING_PROVIDER=jina
JINA_API_KEY=your_key_here
SCOPE_PROBABILITY_THRESHOLD=0.08
EOF

# Run analysis (uses .env settings)
scope conversations.csv -v
```

### Example 5: With PostgreSQL Vector Storage

```bash
# Use PostgreSQL for embedding storage
scope --use-postgres \
  --postgres-host localhost \
  --postgres-db scope \
  --postgres-user postgres \
  conversations.csv -v
```

## Embedding Providers

### SentenceTransformers (Default)

- **Pros**: Runs locally, no API key needed, fast
- **Cons**: Requires more disk space for models
- **Model**: `all-MiniLM-L12-v2` (default)

```bash
scope data.csv -e sentence-transformers
```

### Jina AI

- **Pros**: No local storage, state-of-the-art models, multilingual, best quality
- **Cons**: Requires API key, internet connection, rate limits, 3.3x slower than local
- **Model**: `jina-embeddings-v3` (default)
- **Configuration**: Optimized for text-matching with 384 dimensions
- **Get API Key**: https://jina.ai/?sui=apikey

```bash
export JINA_API_KEY=your_key
scope data.csv -e jina
```

**Performance**: JINA (optimized) detects 10% more segments and 13% more topics than SentenceTransformers, but takes 3.3x longer. Recommended for quality-critical applications.

## Evaluation

Benchmark different embedding providers and configurations:

```bash
# Run evaluation
python evaluate.py test_conversation.csv -n "my_baseline" -e sentence-transformers

# Compare runs
python evaluate.py --compare baseline_1 baseline_2
```

Tracks performance (execution time, memory, cache efficiency) and quality metrics (segments detected, coverage, topic distribution). Results saved to `results/evaluation/<run_name>/` with JSON metrics and human-readable summaries.

## Development

### Running Tests

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v
```

### Code Formatting

```bash
uv run black scope/
uv run ruff check scope/
```

### Type Checking

```bash
uv run mypy scope/
```

## Algorithm

SCOPE uses a Hybrid Cosine-KeyBERT approach with greedy block detection:

1. **Preprocessing**: Clean text for frequency counting while preserving original text for embeddings
2. **Keyword Extraction**: Use KeyBERT to extract relevant keywords from original text (preserves context)
3. **Embedding Generation**: Generate semantic embeddings for keywords and topics using SentenceTransformers or Jina AI
4. **Similarity Calculation**: Calculate cosine similarity between keyword and topic embeddings
5. **Probability Weighting**: Combine similarity × KeyBERT relevance × word frequency, then apply softmax normalization
6. **Hourly Aggregation**: Group messages by user and hour with probability scores
7. **Block Detection**: Find contiguous hours exceeding threshold using greedy approach
8. **Segment Processing**: Format results with timestamps and message aggregation

### Key Innovation: Dual Text Processing

SCOPE uses a novel dual-text approach that improves embedding quality:

- **Original Text**: Used for KeyBERT keyword extraction and embeddings (preserves context and stop words)
- **Cleaned Text**: Used for word frequency counting (removes noise)

This approach produces higher-quality semantic embeddings compared to using preprocessed text alone, as transformer models perform better with full contextual information.

**Example:**
```
Original: "I love machine learning and neural networks"
  → KeyBERT extracts: [('machine learning', 0.75), ('neural networks', 0.68)]
  → High-quality contextual embeddings

Cleaned: ['love', 'machine', 'learning', 'neural', 'network']
  → Used for frequency weighting only
```

## Performance

Typical performance on a standard machine:

- **May 2018 (31 days)**: ~8-10 seconds
- **~155,000 messages**: ~5,245 segments detected
- **Memory**: ~500MB RAM

## Troubleshooting

### spaCy Model Not Found

```bash
python -m spacy download en_core_web_sm
```

### NLTK Data Missing

The package automatically downloads required NLTK data (wordnet, omw-1.4) on first run.

### Jina API Errors

- Verify API key: `echo $JINA_API_KEY`
- Check rate limits (500 RPM for standard keys)
- Use smaller date ranges for large datasets

## Implementation Notes

This codebase implements the **Hybrid Cosine-KeyBERT** approach from the research notebooks. The LDA-based approach has been completely removed in favor of this semantic similarity method for the following reasons:

**Advantages of Cosine-KeyBERT:**
- Better semantic understanding of topics
- More accurate keyword identification
- Works well with small amounts of data per user
- Leverages state-of-the-art transformer embeddings
- Flexible embedding providers (local or API-based)

**Trade-offs:**
- Slightly slower than LDA (embedding-based vs statistical)
- Requires more memory for embedding cache
- Depends on embedding model quality

For implementation details, see `IMPLEMENTATION_SUMMARY.md`.

## Related Documentation

- `IMPLEMENTATION_SUMMARY.md` - Detailed technical implementation notes
- `MIGRATION_GUIDE.md` - Migration from LDA to Cosine-KeyBERT
- `SCOPE.pdf` - Research paper and methodology
- Jupyter notebooks in root directory - Experimental implementations

## License

MIT

## Citation

If you use SCOPE in your research, please cite:

```
@software{scope2024,
  title={SCOPE: Segmented Contiguous Probability Extraction},
  year={2024},
  version={0.1.0}
}
```
