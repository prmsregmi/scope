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
- **Fast Performance**: Processes ~155k messages in 8-10 seconds

## Installation

### Requirements

- Python 3.12+
- uv (for package management)

### Quick Install (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install with all features
cd SCOPE
uv sync --extra all

# Download required spaCy model
uv run python -m spacy download en_core_web_sm
```

### Minimal Install

```bash
# Install with only sentence-transformers (local embeddings)
uv sync --extra sentence-transformers
uv run python -m spacy download en_core_web_sm
```

For PostgreSQL support, install with `--extra postgres`. Requires pgvector extension: https://github.com/pgvector/pgvector

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

### Using PostgreSQL (Optional)

SCOPE supports PostgreSQL with pgvector for persistent embedding storage and faster similarity search. This provides significant speedup on subsequent runs with cached embeddings.

**Setup:**

1. Install PostgreSQL with pgvector extension (if not already installed)

2. Install SCOPE with PostgreSQL support:
   ```bash
   uv sync --extra postgres
   ```

3. Configure database connection in `.env` file:
   ```bash
   SCOPE_USE_POSTGRES=true
   DATABASE_HOST=localhost
   DATABASE_PORT=5432
   DATABASE_NAME=scope
   DATABASE_USER=your_username
   DATABASE_PASSWORD=your_password
   ```

4. Run analysis (schema auto-created on first run):
   ```bash
   scope conversations.csv
   ```

**Alternative: Using CLI flags**
```bash
scope conversations.csv \
  --use-postgres \
  --postgres-host localhost \
  --postgres-db scope \
  --postgres-user your_username
```

**Docker Setup (Recommended for quick start):**
```bash
docker run -d --name scope-postgres \
  -p 5432:5432 \
  -e POSTGRES_DB=scope \
  ankane/pgvector
```

For vector indexing details, advanced configuration, and troubleshooting, see [POSTGRES_SETUP.md](POSTGRES_SETUP.md)

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

## Algorithm

SCOPE uses a Hybrid Cosine-KeyBERT approach:

1. **Dual Text Processing**: Preserves original text for embeddings while using cleaned text for frequency weighting
2. **Keyword Extraction**: KeyBERT extracts relevant keywords from original text
3. **Semantic Similarity**: Generates embeddings and calculates cosine similarity between keywords and topics
4. **Probability Scoring**: Combines similarity × KeyBERT relevance × word frequency with softmax normalization
5. **Block Detection**: Identifies contiguous hourly segments exceeding probability threshold

## Evaluation

Run SCOPE in evaluation mode to collect detailed performance metrics and accuracy against labeled ground truth:

```bash
# Run evaluation with accuracy calculation
uv run scope data/Conversation.csv \
  --evaluate \
  --run-name "experiment_name" \
  --labeled-data data/labeled_test_data.csv \
  -t 0.07

# Compare multiple runs
uv run scope --compare-runs run1 run2 run3
```

Results saved to `results/evaluation/<run_name>/` with metrics.json, summary.txt, and per-topic accuracy.

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

## Additional Resources

See `SCOPE.pdf` for detailed research paper and methodology.
