# SCOPE

**S**egmented **CO**ntiguous **P**robability **E**xtraction

A CLI tool for topic modeling that identifies contiguous time blocks in conversation data using Hybrid Cosine-KeyBERT semantic similarity.

## Overview

SCOPE analyzes conversational data to find temporal segments where users are discussing particular subjects. It uses a Hybrid Cosine-KeyBERT approach combining keyword extraction, semantic embeddings, and cosine similarity to calculate topic relevance probabilities, then identifies contiguous hourly blocks that exceed a specified threshold.

---

## Setup

```bash
./setup.sh
```

**That's it!** One command installs everything (10-15 minutes):

- Homebrew (macOS, if needed)
- uv + Python 3.14
- PostgreSQL 15 + pgvector
- All dependencies (~2-3GB)
- Auto-configured `.env`

> **Note:** For manual installation without using the setup script, see the [**Manual Setup**](#manual-setup) section at the bottom of this document.

---

## Usage

```bash
# Basic analysis (with default verbose output)
uv run scope data/Conversation.csv

# With JINA embeddings (recommended - best quality)
uv run scope data/Conversation.csv -e jina

# Custom threshold
uv run scope data/Conversation.csv -t 0.08

# Date range filtering
uv run scope data/Conversation.csv --start-date 2018-05-01 --end-date 2018-05-31

# Quiet mode (minimal output)
uv run scope data/Conversation.csv --quiet
```

**Results**: CSV file at `results/scope_results.csv` with temporal segments and topic probabilities.

> **Defaults**: Verbose output **ON** (use `--quiet` for minimal output), JINA parallel processing **enabled** (5 workers), spell checking **OFF** (use `--spell-check` to enable).

---

## Examples

### Example 1: JINA Embeddings (Recommended - Best Quality)

```bash
# JINA with parallel processing (enabled by default)
uv run scope data/Conversation.csv -e jina
```

### Example 2: Custom Topics and Threshold

```bash
# JINA with custom topics and threshold
uv run scope data/Conversation.csv \
  -e jina \
  --topics "Politics,Technology,Sports,Education" \
  --threshold 0.10 \
  -o filtered_results.csv
```

### Example 3: Date Range Filtering

```bash
# Analyze specific date range with JINA
uv run scope data/Conversation.csv \
  -e jina \
  --start-date 2018-05-01 \
  --end-date 2018-05-31 \
  -o may_results.csv
```

### Example 4: Local Embeddings (Faster, No API Key)

```bash
# SentenceTransformers for fast local processing
uv run scope data/Conversation.csv
```

### Example 5: Quiet Mode (Minimal Output)

```bash
# Run with minimal output (verbose is on by default)
uv run scope data/Conversation.csv --quiet
```

> **For a complete list of available CLI arguments, see the [CLI Arguments](#cli-arguments) section below.**

---

---

# **ADVANCED DOCUMENTATION BELOW**

**The following sections contain advanced configuration, technical details, and manual installation options.**

---

---

## Features

- **Hybrid Cosine-KeyBERT**: Advanced semantic similarity using KeyBERT keyword extraction and cosine similarity
- **Dual Text Processing**: Uses original text for embeddings (preserving context) and cleaned text for frequency weighting
- **Contiguous Block Detection**: Greedy algorithm to find temporal segments of related conversation
- **Flexible Embedding Support**: Choose between SentenceTransformers (local) or Jina AI (API-based) embeddings
- **PostgreSQL Vector Storage**: pgvector integration for scalable embedding storage and 8-10x speedup on cached runs
- **Comprehensive Preprocessing**: Text cleaning, stop word removal, spell checking, and lemmatization
- **Configurable**: Environment variables (.env) or command-line arguments
- **Smart Caching**: Caches embeddings and probability calculations for performance
- **Fast Performance**: Processes ~155k messages in 8-10 seconds

## Algorithm

1. **Dual Text Processing**: Preserves original text for embeddings while using cleaned text for frequency weighting
2. **Keyword Extraction**: KeyBERT extracts relevant keywords from original text
3. **Semantic Similarity**: Generates embeddings and calculates cosine similarity between keywords and topics
4. **Probability Scoring**: Combines similarity × KeyBERT relevance × word frequency with softmax normalization
5. **Block Detection**: Identifies contiguous hourly segments exceeding probability threshold

## CLI Arguments

CLI arguments override `.env` values:

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

## Configuration

### Environment Variables

Create a `.env` file (auto-created by `setup.sh`):

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

# PostgreSQL (auto-configured by setup.sh)
SCOPE_USE_POSTGRES=true
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=scope
DATABASE_USER=your_username
DATABASE_PASSWORD=
```

See `.env.example` for all available options.

## Input Data Format

Your CSV file must contain these columns:

- **Chatroom**: Identifier for the discussion channel/room
- **Sender**: User identifier
- **Timestamp**: Date and time in `YYYY-MM-DD HH:MM:SS` format
- **Text**: Message content
- **Prompt** (optional): Associated prompt or context

**Example:**

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

When enabled (default), a `.summary.txt` file is generated with:

- Number of extracted segments
- Average segment length
- Total messages captured
- Processing time
- Average topic relevance score
- Topic distribution

## Embedding Providers

### SentenceTransformers (Default)

- **Pros**: Runs locally, no API key needed, fast
- **Cons**: Requires more disk space for models
- **Model**: `all-MiniLM-L12-v2` (default)

```bash
uv run scope data.csv -e sentence-transformers
```

### Jina AI

- **Pros**: No local storage, state-of-the-art models, multilingual, best quality
- **Cons**: Requires API key, internet connection, rate limits, 3.3x slower than local
- **Model**: `jina-embeddings-v3` (default)
- **Configuration**: Optimized for text-matching with 384 dimensions
- **Get API Key**: https://jina.ai/?sui=apikey

```bash
export JINA_API_KEY=your_key
uv run scope data.csv -e jina
```

**Performance**: JINA (optimized) detects 10% more segments and 13% more topics than SentenceTransformers, but takes 3.3x longer. Recommended for quality-critical applications.

## Evaluation

Evaluation runs by default, collecting performance metrics (speed, memory) and accuracy against labeled test data.

```bash
# Run with auto-generated name
uv run scope data/Conversation.csv -e jina -t 0.08

# Or specify run name
uv run scope data/Conversation.csv --run-name "jina_t008" -e jina -t 0.08

# Disable evaluation (just get CSV output)
uv run scope data/Conversation.csv --no-evaluation -o output.csv

# Compare experiments
uv run scope --compare-runs jina_t008 st_t007
```

**Results**: `results/evaluation/<run_name>/` with metrics (accuracy, precision, recall, F1), performance stats, and confusion matrix.

Configure test data path in `.env`: `SCOPE_LABELED_TEST_DATA=data/labeled_test_data_clean.csv`

## Troubleshooting

### spaCy Model Not Found

```bash
uv run python -m spacy download en_core_web_sm
```

### NLTK Data Missing

The package automatically downloads required NLTK data (wordnet, omw-1.4) on first run.

### Jina API Errors

- Verify API key: `echo $JINA_API_KEY`
- Check rate limits (500 RPM for standard keys)
- Use smaller date ranges for large datasets

### PostgreSQL Connection Issues

```bash
# Test connection
pg_isready

# Check if PostgreSQL is running
brew services list | grep postgresql  # macOS
sudo systemctl status postgresql      # Linux

# Check database exists
psql -l | grep scope
```

### Help

```bash
uv run scope --help
```

## Additional Resources

See `SCOPE.pdf` for detailed research paper and methodology.

---

---

# **MANUAL SETUP**

**The following section provides step-by-step manual installation instructions for users who prefer not to use the automated setup script.**

---

---

## Alternative Manual Installation

> **⚠️ NOTE: Manual installation is NOT required if you used `./setup.sh` above!**

<details>
<summary><b>Click here only if you need manual installation steps</b></summary>

### Requirements

- Python 3.12+ (or let uv handle it)
- uv (for package management)
- PostgreSQL 15+ with pgvector (optional but recommended)

### Step 1: Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

### Step 2: Install PostgreSQL with pgvector

**macOS:**

```bash
brew install postgresql@15 pgvector
brew services start postgresql@15
createdb scope
psql scope -c "CREATE EXTENSION vector;"
```

**Linux (Ubuntu/Debian):**

```bash
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get install postgresql-15 postgresql-15-pgvector
sudo -u postgres createdb scope
sudo -u postgres psql scope -c "CREATE EXTENSION vector;"
```

### Step 3: Install SCOPE Dependencies

```bash
cd SCOPE
uv sync --python 3.14 --extra all
uv run python -m spacy download en_core_web_sm
```

### Step 4: Configure Environment

```bash
cp .env.example .env
# Edit .env and set your database credentials and API keys
```

### Step 5: PostgreSQL Management (Optional)

If you installed PostgreSQL manually, here are useful management commands:

**macOS:**

```bash
# Start/stop/restart
brew services start postgresql@15
brew services stop postgresql@15
brew services restart postgresql@15

# Connect
psql scope

# View stats
psql scope -c "SELECT COUNT(*) FROM keywords;"

# Clear cache
psql scope -c "DELETE FROM keywords;"
```

**Linux:**

```bash
# Start/stop/restart
sudo systemctl start postgresql
sudo systemctl stop postgresql
sudo systemctl restart postgresql

# Connect
psql scope

# View stats
psql scope -c "SELECT COUNT(*) FROM keywords;"

# Clear cache
psql scope -c "DELETE FROM keywords;"
```

**Using CLI flags:**

```bash
uv run scope conversations.csv \
  --use-postgres \
  --postgres-host localhost \
  --postgres-db scope \
  --postgres-user your_username
```

For vector indexing details and advanced configuration, see [POSTGRES_SETUP.md](POSTGRES_SETUP.md)

</details>
