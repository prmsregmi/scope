# PostgreSQL Setup with Vector Indexing

> **Note**: This guide assumes you've already followed the basic setup in the main README. Refer there first for installation and initial configuration.

---

## Why Vector Indexing?

SCOPE compares embeddings (384-dimensional vectors) to find semantic similarity between keywords and topics. Without indexing, this requires comparing every keyword to every topic using brute-force calculation—slow for large datasets.

**Vector indexes solve this by organizing embeddings in data structures that enable approximate nearest neighbor (ANN) search**, reducing similarity queries from O(n) to O(log n) complexity.

### How It Works

Traditional approach (without indexing):
```
For each keyword embedding:
  For each topic embedding:
    Calculate cosine similarity  ← Does this millions of times
```

With vector indexing:
```
For each keyword embedding:
  Navigate index structure       ← Logarithmic lookup
  Find most similar topics       ← Orders of magnitude faster
```

**Result**: 8-10x speedup on cached runs, scaling better with dataset size.

---

## Vector Index Types

### HNSW (Hierarchical Navigable Small World)

**How it works**: Builds a multi-layer graph where each node (embedding) connects to nearby nodes. Search starts at the top layer and navigates down, efficiently finding similar vectors.

**Why it's fast**:
- Graph structure enables logarithmic search
- Multiple layers provide different granularities
- Connections optimized during build time

**Trade-offs**:
- Faster queries (production use)
- Higher memory usage
- Slower inserts (graph updates)

**Configuration**:
```bash
SCOPE_INDEX_TYPE=hnsw
SCOPE_HNSW_M=16              # Connections per node (2-100)
SCOPE_HNSW_EF_CONSTRUCTION=64  # Build quality (4-1000)
```

**Parameter guide**:
- `m`: More connections = better recall, more memory
- `ef_construction`: Higher = better index quality, slower build

### IVFFlat (Inverted File Flat)

**How it works**: Partitions embedding space into clusters (Voronoi cells). Search only checks embeddings in nearest clusters, not the entire dataset.

**Why it's fast**:
- Reduces search space via clustering
- Simple structure, lower memory
- Faster updates

**Trade-offs**:
- Good for development/testing
- Slightly lower recall than HNSW
- Needs periodic retraining for optimal performance

**Configuration**:
```bash
SCOPE_INDEX_TYPE=ivfflat
SCOPE_IVFFLAT_LISTS=100  # Number of clusters (recommend: rows/1000)
```

**When to use**: Memory-constrained environments or rapid iteration during development.

---

## Manual PostgreSQL Installation

If not using Docker, install PostgreSQL with pgvector:

### macOS
```bash
brew install postgresql@15 pgvector
brew services start postgresql@15
createdb scope
psql scope -c "CREATE EXTENSION vector;"
```

### Ubuntu/Debian
```bash
sudo apt-get install postgresql-15 postgresql-15-pgvector
sudo systemctl start postgresql
sudo -u postgres createdb scope
sudo -u postgres psql scope -c "CREATE EXTENSION vector;"
```


---

## Database Management

### Check Statistics

```sql
psql -h localhost -U your_username -d scope

-- Check counts and size
SELECT COUNT(*) FROM topics;
SELECT COUNT(*) FROM keywords;
SELECT pg_size_pretty(pg_database_size('scope'));
```

### Clear Cache

```bash
# Clear keywords only
psql scope -c "DELETE FROM keywords;"

# Clear everything
psql scope -c "DELETE FROM keywords; DELETE FROM topics;"
```

### Backup/Restore

```bash
pg_dump scope > backup.sql
psql scope < backup.sql
```

---

## Troubleshooting

**Connection failed**: Check PostgreSQL is running
```bash
pg_isready -h localhost -p 5432
```

**Extension missing**: Install pgvector
```bash
psql scope -c "CREATE EXTENSION vector;"
```

**Permission denied**: Grant permissions
```sql
GRANT ALL ON ALL TABLES IN SCHEMA public TO your_username;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_username;
```

**Slow queries**: Rebuild indexes
```sql
REINDEX TABLE keywords;
ANALYZE keywords;
```

---

## Testing Setup

```bash
uv run python -c "
from scope.database import DatabaseConfig, VectorStore
config = DatabaseConfig.from_env()
store = VectorStore(config)
store.connect()
print('✓ Connected:', store.get_stats())
store.close()
"
```

---

## Quick Reference

```bash
# Start/stop Docker
docker start scope-postgres
docker stop scope-postgres

# Database size
psql scope -c "SELECT pg_size_pretty(pg_database_size('scope'));"

# Connection test
pg_isready -h localhost -p 5432

# Clear cache
psql scope -c "DELETE FROM keywords;"
```

---

For detailed performance assessment, see `results/docs/POSTGRES_ASSESSMENT.md`
