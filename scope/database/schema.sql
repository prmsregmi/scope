-- PostgreSQL schema for SCOPE vector embeddings
-- Requires pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Topics table
-- Stores embeddings for each topic (e.g., "News", "Technology", "Sports")
CREATE TABLE IF NOT EXISTS topics (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    embedding VECTOR(384) NOT NULL,  -- 384 dimensions for MiniLM, adjust for other models
    created_at TIMESTAMP DEFAULT NOW()
);

-- Keywords table
-- Stores embeddings for keywords extracted by KeyBERT
CREATE TABLE IF NOT EXISTS keywords (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL UNIQUE,
    embedding VECTOR(384) NOT NULL,  -- Match dimension with topics
    frequency INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    last_used TIMESTAMP DEFAULT NOW()
);

-- Text search index for keywords
CREATE INDEX IF NOT EXISTS idx_keywords_text ON keywords(text);

-- Vector indexes for similarity search
-- Choose ONE index type based on your needs:

-- Option 1: HNSW (Hierarchical Navigable Small World)
-- Best for: High query performance, when you have enough memory
-- Pros: Faster queries, better recall
-- Cons: Slower inserts, higher memory usage
-- Parameters:
--   m: Number of bi-directional links (default 16, range 2-100)
--   ef_construction: Size of dynamic candidate list during build (default 64, range 4-1000)

CREATE INDEX IF NOT EXISTS idx_topics_embedding_hnsw
ON topics
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_keywords_embedding_hnsw
ON keywords
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Option 2: IVFFlat (Inverted File Flat)
-- Best for: Balanced performance, when memory is limited
-- Pros: Faster inserts, lower memory usage
-- Cons: Slightly slower queries
-- Parameters:
--   lists: Number of inverted lists (default 100, recommendation: rows / 1000)

-- Uncomment to use IVFFlat instead of HNSW:
-- CREATE INDEX IF NOT EXISTS idx_topics_embedding_ivf
-- ON topics
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- CREATE INDEX IF NOT EXISTS idx_keywords_embedding_ivf
-- ON keywords
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- Grants (adjust user as needed)
-- GRANT ALL ON TABLE topics TO your_user;
-- GRANT ALL ON TABLE keywords TO your_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_user;
