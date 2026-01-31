"""PostgreSQL vector storage for SCOPE embeddings using pgvector."""

import logging
from typing import Optional

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from scope.database.config import DatabaseConfig

logger = logging.getLogger(__name__)


class VectorStore:
    """PostgreSQL-backed vector store for embeddings.

    Provides efficient storage and retrieval of embeddings using pgvector extension.
    Supports fast similarity search using HNSW or IVFFlat indexes.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """Initialize vector store with database connection.

        Args:
            config: Database configuration. If None, loads from environment.
        """
        if config is None:
            config = DatabaseConfig.from_env()

        self.config = config
        self.conn: Optional[psycopg.Connection] = None
        self._connected = False

    def connect(self) -> None:
        """Establish database connection and register pgvector types."""
        if self._connected:
            return

        try:
            # Connect to PostgreSQL
            conn_params = self.config.get_connection_params()
            self.conn = psycopg.connect(**conn_params, row_factory=dict_row)

            # Register pgvector types
            register_vector(self.conn)

            self._connected = True
            logger.info(f"Connected to PostgreSQL at {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self._connected = False
            logger.info("Closed PostgreSQL connection")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def initialize_schema(self) -> None:
        """Create tables and indexes if they don't exist.

        Creates:
        - topics table with embeddings
        - keywords table with embeddings
        - Vector indexes for fast similarity search
        """
        if not self._connected:
            self.connect()

        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create topics table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS topics (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL UNIQUE,
                    embedding VECTOR({self.config.embedding_dim}) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Create keywords table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS keywords (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL UNIQUE,
                    embedding VECTOR({self.config.embedding_dim}) NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_used TIMESTAMP DEFAULT NOW()
                )
            """)

            # Create indexes for text lookups
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_keywords_text
                ON keywords(text)
            """)

            # Create vector indexes based on configuration
            if self.config.index_type == "hnsw":
                # HNSW index for topics
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_topics_embedding_hnsw
                    ON topics
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {self.config.hnsw_m}, ef_construction = {self.config.hnsw_ef_construction})
                """)

                # HNSW index for keywords
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_keywords_embedding_hnsw
                    ON keywords
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {self.config.hnsw_m}, ef_construction = {self.config.hnsw_ef_construction})
                """)

            elif self.config.index_type == "ivfflat":
                # IVFFlat index for topics
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_topics_embedding_ivf
                    ON topics
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {self.config.ivfflat_lists})
                """)

                # IVFFlat index for keywords
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_keywords_embedding_ivf
                    ON keywords
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {self.config.ivfflat_lists})
                """)

            self.conn.commit()
            logger.info(f"Database schema initialized with {self.config.index_type} indexes")

    def store_topic_embeddings(
        self,
        topics: dict[str, np.ndarray],
    ) -> None:
        """Store topic embeddings in PostgreSQL.

        Args:
            topics: Dictionary mapping topic names to embedding vectors
        """
        if not self._connected:
            self.connect()

        with self.conn.cursor() as cur:
            for name, embedding in topics.items():
                # Convert numpy array to list for pgvector
                emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

                cur.execute(
                    """
                    INSERT INTO topics (name, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (name) DO UPDATE SET
                        embedding = EXCLUDED.embedding
                    """,
                    (name, emb_list),
                )

            self.conn.commit()
            logger.info(f"Stored {len(topics)} topic embeddings")

    def store_keyword_embeddings(
        self,
        keywords: dict[str, np.ndarray],
    ) -> None:
        """Store keyword embeddings in PostgreSQL.

        Args:
            keywords: Dictionary mapping keyword text to embedding vectors
        """
        if not self._connected:
            self.connect()

        with self.conn.cursor() as cur:
            for text, embedding in keywords.items():
                # Convert numpy array to list for pgvector
                emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

                cur.execute(
                    """
                    INSERT INTO keywords (text, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (text) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        frequency = keywords.frequency + 1,
                        last_used = NOW()
                    """,
                    (text, emb_list),
                )

            self.conn.commit()
            logger.debug(f"Stored {len(keywords)} keyword embeddings")

    def get_keyword_embeddings(
        self,
        keywords: list[str],
    ) -> dict[str, Optional[np.ndarray]]:
        """Retrieve keyword embeddings from PostgreSQL.

        Args:
            keywords: List of keyword texts to retrieve

        Returns:
            Dictionary mapping keyword text to embedding vector (or None if not found)
        """
        if not self._connected:
            self.connect()

        if not keywords:
            return {}

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT text, embedding
                FROM keywords
                WHERE text = ANY(%s)
                """,
                (keywords,),
            )

            results = {}
            for row in cur.fetchall():
                # Convert from pgvector to numpy array
                results[row["text"]] = np.array(row["embedding"])

            # Add None for missing keywords
            for keyword in keywords:
                if keyword not in results:
                    results[keyword] = None

            return results

    def calculate_similarities(
        self,
        keywords: list[str],
        topics: list[str],
    ) -> dict[tuple[str, str], float]:
        """Calculate keyword-topic similarities using PostgreSQL vector operations.

        Uses vector indexes for fast cosine similarity calculations.

        Args:
            keywords: List of keyword texts
            topics: List of topic names

        Returns:
            Dictionary mapping (keyword, topic) tuples to similarity scores
        """
        if not self._connected:
            self.connect()

        if not keywords or not topics:
            return {}

        with self.conn.cursor() as cur:
            # Use <=> operator for cosine distance (1 - cosine_similarity)
            cur.execute(
                """
                SELECT
                    k.text as keyword,
                    t.name as topic,
                    1 - (k.embedding <=> t.embedding) as similarity
                FROM keywords k
                CROSS JOIN topics t
                WHERE k.text = ANY(%s)
                  AND t.name = ANY(%s)
                """,
                (keywords, topics),
            )

            results = {}
            for row in cur.fetchall():
                results[(row["keyword"], row["topic"])] = float(row["similarity"])

            return results

    def get_topic_embeddings(self) -> dict[str, np.ndarray]:
        """Retrieve all topic embeddings.

        Returns:
            Dictionary mapping topic names to embedding vectors
        """
        if not self._connected:
            self.connect()

        with self.conn.cursor() as cur:
            cur.execute("SELECT name, embedding FROM topics ORDER BY name")

            results = {}
            for row in cur.fetchall():
                results[row["name"]] = np.array(row["embedding"])

            return results

    def get_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with database stats (topic count, keyword count, etc.)
        """
        if not self._connected:
            self.connect()

        with self.conn.cursor() as cur:
            # Count topics
            cur.execute("SELECT COUNT(*) as count FROM topics")
            topic_count = cur.fetchone()["count"]

            # Count keywords
            cur.execute("SELECT COUNT(*) as count FROM keywords")
            keyword_count = cur.fetchone()["count"]

            # Get database size (approximate)
            cur.execute("""
                SELECT pg_size_pretty(pg_database_size(current_database())) as size
            """)
            db_size = cur.fetchone()["size"]

            return {
                "topic_count": topic_count,
                "keyword_count": keyword_count,
                "database_size": db_size,
                "index_type": self.config.index_type,
                "embedding_dim": self.config.embedding_dim,
            }

    def clear_keywords(self) -> None:
        """Clear all keyword embeddings (useful for testing)."""
        if not self._connected:
            self.connect()

        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM keywords")
            self.conn.commit()
            logger.info("Cleared all keyword embeddings")

    def clear_all(self) -> None:
        """Clear all data (topics and keywords)."""
        if not self._connected:
            self.connect()

        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM keywords")
            cur.execute("DELETE FROM topics")
            self.conn.commit()
            logger.info("Cleared all embeddings")
