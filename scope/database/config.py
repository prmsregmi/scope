"""Database configuration for PostgreSQL + pgvector."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class DatabaseConfig:
    """Configuration for PostgreSQL database connection."""

    host: str = "localhost"
    port: int = 5432
    dbname: str = "scope"
    user: str = "postgres"
    password: str = ""

    # Vector index configuration
    index_type: str = "hnsw"  # or "ivfflat"
    embedding_dim: int = 384  # 384 for MiniLM, 1024 for Jina

    # HNSW parameters
    hnsw_m: int = 16  # Number of bi-directional links
    hnsw_ef_construction: int = 64  # Size of dynamic candidate list during build

    # IVFFlat parameters
    ivfflat_lists: int = 100  # Number of inverted lists

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load configuration from environment variables.

        Checks for DATABASE_URL first, then individual variables.

        Returns:
            DatabaseConfig instance
        """
        database_url = os.getenv("DATABASE_URL")

        if database_url:
            # Parse DATABASE_URL (postgresql://user:password@host:port/dbname)
            return cls.from_url(database_url)

        # Load individual parameters
        return cls(
            host=os.getenv("DATABASE_HOST", "localhost"),
            port=int(os.getenv("DATABASE_PORT", "5432")),
            dbname=os.getenv("DATABASE_NAME", "scope"),
            user=os.getenv("DATABASE_USER", "postgres"),
            password=os.getenv("DATABASE_PASSWORD", ""),
            index_type=os.getenv("SCOPE_INDEX_TYPE", "hnsw"),
            embedding_dim=int(os.getenv("SCOPE_EMBEDDING_DIM", "384")),
            hnsw_m=int(os.getenv("SCOPE_HNSW_M", "16")),
            hnsw_ef_construction=int(os.getenv("SCOPE_HNSW_EF_CONSTRUCTION", "64")),
            ivfflat_lists=int(os.getenv("SCOPE_IVFFLAT_LISTS", "100")),
        )

    @classmethod
    def from_url(cls, url: str) -> "DatabaseConfig":
        """Parse database configuration from URL.

        Args:
            url: PostgreSQL connection URL

        Returns:
            DatabaseConfig instance
        """
        # Remove postgresql:// prefix
        url = url.replace("postgresql://", "").replace("postgres://", "")

        # Split credentials and connection parts
        if "@" in url:
            credentials, connection = url.split("@", 1)
            if ":" in credentials:
                user, password = credentials.split(":", 1)
            else:
                user = credentials
                password = ""
        else:
            user = "postgres"
            password = ""
            connection = url

        # Parse host, port, and database
        if "/" in connection:
            host_port, dbname = connection.split("/", 1)
        else:
            host_port = connection
            dbname = "scope"

        if ":" in host_port:
            host, port_str = host_port.split(":", 1)
            port = int(port_str)
        else:
            host = host_port
            port = 5432

        return cls(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )

    def get_connection_string(self) -> str:
        """Generate PostgreSQL connection string.

        Returns:
            Connection string for psycopg
        """
        if self.password:
            return (
                f"postgresql://{self.user}:{self.password}@"
                f"{self.host}:{self.port}/{self.dbname}"
            )
        else:
            return f"postgresql://{self.user}@{self.host}:{self.port}/{self.dbname}"

    def get_connection_params(self) -> dict:
        """Get connection parameters as dictionary.

        Returns:
            Dictionary of connection parameters
        """
        params = {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
        }

        if self.password:
            params["password"] = self.password

        return params
