"""
connection.py - Database connection management using SQLAlchemy.
Provides a thread-safe session factory and connection pooling.
"""

import logging
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


def _build_database_url() -> str:
    """
    Build the PostgreSQL connection URL from environment variables.
    Falls back to safe defaults suitable for local / Docker development.
    """
    user = os.getenv("POSTGRES_USER", "ingestion_user")
    password = os.getenv("POSTGRES_PASSWORD", "ingestion_pass")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "ingestion_db")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


class DatabaseConnection:
    """
    Singleton-style class that manages the SQLAlchemy engine and session factory.
    """

    _instance: "DatabaseConnection | None" = None

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or _build_database_url()
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        logger.info("DatabaseConnection initialized.")

    @classmethod
    def get_instance(cls) -> "DatabaseConnection":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _create_engine(self):
        engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,          # Verify connections before use
            pool_recycle=3600,           # Recycle connections every hour
            echo=False,                  # Set True to log SQL statements
            connect_args={
                "connect_timeout": 10,
                "application_name": "universal_ingestion_layer",
            },
        )

        # Ensure JSONB support
        @event.listens_for(engine, "connect")
        def set_search_path(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("SET search_path TO public")
            cursor.close()

        return engine

    def verify_connection(self) -> bool:
        """Verify the database is reachable."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection verified.")
            return True
        except OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            return False

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Provide a transactional session scope."""
        session: Session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        """Dispose the connection pool (for clean shutdown)."""
        self.engine.dispose()
        logger.info("Database engine disposed.")
