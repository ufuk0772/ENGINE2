"""
models.py - SQLAlchemy ORM models for the Universal Data Ingestion Layer.

Schema Design:
  datasets    - Metadata about each ingested file/source.
  data_points - Individual rows stored with a timestamp and JSONB payload.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Dataset(Base):
    """
    Represents a single ingested dataset (file or API source).
    Stores metadata about the source, schema profile, and ingestion status.
    """

    __tablename__ = "datasets"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    name = Column(String(255), nullable=False, index=True)
    source_type = Column(String(50), nullable=False, default="file")  # file | api
    source_path = Column(Text, nullable=True)
    row_count = Column(Integer, nullable=True)
    datetime_column = Column(String(255), nullable=True)
    numeric_columns = Column(JSONB, nullable=True)   # list of column names
    text_columns = Column(JSONB, nullable=True)      # list of column names
    schema_profile = Column(JSONB, nullable=True)    # full SchemaProfile dict
    status = Column(String(50), nullable=False, default="pending")  # pending | success | failed
    error_message = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=_utcnow,
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=_utcnow,
        onupdate=_utcnow,
    )

    # Relationship
    data_points = relationship(
        "DataPoint",
        back_populates="dataset",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    __table_args__ = (
        UniqueConstraint("name", "source_path", name="uq_dataset_name_path"),
        Index("ix_datasets_status", "status"),
        Index("ix_datasets_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Dataset id={self.id} name='{self.name}' status='{self.status}'>"


class DataPoint(Base):
    """
    Stores a single row of ingested data.

    - dataset_id: FK to the parent dataset.
    - timestamp:  The parsed datetime value from the datetime column (if any).
    - payload:    All other column values as a JSONB document.
    - row_index:  The original row position in the source file (0-based).
    """

    __tablename__ = "data_points"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    row_index = Column(Integer, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=True, index=True)
    payload = Column(JSONB, nullable=False)
    is_valid = Column(Boolean, nullable=False, default=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=_utcnow,
    )

    # Relationship
    dataset = relationship("Dataset", back_populates="data_points")

    __table_args__ = (
        Index("ix_data_points_dataset_id", "dataset_id"),
        Index("ix_data_points_timestamp", "timestamp"),
        Index("ix_data_points_dataset_timestamp", "dataset_id", "timestamp"),
        # GIN index on payload for fast JSONB key/value queries
        Index(
            "ix_data_points_payload_gin",
            "payload",
            postgresql_using="gin",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<DataPoint id={self.id} dataset_id={self.dataset_id} "
            f"row={self.row_index} ts={self.timestamp}>"
        )
