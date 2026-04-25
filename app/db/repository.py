"""
repository.py - Data Access Layer: All database read/write operations.
Follows the Repository pattern to decouple business logic from persistence.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db.models import DataPoint, Dataset
from app.ingestion.detector import SchemaProfile

logger = logging.getLogger(__name__)

# Batch size for bulk inserts
INSERT_BATCH_SIZE = 500


class DatasetRepository:
    """Handles CRUD operations for the Dataset entity."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        name: str,
        source_path: str,
        profile: SchemaProfile,
        source_type: str = "file",
    ) -> Dataset:
        """Create and persist a new Dataset record."""
        dataset = Dataset(
            id=uuid.uuid4(),
            name=name,
            source_type=source_type,
            source_path=source_path,
            datetime_column=profile.datetime_column,
            numeric_columns=profile.numeric_columns,
            text_columns=profile.text_columns,
            schema_profile=profile.to_dict(),
            status="pending",
        )
        self.session.add(dataset)
        self.session.flush()  # Get the ID without committing
        logger.debug(f"Dataset record created: {dataset.id}")
        return dataset

    def update_status(
        self,
        dataset_id: uuid.UUID,
        status: str,
        row_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update the status of an existing Dataset."""
        dataset = self.session.get(Dataset, dataset_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_id}")

        dataset.status = status
        if row_count is not None:
            dataset.row_count = row_count
        if error_message is not None:
            dataset.error_message = error_message

        logger.debug(f"Dataset {dataset_id} status → '{status}'")

    def get_by_id(self, dataset_id: uuid.UUID) -> Optional[Dataset]:
        return self.session.get(Dataset, dataset_id)

    def list_all(self, limit: int = 100, offset: int = 0) -> List[Dataset]:
        return (
            self.session.query(Dataset)
            .order_by(Dataset.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )


class DataPointRepository:
    """Handles bulk insert and query operations for DataPoint entities."""

    def __init__(self, session: Session):
        self.session = session

    def bulk_insert(
        self,
        dataset_id: uuid.UUID,
        records: List[Tuple[Optional[datetime], Dict[str, Any]]],
    ) -> int:
        """
        Insert records in batches using PostgreSQL's efficient COPY-style bulk insert.
        Returns the total number of rows inserted.
        """
        if not records:
            logger.warning("No records to insert.")
            return 0

        total_inserted = 0
        batches = self._chunk(records, INSERT_BATCH_SIZE)

        for batch_num, batch in enumerate(batches, start=1):
            rows = [
                {
                    "dataset_id": str(dataset_id),
                    "row_index": idx,
                    "timestamp": timestamp,
                    "payload": payload,
                    "is_valid": True,
                }
                for idx, (timestamp, payload) in enumerate(
                    batch, start=total_inserted
                )
            ]

            stmt = pg_insert(DataPoint).values(rows)
            result = self.session.execute(stmt)
            total_inserted += len(rows)
            logger.debug(
                f"Batch {batch_num}: inserted {len(rows)} rows "
                f"(total so far: {total_inserted})"
            )

        logger.info(f"Bulk insert complete: {total_inserted} rows for dataset {dataset_id}")
        return total_inserted

    def query_by_dataset(
        self,
        dataset_id: uuid.UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[DataPoint]:
        """Query data points for a dataset with optional time-range filter."""
        q = self.session.query(DataPoint).filter(
            DataPoint.dataset_id == dataset_id
        )
        if start_time:
            q = q.filter(DataPoint.timestamp >= start_time)
        if end_time:
            q = q.filter(DataPoint.timestamp <= end_time)

        return q.order_by(DataPoint.timestamp.asc()).limit(limit).all()

    def query_payload_field(
        self,
        dataset_id: uuid.UUID,
        field_name: str,
        field_value: Any,
        limit: int = 100,
    ) -> List[DataPoint]:
        """
        Query data points where payload contains a specific key-value pair.
        Uses the GIN index for performance.
        """
        return (
            self.session.query(DataPoint)
            .filter(
                DataPoint.dataset_id == dataset_id,
                DataPoint.payload[field_name].astext == str(field_value),
            )
            .limit(limit)
            .all()
        )

    def aggregate_numeric(
        self,
        dataset_id: uuid.UUID,
        field_name: str,
    ) -> Dict[str, Any]:
        """
        Run min/max/avg aggregation on a JSONB numeric field using raw SQL.
        Returns a dict with min, max, avg, count.
        """
        sql = text("""
            SELECT
                COUNT(*)::int                                         AS count,
                MIN((payload->>:field)::numeric)                      AS min,
                MAX((payload->>:field)::numeric)                      AS max,
                AVG((payload->>:field)::numeric)                      AS avg
            FROM data_points
            WHERE dataset_id = :dataset_id
              AND payload->>:field IS NOT NULL
              AND (payload->>:field) ~ '^-?[0-9]+(\\.[0-9]+)?$'
        """)
        row = self.session.execute(
            sql, {"dataset_id": str(dataset_id), "field": field_name}
        ).fetchone()

        if row:
            return {
                "field": field_name,
                "count": row.count,
                "min": float(row.min) if row.min is not None else None,
                "max": float(row.max) if row.max is not None else None,
                "avg": float(row.avg) if row.avg is not None else None,
            }
        return {}

    @staticmethod
    def _chunk(lst: list, size: int):
        for i in range(0, len(lst), size):
            yield lst[i : i + size]
