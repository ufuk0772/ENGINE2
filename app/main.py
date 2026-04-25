"""
main.py - Orchestration Entry Point for the Universal Data Ingestion Layer.

Coordinates the full pipeline:
  1. Load file (CSV/Excel)
  2. Detect schema
  3. Transform rows into (timestamp, payload) pairs
  4. Persist to PostgreSQL

Usage:
    python -m app.main --file data/sample_data.csv
    python -m app.main --file data/sales.xlsx --name "Q1 Sales"
"""

import argparse
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

from app.db.connection import DatabaseConnection
from app.db.models import Base
from app.db.repository import DataPointRepository, DatasetRepository
from app.ingestion.detector import SchemaDetector
from app.ingestion.loader import FileLoader
from app.ingestion.report import IngestionReportBuilder, format_report
from app.ingestion.transformer import DataTransformer

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestionPipeline:
    """
    Orchestrates the complete data ingestion workflow.
    Designed to be reusable (can be called multiple times with different files).
    """

    def __init__(self, db: DatabaseConnection):
        self.db = db

    def run(
        self,
        file_path: str,
        dataset_name: Optional[str] = None,
        encoding: str = "utf-8",
        sheet_name: Optional[str] = None,
    ) -> uuid.UUID:
        """
        Execute the full ingestion pipeline.

        Args:
            file_path:    Path to the CSV or Excel file.
            dataset_name: Human-readable name for the dataset. Defaults to filename.
            encoding:     File encoding (CSV only).
            sheet_name:   Sheet name (Excel only).

        Returns:
            The UUID of the created Dataset record.
        """
        path = Path(file_path)
        name = dataset_name or path.stem

        logger.info("=" * 60)
        logger.info(f"INGESTION PIPELINE STARTED: '{name}'")
        logger.info(f"  Source: {path.resolve()}")
        logger.info("=" * 60)

        dataset_id: Optional[uuid.UUID] = None

        try:
            # ------------------------------------------------------------------
            # Step 1: Load file
            # ------------------------------------------------------------------
            logger.info("[1/4] Loading file...")
            loader = FileLoader(file_path=str(path), encoding=encoding, sheet_name=sheet_name)
            df = loader.load()
            logger.info(f"      Loaded {len(df)} rows × {len(df.columns)} columns")

            # ------------------------------------------------------------------
            # Step 2: Detect schema
            # ------------------------------------------------------------------
            logger.info("[2/4] Detecting schema...")
            detector = SchemaDetector(df)
            profile = detector.detect()
            logger.info(f"      Datetime column : {profile.datetime_column or 'None'}")
            logger.info(f"      Numeric columns : {profile.numeric_columns}")
            logger.info(f"      Text columns    : {profile.text_columns}")
            logger.info(f"      Boolean columns : {profile.boolean_columns}")

            # ------------------------------------------------------------------
            # Step 3: Transform rows
            # ------------------------------------------------------------------
            logger.info("[3/4] Transforming data...")
            transformer = DataTransformer(df=df, profile=profile)
            transform_result = transformer.transform()
            records = transform_result["records"]
            transform_report = transform_result["report"]
            logger.info(f"      {len(records)} records ready for insertion")

            # ------------------------------------------------------------------
            # Step 4: Persist to database
            # ------------------------------------------------------------------
            logger.info("[4/4] Persisting to PostgreSQL...")
            with self.db.get_session() as session:
                ds_repo = DatasetRepository(session)
                dp_repo = DataPointRepository(session)

                # Create dataset metadata record
                dataset = ds_repo.create(
                    name=name,
                    source_path=str(path.resolve()),
                    profile=profile,
                    source_type="file",
                )
                dataset_id = dataset.id

                # Bulk insert data points
                inserted = dp_repo.bulk_insert(dataset_id=dataset_id, records=records)

                # Mark success
                ds_repo.update_status(
                    dataset_id=dataset_id,
                    status="success",
                    row_count=inserted,
                )

            logger.info("=" * 60)
            logger.info(f"INGESTION COMPLETE")
            logger.info(f"  Dataset ID : {dataset_id}")
            logger.info(f"  Rows stored: {len(records)}")
            logger.info("=" * 60)

            # Emit structured report
            report_builder = IngestionReportBuilder(
                dataset_id=str(dataset_id),
                source_path=str(path.resolve()),
                schema_profile=profile,
                transformation_report=transform_report,
            )
            summary = report_builder.build()
            logger.info("\n" + format_report(summary))

            return dataset_id

        except Exception as exc:
            logger.error(f"Pipeline failed: {exc}", exc_info=True)

            # Attempt to mark the dataset as failed if it was created
            if dataset_id:
                try:
                    with self.db.get_session() as session:
                        DatasetRepository(session).update_status(
                            dataset_id=dataset_id,
                            status="failed",
                            error_message=str(exc),
                        )
                except Exception as update_exc:
                    logger.error(f"Could not update dataset status: {update_exc}")

            raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Universal Data Ingestion Layer — ingest CSV/Excel into PostgreSQL"
    )
    parser.add_argument("--file", required=True, help="Path to the CSV or Excel file")
    parser.add_argument("--name", default=None, help="Dataset name (defaults to filename)")
    parser.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    parser.add_argument("--sheet", default=None, help="Excel sheet name (default: first sheet)")
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Create all tables before ingestion (safe to run multiple times)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Initialize DB connection
    db = DatabaseConnection.get_instance()

    if not db.verify_connection():
        logger.error("Cannot connect to PostgreSQL. Check your environment variables.")
        sys.exit(1)

    if args.init_db:
        logger.info("Initializing database schema...")
        Base.metadata.create_all(bind=db.engine)
        logger.info("Schema ready.")

    # Run pipeline
    pipeline = IngestionPipeline(db=db)
    pipeline.run(
        file_path=args.file,
        dataset_name=args.name,
        encoding=args.encoding,
        sheet_name=args.sheet,
    )


if __name__ == "__main__":
    main()
