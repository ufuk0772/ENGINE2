"""
report.py — Ingestion Report Builder (v2).

Combines the transformation report (from transformer.py) and schema profile
(from detector.py) into a unified ingestion summary, with a human-readable
formatter for logs and CLI output.

Usage:
    from app.ingestion.report import IngestionReportBuilder, format_report

    builder = IngestionReportBuilder(
        dataset_id=str(dataset.id),
        source_path="data/sales.csv",
        schema_profile=profile,
        transformation_report=result["report"],
    )
    summary = builder.build()
    print(format_report(summary))
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.ingestion.detector import SchemaProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report Builder
# ---------------------------------------------------------------------------


class IngestionReportBuilder:
    """
    Assembles a unified ingestion summary from schema and transformation data.

    Args:
        dataset_id:             UUID string of the stored Dataset record.
        source_path:            Absolute or relative path of the source file.
        schema_profile:         SchemaProfile produced by SchemaDetector.
        transformation_report:  Metrics dict produced by DataTransformer.transform().
        extra_metadata:         Any additional key/value pairs to include.
    """

    def __init__(
        self,
        dataset_id: str,
        source_path: str,
        schema_profile: SchemaProfile,
        transformation_report: Dict[str, Any],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_id = dataset_id
        self.source_path = source_path
        self.schema_profile = schema_profile
        self.transformation_report = transformation_report
        self.extra_metadata = extra_metadata or {}

    def build(self) -> Dict[str, Any]:
        """
        Build and return the full ingestion summary dict.

        Returns:
            {
                "dataset_id":          str,
                "source_path":         str,
                "generated_at":        ISO-8601 timestamp,
                "schema": {
                    "datetime_column":  str | None,
                    "numeric_columns":  List[str],
                    "boolean_columns":  List[str],
                    "text_columns":     List[str],
                    "total_columns":    int,
                    "column_profiles":  Dict[str, Dict],
                },
                "transformation": {
                    "total_rows":                int,
                    "valid_rows":                int,
                    "invalid_rows":              int,
                    "dropped_empty_rows":        int,
                    "timestamp_parse_failures":  int,
                    "numeric_coercion_failures": Dict[str, int],
                    "valid_ratio":               float,
                },
                "quality_flags":   List[str],   ← warnings / issues detected
                "extra_metadata":  Dict,
            }
        """
        report = {
            "dataset_id": self.dataset_id,
            "source_path": self.source_path,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "schema": self._build_schema_section(),
            "transformation": self.transformation_report,
            "quality_flags": self._build_quality_flags(),
            "extra_metadata": self.extra_metadata,
        }
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_schema_section(self) -> Dict[str, Any]:
        """Flatten the SchemaProfile into the report schema section."""
        sp = self.schema_profile
        return {
            "datetime_column": sp.datetime_column,
            "numeric_columns": sp.numeric_columns,
            "boolean_columns": sp.boolean_columns,
            "text_columns": sp.text_columns,
            "total_columns": len(sp.all_columns),
            "column_profiles": sp.column_profiles,
        }

    def _build_quality_flags(self) -> List[str]:
        """
        Generate human-readable quality warnings based on the ingestion data.

        Returns:
            List of warning strings (empty list = no issues detected).
        """
        flags: List[str] = []
        tr = self.transformation_report
        sp = self.schema_profile

        # --- Missing datetime column ---
        if sp.datetime_column is None:
            flags.append("NO_DATETIME_COLUMN: No datetime column was detected.")

        # --- Timestamp parse failures ---
        ts_failures = tr.get("timestamp_parse_failures", 0)
        total = tr.get("total_rows", 1) or 1
        if ts_failures > 0:
            ratio = ts_failures / total
            flags.append(
                f"TIMESTAMP_PARSE_FAILURES: {ts_failures} rows ({ratio:.1%}) "
                "had unparseable timestamp values."
            )

        # --- Numeric coercion failures ---
        for col, count in tr.get("numeric_coercion_failures", {}).items():
            if count > 0:
                flags.append(
                    f"NUMERIC_COERCION_FAILURE [{col}]: "
                    f"{count} values could not be coerced to numeric."
                )

        # --- High null ratio columns ---
        for col, prof in sp.column_profiles.items():
            null_ratio = prof.get("null_ratio", 0.0)
            if null_ratio > 0.5:
                flags.append(
                    f"HIGH_NULL_RATIO [{col}]: {null_ratio:.1%} of values are null."
                )

        # --- Low valid row ratio ---
        valid_ratio = tr.get("valid_ratio", 1.0)
        if valid_ratio < 0.9:
            flags.append(
                f"LOW_VALID_RATIO: Only {valid_ratio:.1%} of rows are valid. "
                "Check source data quality."
            )

        # --- Dropped rows ---
        dropped = tr.get("dropped_empty_rows", 0)
        if dropped > 0:
            flags.append(f"EMPTY_ROWS_DROPPED: {dropped} fully-empty rows were removed.")

        return flags


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def format_report(report: Dict[str, Any], width: int = 64) -> str:
    """
    Format an ingestion summary dict as a human-readable text block.

    Args:
        report: Dict produced by IngestionReportBuilder.build().
        width:  Line width for the separator (default 64).

    Returns:
        Multi-line string ready for logging or CLI output.
    """
    sep = "─" * width
    lines: List[str] = [
        sep,
        "  INGESTION REPORT",
        sep,
        f"  Dataset ID   : {report.get('dataset_id', 'N/A')}",
        f"  Source       : {report.get('source_path', 'N/A')}",
        f"  Generated At : {report.get('generated_at', 'N/A')}",
        sep,
        "  SCHEMA",
        sep,
    ]

    schema = report.get("schema", {})
    lines += [
        f"  Total Columns    : {schema.get('total_columns', 0)}",
        f"  Datetime Column  : {schema.get('datetime_column') or '— not detected —'}",
        f"  Numeric Columns  : {schema.get('numeric_columns', [])}",
        f"  Boolean Columns  : {schema.get('boolean_columns', [])}",
        f"  Text Columns     : {schema.get('text_columns', [])}",
        sep,
        "  TRANSFORMATION METRICS",
        sep,
    ]

    tr = report.get("transformation", {})
    lines += [
        f"  Total Rows       : {tr.get('total_rows', 0):,}",
        f"  Valid Rows       : {tr.get('valid_rows', 0):,}",
        f"  Invalid Rows     : {tr.get('invalid_rows', 0):,}",
        f"  Dropped (empty)  : {tr.get('dropped_empty_rows', 0):,}",
        f"  Valid Ratio      : {tr.get('valid_ratio', 0.0):.1%}",
        f"  TS Parse Failures: {tr.get('timestamp_parse_failures', 0):,}",
    ]

    num_failures = tr.get("numeric_coercion_failures", {})
    if num_failures:
        lines.append("  Numeric Failures :")
        for col, count in num_failures.items():
            if count > 0:
                lines.append(f"    └─ {col}: {count:,} values")

    flags = report.get("quality_flags", [])
    if flags:
        lines += [sep, "  ⚠  QUALITY FLAGS"]
        for flag in flags:
            lines.append(f"  ! {flag}")
    else:
        lines += [sep, "  ✓  No quality issues detected."]

    lines.append(sep)
    return "\n".join(lines)
