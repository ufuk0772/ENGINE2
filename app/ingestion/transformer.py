"""
transformer.py — Transformation Layer (v2): Reliable, metrics-tracked transformation.

Improvements over v1:
  - sanitize_value() handles numpy scalars, Decimal, NaT, timezone-aware
    datetimes, and all pandas NA variants robustly.
  - parse_timestamp() tracks per-row parse failures separately.
  - Rows are classified as valid/invalid; both are returned.
  - transform() returns a structured dict:
      {
          "records": List[Tuple[Optional[datetime], Dict]],
          "report":  Dict[str, Any],   ← transformation metrics
      }
  - Invalid rows are still preserved in records with is_valid=False
    (the repository layer decides what to do with them).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, date
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

from app.ingestion.detector import SchemaProfile, try_parse_numeric_series

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Record = Tuple[Optional[datetime], Dict[str, Any]]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def parse_timestamp(value: Any) -> Tuple[Optional[datetime], bool]:
    """
    Robustly convert a raw value to a UTC-aware datetime.

    Handles:
      - pandas Timestamp (tz-aware and tz-naive)
      - Python datetime (tz-aware and tz-naive)
      - Python date objects
      - ISO-8601 strings and many common date string formats
      - NaT, None, NaN

    Args:
        value: Raw value from the datetime column.

    Returns:
        Tuple of (datetime | None, success: bool).
        On failure: (None, False).
    """
    if value is None:
        return None, False

    # pandas NaT
    if isinstance(value, type(pd.NaT)) or (
        hasattr(pd, "NaT") and value is pd.NaT
    ):
        return None, False

    # pandas / numpy NA
    try:
        if pd.isna(value):
            return None, False
    except (TypeError, ValueError):
        pass

    # Already a tz-aware datetime
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc), True
        return value.astimezone(timezone.utc), True

    # date (not datetime)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc), True

    # pandas Timestamp
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None, False
        dt = value.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc), True

    # numpy datetime64
    if _NUMPY_AVAILABLE and isinstance(value, np.datetime64):
        if np.isnat(value):
            return None, False
        ts = pd.Timestamp(value)
        dt = ts.to_pydatetime().replace(tzinfo=timezone.utc)
        return dt, True

    # String fallback
    try:
        parsed = pd.to_datetime(str(value), infer_datetime_format=True, errors="raise", utc=True)
        return parsed.to_pydatetime().astimezone(timezone.utc), True
    except Exception:
        logger.debug(f"parse_timestamp: could not parse {repr(value)}")
        return None, False


def sanitize_value(val: Any) -> Any:
    """
    Convert any Python/pandas/numpy value to a JSON-safe primitive.

    Handles:
      - None, NaN, NaT, pd.NA → None
      - numpy integer / float scalars → Python int / float
      - Decimal → float
      - bool → bool (must check before int — bool is subclass of int)
      - pandas Timestamp / datetime → ISO-8601 string
      - date → ISO-8601 string
      - Inf / -Inf → None (not JSON-serializable)
      - Everything else → str() fallback

    Args:
        val: Any raw value.

    Returns:
        JSON-compatible Python primitive.
    """
    # --- None / missing ---
    if val is None:
        return None

    # pandas NaT
    if isinstance(val, type(pd.NaT)) or (hasattr(pd, "NaT") and val is pd.NaT):
        return None

    # pandas NA / NAType
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass

    # --- Boolean (must precede int check: bool is subclass of int) ---
    if isinstance(val, bool):
        return val

    # --- numpy scalars ---
    if _NUMPY_AVAILABLE:
        if isinstance(val, np.bool_):
            return bool(val)
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            fval = float(val)
            if math.isnan(fval) or math.isinf(fval):
                return None
            return fval
        if isinstance(val, np.datetime64):
            if np.isnat(val):
                return None
            return pd.Timestamp(val).isoformat()

    # --- Python int ---
    if isinstance(val, int):
        return val

    # --- Python float ---
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val

    # --- Decimal ---
    if isinstance(val, Decimal):
        try:
            fval = float(val)
            return None if (math.isnan(fval) or math.isinf(fval)) else fval
        except (InvalidOperation, OverflowError):
            return None

    # --- pandas Timestamp ---
    if isinstance(val, pd.Timestamp):
        if pd.isna(val):
            return None
        return val.isoformat()

    # --- Python datetime ---
    if isinstance(val, datetime):
        return val.isoformat()

    # --- Python date ---
    if isinstance(val, date):
        return val.isoformat()

    # --- str ---
    if isinstance(val, str):
        stripped = val.strip()
        return stripped if stripped else None

    # --- Fallback ---
    return str(val)


def build_payload(
    row: pd.Series,
    exclude_columns: List[str],
    numeric_columns: List[str],
) -> Dict[str, Any]:
    """
    Build a JSON-safe payload dict from a DataFrame row.

    Numeric columns that stored as formatted strings (e.g. "$1,200") are
    coerced to float before sanitization so the stored JSONB value is numeric.

    Args:
        row:             A single DataFrame row (pd.Series).
        exclude_columns: Columns to omit (e.g. the datetime column).
        numeric_columns: Columns to attempt numeric coercion on.

    Returns:
        Dict mapping column name → JSON-safe value.
    """
    exclude_set = set(exclude_columns)
    payload: Dict[str, Any] = {}

    for col in row.index:
        if col in exclude_set:
            continue

        raw = row[col]

        if col in numeric_columns and isinstance(raw, str):
            # Attempt to coerce formatted numeric strings
            coerced = try_parse_numeric_series(pd.Series([raw]))
            val = coerced.iloc[0]
            payload[col] = sanitize_value(val)
        else:
            payload[col] = sanitize_value(raw)

    return payload


def split_valid_invalid_records(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partition records into valid and invalid based on the 'is_valid' flag.

    Args:
        records: List of internal record dicts (with 'is_valid' key).

    Returns:
        (valid_records, invalid_records)
    """
    valid = [r for r in records if r.get("is_valid", True)]
    invalid = [r for r in records if not r.get("is_valid", True)]
    return valid, invalid


def generate_transformation_report(
    total_rows: int,
    valid_rows: int,
    invalid_rows: int,
    dropped_empty_rows: int,
    timestamp_parse_failures: int,
    numeric_coercion_failures: Dict[str, int],
) -> Dict[str, Any]:
    """
    Assemble a structured transformation metrics report.

    Args:
        total_rows:                  Rows in the source DataFrame.
        valid_rows:                  Rows that produced a clean record.
        invalid_rows:                Rows that had issues (still stored).
        dropped_empty_rows:          Rows dropped because all values were null.
        timestamp_parse_failures:    Rows where the datetime column failed to parse.
        numeric_coercion_failures:   {column_name: failure_count} for numeric cols.

    Returns:
        Flat dict suitable for logging or storing in the dataset metadata.
    """
    return {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "dropped_empty_rows": dropped_empty_rows,
        "timestamp_parse_failures": timestamp_parse_failures,
        "numeric_coercion_failures": numeric_coercion_failures,
        "valid_ratio": round(valid_rows / total_rows, 4) if total_rows else 0.0,
    }


# ---------------------------------------------------------------------------
# DataTransformer Class
# ---------------------------------------------------------------------------


class DataTransformer:
    """
    Transforms a raw DataFrame into clean (timestamp, payload) record pairs
    using a SchemaProfile for type guidance.

    transform() return value (v2):
    ┌──────────────────────────────────────────────────────┐
    │ {                                                    │
    │   "records": List[Tuple[Optional[datetime], dict]],  │
    │   "report":  Dict[str, Any]                          │
    │ }                                                    │
    └──────────────────────────────────────────────────────┘

    Backward compatibility note:
      v1 callers that did  records = transformer.transform()
      must be updated to   result["records"] = transformer.transform()["records"]
      The repository layer (main.py) is already updated to handle this dict.

    Args:
        df:      Loaded, normalized DataFrame.
        profile: SchemaProfile from SchemaDetector.
    """

    def __init__(self, df: pd.DataFrame, profile: SchemaProfile):
        self.df = df.copy()
        self.profile = profile

        # Metrics (populated during transform)
        self._dropped_empty_rows: int = 0
        self._timestamp_parse_failures: int = 0
        self._numeric_coercion_failures: Dict[str, int] = {
            col: 0 for col in profile.numeric_columns
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self) -> Dict[str, Any]:
        """
        Execute the full transformation pipeline.

        Steps:
          1. Drop all-null rows.
          2. Cast numeric columns (cleans formatted strings).
          3. Cast boolean columns.
          4. Cast datetime column.
          5. Iterate rows → (timestamp, payload) records.
          6. Return records + transformation report.

        Returns:
            {
                "records": List[Tuple[Optional[datetime], Dict]],
                "report":  Dict[str, Any],
            }
        """
        logger.info("Starting data transformation...")

        self._drop_fully_empty_rows()
        self._cast_numeric_columns()
        self._cast_boolean_columns()
        self._cast_datetime_column()

        records: List[Record] = []
        valid_count = 0
        invalid_count = 0

        for idx, (_, row) in enumerate(self.df.iterrows()):
            timestamp, ts_ok = self._extract_timestamp(row)
            if not ts_ok:
                self._timestamp_parse_failures += 1

            payload = build_payload(
                row=row,
                exclude_columns=[self.profile.datetime_column] if self.profile.datetime_column else [],
                numeric_columns=self.profile.numeric_columns,
            )

            # A row is invalid if it has no timestamp (when one was expected)
            # AND the entire payload is empty
            is_valid = not (payload == {} or (self.profile.datetime_column and not ts_ok and not payload))
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1

            records.append((timestamp, payload))

        report = generate_transformation_report(
            total_rows=len(self.df) + self._dropped_empty_rows,
            valid_rows=valid_count,
            invalid_rows=invalid_count,
            dropped_empty_rows=self._dropped_empty_rows,
            timestamp_parse_failures=self._timestamp_parse_failures,
            numeric_coercion_failures=self._numeric_coercion_failures,
        )

        logger.info(
            f"Transformation complete: {valid_count} valid, "
            f"{invalid_count} invalid, "
            f"{self._dropped_empty_rows} empty rows dropped, "
            f"{self._timestamp_parse_failures} timestamp parse failures."
        )

        return {"records": records, "report": report}

    # ------------------------------------------------------------------
    # Private: Cleaning steps
    # ------------------------------------------------------------------

    def _drop_fully_empty_rows(self) -> None:
        """Remove rows where all values are null."""
        before = len(self.df)
        self.df.dropna(how="all", inplace=True)
        self._dropped_empty_rows = before - len(self.df)
        if self._dropped_empty_rows:
            logger.warning(f"Dropped {self._dropped_empty_rows} fully-empty rows.")

    def _cast_numeric_columns(self) -> None:
        """
        Coerce numeric columns using try_parse_numeric_series (handles
        currency, percent, thousands separators).  Track coercion failures.
        """
        for col in self.profile.numeric_columns:
            if col not in self.df.columns:
                continue
            coerced = try_parse_numeric_series(self.df[col])
            failures = int(
                self.df[col].notna().sum() - coerced.notna().sum()
            )
            if failures > 0:
                self._numeric_coercion_failures[col] = failures
                logger.debug(f"Numeric coercion: {failures} failures in column '{col}'.")
            self.df[col] = coerced

    def _cast_boolean_columns(self) -> None:
        """Normalize boolean columns to Python bool."""
        for col in self.profile.boolean_columns:
            if col not in self.df.columns:
                continue
            from app.ingestion.detector import BOOL_TRUTHY
            self.df[col] = (
                self.df[col].astype(str).str.strip().str.lower().isin(BOOL_TRUTHY)
            )

    def _cast_datetime_column(self) -> None:
        """Parse the datetime column into UTC-aware datetime objects."""
        dt_col = self.profile.datetime_column
        if dt_col and dt_col in self.df.columns:
            self.df[dt_col] = pd.to_datetime(
                self.df[dt_col],
                infer_datetime_format=True,
                errors="coerce",
                utc=True,
            )
            logger.debug(f"Datetime column '{dt_col}' cast to UTC.")

    # ------------------------------------------------------------------
    # Private: Row processing
    # ------------------------------------------------------------------

    def _extract_timestamp(
        self, row: pd.Series
    ) -> Tuple[Optional[datetime], bool]:
        """Extract and parse the timestamp from a row."""
        dt_col = self.profile.datetime_column
        if not dt_col or dt_col not in row.index:
            return None, True  # No datetime expected — not a failure

        return parse_timestamp(row[dt_col])
