"""
detector.py — Schema Detection Layer (v2): Smart, scored schema detection.

Improvements over v1:
  - Datetime column selected by composite score (name hints + parse ratio),
    not first-match.
  - Numeric detection handles currency ("$1,200"), percentages ("12.5%"),
    European decimals ("1.200,50"), and comma-separated thousands ("1,000").
  - Boolean detection: true/false, yes/no, 0/1, t/f, y/n.
  - Full column profiling: inferred_type, null_ratio, unique_ratio,
    sample_values, parse_success_ratio.
  - Unified SchemaProfile output — backward-compatible with v1 consumers.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Name fragments that suggest a datetime column (ordered by strength)
DATETIME_NAME_HINTS: list[str] = [
    "timestamp", "datetime", "date_time",
    "date", "time", "created_at", "updated_at",
    "recorded_at", "event_time", "period",
    "month", "year", "day",
]

# A column must parse as datetime for at least this fraction of non-null values
DATETIME_PARSE_THRESHOLD: float = 0.60

# A column is numeric if at least this fraction of non-null values coerce successfully
NUMERIC_PARSE_THRESHOLD: float = 0.70

# Patterns to strip before numeric coercion
NUMERIC_CLEAN_PATTERN = re.compile(
    r"""
    ^\s*              # leading whitespace
    [+\-]?            # optional sign
    [\$£€¥₹]?        # optional currency symbol
    \s*
    (.*?)             # the actual number part (captured)
    \s*
    %?                # optional trailing percent
    \s*$
    """,
    re.VERBOSE,
)

# Boolean canonical values (lowercased)
BOOL_TRUTHY = {"true", "yes", "1", "t", "y", "on"}
BOOL_FALSY = {"false", "no", "0", "f", "n", "off"}
BOOL_ALL = BOOL_TRUTHY | BOOL_FALSY

# Maximum unique values in a sample returned for profiling
SAMPLE_SIZE = 5


# ---------------------------------------------------------------------------
# SchemaProfile
# ---------------------------------------------------------------------------


@dataclass
class SchemaProfile:
    """
    Unified schema description for a dataset.

    Attributes:
        datetime_column:  Best candidate datetime column name (or None).
        numeric_columns:  List of numeric column names.
        boolean_columns:  List of boolean column names.
        text_columns:     List of text / categorical column names.
        all_columns:      All column names in original order.
        column_profiles:  Per-column profiling metadata dict.
    """

    datetime_column: Optional[str] = None
    numeric_columns: List[str] = field(default_factory=list)
    boolean_columns: List[str] = field(default_factory=list)
    text_columns: List[str] = field(default_factory=list)
    all_columns: List[str] = field(default_factory=list)
    column_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "datetime_column": self.datetime_column,
            "numeric_columns": self.numeric_columns,
            "boolean_columns": self.boolean_columns,
            "text_columns": self.text_columns,
            "all_columns": self.all_columns,
            "column_profiles": self.column_profiles,
        }


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def score_datetime_column(col_name: str, series: pd.Series) -> float:
    """
    Return a composite score [0.0–2.0] indicating how likely this column
    is the primary datetime column.

    Scoring components:
      - Name hint match (0.0–1.0):  higher for stronger name hints.
      - Parse success ratio (0.0–1.0): fraction of non-null values that
        parse successfully as datetimes.

    Args:
        col_name: Normalized column name.
        series:   The pandas Series for that column.

    Returns:
        Float score; higher = more likely to be the datetime column.
    """
    # --- Name score ---
    name_score = 0.0
    col_lower = col_name.lower()
    for idx, hint in enumerate(DATETIME_NAME_HINTS):
        if hint in col_lower:
            # Earlier (stronger) hints score higher
            name_score = 1.0 - (idx / len(DATETIME_NAME_HINTS)) * 0.5
            break

    # --- Parse ratio score ---
    parse_ratio = _datetime_parse_ratio(series)
    if parse_ratio < DATETIME_PARSE_THRESHOLD:
        return 0.0  # Below threshold → not eligible at all

    total = name_score + parse_ratio
    logger.debug(
        f"DateTime score for '{col_name}': name={name_score:.2f} "
        f"parse={parse_ratio:.2f} total={total:.2f}"
    )
    return total


def _datetime_parse_ratio(series: pd.Series) -> float:
    """Return fraction of non-null values that parse as datetime."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return 1.0

    non_null = series.dropna()
    if non_null.empty:
        return 0.0

    try:
        parsed = pd.to_datetime(
            non_null.astype(str), infer_datetime_format=True, errors="coerce"
        )
        return float(parsed.notna().sum() / len(non_null))
    except Exception:
        return 0.0


def try_parse_numeric_series(series: pd.Series) -> pd.Series:
    """
    Attempt to coerce a Series to numeric, cleaning common formatting first.

    Handles:
      - Currency symbols: $, £, €, ¥, ₹
      - Trailing percent: "12.5%"
      - Thousands separators: "1,000" → "1000"
      - European format: "1.200,50" → "1200.50"
      - Leading/trailing whitespace

    Args:
        series: Raw object/string Series.

    Returns:
        Numeric Series (NaN where coercion failed).
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = series.astype(str).str.strip()

    # Strip currency + percent wrapper
    cleaned = cleaned.str.replace(
        r"^\s*[+\-]?[\$£€¥₹]?\s*|\s*%?\s*$", "", regex=True
    )

    # Detect European number format: "1.234,56" → "1234.56"
    european_mask = cleaned.str.match(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$")
    if european_mask.any():
        cleaned = cleaned.copy()
        cleaned[european_mask] = (
            cleaned[european_mask]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
    else:
        # Standard format: remove thousands commas "1,000" → "1000"
        cleaned = cleaned.str.replace(",", "", regex=False)

    return pd.to_numeric(cleaned, errors="coerce")


def detect_boolean_columns(
    df: pd.DataFrame,
    exclude: List[Optional[str]],
) -> List[str]:
    """
    Detect columns whose values are exclusively boolean-like tokens.

    Recognized values (case-insensitive):
      Truthy : true, yes, 1, t, y, on
      Falsy  : false, no, 0, f, n, off

    Args:
        df:      DataFrame to scan.
        exclude: Column names to skip.

    Returns:
        List of boolean column names.
    """
    exclude_set = {c for c in exclude if c is not None}
    bool_cols: list[str] = []

    for col in df.columns:
        if col in exclude_set:
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            bool_cols.append(col)
            continue

        non_null = df[col].dropna()
        if non_null.empty:
            continue

        unique_vals = set(non_null.astype(str).str.lower().unique())
        if unique_vals and unique_vals.issubset(BOOL_ALL):
            bool_cols.append(col)

    logger.debug(f"Boolean columns detected: {bool_cols}")
    return bool_cols


def build_column_profile(col_name: str, series: pd.Series) -> Dict[str, Any]:
    """
    Build a statistical/type profile for a single column.

    Returns a dict containing:
      - inferred_type:        'datetime' | 'numeric' | 'boolean' | 'text'
      - null_ratio:           fraction of null values
      - unique_ratio:         fraction of unique non-null values
      - sample_values:        up to SAMPLE_SIZE representative values
      - parse_success_ratio:  (numeric/datetime only) fraction that parsed

    Args:
        col_name: Column name.
        series:   The pandas Series.

    Returns:
        Profile dict.
    """
    total = len(series)
    null_count = int(series.isnull().sum())
    null_ratio = null_count / total if total > 0 else 1.0

    non_null = series.dropna()
    unique_count = int(non_null.nunique())
    unique_ratio = unique_count / len(non_null) if len(non_null) > 0 else 0.0

    sample_raw = non_null.head(SAMPLE_SIZE).tolist()
    sample_values = [_safe_repr(v) for v in sample_raw]

    profile: Dict[str, Any] = {
        "null_ratio": round(null_ratio, 4),
        "unique_ratio": round(unique_ratio, 4),
        "total_count": total,
        "null_count": null_count,
        "unique_count": unique_count,
        "sample_values": sample_values,
        "inferred_type": "text",           # overwritten below
        "parse_success_ratio": None,
    }

    # --- Infer type ---
    if pd.api.types.is_bool_dtype(series):
        profile["inferred_type"] = "boolean"
        return profile

    # Boolean via value set
    unique_str = set(non_null.astype(str).str.lower().unique())
    if unique_str and unique_str.issubset(BOOL_ALL):
        profile["inferred_type"] = "boolean"
        return profile

    # Datetime
    dt_ratio = _datetime_parse_ratio(series)
    if dt_ratio >= DATETIME_PARSE_THRESHOLD:
        profile["inferred_type"] = "datetime"
        profile["parse_success_ratio"] = round(dt_ratio, 4)
        return profile

    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        profile["inferred_type"] = "numeric"
        profile["parse_success_ratio"] = 1.0
        return profile

    coerced = try_parse_numeric_series(series)
    if len(non_null) > 0:
        num_ratio = float(coerced.notna().sum() / len(non_null))
        if num_ratio >= NUMERIC_PARSE_THRESHOLD:
            profile["inferred_type"] = "numeric"
            profile["parse_success_ratio"] = round(num_ratio, 4)
            return profile

    profile["inferred_type"] = "text"
    return profile


def generate_schema_profile(df: pd.DataFrame) -> SchemaProfile:
    """
    Analyze the full DataFrame and return a populated SchemaProfile.

    This is the main entry point used by SchemaDetector.detect().

    Steps:
      1. Build per-column profiles.
      2. Score all columns for datetime candidacy; pick best above threshold.
      3. Detect boolean columns (excluding datetime).
      4. Detect numeric columns (excluding datetime + boolean).
      5. Remaining columns → text.

    Args:
        df: Normalized DataFrame (column names already cleaned).

    Returns:
        Populated SchemaProfile.
    """
    if df.empty:
        raise ValueError("Cannot generate schema profile for an empty DataFrame.")

    profile = SchemaProfile(all_columns=list(df.columns))

    # --- 1. Build column profiles ---
    for col in df.columns:
        profile.column_profiles[col] = build_column_profile(col, df[col])

    # --- 2. Detect datetime column (scored) ---
    dt_scores: dict[str, float] = {}
    for col in df.columns:
        score = score_datetime_column(col, df[col])
        if score > 0:
            dt_scores[col] = score

    if dt_scores:
        best_dt = max(dt_scores, key=lambda c: dt_scores[c])
        profile.datetime_column = best_dt
        profile.column_profiles[best_dt]["inferred_type"] = "datetime"
        logger.info(
            f"Datetime column: '{best_dt}' (score={dt_scores[best_dt]:.3f})"
        )
    else:
        logger.warning("No datetime column detected.")

    # --- 3. Detect boolean columns ---
    profile.boolean_columns = detect_boolean_columns(
        df, exclude=[profile.datetime_column]
    )
    for col in profile.boolean_columns:
        profile.column_profiles[col]["inferred_type"] = "boolean"

    # --- 4. Detect numeric columns ---
    excluded = {profile.datetime_column} | set(profile.boolean_columns)
    for col in df.columns:
        if col in excluded:
            continue
        col_profile = profile.column_profiles[col]
        if col_profile["inferred_type"] == "numeric":
            profile.numeric_columns.append(col)
        elif col_profile["inferred_type"] != "datetime":
            # Re-check with try_parse_numeric_series for formatted numbers
            coerced = try_parse_numeric_series(df[col])
            non_null = df[col].dropna()
            if len(non_null) > 0:
                ratio = float(coerced.notna().sum() / len(non_null))
                if ratio >= NUMERIC_PARSE_THRESHOLD:
                    profile.numeric_columns.append(col)
                    col_profile["inferred_type"] = "numeric"
                    col_profile["parse_success_ratio"] = round(ratio, 4)

    # --- 5. Text columns ---
    used = (
        {profile.datetime_column}
        | set(profile.numeric_columns)
        | set(profile.boolean_columns)
    )
    profile.text_columns = [c for c in df.columns if c not in used]

    logger.info(
        f"Schema profile complete — "
        f"datetime={profile.datetime_column}, "
        f"numeric={len(profile.numeric_columns)}, "
        f"boolean={len(profile.boolean_columns)}, "
        f"text={len(profile.text_columns)}"
    )
    return profile


def _safe_repr(value: Any) -> Any:
    """Convert a value to a JSON-safe primitive for sample display."""
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# SchemaDetector Class (public API — unchanged interface)
# ---------------------------------------------------------------------------


class SchemaDetector:
    """
    Analyzes a DataFrame and produces a SchemaProfile.

    The public interface (detect() → SchemaProfile) is unchanged from v1,
    but the internal implementation now uses scored selection and richer
    column profiling.

    Args:
        df: Normalized DataFrame (column names already cleaned by FileLoader).
    """

    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Cannot detect schema of an empty DataFrame.")
        self.df = df.copy()

    def detect(self) -> SchemaProfile:
        """
        Run full schema detection and return a populated SchemaProfile.

        Returns:
            SchemaProfile with datetime_column, numeric_columns,
            boolean_columns, text_columns, and column_profiles populated.
        """
        logger.info("Starting schema detection...")
        profile = generate_schema_profile(self.df)
        return profile
