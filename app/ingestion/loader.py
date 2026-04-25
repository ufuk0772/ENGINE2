"""
app/ingestion/loader.py
=======================
Responsible for:
  • Computing a deterministic hash for uploaded file bytes
    (used by the orchestrator for hash-based cache invalidation).
  • Loading CSV / Excel files into a normalised pd.DataFrame.
  • Resetting session-state dataset keys when a file is cleared.
  • Lightweight column-level validation helpers consumed by tabs and analytics.

No Streamlit UI commands live here except the single st.error() call in
load_uploaded_file(), which is intentional — it surfaces parse errors right
where the upload widget lives without coupling the caller to error handling.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Public: hash
# ---------------------------------------------------------------------------

def compute_file_hash(file_bytes: bytes) -> str:
    """Return an MD5 hex-digest of *file_bytes* for cache-control purposes."""
    return hashlib.md5(file_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Public: session-state reset
# ---------------------------------------------------------------------------

def reset_dataset_state() -> None:
    """
    Clear all dataset-related keys from st.session_state.
    Called by the orchestrator when the file uploader returns None.
    """
    keys_to_clear = [
        "df", "profile", "filename", "file_hash",
        "col_date", "col_sales", "col_production",
        "col_stock", "col_defect", "col_category",
    ]
    for key in keys_to_clear:
        st.session_state[key] = None if key not in {"filename", "file_hash"} else ""


# ---------------------------------------------------------------------------
# Public: file loading
# ---------------------------------------------------------------------------

def load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Persist the Streamlit UploadedFile to a temp path, parse it with
    FileLoader, normalise column names, and return a pd.DataFrame.

    Returns None and surfaces an st.error() if anything goes wrong.
    """
    suffix = Path(uploaded_file.name).suffix.lower()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        loader = FileLoader(file_path=tmp_path)
        df = loader.load()

        if df is None or df.empty:
            st.error("The file was uploaded successfully but contains no usable rows.")
            return None

        df = normalize_column_names(df)
        return df

    except Exception as exc:
        st.error(f"Could not parse file: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public: column-level validation helpers
# ---------------------------------------------------------------------------

def is_valid_datetime_column(df: pd.DataFrame, column: Optional[str]) -> bool:
    """
    Return True if *column* exists in *df* and contains at least 3
    parseable datetime values.
    """
    if not column or column not in df.columns:
        return False
    parsed = pd.to_datetime(df[column], errors="coerce")
    return int(parsed.notna().sum()) >= 3


def is_valid_numeric_column(df: pd.DataFrame, column: Optional[str]) -> bool:
    """
    Return True if *column* exists in *df* and contains at least 3
    parseable numeric values.
    """
    if not column or column not in df.columns:
        return False
    numeric = pd.to_numeric(df[column], errors="coerce")
    return int(numeric.notna().sum()) >= 3


# ---------------------------------------------------------------------------
# Internal: FileLoader + normalize_column_names
# (kept here so ingestion is self-contained; previously in loader.py)
# ---------------------------------------------------------------------------

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase all column names, strip surrounding whitespace, and replace
    spaces / hyphens with underscores for consistent programmatic access.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    return df


class FileLoader:
    """
    Thin wrapper that dispatches to pandas read_csv / read_excel based on
    the file extension detected from *file_path*.
    """

    _CSV_EXTENSIONS  = {".csv", ".tsv", ".txt"}
    _EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".xlsb"}

    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)

    def load(self) -> Optional[pd.DataFrame]:
        ext = self._path.suffix.lower()
        if ext in self._CSV_EXTENSIONS:
            return self._load_csv()
        if ext in self._EXCEL_EXTENSIONS:
            return self._load_excel()
        raise ValueError(f"Unsupported file extension: '{ext}'")

    # ── private helpers ──────────────────────────────────────────────────

    def _load_csv(self) -> pd.DataFrame:
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(self._path, encoding=enc)
            except UnicodeDecodeError:
                continue
        # Last resort — ignore bad bytes
        return pd.read_csv(self._path, encoding="utf-8", errors="replace")

    def _load_excel(self) -> pd.DataFrame:
        return pd.read_excel(self._path)