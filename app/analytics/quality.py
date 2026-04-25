"""
app/analytics/quality.py
========================
Pure-logic quality metrics.  No Streamlit imports.

All functions accept a pd.DataFrame and return plain Python dicts or
scalar values that UI components can render however they choose.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_quality_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Return a comprehensive quality snapshot for *df*.

    Returns
    -------
    dict with keys:
        total_rows        int
        total_cols        int
        missing_pct       float   — overall missing-cell percentage
        duplicate_count   int
        column_types      dict[str, int]  — dtype category → column count
        missing_per_col   dict[str, float] — column → missing % (only >0)
    """
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    missing_pct = (missing_cells / total_cells * 100) if total_cells else 0.0

    missing_per_col: dict[str, float] = {}
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 0:
            missing_per_col[col] = round(pct, 2)

    return {
        "total_rows":      len(df),
        "total_cols":      len(df.columns),
        "missing_pct":     round(missing_pct, 2),
        "duplicate_count": int(df.duplicated().sum()),
        "column_types":    _column_type_summary(df),
        "missing_per_col": missing_per_col,
    }


def missing_ratio(df: pd.DataFrame) -> float:
    """Overall percentage of missing values across the entire DataFrame."""
    total = df.shape[0] * df.shape[1]
    return (df.isna().sum().sum() / total * 100) if total else 0.0


def missing_ratio_per_column(df: pd.DataFrame) -> dict[str, float]:
    """Per-column missing percentage (only columns with at least one missing value)."""
    result = {}
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 0:
            result[col] = round(pct, 2)
    return result


def duplicate_row_count(df: pd.DataFrame) -> int:
    """Number of exactly duplicated rows."""
    return int(df.duplicated().sum())


def column_type_summary(df: pd.DataFrame) -> dict[str, int]:
    """Alias kept for backward compatibility."""
    return _column_type_summary(df)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _column_type_summary(df: pd.DataFrame) -> dict[str, int]:
    """
    Categorise each column by a human-readable type label and count them.
    Categories: Numeric, DateTime, Text, Boolean, Other.
    """
    counts: dict[str, int] = {}
    for dtype in df.dtypes:
        if pd.api.types.is_numeric_dtype(dtype):
            label = "Numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            label = "DateTime"
        elif pd.api.types.is_bool_dtype(dtype):
            label = "Boolean"
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            label = "Text"
        else:
            label = "Other"
        counts[label] = counts.get(label, 0) + 1
    return counts