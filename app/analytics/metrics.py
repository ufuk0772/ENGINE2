"""
metrics.py — Analytics helpers for data quality, trend profiling, and KPI extraction.

All functions are pure (no side-effects) and operate on pandas Series / DataFrames.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data Quality Metrics
# ---------------------------------------------------------------------------


def missing_ratio(df: pd.DataFrame) -> float:
    """Return the fraction of cells that are null across the entire DataFrame."""
    if df.empty:
        return 0.0
    return float(df.isnull().values.mean())


def missing_ratio_per_column(df: pd.DataFrame) -> Dict[str, float]:
    """Return null ratio for each column as {col: ratio}."""
    return {col: float(df[col].isnull().mean()) for col in df.columns}


def duplicate_row_count(df: pd.DataFrame) -> int:
    """Return the number of fully duplicate rows."""
    return int(df.duplicated().sum())


def column_type_summary(df: pd.DataFrame) -> Dict[str, str]:
    """
    Return a human-readable dtype label for each column.
    Maps pandas dtypes to: 'numeric', 'datetime', 'boolean', 'text', 'categorical'.
    """
    summary: Dict[str, str] = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            summary[col] = "boolean"
        elif pd.api.types.is_numeric_dtype(dtype):
            summary[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            summary[col] = "datetime"
        else:
            n_unique = df[col].nunique()
            n_total = len(df[col].dropna())
            ratio = n_unique / n_total if n_total > 0 else 0
            summary[col] = "categorical" if ratio < 0.5 else "text"
    return summary


def descriptive_stats(series: pd.Series) -> Dict[str, float]:
    """Return min, max, mean, median, std, p25, p75 for a numeric Series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {}
    return {
        "min":    float(clean.min()),
        "max":    float(clean.max()),
        "mean":   float(clean.mean()),
        "median": float(clean.median()),
        "std":    float(clean.std()),
        "p25":    float(clean.quantile(0.25)),
        "p75":    float(clean.quantile(0.75)),
    }


# ---------------------------------------------------------------------------
# Rolling / Trend Metrics
# ---------------------------------------------------------------------------


def rolling_mean(series: pd.Series, window: int = 7) -> pd.Series:
    """Compute rolling mean with min_periods=1."""
    return series.rolling(window=window, min_periods=1).mean()


def rolling_std(series: pd.Series, window: int = 7) -> pd.Series:
    """Compute rolling standard deviation (volatility proxy)."""
    return series.rolling(window=window, min_periods=2).std()


def recent_trend_direction(
    series: pd.Series,
    lookback: int = 14,
    threshold_pct: float = 2.0,
) -> str:
    """
    Determine recent trend direction: 'upward' | 'downward' | 'flat'.

    Compares mean of last `lookback` periods vs prior `lookback` periods.
    Falls back to shorter windows for small datasets.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)

    if n < 4:
        return "flat"

    # Adapt lookback for small datasets
    lookback = min(lookback, n // 2)
    lookback = max(lookback, 2)

    recent = clean.iloc[-lookback:].mean()
    prior  = clean.iloc[max(0, -lookback * 2) : -lookback].mean()

    if prior == 0 or pd.isna(prior):
        return "flat"

    pct_change = ((recent - prior) / abs(prior)) * 100

    if pct_change > threshold_pct:
        return "upward"
    elif pct_change < -threshold_pct:
        return "downward"
    return "flat"


def trend_percentage_change(
    series: pd.Series,
    lookback: int = 14,
) -> float:
    """
    Return % change between the recent and prior period means.
    Returns 0.0 if not enough data.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)
    if n < 4:
        return 0.0
    lookback = min(lookback, n // 2)
    lookback = max(lookback, 2)
    recent = float(clean.iloc[-lookback:].mean())
    prior  = float(clean.iloc[max(0, -lookback * 2) : -lookback].mean())
    if prior == 0:
        return 0.0
    return ((recent - prior) / abs(prior)) * 100


def volatility_level(series: pd.Series) -> Tuple[str, float]:
    """
    Classify coefficient of variation as 'low' | 'moderate' | 'high'.

    Returns:
        (label, cv)
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty or clean.mean() == 0:
        return "unknown", 0.0

    cv = float(clean.std() / abs(clean.mean()))

    if cv < 0.15:
        label = "low"
    elif cv < 0.40:
        label = "moderate"
    else:
        label = "high"

    return label, round(cv, 4)


# ---------------------------------------------------------------------------
# KPI extraction (new — for the top KPI bar)
# ---------------------------------------------------------------------------


def extract_kpis(
    df: pd.DataFrame,
    value_col: Optional[str] = None,
    lookback: int = 14,
) -> Dict[str, Any]:
    """
    Extract the four KPIs shown in the top dashboard bar.

    Args:
        df:        The loaded DataFrame.
        value_col: The target numeric column name (or None if not selected).
        lookback:  Periods for trend calculation.

    Returns:
        {
            "total_rows":       int,
            "missing_pct":      float (0–1),
            "trend_direction":  str  ('upward'|'downward'|'flat'|'unknown'),
            "trend_pct_change": float,
            "volatility_label": str  ('low'|'moderate'|'high'|'unknown'),
            "volatility_cv":    float,
        }
    """
    kpis: Dict[str, Any] = {
        "total_rows":       len(df),
        "missing_pct":      missing_ratio(df),
        "trend_direction":  "unknown",
        "trend_pct_change": 0.0,
        "volatility_label": "unknown",
        "volatility_cv":    0.0,
    }

    if value_col and value_col in df.columns:
        series = pd.to_numeric(df[value_col], errors="coerce").dropna()
        kpis["trend_direction"]  = recent_trend_direction(series, lookback)
        kpis["trend_pct_change"] = trend_percentage_change(series, lookback)
        vol_label, cv            = volatility_level(series)
        kpis["volatility_label"] = vol_label
        kpis["volatility_cv"]    = cv

    return kpis


# ---------------------------------------------------------------------------
# Forecast interpretation (new — rule-based insight text)
# ---------------------------------------------------------------------------


def interpret_forecast(
    forecast_mean: float,
    recent_mean: float,
    horizon: int,
    value_col: str,
) -> str:
    """
    Return a short, client-facing forecast interpretation sentence.

    Example: "Forecast suggests a short-term upward movement (+2.3%) over 14 periods."
    """
    col_display = value_col.replace("_", " ").title()

    if recent_mean == 0:
        return f"Forecast for {col_display}: insufficient baseline to determine direction."

    pct = ((forecast_mean - recent_mean) / abs(recent_mean)) * 100

    if pct > 2:
        direction = f"an upward movement ({pct:+.1f}%)"
        tone = "positive outlook"
    elif pct < -2:
        direction = f"a downward movement ({pct:+.1f}%)"
        tone = "caution is advised"
    else:
        direction = f"broadly stable conditions ({pct:+.1f}%)"
        tone = "no significant change expected"

    return (
        f"The forecast over the next {horizon} periods suggests {direction} "
        f"for {col_display}. {tone.capitalize()}."
    )


# ---------------------------------------------------------------------------
# Time-series preparation
# ---------------------------------------------------------------------------


def prepare_time_series(
    df: pd.DataFrame,
    datetime_col: str,
    value_col: str,
    freq: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse datetime column, set index, sort, optionally resample.

    Args:
        df:           Source DataFrame.
        datetime_col: Name of the datetime column.
        value_col:    Name of the numeric target column.
        freq:         Optional pandas resample frequency ('D', 'W', 'ME', etc.).

    Returns:
        DataFrame with DatetimeIndex, sorted ascending, NaNs dropped.

    Raises:
        ValueError: If columns are missing or data cannot be parsed.
    """
    if datetime_col not in df.columns:
        raise ValueError(
            f"Datetime column '{datetime_col}' was not found in the dataset. "
            "Please select a valid date column."
        )
    if value_col not in df.columns:
        raise ValueError(
            f"Metric column '{value_col}' was not found in the dataset. "
            "Please select a valid numeric column."
        )

    ts = df[[datetime_col, value_col]].copy()
    ts[datetime_col] = pd.to_datetime(ts[datetime_col], errors="coerce", utc=True)
    ts = ts.dropna(subset=[datetime_col])
    ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")
    ts = ts.set_index(datetime_col).sort_index()

    if freq:
        ts = ts.resample(freq)[value_col].mean().to_frame()
    else:
        ts = ts[[value_col]]

    result = ts.dropna()

    if len(result) < 4:
        raise ValueError(
            f"After parsing, only {len(result)} valid data points remain in '{value_col}'. "
            "Please choose a column with more numeric data, or check your date column."
        )

    return result
