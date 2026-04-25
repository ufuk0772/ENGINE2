"""
app/analytics/trends.py
=======================
Pure-logic trend calculations for the FMCG Analytics Platform.

Responsibilities:
- prepare clean time series data
- compute rolling trend statistics
- compute demand variability
- compute descriptive statistics
- build aggregated views (Daily / Weekly / Monthly)

No Streamlit imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data transfer object
# ---------------------------------------------------------------------------

@dataclass
class TrendResult:
    """
    All outputs produced by build_trend_data().

    Fields
    ------
    series        pd.Series              cleaned numeric series indexed by date
    rolling_mean  pd.Series              rolling mean
    rolling_std   pd.Series              rolling std deviation
    stats         dict                   descriptive statistics
    aggregated    pd.DataFrame | None    aggregated roll-up view
    trend_dir     str                    "upward" | "downward" | "stable"
    pct_change    float                  recent vs prior mean change (%)
    """
    series: pd.Series
    rolling_mean: pd.Series
    rolling_std: pd.Series
    stats: dict
    aggregated: Optional[pd.DataFrame] = None
    trend_dir: str = "stable"
    pct_change: float = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_trend_data(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    rolling_window: int = 7,
    agg_level: str = "Daily",
) -> Optional[TrendResult]:
    """
    Prepare a time series and compute trend / rolling statistics.

    Parameters
    ----------
    df
        Source DataFrame.
    date_col
        Name of the datetime column.
    value_col
        Name of the numeric column to analyse.
    rolling_window
        Window size for rolling mean / std.
    agg_level
        One of "Daily" | "Weekly" | "Monthly".

    Returns
    -------
    TrendResult or None if insufficient valid data is available.

    Raises
    ------
    ValueError
        If required columns do not exist.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in DataFrame.")

    ts = _prepare_time_series(df, date_col, value_col)
    if ts is None or len(ts) < 4:
        return None

    series = ts.set_index(date_col)[value_col].sort_index()
    series = pd.to_numeric(series, errors="coerce").dropna()

    if len(series) < 4:
        return None

    window = max(2, min(int(rolling_window), len(series)))

    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std().fillna(0.0)

    stats = descriptive_stats(series)
    pct_change = recent_pct_change(series)
    trend_dir = classify_trend_from_pct(pct_change)
    aggregated = _aggregate(ts, date_col, value_col, agg_level)

    return TrendResult(
        series=series,
        rolling_mean=rolling_mean,
        rolling_std=rolling_std,
        stats=stats,
        aggregated=aggregated,
        trend_dir=trend_dir,
        pct_change=round(pct_change, 2),
    )


def compute_demand_variability(series: pd.Series) -> dict:
    """
    Compute Demand Variability using coefficient of variation (CV).

    Returns
    -------
    dict with keys:
        level  str    — "low" | "moderate" | "high"
        cv     float  — std / mean
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) == 0:
        return {"level": "unknown", "cv": 0.0}

    std = float(clean.std())
    mean = float(clean.mean())
    cv = std / abs(mean) if mean != 0 else 0.0

    if cv < 0.15:
        level = "low"
    elif cv < 0.40:
        level = "moderate"
    else:
        level = "high"

    return {"level": level, "cv": round(cv, 4)}


def descriptive_stats(series: pd.Series) -> dict:
    """
    Return summary statistics for a numeric series.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) == 0:
        return {}

    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
    }


def recent_pct_change(series: pd.Series, window: int = 7) -> float:
    """
    Compare recent average vs prior average and return % change.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 4:
        return 0.0

    w = min(window, max(2, len(clean) // 2))
    recent = float(clean.iloc[-w:].mean())

    if len(clean) >= w * 2:
        prior = float(clean.iloc[-2 * w:-w].mean())
    else:
        prior = float(clean.iloc[:-w].mean()) if len(clean) > w else recent

    if prior == 0:
        return 0.0

    return ((recent - prior) / abs(prior)) * 100.0


def classify_trend_from_pct(pct_change: float) -> str:
    """
    Convert recent % movement into a simple trend label.
    """
    if pct_change > 2:
        return "upward"
    if pct_change < -2:
        return "downward"
    return "stable"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _prepare_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> Optional[pd.DataFrame]:
    """
    Clean and sort the date/value pair.

    Returns a two-column DataFrame with valid dates and numeric values.
    If multiple rows share the same date, they are summed.
    """
    ts = df[[date_col, value_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")
    ts = ts.dropna(subset=[date_col, value_col])

    if len(ts) < 3:
        return None

    ts = (
        ts.groupby(date_col, as_index=False)[value_col]
        .sum()
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    return ts if len(ts) >= 3 else None


_AGG_FREQ_MAP = {
    "Daily": "D",
    "Weekly": "W",
    "Monthly": "ME",
}


def _aggregate(
    ts: pd.DataFrame,
    date_col: str,
    value_col: str,
    agg_level: str,
) -> Optional[pd.DataFrame]:
    """
    Build aggregated sales view for UI tables/charts.
    """
    freq = _AGG_FREQ_MAP.get(agg_level)
    if freq is None:
        return None

    try:
        agg = (
            ts.set_index(date_col)[value_col]
            .resample(freq)
            .sum()
            .reset_index()
            .rename(columns={date_col: "Period", value_col: "Sales (Sum)"})
        )
        return agg
    except Exception:
        return None