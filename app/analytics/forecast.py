"""
app/analytics/forecast.py
=========================
Pure-logic forecasting utilities for the FMCG Analytics Platform.

Responsibilities:
- prepare historical demand series
- run simple short-term forecast methods
- preserve datetime index when possible
- return a clean ForecastResult object for UI rendering

No Streamlit imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Data transfer object
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """
    Output returned by run_fmcg_forecast().

    Fields
    ------
    historical     pd.Series   cleaned historical demand series
    forecast       pd.Series   future forecast series
    method         str         raw method key
    method_label   str         human-readable label
    horizon        int         forecast horizon
    recent_mean    float       recent average demand
    forecast_mean  float       average forecast value
    pct_change     float       forecast mean vs recent mean (%)
    """
    historical: pd.Series
    forecast: pd.Series
    method: str
    method_label: str
    horizon: int
    recent_mean: float
    forecast_mean: float
    pct_change: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_fmcg_forecast(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    horizon: int = 14,
    method: str = "moving_average",
) -> Optional[ForecastResult]:
    """
    Main forecast entry point used by tabs.py.

    Parameters
    ----------
    df
        Source DataFrame.
    date_col
        Datetime column name.
    value_col
        Numeric demand / sales column name.
    horizon
        Number of future periods to forecast.
    method
        One of: "moving_average", "naive", "exponential_smoothing"

    Returns
    -------
    ForecastResult or None if there is insufficient usable data.

    Raises
    ------
    ValueError
        If required columns are missing or method is unsupported.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in DataFrame.")
    if horizon <= 0:
        raise ValueError("Forecast horizon must be greater than zero.")

    historical = _prepare_historical_series(df, date_col, value_col)
    if historical is None or len(historical) < 5:
        return None

    method = method.lower().strip()
    if method == "moving_average":
        forecast = _moving_average_forecast(historical, horizon=horizon)
        method_label = "Moving Average"
    elif method == "naive":
        forecast = _naive_forecast(historical, horizon=horizon)
        method_label = "Naïve"
    elif method == "exponential_smoothing":
        forecast = _exponential_smoothing_forecast(historical, horizon=horizon)
        method_label = "Exponential Smoothing"
    else:
        raise ValueError(
            f"Unsupported forecast method '{method}'. "
            "Use 'moving_average', 'naive', or 'exponential_smoothing'."
        )

    if forecast is None or len(forecast) == 0:
        return None

    recent_window = min(14, len(historical))
    recent_mean = float(historical.iloc[-recent_window:].mean())
    forecast_mean = float(forecast.mean())
    pct_change = ((forecast_mean - recent_mean) / abs(recent_mean) * 100.0) if recent_mean else 0.0

    return ForecastResult(
        historical=historical,
        forecast=forecast,
        method=method,
        method_label=method_label,
        horizon=horizon,
        recent_mean=round(recent_mean, 2),
        forecast_mean=round(forecast_mean, 2),
        pct_change=round(pct_change, 2),
    )


# ---------------------------------------------------------------------------
# Public helper (optional reuse)
# ---------------------------------------------------------------------------

def prepare_forecast_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> Optional[pd.Series]:
    """
    Expose cleaned historical series for debugging / reuse if needed.
    """
    return _prepare_historical_series(df, date_col, value_col)


# ---------------------------------------------------------------------------
# Historical series preparation
# ---------------------------------------------------------------------------

def _prepare_historical_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> Optional[pd.Series]:
    """
    Clean, aggregate, and index the historical series by date.

    If multiple rows share the same date, values are summed.
    """
    ts = df[[date_col, value_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")
    ts = ts.dropna(subset=[date_col, value_col])

    if len(ts) < 5:
        return None

    ts = (
        ts.groupby(date_col, as_index=False)[value_col]
        .sum()
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    if len(ts) < 5:
        return None

    series = ts.set_index(date_col)[value_col].astype(float).sort_index()
    return series if len(series) >= 5 else None


# ---------------------------------------------------------------------------
# Forecast methods
# ---------------------------------------------------------------------------

def _naive_forecast(
    historical: pd.Series,
    horizon: int,
) -> pd.Series:
    """
    Repeat the last observed value into the future.
    """
    last_value = float(historical.iloc[-1])
    future_index = _build_future_index(historical.index, horizon)
    values = [last_value] * horizon
    return pd.Series(values, index=future_index, name="forecast")


def _moving_average_forecast(
    historical: pd.Series,
    horizon: int,
    window: Optional[int] = None,
) -> pd.Series:
    """
    Forecast using the mean of the most recent observations.

    Default window:
    min(7, len(historical))
    """
    if window is None:
        window = min(7, len(historical))

    window = max(2, min(window, len(historical)))
    avg_value = float(historical.iloc[-window:].mean())

    future_index = _build_future_index(historical.index, horizon)
    values = [avg_value] * horizon
    return pd.Series(values, index=future_index, name="forecast")


def _exponential_smoothing_forecast(
    historical: pd.Series,
    horizon: int,
    alpha: float = 0.3,
) -> pd.Series:
    """
    Simple exponential smoothing forecast.

    Produces a flat future path equal to the final smoothed level.
    This is intentionally simple and robust for MVP dashboard usage.
    """
    values = historical.astype(float).tolist()
    level = values[0]

    for obs in values[1:]:
        level = alpha * obs + (1.0 - alpha) * level

    future_index = _build_future_index(historical.index, horizon)
    forecast_values = [float(level)] * horizon
    return pd.Series(forecast_values, index=future_index, name="forecast")


# ---------------------------------------------------------------------------
# Future index generation
# ---------------------------------------------------------------------------

def _build_future_index(
    index: pd.Index,
    horizon: int,
) -> pd.Index:
    """
    Build future index aligned with the historical series.

    If the input index is DatetimeIndex, continue it using inferred or fallback freq.
    Otherwise, return a simple RangeIndex-like numeric index continuation.
    """
    if isinstance(index, pd.DatetimeIndex):
        return _build_future_datetime_index(index, horizon)

    start = len(index)
    return pd.Index(range(start, start + horizon))


def _build_future_datetime_index(
    dt_index: pd.DatetimeIndex,
    horizon: int,
) -> pd.DatetimeIndex:
    """
    Continue a DatetimeIndex into the future.

    Strategy:
    1. Try inferred_freq
    2. Try median delta
    3. Fallback to daily frequency
    """
    if len(dt_index) == 0:
        return pd.date_range(start=pd.Timestamp.today().normalize(), periods=horizon, freq="D")

    inferred = dt_index.inferred_freq
    if inferred:
        start = dt_index[-1] + pd.tseries.frequencies.to_offset(inferred)
        return pd.date_range(start=start, periods=horizon, freq=inferred)

    if len(dt_index) >= 2:
        deltas = dt_index.to_series().diff().dropna()
        if len(deltas) > 0:
            median_delta = deltas.median()
            if pd.notna(median_delta) and median_delta > pd.Timedelta(0):
                start = dt_index[-1] + median_delta
                return pd.DatetimeIndex([start + i * median_delta for i in range(horizon)])

    start = dt_index[-1] + pd.Timedelta(days=1)
    return pd.date_range(start=start, periods=horizon, freq="D")