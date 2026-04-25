"""
app/analytics/operations.py
============================
Pure-logic FMCG operations analytics.  No Streamlit imports.

Three main functions:
    compute_stock_to_sales      — stock-to-sales ratio & stock-out risk
    compute_defect_summary      — defect rate statistics & high-defect periods
    compute_production_summary  — production volume vs sales demand
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Stock-to-Sales & Stock-out Risk
# ---------------------------------------------------------------------------

STOCKOUT_THRESHOLD = 0.0   # stock <= 0 is considered a stock-out day


def compute_stock_to_sales(
    df:        pd.DataFrame,
    col_stock: str,
    col_sales: Optional[str] = None,
) -> dict[str, Any]:
    """
    Compute stock health metrics.

    Parameters
    ----------
    df          Source DataFrame.
    col_stock   Name of the closing-stock / inventory-level column.
    col_sales   (optional) Name of the sales column; used for ratio.

    Returns
    -------
    dict with keys:
        avg_stock             float
        min_stock             float
        max_stock             float
        stock_to_sales_ratio  float | None   — avg_stock / avg_sales
        stockout_risk_days    int             — periods where stock <= 0
        stockout_risk_pct     float           — % of total periods
        stock_series          pd.Series | None
    """
    stock_raw = pd.to_numeric(df[col_stock], errors="coerce").dropna()
    if len(stock_raw) == 0:
        return _empty_stock_result()

    avg_stock = float(stock_raw.mean())
    min_stock = float(stock_raw.min())
    max_stock = float(stock_raw.max())

    stockout_days = int((stock_raw <= STOCKOUT_THRESHOLD).sum())
    stockout_pct  = stockout_days / len(stock_raw) * 100

    # Stock-to-sales ratio
    ratio: Optional[float] = None
    if col_sales and col_sales in df.columns:
        sales_raw = pd.to_numeric(df[col_sales], errors="coerce").dropna()
        avg_sales = float(sales_raw.mean()) if len(sales_raw) else 0.0
        if avg_sales > 0:
            ratio = round(avg_stock / avg_sales, 3)

    return {
        "avg_stock":            round(avg_stock, 2),
        "min_stock":            round(min_stock, 2),
        "max_stock":            round(max_stock, 2),
        "stock_to_sales_ratio": ratio,
        "stockout_risk_days":   stockout_days,
        "stockout_risk_pct":    round(stockout_pct, 2),
        "stock_series":         stock_raw.reset_index(drop=True),
    }


def _empty_stock_result() -> dict[str, Any]:
    return {
        "avg_stock": 0.0,
        "min_stock": 0.0,
        "max_stock": 0.0,
        "stock_to_sales_ratio": None,
        "stockout_risk_days": 0,
        "stockout_risk_pct": 0.0,
        "stock_series": None,
    }


# ---------------------------------------------------------------------------
# Defect Rate
# ---------------------------------------------------------------------------

HIGH_DEFECT_THRESHOLD = 5.0   # percent — periods above this are flagged


def compute_defect_summary(
    df:         pd.DataFrame,
    col_defect: str,
) -> dict[str, Any]:
    """
    Summarise defect rate data.

    Returns
    -------
    dict with keys:
        mean               float   — average defect rate
        max                float   — peak defect rate
        min                float
        std                float
        high_defect_periods int    — periods exceeding HIGH_DEFECT_THRESHOLD
        series             pd.Series | None
    """
    defect_raw = pd.to_numeric(df[col_defect], errors="coerce").dropna()
    if len(defect_raw) == 0:
        return {
            "mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0,
            "high_defect_periods": 0, "series": None,
        }

    return {
        "mean":               round(float(defect_raw.mean()), 3),
        "max":                round(float(defect_raw.max()),  3),
        "min":                round(float(defect_raw.min()),  3),
        "std":                round(float(defect_raw.std()),  3),
        "high_defect_periods": int((defect_raw > HIGH_DEFECT_THRESHOLD).sum()),
        "series":             defect_raw.reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# Production Volume
# ---------------------------------------------------------------------------

def compute_production_summary(
    df:             pd.DataFrame,
    col_production: str,
    col_sales:      Optional[str] = None,
) -> dict[str, Any]:
    """
    Summarise production volume and compare to sales demand.

    Returns
    -------
    dict with keys:
        total                float
        mean                 float
        max                  float
        min                  float
        prod_to_sales_ratio  float | None   — avg_production / avg_sales
        series               pd.Series | None
    """
    prod_raw = pd.to_numeric(df[col_production], errors="coerce").dropna()
    if len(prod_raw) == 0:
        return {
            "total": 0.0, "mean": 0.0, "max": 0.0, "min": 0.0,
            "prod_to_sales_ratio": None, "series": None,
        }

    ratio: Optional[float] = None
    if col_sales and col_sales in df.columns:
        sales_raw = pd.to_numeric(df[col_sales], errors="coerce").dropna()
        avg_sales = float(sales_raw.mean()) if len(sales_raw) else 0.0
        avg_prod  = float(prod_raw.mean())
        if avg_sales > 0:
            ratio = round(avg_prod / avg_sales, 3)

    return {
        "total":               round(float(prod_raw.sum()), 2),
        "mean":                round(float(prod_raw.mean()), 2),
        "max":                 round(float(prod_raw.max()),  2),
        "min":                 round(float(prod_raw.min()),  2),
        "prod_to_sales_ratio": ratio,
        "series":              prod_raw.reset_index(drop=True),
    }