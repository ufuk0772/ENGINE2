"""
app/ui/sidebar.py
=================
Renders the entire left sidebar and writes all user selections directly into
st.session_state so every tab can read them without prop-drilling.

FMCG Column Mapping
-------------------
Instead of the generic "datetime column / target metric" pattern from the
MVP, the sidebar exposes six semantically labelled FMCG selectors:

    col_date        — transaction / reporting date
    col_sales       — sales volume or revenue
    col_production  — production volume (optional)
    col_stock       — closing stock / inventory level (optional)
    col_defect      — defect rate / reject count (optional)
    col_category    — product / SKU category for slice-and-dice (optional)

Analysis Settings
-----------------
    rolling_window      — smoothing window for trend charts (days)
    forecast_horizon    — periods ahead for the forecast module
    forecast_method     — naive | moving_average | exponential_smoothing
    agg_level           — Daily | Weekly | Monthly roll-up
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame, profile) -> None:
    """
    Render all sidebar widgets.  All selections are stored in
    st.session_state; callers do not need the return value.
    """
    st.sidebar.markdown(
        """
        <div style="margin-bottom:4px;">
            <span style="font-size:1.25rem; font-weight:800;
                         color:#0f4c81; letter-spacing:-0.01em;">
                ⚙️ Platform Controls
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    all_cols   = list(df.columns)
    num_cols   = _numeric_columns(df)
    cat_cols   = _categorical_columns(df)
    date_cols  = _datetime_hint_columns(df, profile)

    _render_column_mapping(
        all_cols=all_cols,
        date_cols=date_cols,
        num_cols=num_cols,
        cat_cols=cat_cols,
        profile=profile,
    )

    st.sidebar.divider()
    _render_analysis_settings()

    st.sidebar.divider()
    _render_forecast_settings()

    st.sidebar.divider()
    _render_data_info(df)

    st.sidebar.caption("FMCG Analytics Platform · pandas + Plotly")


# ---------------------------------------------------------------------------
# Column-mapping section
# ---------------------------------------------------------------------------

def _render_column_mapping(
    all_cols:  list[str],
    date_cols: list[str],
    num_cols:  list[str],
    cat_cols:  list[str],
    profile,
) -> None:
    st.sidebar.markdown("#### 📋 FMCG Column Mapping")
    st.sidebar.markdown(
        "<span style='font-size:0.78rem; color:#94a3b8;'>"
        "Map your dataset columns to FMCG KPIs</span>",
        unsafe_allow_html=True,
    )

    # ── Date ────────────────────────────────────────────────────────────
    detected_date = getattr(profile, "datetime_column", None)
    date_options  = _options_with_none(date_cols or all_cols)
    default_date  = _safe_index(date_options, detected_date or st.session_state.get("col_date"))

    st.session_state["col_date"] = st.sidebar.selectbox(
        "📅 Date Column",
        options=date_options,
        index=default_date,
        help="Column containing transaction or reporting dates.",
        key="_sb_col_date",
    )

    # ── Sales ────────────────────────────────────────────────────────────
    detected_num = getattr(profile, "numeric_columns", []) or []
    sales_opts   = _options_with_none(num_cols)
    default_sales = _safe_index(sales_opts, st.session_state.get("col_sales") or _first_or_none(detected_num))

    st.session_state["col_sales"] = st.sidebar.selectbox(
        "💰 Sales / Revenue KPI",
        options=sales_opts,
        index=default_sales,
        help="Primary sales volume or revenue column.",
        key="_sb_col_sales",
    )

    # ── Production ───────────────────────────────────────────────────────
    prod_opts     = _options_with_none(num_cols)
    default_prod  = _safe_index(prod_opts, st.session_state.get("col_production") or _best_guess(num_cols, ["production", "output", "produced", "units_produced"]))

    st.session_state["col_production"] = st.sidebar.selectbox(
        "🏗️ Production Volume (optional)",
        options=prod_opts,
        index=default_prod,
        help="Units produced in the period — used in Operations tab.",
        key="_sb_col_production",
    )

    # ── Stock ────────────────────────────────────────────────────────────
    stock_opts    = _options_with_none(num_cols)
    default_stock = _safe_index(stock_opts, st.session_state.get("col_stock") or _best_guess(num_cols, ["stock", "inventory", "closing_stock", "on_hand"]))

    st.session_state["col_stock"] = st.sidebar.selectbox(
        "📦 Stock / Inventory Level (optional)",
        options=stock_opts,
        index=default_stock,
        help="Closing stock or on-hand inventory — used for Stock-out Risk.",
        key="_sb_col_stock",
    )

    # ── Defect Rate ──────────────────────────────────────────────────────
    defect_opts    = _options_with_none(num_cols)
    default_defect = _safe_index(defect_opts, st.session_state.get("col_defect") or _best_guess(num_cols, ["defect", "reject", "defect_rate", "quality"]))

    st.session_state["col_defect"] = st.sidebar.selectbox(
        "⚠️ Defect Rate (optional)",
        options=defect_opts,
        index=default_defect,
        help="Defect rate or reject count — used in Quality panel.",
        key="_sb_col_defect",
    )

    # ── Category ────────────────────────────────────────────────────────
    cat_opts     = _options_with_none(cat_cols)
    default_cat  = _safe_index(cat_opts, st.session_state.get("col_category") or _best_guess(cat_cols, ["category", "sku", "product", "brand", "segment"]))

    st.session_state["col_category"] = st.sidebar.selectbox(
        "🏷️ Category / SKU (optional)",
        options=cat_opts,
        index=default_cat,
        help="Product or SKU category — enables segment-level filtering.",
        key="_sb_col_category",
    )

    # ── Category filter (only when category column is set) ───────────────
    cat_col = st.session_state.get("col_category")
    if cat_col and cat_col in (all_cols):
        import pandas as _pd  # local import to avoid top-level circular issues
        # We need the full df here but only have profile — pass via session state
        _df = st.session_state.get("df")
        if _df is not None and cat_col in _df.columns:
            unique_cats = sorted(_df[cat_col].dropna().astype(str).unique().tolist())
            if 1 < len(unique_cats) <= 50:
                selected = st.sidebar.multiselect(
                    "Filter by Category",
                    options=unique_cats,
                    default=unique_cats,
                    key="_sb_cat_filter",
                )
                st.session_state["cat_filter"] = selected
            else:
                st.session_state["cat_filter"] = None
        else:
            st.session_state["cat_filter"] = None
    else:
        st.session_state["cat_filter"] = None


# ---------------------------------------------------------------------------
# Analysis-settings section
# ---------------------------------------------------------------------------

def _render_analysis_settings() -> None:
    st.sidebar.markdown("#### 📊 Analysis Settings")

    # Rolling window
    st.session_state["rolling_window"] = st.sidebar.slider(
        "Smoothing Window (days)",
        min_value=3,
        max_value=90,
        value=st.session_state.get("rolling_window", 7),
        step=1,
        help="Number of periods used for rolling-average smoothing in trend charts.",
        key="_sb_rolling",
    )

    # Aggregation level
    agg_options = ["Daily", "Weekly", "Monthly"]
    current_agg = st.session_state.get("agg_level", "Daily")
    st.session_state["agg_level"] = st.sidebar.radio(
        "Aggregation Level",
        options=agg_options,
        index=agg_options.index(current_agg) if current_agg in agg_options else 0,
        horizontal=True,
        help="Roll-up granularity applied before charting and forecasting.",
        key="_sb_agg",
    )


# ---------------------------------------------------------------------------
# Forecast-settings section
# ---------------------------------------------------------------------------

def _render_forecast_settings() -> None:
    st.sidebar.markdown("#### 🔮 Demand Forecast Settings")

    st.session_state["forecast_horizon"] = st.sidebar.number_input(
        "Forecast Horizon (periods)",
        min_value=1,
        max_value=365,
        value=st.session_state.get("forecast_horizon", 14),
        step=1,
        help="Number of future periods to forecast.",
        key="_sb_horizon",
    )

    method_labels = {
        "Naïve (Last Value)":         "naive",
        "Moving Average":             "moving_average",
        "Exponential Smoothing":      "exponential_smoothing",
    }
    current_method = st.session_state.get("forecast_method", "moving_average")
    # Reverse lookup for current label
    reverse = {v: k for k, v in method_labels.items()}
    current_label = reverse.get(current_method, "Moving Average")

    chosen_label = st.sidebar.selectbox(
        "Forecasting Method",
        options=list(method_labels.keys()),
        index=list(method_labels.keys()).index(current_label),
        key="_sb_method",
    )
    st.session_state["forecast_method"] = method_labels[chosen_label]


# ---------------------------------------------------------------------------
# Data info footer
# ---------------------------------------------------------------------------

def _render_data_info(df: pd.DataFrame) -> None:
    filename = st.session_state.get("filename", "")
    st.sidebar.markdown("#### 📁 Loaded Dataset")
    st.sidebar.markdown(
        f"<div style='font-size:0.8rem; color:#64748b;'>"
        f"<b>File:</b> {filename or '—'}<br>"
        f"<b>Rows:</b> {len(df):,} &nbsp;|&nbsp; <b>Cols:</b> {len(df.columns)}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _options_with_none(columns: list[str]) -> list[Optional[str]]:
    """Return [None, *columns] so the first option is always 'not set'."""
    return [None, *columns]


def _safe_index(options: list, value: Optional[str]) -> int:
    """Return the index of *value* in *options*, falling back to 0."""
    try:
        return options.index(value)
    except (ValueError, TypeError):
        return 0


def _first_or_none(lst: list[str]) -> Optional[str]:
    return lst[0] if lst else None


def _best_guess(columns: list[str], keywords: list[str]) -> Optional[str]:
    """Return the first column name whose lower-case contains any keyword."""
    for col in columns:
        for kw in keywords:
            if kw in col.lower():
                return col
    return None


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _datetime_hint_columns(df: pd.DataFrame, profile) -> list[str]:
    """
    Combine columns detected by SchemaDetector with any column whose name
    contains date-like keywords.
    """
    detected = {getattr(profile, "datetime_column", None)} - {None}
    keywords = {"date", "time", "period", "week", "month", "year", "day"}
    hinted   = {c for c in df.columns if any(kw in c.lower() for kw in keywords)}
    return list(detected | hinted) or df.columns.tolist()