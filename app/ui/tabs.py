from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui.components import section, end_section
from app.ui import charts
from app.ui.charts import anomaly_chart
from app.analytics.anomaly import build_anomaly_detection
from app.analytics.quality import compute_quality_metrics
from app.analytics.trends import (
    build_trend_data,
    compute_demand_variability,
    TrendResult,
)
from app.analytics.operations import (
    compute_stock_to_sales,
    compute_defect_summary,
    compute_production_summary,
)
from app.analytics.forecast import run_fmcg_forecast, ForecastResult
from app.ingestion.loader import is_valid_datetime_column, is_valid_numeric_column
from app.optimization.transportation import (
    balance_problem,
    least_cost_method,
    northwest_corner_method,
    vogel_approximation_method,
)

import app.ui.charts as charts

def insight_card(title, body, icon="💡"):
    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #e2e8f0;
            border-radius:12px;
            padding:16px;
            margin-top:12px;
        ">
            <div style="font-weight:700; color:#0f4c81;">
                {icon} {title}
            </div>
            <div style="margin-top:8px; color:#334155; font-size:0.9rem;">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def hero_banner(title, subtitle="", badge=None):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0f4c81 0%, #2563eb 100%);
            border-radius: 18px;
            padding: 24px 28px;
            margin: 8px 0 22px 0;
            color: white;
            box-shadow: 0 10px 28px rgba(15, 76, 129, 0.22);
        ">
            <div style="
                display:inline-block;
                background:rgba(255,255,255,0.16);
                border:1px solid rgba(255,255,255,0.22);
                border-radius:999px;
                padding:5px 12px;
                font-size:0.78rem;
                font-weight:700;
                margin-bottom:12px;
            ">
                {badge or "Decision Intelligence"}
            </div>
            <div style="font-size:1.65rem; font-weight:850; margin-bottom:8px;">
                {title}
            </div>
            <div style="font-size:0.95rem; line-height:1.6; opacity:0.92;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
# ---------------------------------------------------------------------------
# Session-state key — optimizasyon sonuçları için tek bir sabit
# ---------------------------------------------------------------------------
_OPT_KEY = "opt_results"


# ---------------------------------------------------------------------------
# Shared banner helpers
# ---------------------------------------------------------------------------

def _info(msg: str) -> None:
    st.markdown(
        f"<div class='fmcg-banner fmcg-banner-info'>ℹ️ &nbsp;{msg}</div>",
        unsafe_allow_html=True,
    )


def _warn(msg: str) -> None:
    st.markdown(
        f"<div class='fmcg-banner fmcg-banner-warn'>⚠️ &nbsp;{msg}</div>",
        unsafe_allow_html=True,
    )


def _err(msg: str) -> None:
    st.markdown(
        f"<div class='fmcg-banner fmcg-banner-error'>🚨 &nbsp;{msg}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_tabs(df: pd.DataFrame, profile) -> None:
    """Render all FMCG tabs. Reads config from st.session_state."""

    col_date       = st.session_state.get("col_date")
    col_sales      = st.session_state.get("col_sales")
    col_production = st.session_state.get("col_production")
    col_stock      = st.session_state.get("col_stock")
    col_defect     = st.session_state.get("col_defect")
    col_category   = st.session_state.get("col_category")
    cat_filter     = st.session_state.get("cat_filter")

    rolling_window = st.session_state.get("rolling_window", 7)
    agg_level      = st.session_state.get("agg_level", "Daily")
    fc_horizon     = st.session_state.get("forecast_horizon", 14)
    fc_method      = st.session_state.get("forecast_method", "moving_average")

    working_df = _apply_category_filter(df, col_category, cat_filter)

    tabs = st.tabs([
        "📊 Overview",
        "📈 Sales & Demand",
        "🏗️ Inventory & Operations",
        "🔮 Forecast & Insights",
        "🚀 Optimization",
    ])

    with tabs[0]:
        _render_overview(working_df, profile, col_date, col_sales)

    with tabs[1]:
        _render_sales_demand(working_df, col_date, col_sales, rolling_window, agg_level)

    with tabs[2]:
        _render_inventory_operations(
            working_df, col_date, col_sales, col_stock, col_production, col_defect
        )

    with tabs[3]:
        _render_forecast_insights(working_df, col_date, col_sales, fc_horizon, fc_method)

    with tabs[4]:
        _render_optimization()


# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------

def _render_overview(
    df: pd.DataFrame,
    profile,
    col_date: Optional[str],
    col_sales: Optional[str],
) -> None:
    st.markdown("### 📊 Platform Overview")
    st.caption("High-level business health, data quality, and dataset readiness for FMCG analysis.")

    qm = compute_quality_metrics(df)
    series = _safe_numeric_series(df, col_sales)

    demand_stability = "N/A"
    operational_risk = "N/A"
    demand_trend     = "N/A"
    quality_score    = _compute_quality_score(qm)

    if series is not None and len(series) >= 4:
        variability      = compute_demand_variability(series)
        demand_stability = variability["level"].title()
        operational_risk = _risk_label_from_variability(variability["level"])
        demand_trend     = _simple_trend_label(series)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",      f"{qm['total_rows']:,}")
    c2.metric("Data Quality Score", f"{quality_score}/100")
    c3.metric("Demand Trend",       demand_trend)
    c4.metric("Operational Risk",   operational_risk)
    c5.metric("Demand Stability",   demand_stability)

    st.divider()

    detected_items = []
    if getattr(profile, "datetime_column", None):
        detected_items.append(f"**Date:** `{profile.datetime_column}`")

    num_cols = getattr(profile, "numeric_columns", []) or []
    if num_cols:
        sample = ", ".join(f"`{c}`" for c in num_cols[:4])
        extra  = f" + {len(num_cols) - 4} more" if len(num_cols) > 4 else ""
        detected_items.append(f"**Numeric KPIs:** {sample}{extra}")

    if getattr(profile, "boolean_columns", None):
        detected_items.append(f"**Boolean:** `{'`, `'.join(profile.boolean_columns)}`")

    if detected_items:
        _info("Auto-detected schema — " + " · ".join(detected_items))

    if not col_date:
        _warn("No Date column mapped. Assign one in the sidebar to unlock trend and forecast views.")
    if not col_sales:
        _warn("No Sales / Revenue KPI mapped. Assign one in the sidebar.")

    st.divider()
    st.markdown("#### 🔍 Data Quality Report")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**Column Type Distribution**")
        type_df = pd.DataFrame(
            list(qm["column_types"].items()),
            columns=["Data Type", "Count"],
        )
        st.dataframe(type_df, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**Missing Values by Column**")
        miss = qm["missing_per_col"]
        if miss:
            miss_df = (
                pd.DataFrame(list(miss.items()), columns=["Column", "Missing %"])
                .sort_values("Missing %", ascending=False)
            )
            st.dataframe(
                miss_df.style.format({"Missing %": "{:.1f}%"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("✅ Dataset is clean and complete for analysis.")

    st.divider()
    st.markdown("#### 💡 Key Takeaways")
    for item in _build_overview_insights(qm, demand_trend, demand_stability, operational_risk):
        st.markdown(f"- {item}")

    st.divider()
    st.markdown("#### 👁️ Uploaded Dataset Snapshot")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    if len(df) > 100:
        st.caption(f"Showing first 100 of {len(df):,} rows.")


# ---------------------------------------------------------------------------
# Tab 2 — Sales & Demand
# ---------------------------------------------------------------------------

def _render_sales_demand(
    df: pd.DataFrame,
    col_date: Optional[str],
    col_sales: Optional[str],
    rolling_window: int,
    agg_level: str,
) -> None:
    st.markdown("### 📈 Sales & Demand Analysis")
    st.caption("Trend analysis, demand variability, and aggregated sales performance.")

    if not _guard_columns(col_date, col_sales, df):
        return

    try:
        result: Optional[TrendResult] = build_trend_data(
            df=df,
            date_col=col_date,
            value_col=col_sales,
            rolling_window=rolling_window,
            agg_level=agg_level,
        )
    except Exception as exc:
        _err(f"Could not build trend data: {exc}")
        return

    if result is None or result.series is None or len(result.series) < 4:
        _warn("Not enough valid data points for trend analysis (minimum 4 required).")
        return

    s           = result.stats
    variability = compute_demand_variability(result.series)
    growth_pct  = _recent_growth_pct(result.series)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Min Sales",        f"{s['min']:,.2f}")
    c2.metric("Max Sales",        f"{s['max']:,.2f}")
    c3.metric("Average Sales",    f"{s['mean']:,.2f}")
    c4.metric("Demand Stability", variability["level"].title())
    c5.metric("Recent Change",    f"{growth_pct:+.1f}%")

    st.markdown("")

    # ── Anomaly Detection ───────────────────────────────────────────────────
    # build_anomaly_detection(df, date_col, value_col) → AnomalyResult | None
    # anomaly_chart(anomaly_df, date_col, value_col)  ← series_df kullanır
    _anom_result = build_anomaly_detection(
        df=df,
        date_col=col_date,
        value_col=col_sales,
        z_thresh=3.0,
    )

    col_x, col_y = st.columns([2, 1])

    with col_x:
        section("🚨 Demand Anomaly Detection")
        if _anom_result is not None:
            fig_anom = anomaly_chart(
                anomaly_df=_anom_result.series_df,
                date_col=col_date,
                value_col=col_sales,
            )
            st.plotly_chart(fig_anom, use_container_width=True)
        else:
            st.info("Anomaly detection requires more data points (minimum window + 5).")
        end_section()

    with col_y:
        section("🧠 Anomaly Insight")

        if _anom_result is not None:
            _a_count = _anom_result.anomaly_count
            _a_ratio = _anom_result.anomaly_ratio

            if _a_ratio > 0.10:
                _a_label, _a_color, _a_bg = "High Risk",     "#ef4444", "#fef2f2"
            elif _a_ratio > 0.05:
                _a_label, _a_color, _a_bg = "Moderate Risk", "#f59e0b", "#fffbeb"
            else:
                _a_label, _a_color, _a_bg = "Stable",        "#10b981", "#ecfdf5"

            _a_summary = (
                f"• Total anomalies detected: {_a_count}<br>"
                f"• Anomaly ratio: {_a_ratio:.2%}<br>"
                f"• Spikes: {_anom_result.spike_count} &nbsp;|&nbsp; "
                f"Drops: {_anom_result.drop_count}<br><br>"
                f"<b>Operational interpretation:</b><br>"
                f"• High volatility suggests supply-demand mismatch<br>"
                f"• Possible causes: promotion spikes, stock issues, or demand shocks"
            )

            st.markdown(
                f'''
                <div style="background:{_a_bg};border:1px solid #e2e8f0;
                            border-radius:12px;padding:18px;">
                    <div style="font-size:0.82rem;color:#64748b;margin-bottom:6px;">
                        Anomaly Status
                    </div>
                    <div style="font-size:2rem;font-weight:800;color:{_a_color};line-height:1.1;">
                        {_a_label}
                    </div>
                    <div style="color:#64748b;font-size:0.85rem;margin-top:8px;">
                        Detected points: <b>{_a_count}</b>
                    </div>
                    <div style="color:#475569;font-size:0.82rem;margin-top:12px;line-height:1.6;">
                        {_a_summary}
                    </div>
                </div>
                ''',
                unsafe_allow_html=True,
            )
            st.metric("Anomaly Ratio", f"{_a_ratio:.2%}")
        else:
            st.info("Not enough data for anomaly insight.")

        end_section()

    fig_trend = charts.trend_chart(
        result.series, result.rolling_mean, result.rolling_std, col_sales, rolling_window
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_var = charts.volatility_chart(result.rolling_std, col_sales, rolling_window)
        st.plotly_chart(fig_var, use_container_width=True)
    with col_b:
        _render_variability_card(variability)

    if result.aggregated is not None and len(result.aggregated) > 0:
        st.divider()
        st.markdown(f"#### {agg_level} Aggregated Sales View")
        st.dataframe(result.aggregated.head(50), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 💡 Business Interpretation")
    st.markdown(
        f"- Demand trend is currently **{_simple_trend_label(result.series).lower()}**.\n"
        f"- Variability is **{variability['level']}**, which implies "
        f"**{_variability_guidance(variability['level'])}**.\n"
        f"- Recent movement vs prior period is **{growth_pct:+.1f}%**."
    )


def _render_variability_card(variability: dict) -> None:
    level = variability["level"]
    cv    = variability["cv"]

    color_map = {"low": "#10b981", "moderate": "#f59e0b", "high": "#ef4444"}
    bg_map    = {"low": "#ecfdf5", "moderate": "#fffbeb", "high": "#fef2f2"}

    color = color_map.get(level, "#64748b")
    bg    = bg_map.get(level, "#f8fafc")

    st.markdown(
        f"""
        <div style="background:{bg};border:1px solid #e2e8f0;
                    border-radius:12px;padding:18px;">
            <div style="font-size:0.82rem;color:#64748b;margin-bottom:6px;">
                Demand Stability Index
            </div>
            <div style="font-size:2rem;font-weight:800;color:{color};line-height:1.1;">
                {level.title()}
            </div>
            <div style="color:#64748b;font-size:0.85rem;margin-top:8px;">
                Coefficient of Variation: <b>{cv:.3f}</b>
            </div>
            <div style="color:#475569;font-size:0.82rem;margin-top:12px;line-height:1.6;">
                {_variability_guidance(level).capitalize()}.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 3 — Inventory & Operations
# ---------------------------------------------------------------------------

def _render_inventory_operations(
    df: pd.DataFrame,
    col_date: Optional[str],
    col_sales: Optional[str],
    col_stock: Optional[str],
    col_production: Optional[str],
    col_defect: Optional[str],
) -> None:
    st.markdown("### 🏗️ Inventory & Operations")
    st.caption("Inventory health, stock-out risk, production coverage, and quality performance.")

    has_any = any(c and c in df.columns for c in [col_stock, col_production, col_defect])
    if not has_any:
        _info(
            "Map at least one of **Stock**, **Production**, or **Defect Rate** in the sidebar "
            "to activate this section."
        )
        return

    if col_stock and is_valid_numeric_column(df, col_stock):
        st.markdown("#### 📦 Stock-to-Sales & Stock-out Risk")
        sts = compute_stock_to_sales(df, col_stock=col_stock, col_sales=col_sales)
        _render_stock_panel(sts, col_stock)
        st.divider()

    if col_production and is_valid_numeric_column(df, col_production):
        st.markdown("#### 🏗️ Production Coverage")
        prod = compute_production_summary(df, col_production=col_production, col_sales=col_sales)
        _render_production_panel(df, prod, col_sales, col_production)
        st.divider()

    if col_defect and is_valid_numeric_column(df, col_defect):
        st.markdown("#### ⚠️ Defect Rate & Quality Overview")
        defect = compute_defect_summary(df, col_defect=col_defect)
        _render_defect_panel(defect)


def _render_stock_panel(sts: dict, col_stock: str) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average Stock", f"{sts['avg_stock']:,.2f}")
    c2.metric(
        "Stock-to-Sales Ratio",
        f"{sts['stock_to_sales_ratio']:.2f}x" if sts["stock_to_sales_ratio"] is not None else "N/A",
    )
    c3.metric("Stock-out Days", f"{sts['stockout_risk_days']:,}")

    risk_pct   = sts["stockout_risk_pct"]
    risk_label = "Low" if risk_pct <= 5 else "Moderate" if risk_pct <= 15 else "High"
    c4.metric("Stock-out Risk", f"{risk_label} ({risk_pct:.1f}%)")

    if risk_pct > 15:
        _warn("Stock-out risk exceeds 15% of periods. Review replenishment policy and safety stock thresholds.")
    elif risk_pct > 5:
        _info("Moderate stock-out risk detected. Monitor inventory closely.")
    else:
        st.success("✅ Inventory profile looks stable.")

    if sts.get("stock_series") is not None and hasattr(charts, "stock_level_chart"):
        st.plotly_chart(charts.stock_level_chart(sts["stock_series"], col_stock), use_container_width=True)


def _render_production_panel(
    df: pd.DataFrame,
    prod: dict,
    col_sales: Optional[str],
    col_production: str,
) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Production", f"{prod['total']:,.0f}")
    c2.metric("Average / Period", f"{prod['mean']:,.2f}")
    c3.metric(
        "Production vs Sales",
        f"{prod['prod_to_sales_ratio']:.2f}x" if prod["prod_to_sales_ratio"] is not None else "N/A",
    )

    ratio = prod["prod_to_sales_ratio"]
    if ratio is not None:
        if ratio < 0.95:
            _warn("Production is running below demand. Supply shortage risk may be increasing.")
        elif ratio > 1.30:
            _info("Production exceeds sales by a wide margin. Review build plans to avoid overstock.")
        else:
            st.success("✅ Production is broadly aligned with demand.")

    if (
        col_sales and col_sales in df.columns
        and col_production in df.columns
        and hasattr(charts, "production_vs_sales_chart")
    ):
        sales_s = pd.to_numeric(df[col_sales],      errors="coerce").dropna().reset_index(drop=True)
        prod_s  = pd.to_numeric(df[col_production], errors="coerce").dropna().reset_index(drop=True)
        n       = min(len(sales_s), len(prod_s))
        if n >= 3:
            fig = charts.production_vs_sales_chart(
                production=prod_s.iloc[:n], sales=sales_s.iloc[:n]
            )
            st.plotly_chart(fig, use_container_width=True)
            return

    if prod.get("series") is not None and hasattr(charts, "production_chart"):
        st.plotly_chart(charts.production_chart(prod["series"]), use_container_width=True)


def _render_defect_panel(defect: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Defect Rate",  f"{defect['mean']:.2f}%")
    c2.metric("Max Defect Rate",      f"{defect['max']:.2f}%")
    c3.metric("High-Defect Periods",  f"{defect['high_defect_periods']:,}")

    if defect["mean"] > 5:
        _warn(f"Average defect rate of {defect['mean']:.1f}% is above the 5% threshold. Escalate to QA.")
    elif defect["mean"] > 2:
        _info(f"Defect rate is {defect['mean']:.1f}% — acceptable but should be monitored.")
    else:
        st.success(f"✅ Defect rate {defect['mean']:.1f}% — strong quality performance.")

    if defect.get("series") is not None and hasattr(charts, "defect_rate_chart"):
        st.plotly_chart(charts.defect_rate_chart(defect["series"]), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — Forecast & Insights
# ---------------------------------------------------------------------------

def _render_forecast_insights(
    df: pd.DataFrame,
    col_date: Optional[str],
    col_sales: Optional[str],
    fc_horizon: int,
    fc_method: str,
) -> None:
    st.markdown("### 🔮 Forecast & Executive Insights")
    st.caption("Short-term demand forecast and decision-ready summary for FMCG planning.")

    if not _guard_columns(col_date, col_sales, df):
        return
    if not is_valid_datetime_column(df, col_date):
        _warn("Date column does not contain enough parseable dates for forecasting.")
        return
    if not is_valid_numeric_column(df, col_sales):
        _warn("Sales / Revenue KPI column does not contain enough numeric values.")
        return

    try:
        result: Optional[ForecastResult] = run_fmcg_forecast(
            df=df, date_col=col_date, value_col=col_sales,
            horizon=fc_horizon, method=fc_method,
        )
    except Exception as exc:
        _err(f"Forecast failed: {exc}")
        return

    if result is None:
        _warn("Not enough data points to produce a forecast (minimum 5 required).")
        return

    st.plotly_chart(
        charts.forecast_chart(result.historical, result.forecast, col_sales, result.method_label),
        use_container_width=True,
    )

    direction_label = (
        "Upward"   if result.pct_change > 1  else
        "Downward" if result.pct_change < -1 else
        "Stable"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast Horizon",     f"{fc_horizon} periods")
    c2.metric("Forecast Method",      result.method_label)
    c3.metric("Projected Avg Demand", f"{result.forecast_mean:,.2f}")
    c4.metric("vs Recent Average",    f"{result.recent_mean:,.2f}", delta=f"{result.pct_change:+.1f}%")

    _info(_forecast_action_message(direction_label, fc_horizon, result.method_label))

    st.divider()
    st.markdown("#### 🧠 Executive Summary")
    _render_executive_summary(df, col_sales, result)


def _render_executive_summary(
    df: pd.DataFrame,
    col_sales: Optional[str],
    fc_result: ForecastResult,
) -> None:
    qm     = compute_quality_metrics(df)
    series = _safe_numeric_series(df, col_sales)

    if series is None or len(series) < 3:
        _info("Not enough Sales data to generate an executive summary.")
        return

    variability = compute_demand_variability(series)
    direction   = (
        "upward"   if fc_result.pct_change > 1  else
        "downward" if fc_result.pct_change < -1 else
        "stable"
    )

    short_summary   = _build_short_summary(qm, variability["level"], direction)
    recommendations = _build_recommendations(variability["level"], direction)

    st.markdown(
        f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
            padding:20px 24px;color:#1e293b;">
    <div style="font-size:1.03rem;font-weight:700;color:#0f4c81;margin-bottom:10px;">
        FMCG Executive Summary
    </div>
    <div style="margin-bottom:14px;font-size:0.92rem;line-height:1.7;">
        {short_summary}
    </div>
    <div style="font-size:0.88rem;color:#475569;line-height:1.7;">
        <b>Forecast Outlook:</b> Demand is projected to be <b>{direction}</b> over the next
        <b>{fc_result.horizon}</b> periods using <b>{fc_result.method_label}</b>.
        Forecast mean is <b>{fc_result.forecast_mean:,.2f}</b> versus recent average of
        <b>{fc_result.recent_mean:,.2f}</b> ({fc_result.pct_change:+.1f}%).
    </div>
    <div style="margin-top:14px;font-size:0.88rem;line-height:1.8;">
        <b>Recommended Actions:</b><br>
        {'<br>'.join(recommendations)}
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 5 — Optimization
# ---------------------------------------------------------------------------
from app.optimization.transportation import linear_programming_method
# Sabit girdi verisi — ileride sidebar'dan alınabilir
_DEFAULT_SUPPLY = [120, 80, 100]
_DEFAULT_DEMAND = [70, 90, 60, 80]
_DEFAULT_COSTS  = np.array(
    [[8, 6, 10, 9], [9, 12, 13, 7], [14, 9, 16, 5]],
    dtype=float,
)


def _run_optimization(supply: list, demand: list, costs: np.ndarray) -> dict:
    """
    Üç algoritmayı çalıştırır, en iyisini seçer ve sonuç sözlüğü döndürür.
    Her algoritma kendi verisinin kopyasını alır — in-place bozulma olmaz.
    """
    supply_b, demand_b, costs_b = balance_problem(supply, demand, costs)

    alloc_lc,  cost_lc  = least_cost_method(list(supply_b),  list(demand_b),  costs_b.copy())
    alloc_nw,  cost_nw  = northwest_corner_method(list(supply_b), list(demand_b), costs_b.copy())
    alloc_vam, cost_vam = vogel_approximation_method(list(supply_b), list(demand_b), costs_b.copy())
    alloc_lp, cost_lp = linear_programming_method(
        list(supply_b),
        list(demand_b),
        costs_b.copy(),
    )
    alloc_lc  = np.array(alloc_lc,  dtype=float)
    alloc_nw  = np.array(alloc_nw,  dtype=float)
    alloc_vam = np.array(alloc_vam, dtype=float)

    method_map = {
        "Least Cost": (alloc_lc,  cost_lc),
        "Vogel":      (alloc_vam, cost_vam),
        "Northwest":  (alloc_nw,  cost_nw),
    }
    if  alloc_lp is not None:
         method_map["Linear Programming"] = (alloc_lp, cost_lp)

    best_name = min(method_map, key=lambda k: method_map[k][1])
    best_alloc, best_cost = method_map[best_name]

    return {
        "best_method": best_name,
        "best_cost":   float(best_cost),
        "best_alloc":  best_alloc,
        "costs_map":   {k: float(v[1]) for k, v in method_map.items()},
        # Ham veriyi senaryo analizinde yeniden kullanmak için sakla
        "supply":      list(supply_b),
        "demand":      list(demand_b),
        "costs":       costs_b,
    }


def _render_optimization() -> None:
    """
    Tab 5 render fonksiyonu.

    Render akışı:
    ┌─────────────────────────────────────────────┐
    │  Her zaman görünen alan                     │
    │  - Başlık + açıklama                        │
    │  - "Optimizasyonu Çalıştır" butonu          │
    └─────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────┐
    │  Sadece opt_done == True ise görünen alan   │
    │  - KPI kartları                             │
    │  - Yöntem karşılaştırma grafiği             │
    │  - Optimal tahsis tablosu                   │
    │  - Isı haritası                             │
    │  - Senaryo analizi                          │
    │  - İş yorumu                                │
    │  - Yönetici özeti                           │
    └─────────────────────────────────────────────┘
    """
    hero_banner(
    title="🚀 Logistics Optimization Engine",
    subtitle="Transportation cost minimization, route efficiency comparison, and scenario-based planning in a single decision module.",
    badge="Operations Intelligence"
)
    # ── Başlık (her zaman görünür) ────────────────────────────────────────────
    st.markdown("### 🚀 Lojistik Optimizasyonu")
    st.caption("Sevkiyat maliyetlerini minimize eden optimal dağıtım planı.")

    # ── Buton (her zaman görünür) ─────────────────────────────────────────────
    # st.button() True döndürdüğü render'da hesaplamayı tetikler,
    # ardından Streamlit sayfayı yeniden çizer; sonuç session_state'te saklandığı
    # için kaybolmaz.
    if st.button("▶ Optimizasyonu Çalıştır", type="primary"):
        with st.spinner("Algoritmalar çalışıyor…"):
            try:
                result = _run_optimization(
                    _DEFAULT_SUPPLY,
                    _DEFAULT_DEMAND,
                    _DEFAULT_COSTS.copy(),
                )
                st.session_state[_OPT_KEY] = result
            except Exception as exc:
                st.error(f"Optimizasyon hatası: {exc}")
                # Hatalı durum varsa eski sonucu temizle
                st.session_state.pop(_OPT_KEY, None)

    # ── Sonuç yoksa buradan çık — hiçbir sonuç bileşeni render edilmez ────────
    if _OPT_KEY not in st.session_state:
        st.info("Hesaplamayı başlatmak için **▶ Optimizasyonu Çalıştır** butonuna tıklayın.")
        return

    # ── Sonuçları state'den oku ───────────────────────────────────────────────
    res         = st.session_state[_OPT_KEY]
    best_method = res["best_method"]
    best_cost   = res["best_cost"]
    best_alloc  = res["best_alloc"]
    costs_map   = res["costs_map"]
    bal_supply  = res["supply"]
    bal_demand  = res["demand"]
    bal_costs   = res["costs"]

    # ── KPI Paneli ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📊 Operasyon Özeti")

    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Maliyet", f"{best_cost:,.0f}")
    c2.metric("En İyi Yöntem",  best_method)
    c3.metric("Aktif Rota",     int((best_alloc > 0).sum()))

    # ── Yöntem Karşılaştırma ──────────────────────────────────────────────────
    st.divider()
    st.markdown("#### ⚖️ Yöntem Karşılaştırma")

    compare_df = pd.DataFrame({
        "Yöntem":  list(costs_map.keys()),
        "Maliyet": list(costs_map.values()),
    })
    fig_bar = px.bar(
        compare_df, x="Yöntem", y="Maliyet",
        title="Yöntemlere Göre Toplam Maliyet",
        color="Yöntem",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Optimal Tahsis Tablosu ────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 📦 Optimal Sevkiyat Planı")

    n_src, n_dst = best_alloc.shape
    alloc_df = pd.DataFrame(
        best_alloc,
        index=[f"Kaynak {i + 1}" for i in range(n_src)],
        columns=[f"Hedef {j + 1}" for j in range(n_dst)],
    )
    st.dataframe(
        alloc_df.style.highlight_max(axis=None, color="#d4edda"),
        use_container_width=True,
    )

    # ── Isı Haritası ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🔥 Sevkiyat Yoğunluğu")

    fig_heat = px.imshow(
        best_alloc,
        text_auto=True,
        title="Sevkiyat Isı Haritası",
        labels={"x": "Hedef", "y": "Kaynak"},
        x=[f"Hedef {j + 1}" for j in range(n_dst)],
        y=[f"Kaynak {i + 1}" for i in range(n_src)],
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    # ── Maliyet Katkısı Heatmap ───────────────────────────────────────────────
    st.divider()
    st.markdown("#### 💸 Rota Bazlı Maliyet Katkısı")
    st.caption("Her rotanın toplam maliyete katkısı: sevkiyat miktarı × birim maliyet.")

    cost_contribution = best_alloc * bal_costs

    fig_cost_heat = px.imshow(
    cost_contribution,
    text_auto=".0f",
    labels={"x": "Hedef", "y": "Kaynak", "color": "Maliyet"},
    x=[f"Hedef {j + 1}" for j in range(cost_contribution.shape[1])],
    y=[f"Kaynak {i + 1}" for i in range(cost_contribution.shape[0])],
    color_continuous_scale="OrRd",
)

    fig_cost_heat.update_layout(
    template="plotly_white",
    height=430,
)

    st.plotly_chart(fig_cost_heat, use_container_width=True)
     
    # ── Senaryo Analizi ───────────────────────────────────────────────────────
    # Bu bloğun tamamı _render_optimization() içinde ve opt_done guard'ından
    # SONRA olduğu için asla erken render edilmez.
    st.divider()
    st.markdown("#### 🧪 Senaryo Analizi")
    st.caption("Parametreleri değiştirerek maliyet etkisini anlık görün.")

    col1, col2, col3 = st.columns(3)
    with col1:
        demand_increase_pct = st.slider("Talep Artışı (%)",                  0, 50,  10, key="opt_demand_pct")
    with col2:
        route_cost_increase = st.slider("Seçili Rota Maliyet Artışı (%)",    0, 100, 20, key="opt_route_cost")
    with col3:
        capacity_drop_pct   = st.slider("Kaynak Kapasite Düşüşü (%)",        0, 50,  10, key="opt_cap_drop")

    if st.button("📊 Senaryoyu Uygula", key="opt_scenario_btn"):
        with st.spinner("Senaryo hesaplanıyor…"):
            try:
                # Dengeli verinin kopyasını kullan — orijinali bozma
                sc_supply = [s * (1 - capacity_drop_pct / 100) if idx == 0 else s
                             for idx, s in enumerate(bal_supply)]
                sc_demand = [d * (1 + demand_increase_pct / 100) for d in bal_demand]
                sc_costs  = bal_costs.copy()
                sc_costs[0, 1] *= (1 + route_cost_increase / 100)  # örnek rota: [0,1]

                sc_supply, sc_demand, sc_costs = balance_problem(sc_supply, sc_demand, sc_costs)
                alloc_sc, cost_sc = least_cost_method(
                    list(sc_supply), list(sc_demand), sc_costs.copy()
                )
                alloc_sc = np.array(alloc_sc, dtype=float)

                st.session_state["opt_scenario"] = {
                    "alloc":    alloc_sc,
                    "cost":     float(cost_sc),
                    "baseline": best_cost,
                }
            except Exception as exc:
                st.error(f"Senaryo hatası: {exc}")
                st.session_state.pop("opt_scenario", None)

    # Senaryo sonuçları — sadece hesaplandıktan sonra görünür
    if "opt_scenario" in st.session_state:
        sc = st.session_state["opt_scenario"]

        st.markdown("##### 📊 Senaryo Sonuçları")

        colA, colB = st.columns(2)
        colA.metric("Mevcut Maliyet", f"{sc['baseline']:,.0f}")
        colB.metric(
            "Yeni Maliyet",
            f"{sc['cost']:,.0f}",
            delta=f"{sc['cost'] - sc['baseline']:+.0f}",
        )

        change_pct = ((sc["cost"] - sc["baseline"]) / sc["baseline"]) * 100
        if change_pct > 0:
            st.warning(
                f"Maliyet **%{change_pct:.1f}** arttı. "
                f"Sistem daha pahalı rotalara kaymak zorunda kaldı."
            )
        else:
            st.success(
                f"Maliyet **%{abs(change_pct):.1f}** azaldı. "
                f"Yeni dağıtım daha verimli hale geldi."
            )

        sc_n_src, sc_n_dst = sc["alloc"].shape
        sc_df = pd.DataFrame(
            sc["alloc"],
            index=[f"Kaynak {i + 1}" for i in range(sc_n_src)],
            columns=[f"Hedef {j + 1}" for j in range(sc_n_dst)],
        )
        st.dataframe(sc_df.style.highlight_max(axis=None, color="#d4edda"), use_container_width=True)

    # ── İş Yorumu ─────────────────────────────────────────────────────────────
    # max route bul (eklemen lazım çünkü yok)
    max_route = np.unravel_index(np.argmax(best_alloc), best_alloc.shape)
    max_value = best_alloc[max_route]

    insight_card(
    title="İş Yorumu",
    icon="🧠",
    body=f"""
    En düşük maliyetli çözüm <b>{best_method}</b> yöntemi ile elde edildi.
    Toplam lojistik maliyet <b>{best_cost:.0f}</b> olarak hesaplandı.
    En yoğun sevkiyat hattı <b>Kaynak {max_route[0]+1} → Hedef {max_route[1]+1}</b> olup
    <b>{max_value:.0f} birim</b> sevkiyat planlandı.
    """,
)

    insight_card(
    title="Yönetici Özeti",
    icon="📌",
    body=f"""
    Sistem, dağıtım ağında maliyet açısından verimli rotaları önceliklendirerek
    toplam sevkiyat maliyetini minimize etti. Bu modül; rota optimizasyonu,
    kapasite baskısı ve talep değişimi gibi koşullarda karar desteği sunar.
    """,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_category_filter(
    df: pd.DataFrame,
    col_category: Optional[str],
    cat_filter: Optional[list],
) -> pd.DataFrame:
    if not col_category or col_category not in df.columns or not cat_filter:
        return df
    return df[df[col_category].astype(str).isin(cat_filter)].copy()


def _guard_columns(
    col_date: Optional[str],
    col_value: Optional[str],
    df: pd.DataFrame,
) -> bool:
    ok = True
    if not col_date or col_date not in df.columns:
        _warn("Please map a **Date Column** in the sidebar to enable this view.")
        ok = False
    if not col_value or col_value not in df.columns:
        _warn("Please map a **Sales / Revenue KPI** column in the sidebar to enable this view.")
        ok = False
    return ok


def _safe_numeric_series(df: pd.DataFrame, column: Optional[str]) -> Optional[pd.Series]:
    if not column or column not in df.columns:
        return None
    s = pd.to_numeric(df[column], errors="coerce").dropna()
    return s if len(s) >= 3 else None


def _compute_quality_score(qm: dict) -> int:
    score = 100
    score -= min(40, int(round(qm["missing_pct"])))
    if qm["total_rows"] > 0:
        dup_pct = (qm["duplicate_count"] / qm["total_rows"]) * 100
        score  -= min(25, int(round(dup_pct)))
    return max(0, min(100, score))


def _risk_label_from_variability(level: str) -> str:
    return {"low": "Low", "moderate": "Moderate", "high": "High"}.get(level, "N/A")


def _simple_trend_label(series: pd.Series) -> str:
    if len(series) < 6:
        return "Stable"
    recent_n = min(7, len(series))
    recent   = float(series.iloc[-recent_n:].mean())
    prior    = (
        float(series.iloc[-recent_n * 2:-recent_n].mean())
        if len(series) >= recent_n * 2
        else float(series.iloc[:-recent_n].mean()) if len(series) > recent_n
        else recent
    )
    if prior == 0:
        return "Stable"
    pct = ((recent - prior) / abs(prior)) * 100
    return "Increasing" if pct > 2 else "Declining" if pct < -2 else "Stable"


def _recent_growth_pct(series: pd.Series) -> float:
    if len(series) < 6:
        return 0.0
    recent_n = min(7, len(series))
    recent   = float(series.iloc[-recent_n:].mean())
    prior    = (
        float(series.iloc[-recent_n * 2:-recent_n].mean())
        if len(series) >= recent_n * 2
        else float(series.iloc[:-recent_n].mean()) if len(series) > recent_n
        else recent
    )
    return 0.0 if prior == 0 else ((recent - prior) / abs(prior)) * 100


def _variability_guidance(level: str) -> str:
    return {
        "low":      "stable demand patterns and lower buffer requirements",
        "moderate": "moderate fluctuations and the need to review safety stock",
        "high":     "high demand instability and a need for stronger stock buffers",
    }.get(level, "unclear demand dynamics")


def _forecast_action_message(direction_label: str, horizon: int, method_label: str) -> str:
    if direction_label == "Upward":
        return (
            f"📈 Short-term demand is expected to increase over the next **{horizon}** periods "
            f"using **{method_label}**. Consider increasing stock coverage and procurement readiness."
        )
    if direction_label == "Downward":
        return (
            f"📉 Short-term demand is expected to soften over the next **{horizon}** periods "
            f"using **{method_label}**. Review replenishment and avoid excess stock build-up."
        )
    return (
        f"➡️ Short-term demand is expected to remain stable over the next **{horizon}** periods "
        f"using **{method_label}**. Current inventory policy can likely be maintained."
    )


def _build_overview_insights(
    qm: dict,
    demand_trend: str,
    demand_stability: str,
    operational_risk: str,
) -> list[str]:
    insights = [
        f"Dataset contains **{qm['total_rows']:,}** rows and **{qm['total_cols']}** columns.",
        f"Current demand trend is **{demand_trend.lower()}**.",
        f"Demand stability is assessed as **{demand_stability.lower()}** with "
        f"**{operational_risk.lower()}** operational risk.",
    ]
    if qm["missing_pct"] == 0 and qm["duplicate_count"] == 0:
        insights.append("Dataset quality is strong and ready for reliable analysis.")
    elif qm["missing_pct"] > 10:
        insights.append("Missing data is material and should be reviewed before high-stakes decisions.")
    else:
        insights.append("Dataset is usable, but some cleanup checks are still advisable.")
    return insights


def _build_short_summary(qm: dict, variability_level: str, direction: str) -> str:
    quality_text = (
        "strong"          if qm["missing_pct"] < 2 and qm["duplicate_count"] == 0 else
        "acceptable"      if qm["missing_pct"] < 10 else
        "needs attention"
    )
    variability_text = {
        "low":      "stable",
        "moderate": "moderately volatile",
        "high":     "highly volatile",
    }.get(variability_level, "unclear")
    direction_text = {
        "upward":   "an upward demand signal",
        "downward": "a downward demand signal",
        "stable":   "a stable near-term demand outlook",
    }.get(direction, "an unclear demand signal")
    return (
        f"The dataset is **{quality_text}** from a quality perspective. "
        f"Demand behavior appears **{variability_text}**, and the forecast indicates "
        f"**{direction_text}**."
    )


def _build_recommendations(variability_level: str, direction: str) -> list[str]:
    actions: list[str] = []
    if variability_level == "low":
        actions.append("✅ Maintain current replenishment cadence.")
    elif variability_level == "moderate":
        actions.append("⚠️ Review safety stock thresholds for volatile SKUs.")
    else:
        actions.append("🚨 Escalate demand variability findings to supply chain leadership.")
        actions.append("⚠️ Increase monitoring frequency for inventory buffers.")

    if direction == "upward":
        actions.append("📈 Consider increasing stock positions ahead of projected demand uplift.")
    elif direction == "downward":
        actions.append("📉 Review procurement plans to avoid overstock exposure.")
    else:
        actions.append("➡️ Maintain current planning assumptions unless new signals emerge.")
    return actions