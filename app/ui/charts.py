"""
app/ui/charts.py
================
All Plotly figure builders for the FMCG Analytics Platform.

Pure charting only:
- no Streamlit calls
- no business logic
- accepts precomputed pd.Series / pd.DataFrame inputs
- returns plotly.graph_objects.Figure objects ready for st.plotly_chart()
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

def apply_premium_layout(
    fig,
    title=None,
    height=420,
    x_title=None,
    y_title=None,
):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color="#0f172a"),
            x=0.02,
            xanchor="left",
        ) if title else None,

        height=height,
        template="plotly_white",

        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",

        font=dict(
            family="Inter, Segoe UI, Arial",
            size=12,
            color="#334155",
        ),

        margin=dict(l=40, r=30, t=60 if title else 30, b=45),

        hovermode="x unified",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=11),
        ),

        xaxis=dict(
            title=x_title,
            showgrid=False,
            zeroline=False,
            linecolor="#e2e8f0",
            tickfont=dict(color="#64748b"),
        ),

        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor="#eef2f7",
            zeroline=False,
            linecolor="#e2e8f0",
            tickfont=dict(color="#64748b"),
        ),
    )

    fig.update_traces(
        hovertemplate=None,
    )

    return fig

# ---------------------------------------------------------------------------
# Shared design tokens
# ---------------------------------------------------------------------------

COLORS = {
    "navy": "#0f4c81",
    "blue": "#1e88e5",
    "green": "#10b981",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "slate": "#64748b",
    "grid": "rgba(226,232,240,0.8)",
    "paper": "rgba(0,0,0,0)",
    "plot": "rgba(248,250,252,0.72)",
    "band_blue": "rgba(30,136,229,0.10)",
    "fill_green": "rgba(16,185,129,0.15)",
    "fill_amber": "rgba(245,158,11,0.15)",
}

FONT_FAMILY = "Inter, system-ui, sans-serif"


# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------

def _base_layout(**kwargs) -> dict:
    return dict(
        font=dict(family=FONT_FAMILY, size=12, color="#1e293b"),
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["plot"],
        hovermode="x unified",
        margin=dict(l=40, r=20, t=48, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            title_font=dict(size=12),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLORS["grid"],
            zeroline=False,
            title_font=dict(size=12),
        ),
        **kwargs,
    )


def _series_x(series: pd.Series) -> list:
    """
    Use the series index when meaningful; otherwise fallback to period index.
    """
    if series is None or len(series) == 0:
        return []

    idx = series.index
    if isinstance(idx, pd.DatetimeIndex):
        return list(idx)

    if idx.nlevels == 1:
        try:
            if not isinstance(idx, pd.RangeIndex):
                return list(idx)
        except Exception:
            pass

    return list(range(len(series)))


def _x_title_from_series(series: pd.Series) -> str:
    return "Date" if isinstance(series.index, pd.DatetimeIndex) else "Period Index"


# ---------------------------------------------------------------------------
# Trend Chart
# ---------------------------------------------------------------------------

def trend_chart(
    series: pd.Series,
    rolling_mean: pd.Series,
    rolling_std: pd.Series,
    value_col: str,
    rolling_window: int,
) -> go.Figure:
    """
    Main trend chart: actual values + rolling mean + std-dev band.
    """
    x = _series_x(series)
    x_title = _x_title_from_series(series)

    fig = go.Figure()

    if series is None or len(series) == 0:
        fig.update_layout(
            **_base_layout(
                title=dict(text=f"📈 Demand Trend — {value_col}", font=dict(size=14, color=COLORS["navy"]))
            )
        )
        return apply_premium_layout(
    fig,
    title=f"Demand Trend — {value_col}",
    x_title="Date",
    y_title=value_col,
)

    upper = rolling_mean + rolling_std
    lower = (rolling_mean - rolling_std).clip(lower=0)

    band_x = x + x[::-1]
    band_y = list(upper) + list(lower)[::-1]

    fig.add_trace(
        go.Scatter(
            x=band_x,
            y=band_y,
            fill="toself",
            fillcolor=COLORS["band_blue"],
            line=dict(width=0),
            name=f"±1 Std Dev Band ({rolling_window}p)",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=f"Actual {value_col}",
            line=dict(width=2.2, color="#2563eb"),
            opacity=0.85,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean.values,
            mode="lines",
            name=f"{rolling_window}-Period Rolling Mean",
            line=dict(width=3.2, color="#0f4c81"),
        )
    )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"📈 Demand Trend — {value_col}",
                font=dict(size=14, color=COLORS["navy"]),
            ),
        ),
        yaxis_title="Demand Volume",
        xaxis_title=x_title,
    )
    return apply_premium_layout(
    fig,
    title=f"Demand Trend — {value_col}",
    x_title=x_title,
    y_title="Demand Volume",
    height=460,
)


# ---------------------------------------------------------------------------
# Demand Variability Chart
# ---------------------------------------------------------------------------

def volatility_chart(
    rolling_std: pd.Series,
    value_col: str,
    rolling_window: int,
) -> go.Figure:
    """
    Rolling standard deviation chart.
    Function name kept for backward compatibility with tabs.py.
    """
    x = _series_x(rolling_std)
    x_title = _x_title_from_series(rolling_std)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=rolling_std,
            mode="lines",
            fill="tozeroy",
            name=f"{rolling_window}-Period Rolling Std Dev",
            line=dict(color=COLORS["amber"], width=2),
            fillcolor=COLORS["fill_amber"],
        )
    )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"📉 Demand Variability — {value_col}",
                font=dict(size=13, color=COLORS["navy"]),
            ),
            showlegend=False,
        ),
        yaxis_title="Standard Deviation",
        xaxis_title=x_title,
    )
    return fig


# ---------------------------------------------------------------------------
# Forecast Chart
# ---------------------------------------------------------------------------

def forecast_chart(
    historical: pd.Series,
    forecast: pd.Series,
    value_col: str,
    method_label: str,
) -> go.Figure:
    """
    Historical demand + forward forecast.
    Supports datetime index when present.
    """
    fig = go.Figure()

    if historical is None or len(historical) == 0:
        fig.update_layout(
            **_base_layout(
                title=dict(text=f"🔮 Demand Forecast — {value_col}", font=dict(size=14, color=COLORS["navy"]))
            )
        )
        return fig

    x_hist = _series_x(historical)
    hist_x_title = _x_title_from_series(historical)

    if forecast is None or len(forecast) == 0:
        fig.add_trace(
            go.Scatter(
                x=x_hist,
                y=historical,
                mode="lines",
                name="Historical Demand",
                line=dict(color=COLORS["blue"], width=2),
            )
        )
        fig.update_layout(
            **_base_layout(
                title=dict(
                    text=f"🔮 Demand Forecast — {value_col} ({method_label})",
                    font=dict(size=14, color=COLORS["navy"]),
                ),
            ),
            yaxis_title="Demand Volume",
            xaxis_title=hist_x_title,
        )
        return fig

    if isinstance(historical.index, pd.DatetimeIndex) and isinstance(forecast.index, pd.DatetimeIndex):
        x_fc = list(forecast.index)
        xaxis_title = "Date"
        divider_x = x_fc[0]
        use_vline = False
    else:
        n_hist = len(historical)
        x_fc = list(range(n_hist, n_hist + len(forecast)))
        xaxis_title = "Period Index"
        divider_x = n_hist - 0.5
        use_vline = True

    fig.add_trace(
        go.Scatter(
            x=x_hist,
            y=historical,
            mode="lines",
            name="Historical Demand",
            line=dict(color=COLORS["blue"], width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_fc,
            y=forecast,
            mode="lines+markers",
            name=f"Forecast ({method_label})",
            line=dict(color=COLORS["red"], width=2.5, dash="dash"),
            marker=dict(size=5, symbol="circle"),
        )
    )

    if use_vline:
        fig.add_vline(
            x=divider_x,
            line=dict(color="#94a3b8", dash="dot", width=1.5),
            annotation_text=" Forecast →",
            annotation_position="top right",
            annotation_font_color=COLORS["slate"],
        )
    else:
        y_max = max(float(historical.max()), float(forecast.max()))
        y_min = min(float(historical.min()), float(forecast.min()))
        fig.add_shape(
            type="line",
            x0=divider_x,
            x1=divider_x,
            y0=y_min,
            y1=y_max,
            line=dict(color="#94a3b8", dash="dot", width=1.5),
        )
        fig.add_annotation(
            x=divider_x,
            y=y_max,
            text="Forecast →",
            showarrow=False,
            yshift=10,
            font=dict(size=10, color=COLORS["slate"]),
            bgcolor="rgba(255,255,255,0.7)",
        )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"🔮 Demand Forecast — {value_col} ({method_label})",
                font=dict(size=14, color=COLORS["navy"]),
            ),
        ),
        yaxis_title="Demand Volume",
        xaxis_title=xaxis_title,
    )
    return apply_premium_layout(
    fig,
    title=f"Demand Forecast — {value_col} ({method_label})",
    x_title=xaxis_title,
    y_title="Demand Volume",
    height=460,
)


# ---------------------------------------------------------------------------
# Stock Level Chart
# ---------------------------------------------------------------------------

def stock_level_chart(
    stock_series: pd.Series,
    col_stock: str,
) -> go.Figure:
    """
    Stock level area chart with stock-out markers.
    """
    x = _series_x(stock_series)
    x_title = _x_title_from_series(stock_series)
    values = list(stock_series)

    above = [v if v > 0 else None for v in values]
    at_risk = [v if v <= 0 else None for v in values]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=above,
            mode="lines",
            fill="tozeroy",
            name="Stock Level",
            line=dict(color=COLORS["green"], width=2),
            fillcolor=COLORS["fill_green"],
            connectgaps=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=at_risk,
            mode="markers",
            name="Stock-out Risk",
            marker=dict(color=COLORS["red"], size=7, symbol="x"),
            connectgaps=False,
        )
    )

    fig.add_hline(
        y=0,
        line=dict(color=COLORS["red"], dash="dot", width=1.2),
        annotation_text="Stock-out line",
        annotation_font_color=COLORS["red"],
    )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"📦 Stock Level — {col_stock}",
                font=dict(size=13, color=COLORS["navy"]),
            ),
        ),
        yaxis_title="Units",
        xaxis_title=x_title,
    )
    return fig


# ---------------------------------------------------------------------------
# Production Chart
# ---------------------------------------------------------------------------

def production_chart(production_series: pd.Series) -> go.Figure:
    """
    Simple production volume chart.
    """
    x = _series_x(production_series)
    x_title = _x_title_from_series(production_series)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=production_series,
            name="Production Volume",
            marker_color=COLORS["navy"],
            opacity=0.85,
        )
    )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="🏗️ Production Volume",
                font=dict(size=13, color=COLORS["navy"]),
            ),
            showlegend=False,
        ),
        yaxis_title="Units Produced",
        xaxis_title=x_title,
    )
    return fig


# ---------------------------------------------------------------------------
# Production vs Sales Chart
# ---------------------------------------------------------------------------

def production_vs_sales_chart(
    production: pd.Series,
    sales: pd.Series,
) -> go.Figure:
    """
    Compare production and sales on the same chart.
    """
    min_len = min(len(production), len(sales))
    production = production.iloc[:min_len]
    sales = sales.iloc[:min_len]

    x = _series_x(production)
    x_title = _x_title_from_series(production)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=production,
            mode="lines",
            name="Production",
            line=dict(color=COLORS["navy"], width=2.4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=sales,
            mode="lines",
            name="Sales",
            line=dict(color=COLORS["blue"], width=2.2, dash="dash"),
        )
    )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="🏗️ Production vs Sales",
                font=dict(size=13, color=COLORS["navy"]),
            ),
        ),
        yaxis_title="Units / Volume",
        xaxis_title=x_title,
    )
    return fig


# ---------------------------------------------------------------------------
# Defect Rate Chart
# ---------------------------------------------------------------------------

def defect_rate_chart(
    defect_series: pd.Series,
    threshold: float = 5.0,
) -> go.Figure:
    """
    Defect-rate chart with threshold line.
    """
    x = _series_x(defect_series)
    x_title = _x_title_from_series(defect_series)
    values = list(defect_series)

    colors = [COLORS["red"] if v > threshold else COLORS["green"] for v in values]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x,
            y=defect_series,
            name="Defect Rate (%)",
            marker_color=colors,
            opacity=0.88,
        )
    )

    fig.add_hline(
        y=threshold,
        line=dict(color=COLORS["amber"], dash="dash", width=2),
        annotation_text=f"Threshold ({threshold}%)",
        annotation_font_color=COLORS["amber"],
    )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text="⚠️ Defect Rate by Period",
                font=dict(size=13, color=COLORS["navy"]),
            ),
            showlegend=False,
        ),
        yaxis_title="Defect Rate (%)",
        xaxis_title=x_title,
    )
    return fig

import plotly.graph_objects as go


def anomaly_chart(anomaly_df, date_col: str, value_col: str):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=anomaly_df[date_col],
            y=anomaly_df[value_col],
            mode="lines",
            name="Gerçek Talep",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=anomaly_df[date_col],
            y=anomaly_df["rolling_mean"],
            mode="lines",
            name="Rolling Mean",
        )
    )

    spikes = anomaly_df[anomaly_df["anomaly_type"] == "spike"]
    drops = anomaly_df[anomaly_df["anomaly_type"] == "drop"]

    fig.add_trace(
        go.Scatter(
            x=spikes[date_col],
            y=spikes[value_col],
            mode="markers",
            name="Spike Anomaly",
            marker=dict(size=9, symbol="triangle-up"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=drops[date_col],
            y=drops[value_col],
            mode="markers",
            name="Drop Anomaly",
            marker=dict(size=9, symbol="triangle-down"),
        )
    )

    fig.update_layout(
        title="Talep Sapmaları ve Anomali Tespiti",
        xaxis_title="Tarih",
        yaxis_title="Satış Hacmi",
        template="plotly_dark",
        height=500,
    )

    return fig

plot_bgcolor = "#ffffff"
paper_bgcolor = "#ffffff"
gridcolor = "#e5e7eb"