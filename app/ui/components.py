"""
components.py — Reusable Streamlit UI components (v2).

Client-ready, polished, zero debug noise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def page_header(title: str, subtitle: str = "") -> None:
    """Styled page-level header with optional subtitle."""
    st.markdown(
        f"<h1 style='font-size:1.7rem; font-weight:800; color:#0f172a; "
        f"margin-bottom:2px'>{title}</h1>",
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(
            f"<p style='color:#64748b; margin-top:0; margin-bottom:10px; "
            f"font-size:0.93rem'>{subtitle}</p>",
            unsafe_allow_html=True,
        )
    st.divider()


def section_header(title: str, icon: str = "") -> None:
    """Styled section-level header."""
    label = f"{icon}&nbsp; {title}" if icon else title
    st.markdown(
        f"<h3 style='color:#1e3a5f; font-size:1.1rem; font-weight:700; "
        f"margin-bottom:6px'>{label}</h3>",
        unsafe_allow_html=True,
    )


def info_box(message: str) -> None:
    st.info(message, icon="ℹ️")


def warning_box(message: str) -> None:
    st.warning(message, icon="⚠️")


def success_box(message: str) -> None:
    st.success(message, icon="✅")


def error_box(message: str) -> None:
    st.error(message, icon="🚨")


def guidance_block(heading: str, body: str) -> None:
    """
    Render a soft guidance/onboarding block — shown instead of blank sections.
    """
    st.markdown(
        f"""
        <div style="
            background:#f8fafc;
            border:1px dashed #cbd5e1;
            border-radius:10px;
            padding:24px 28px;
            text-align:center;
            color:#64748b;
        ">
            <div style="font-size:2.2rem; margin-bottom:8px">💡</div>
            <div style="font-weight:700; font-size:1rem; color:#334155; margin-bottom:6px">
                {heading}
            </div>
            <div style="font-size:0.88rem; line-height:1.6">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# KPI bar (NEW — top of page after upload)
# ---------------------------------------------------------------------------


def kpi_bar(
    total_rows: int,
    missing_pct: float,
    trend_direction: str,
    trend_pct_change: float,
    volatility_label: str,
) -> None:
    """
    Render the top KPI row: 4 styled metric cards.

    Trend and volatility cards use colour-coding to signal good/bad immediately.
    """
    # Trend card config
    trend_configs = {
        "upward":   dict(icon="📈", color="#059669", bg="#f0fdf4", label="Trending Up"),
        "downward": dict(icon="📉", color="#dc2626", bg="#fef2f2", label="Trending Down"),
        "flat":     dict(icon="➡️", color="#0284c7", bg="#f0f9ff", label="Stable"),
        "unknown":  dict(icon="➡️", color="#6b7280", bg="#f9fafb", label="Unknown"),
    }
    # Volatility card config
    vol_configs = {
        "low":     dict(icon="🟢", color="#059669", bg="#f0fdf4"),
        "moderate":dict(icon="🟡", color="#d97706", bg="#fffbeb"),
        "high":    dict(icon="🔴", color="#dc2626", bg="#fef2f2"),
        "unknown": dict(icon="⚪", color="#6b7280", bg="#f9fafb"),
    }

    tc  = trend_configs.get(trend_direction, trend_configs["unknown"])
    vc  = vol_configs.get(volatility_label, vol_configs["unknown"])
    vol_disp = volatility_label.title() if volatility_label != "unknown" else "—"

    pct_sign = f"{trend_pct_change:+.1f}%" if trend_direction != "unknown" else "—"

    # Missing % color
    if missing_pct < 0.05:
        mp_color = "#059669"
    elif missing_pct < 0.20:
        mp_color = "#d97706"
    else:
        mp_color = "#dc2626"

    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:20px">

          <!-- Total Rows -->
          <div style="background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px;
                      padding:18px 16px;">
            <div style="font-size:0.75rem; color:#0284c7; font-weight:600;
                        text-transform:uppercase; letter-spacing:0.05em">
              Total Rows
            </div>
            <div style="font-size:1.8rem; font-weight:800; color:#0f172a; line-height:1.2;
                        margin-top:4px">
              {total_rows:,}
            </div>
            <div style="font-size:0.78rem; color:#64748b; margin-top:3px">records loaded</div>
          </div>

          <!-- Missing Data -->
          <div style="background:#f9fafb; border:1px solid #e2e8f0; border-radius:10px;
                      padding:18px 16px;">
            <div style="font-size:0.75rem; color:#64748b; font-weight:600;
                        text-transform:uppercase; letter-spacing:0.05em">
              Missing Data
            </div>
            <div style="font-size:1.8rem; font-weight:800; color:{mp_color}; line-height:1.2;
                        margin-top:4px">
              {missing_pct:.1%}
            </div>
            <div style="font-size:0.78rem; color:#64748b; margin-top:3px">of all values</div>
          </div>

          <!-- Trend Direction -->
          <div style="background:{tc['bg']}; border:1px solid #e2e8f0; border-radius:10px;
                      padding:18px 16px;">
            <div style="font-size:0.75rem; color:{tc['color']}; font-weight:600;
                        text-transform:uppercase; letter-spacing:0.05em">
              Trend
            </div>
            <div style="font-size:1.8rem; font-weight:800; color:{tc['color']}; line-height:1.2;
                        margin-top:4px">
              {tc['icon']} {tc['label']}
            </div>
            <div style="font-size:0.78rem; color:#64748b; margin-top:3px">
              {pct_sign} vs prior period
            </div>
          </div>

          <!-- Volatility -->
          <div style="background:{vc['bg']}; border:1px solid #e2e8f0; border-radius:10px;
                      padding:18px 16px;">
            <div style="font-size:0.75rem; color:{vc['color']}; font-weight:600;
                        text-transform:uppercase; letter-spacing:0.05em">
              Volatility
            </div>
            <div style="font-size:1.8rem; font-weight:800; color:{vc['color']}; line-height:1.2;
                        margin-top:4px">
              {vc['icon']} {vol_disp}
            </div>
            <div style="font-size:0.78rem; color:#64748b; margin-top:3px">variability level</div>
          </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

def kpi_card(title, value, subtitle="", color="#0f4c81"):
    import streamlit as st
    st.markdown(f"""
    <div class="kpi-card">
        <div style="font-size:12px; color:#64748b;">{title}</div>
        <div style="font-size:26px; font-weight:800; color:{color};">
            {value}
        </div>
        <div style="font-size:11px; color:#94a3b8;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)

with c1: kpi_card("Total Records", "24,079")
with c2: kpi_card("Data Quality", "98/100")
with c3: kpi_card("Demand Trend", "Increasing")
with c4: kpi_card("Operational Risk", "Low")
with c5: kpi_card("Stability", "Moderate")
# ---------------------------------------------------------------------------
# Trend insight badge (NEW)
# ---------------------------------------------------------------------------


def trend_insight_badge(
    trend_direction: str,
    pct_change: float,
    value_col: str,
) -> None:
    """
    Prominent inline trend signal shown above the trend chart.
    Example: '📈  Trend: UPWARD  (+4.3% vs prior period)'
    """
    col_display = value_col.replace("_", " ").title()
    configs = {
        "upward":   dict(icon="📈", color="#059669", bg="#f0fdf4",
                         border="#a7f3d0", label="UPWARD"),
        "downward": dict(icon="📉", color="#dc2626", bg="#fef2f2",
                         border="#fca5a5", label="DOWNWARD"),
        "flat":     dict(icon="➡️", color="#0284c7", bg="#f0f9ff",
                         border="#bae6fd", label="STABLE"),
    }
    cfg = configs.get(trend_direction, configs["flat"])
    pct_str = f"{pct_change:+.1f}%" if trend_direction != "flat" else f"{abs(pct_change):.1f}% change"

    st.markdown(
        f"""
        <div style="
            background:{cfg['bg']};
            border:1.5px solid {cfg['border']};
            border-radius:8px;
            padding:12px 18px;
            margin-bottom:12px;
            display:flex;
            align-items:center;
            gap:12px;
        ">
          <span style="font-size:1.5rem">{cfg['icon']}</span>
          <span>
            <span style="font-weight:700; font-size:1rem; color:{cfg['color']}">
              Trend: {cfg['label']}
            </span>
            <span style="color:#64748b; font-size:0.9rem; margin-left:10px">
              {pct_str} vs prior period &nbsp;·&nbsp; {col_display}
            </span>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Forecast insight badge (NEW)
# ---------------------------------------------------------------------------


def forecast_insight_badge(insight_text: str) -> None:
    """
    Renders a clean forecast interpretation line above the forecast chart.
    """
    st.markdown(
        f"""
        <div style="
            background:#f0fdf4;
            border:1.5px solid #a7f3d0;
            border-radius:8px;
            padding:12px 18px;
            margin-bottom:12px;
            font-size:0.93rem;
            color:#064e3b;
        ">
          🔮 &nbsp; {insight_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Metric rendering
# ---------------------------------------------------------------------------


def metric_row(metrics: List[Dict[str, Any]]) -> None:
    """Render a horizontal row of st.metric cards."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m.get("label", ""),
                value=m.get("value", "—"),
                delta=m.get("delta"),
                help=m.get("help"),
            )


# ---------------------------------------------------------------------------
# Data preview
# ---------------------------------------------------------------------------


def dataframe_preview(
    df: pd.DataFrame,
    max_rows: int = 8,
    label: str = "Data Preview",
) -> None:
    """Display a labelled, row-limited DataFrame preview."""
    section_header(label, "🗂️")
    st.caption(
        f"Showing first {min(max_rows, len(df))} of {len(df):,} rows "
        f"— {len(df.columns)} columns total"
    )
    st.dataframe(df.head(max_rows), use_container_width=True)


# ---------------------------------------------------------------------------
# Quality panel
# ---------------------------------------------------------------------------


def quality_panel(
    total_rows: int,
    total_cols: int,
    missing_pct: float,
    duplicate_count: int,
    column_types: Dict[str, str],
    missing_per_col: Dict[str, float],
) -> None:
    """Render the Data Quality section."""
    section_header("Data Quality", "🔍")

    metric_row([
        {"label": "Total Rows",
         "value": f"{total_rows:,}"},
        {"label": "Total Columns",
         "value": str(total_cols)},
        {"label": "Missing Values",
         "value": f"{missing_pct:.1%}",
         "help": "Percentage of cells that contain no value across the entire dataset"},
        {"label": "Duplicate Rows",
         "value": f"{duplicate_count:,}",
         "help": "Rows that are exact copies of another row"},
    ])

    st.markdown("")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Column Types Detected**")
        type_counts: Dict[str, int] = {}
        for t in column_types.values():
            type_counts[t] = type_counts.get(t, 0) + 1
        type_icons = {
            "numeric":     "🔢",
            "datetime":    "📅",
            "boolean":     "☑️",
            "categorical": "🏷️",
            "text":        "📝",
        }
        for t, cnt in sorted(type_counts.items()):
            icon = type_icons.get(t, "•")
            st.markdown(f"{icon} **{t.title()}** — {cnt} column{'s' if cnt > 1 else ''}")

    with col_right:
        cols_with_nulls = {
            col: ratio
            for col, ratio in missing_per_col.items()
            if ratio > 0.0
        }
        if cols_with_nulls:
            st.markdown("**Missing Values by Column**")
            null_df = (
                pd.DataFrame.from_dict(cols_with_nulls, orient="index", columns=["null_ratio"])
                .sort_values("null_ratio", ascending=False)
                .rename(columns={"null_ratio": "Missing %"})
            )
            null_df["Missing %"] = (null_df["Missing %"] * 100).round(1).astype(str) + "%"
            st.dataframe(null_df, use_container_width=True)
        else:
            success_box("No missing values detected. Your data is complete.")


# ---------------------------------------------------------------------------
# Sidebar column selectors
# ---------------------------------------------------------------------------


def datetime_column_selector(
    detected: Optional[str],
    all_columns: List[str],
    key: str = "dt_col",
) -> Optional[str]:
    """Sidebar selectbox for the datetime column. Defaults to auto-detected value."""
    options = ["— None —"] + all_columns
    default_idx = 0
    if detected and detected in all_columns:
        default_idx = all_columns.index(detected) + 1

    chosen = st.sidebar.selectbox(
        "📅 Date / Time Column",
        options=options,
        index=default_idx,
        key=key,
        help="Auto-detected. Override if the wrong column is selected.",
    )
    return None if chosen == "— None —" else chosen


def numeric_column_selector(
    detected_numeric: List[str],
    all_columns: List[str],
    key: str = "num_col",
) -> Optional[str]:
    """
    Sidebar selectbox for the primary metric column.
    Defaults to the FIRST detected numeric column (not None).
    """
    options = ["— None —"] + all_columns
    default_idx = 0

    # Default to first numeric column if available — never leave blank
    if detected_numeric:
        first = detected_numeric[0]
        if first in all_columns:
            default_idx = all_columns.index(first) + 1

    chosen = st.sidebar.selectbox(
        "📊 Target Metric Column",
        options=options,
        index=default_idx,
        key=key,
        help="The numeric column to analyse and forecast.",
    )
    return None if chosen == "— None —" else chosen


def rolling_window_selector(key: str = "rolling_window") -> int:
    """Sidebar slider for rolling window size."""
    return st.sidebar.slider(
        "📉 Rolling Window (periods)",
        min_value=3,
        max_value=30,
        value=7,
        step=1,
        key=key,
        help="Number of periods for the rolling average and volatility calculation.",
    )


def forecast_controls(key_prefix: str = "fc") -> tuple[int, str]:
    """Sidebar forecast horizon + method selectors. Returns (horizon, method)."""
    horizon = st.sidebar.select_slider(
        "🔮 Forecast Horizon (periods)",
        options=[7, 14, 21, 30, 60, 90],
        value=14,
        key=f"{key_prefix}_horizon",
        help="How many future periods to forecast.",
    )
    method = st.sidebar.selectbox(
        "📐 Forecast Method",
        options=["moving_average", "naive", "exponential_smoothing"],
        index=0,
        key=f"{key_prefix}_method",
        format_func=lambda x: {
            "moving_average":        "Moving Average",
            "naive":                 "Naïve (Last Value)",
            "exponential_smoothing": "Exponential Smoothing",
        }.get(x, x),
        help="Moving Average is recommended for most datasets.",
    )
    return horizon, method


# ---------------------------------------------------------------------------
# Executive summary card
# ---------------------------------------------------------------------------


def executive_summary_card(summary: Dict[str, Any]) -> None:
    """Render the full executive summary panel."""
    section_header("Executive Summary", "📋")

    # Hero verdict bar
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-left: 5px solid #0284c7;
            border-radius: 8px;
            padding: 16px 22px;
            margin-bottom: 20px;
            font-size: 0.97rem;
            line-height: 2;
        ">
          {summary.get('overall_verdict', '')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Four narrative paragraphs
    section_map = [
        ("quality_sentence",    "📊 Data Quality"),
        ("trend_sentence",      "📈 Recent Trend"),
        ("volatility_sentence", "〰️ Variability"),
        ("forecast_sentence",   "🔮 Outlook"),
    ]

    col_a, col_b = st.columns(2)
    halves = [section_map[:2], section_map[2:]]
    for col, pairs in zip([col_a, col_b], halves):
        with col:
            for key, label in pairs:
                sentence = summary.get(key, "")
                if sentence:
                    st.markdown(
                        f"""
                        <div style="
                            background:#f8fafc;
                            border:1px solid #e2e8f0;
                            border-radius:8px;
                            padding:14px 16px;
                            margin-bottom:12px;
                        ">
                          <div style="font-weight:700; font-size:0.85rem; color:#0284c7;
                                      margin-bottom:6px">{label}</div>
                          <div style="font-size:0.9rem; color:#334155; line-height:1.6">
                            {sentence}
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
def section(title):
    import streamlit as st
    st.markdown(f'<div class="section">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def end_section():
    import streamlit as st
    st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st

def hero_banner(title: str, subtitle: str, badge: str | None = None) -> None:
    badge_html = f"""
    <div style="
        display:inline-block;
        padding:6px 10px;
        border-radius:999px;
        background:rgba(59,130,246,0.14);
        color:#93c5fd;
        font-size:0.78rem;
        font-weight:700;
        margin-bottom:10px;
    ">
        {badge}
    </div>
    """ if badge else ""

    st.markdown(f"""
    <div style="
        padding: 22px 24px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.90));
        border: 1px solid rgba(148,163,184,0.16);
        box-shadow: 0 12px 30px rgba(2,6,23,0.25);
        margin-bottom: 18px;
    ">
        {badge_html}
        <div style="font-size:2rem; font-weight:800; color:#f8fafc; line-height:1.1;">
            {title}
        </div>
        <div style="margin-top:8px; color:#94a3b8; font-size:0.98rem; line-height:1.6;">
            {subtitle}
        </div>
    </div>
    """, unsafe_allow_html=True)


def insight_card(title: str, body: str, icon: str = "📌") -> None:
    st.markdown(f"""
    <div style="
        padding:18px;
        border-radius:18px;
        background: linear-gradient(180deg, rgba(15,23,42,0.92), rgba(15,23,42,0.82));
        border: 1px solid rgba(148,163,184,0.14);
        margin: 8px 0 12px 0;
    ">
        <div style="font-size:1rem; font-weight:700; color:#e2e8f0; margin-bottom:8px;">
            {icon} {title}
        </div>
        <div style="color:#94a3b8; line-height:1.7; font-size:0.92rem;">
            {body}
        </div>
    </div>
    """, unsafe_allow_html=True)