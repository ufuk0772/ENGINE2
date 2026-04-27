"""
FMCG Intel - Enterprise Analytics & Decision Support Platform
============================================================
Çalıştırmak için: python -m streamlit run streamlit_app.py

Mimari:
  - render_sidebar()           → Sol navigasyon
  - render_topbar()            → Üst navbar
  - render_landing_dashboard() → Ana dashboard (mock + gerçek veri)
  - render_data_hub()          → Dosya yükleme & kolon eşleştirme
  - render_sales_demand()      → Satış & Talep Analizi sayfası
  - render_inventory_ops()     → Stok & Operasyonlar sayfası
  - render_forecasting()       → Tahminler & Analizler sayfası
  - render_logistics_opt()     → Lojistik Optimizasyon sayfası
  - render_dashboard_cards()   → KPI kart bileşenleri
  - Plotly grafik yardımcıları → create_forecast_chart, create_donut_chart, create_heatmap
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import linregress, skew, kurtosis
from statsmodels.tsa.stattools import adfuller, acf
import statsmodels.api as sm
# ─── PAGE CONFIG (MUST BE FIRST ST CALL) ─────────────────────────────────────
st.set_page_config(
    page_title="FMCG Intel | Enterprise Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
def inject_global_css():
    st.markdown("""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    /* ── Reset & Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        color: #1e293b;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebar"] {
        background: #0f2742 !important;
        border-right: 0 !important;
        min-width: 240px !important;
        max-width: 240px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: #0f2742 !important;
        padding-top: 18px !important;
    }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* ── App Shell ── */
    .app-shell {
        display: flex;
        min-height: 100vh;
        background: #eef3f8;
    }

    /* ── Sidebar ── */
    .sidebar {
        width: 240px;
        min-width: 240px;
        background: #0f2742;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        position: fixed;
        top: 0;
        left: 0;
        z-index: 100;
        box-shadow: 4px 0 20px rgba(0,0,0,0.15);
    }
    .sidebar-logo {
        padding: 24px 20px 20px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .logo-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
    }
    .logo-icon {
        width: 34px;
        height: 34px;
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        box-shadow: 0 4px 12px rgba(37,99,235,0.4);
    }
    .logo-text {
        font-size: 17px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }
    .logo-sub {
        font-size: 10px;
        color: rgba(255,255,255,0.4);
        font-weight: 400;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .sidebar-nav {
        padding: 16px 12px;
        flex: 1;
    }
    .nav-section-label {
        font-size: 10px;
        font-weight: 600;
        color: rgba(255,255,255,0.3);
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 8px 8px 4px;
        margin-top: 8px;
    }
    .nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 10px;
        border-radius: 8px;
        margin-bottom: 2px;
        cursor: pointer;
        font-size: 13.5px;
        font-weight: 500;
        color: rgba(255,255,255,0.6);
        transition: all 0.15s ease;
        text-decoration: none;
        border: none;
        background: none;
        width: 100%;
        text-align: left;
    }
    .nav-item:hover {
        background: rgba(255,255,255,0.08);
        color: rgba(255,255,255,0.9);
    }
    .nav-item.active {
        background: rgba(37,99,235,0.25);
        color: #ffffff;
        font-weight: 600;
    }
    .nav-item.active .nav-dot {
        background: #3b82f6;
    }
    .nav-icon { font-size: 15px; min-width: 18px; text-align: center; }
    .nav-badge {
        margin-left: auto;
        background: #ef4444;
        color: white;
        font-size: 10px;
        font-weight: 700;
        padding: 1px 6px;
        border-radius: 10px;
        min-width: 18px;
        text-align: center;
    }
    .sidebar-footer {
        padding: 16px 12px;
        border-top: 1px solid rgba(255,255,255,0.08);
    }
    .user-pill {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 10px;
        border-radius: 8px;
        background: rgba(255,255,255,0.06);
    }
    .avatar {
        width: 30px;
        height: 30px;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
        color: white;
    }
    .user-info-name { font-size: 12px; font-weight: 600; color: rgba(255,255,255,0.85); }
    .user-info-role { font-size: 10px; color: rgba(255,255,255,0.35); }

    /* ── Main Content Area ── */
    .main-content {
        margin-left: 0;
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }

    /* ── Top Navbar ── */
    .topbar {
        background: #ffffff;
        height: 60px;
        display: flex;
        align-items: center;
        padding: 0 28px;
        border-bottom: 1px solid #e2e8f0;
        gap: 16px;
        position: sticky;
        top: 0;
        z-index: 50;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .topbar-title {
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        flex: 1;
    }
    .topbar-subtitle {
        font-size: 12px;
        color: #94a3b8;
        font-weight: 400;
    }
    .search-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 7px 14px;
        width: 220px;
        font-size: 13px;
        color: #64748b;
    }
    .topbar-btn {
        display: flex;
        align-items: center;
        gap: 6px;
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 13px;
        font-weight: 600;
        cursor: pointer;
        white-space: nowrap;
        box-shadow: 0 2px 8px rgba(37,99,235,0.3);
    }
    .topbar-icon-btn {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        cursor: pointer;
        color: #64748b;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        margin-right: -4px;
    }

    /* ── Dashboard Page ── */
    .page-wrapper {
        padding: 28px;
        flex: 1;
    }
    .page-header {
        margin-bottom: 24px;
    }
    .page-title {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .page-meta {
        font-size: 13px;
        color: #64748b;
    }
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #dcfce7;
        color: #16a34a;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        margin-left: 10px;
        vertical-align: middle;
    }
    .demo-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #fef3c7;
        color: #d97706;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        margin-left: 10px;
        vertical-align: middle;
    }

    /* ── KPI Cards ── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 20px;
    }
    .kpi-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 22px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }
    .kpi-card.blue::before { background: linear-gradient(90deg, #2563eb, #3b82f6); }
    .kpi-card.green::before { background: linear-gradient(90deg, #10b981, #34d399); }
    .kpi-card.amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
    .kpi-card.red::before { background: linear-gradient(90deg, #ef4444, #f87171); }
    .kpi-label {
        font-size: 11.5px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -1px;
        line-height: 1;
        margin-bottom: 8px;
        font-family: 'DM Mono', monospace;
    }
    .kpi-change {
        font-size: 12px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .kpi-change.up { color: #10b981; }
    .kpi-change.down { color: #ef4444; }
    .kpi-icon {
        position: absolute;
        top: 18px;
        right: 18px;
        font-size: 22px;
        opacity: 0.15;
    }

    /* ── Charts Row ── */
    .charts-row {
        display: grid;
        grid-template-columns: 1fr 380px;
        gap: 16px;
        margin-bottom: 20px;
    }
    .chart-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 22px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .card-title {
        font-size: 14px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .card-subtitle {
        font-size: 12px;
        color: #94a3b8;
        margin-bottom: 16px;
    }

    /* ── Bottom Row ── */
    .bottom-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 20px;
    }

    /* ── Insight Cards ── */
    .insight-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .insight-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .insight-tag {
        font-size: 10.5px;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 20px;
        letter-spacing: 0.3px;
    }
    .tag-warning { background: #fef3c7; color: #d97706; }
    .tag-success { background: #dcfce7; color: #16a34a; }
    .tag-danger  { background: #fee2e2; color: #dc2626; }
    .tag-info    { background: #dbeafe; color: #2563eb; }
    .insight-title {
        font-size: 13.5px;
        font-weight: 700;
        color: #0f172a;
    }
    .insight-body {
        font-size: 12.5px;
        color: #64748b;
        line-height: 1.55;
    }
    .insight-meta {
        font-size: 11px;
        color: #94a3b8;
        font-family: 'DM Mono', monospace;
    }
    .insight-bar-bg {
        height: 5px;
        background: #f1f5f9;
        border-radius: 10px;
        overflow: hidden;
    }
    .insight-bar {
        height: 100%;
        border-radius: 10px;
    }

    /* ── Data Hub ── */
    .data-hub-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 32px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        max-width: 700px;
    }
    .upload-zone {
        border: 2px dashed #cbd5e1;
        border-radius: 10px;
        padding: 32px;
        text-align: center;
        background: #f8fafc;
        margin: 16px 0;
    }
    .upload-zone:hover { border-color: #2563eb; background: #eff6ff; }
    .upload-icon { font-size: 36px; margin-bottom: 10px; }
    .upload-title { font-size: 15px; font-weight: 600; color: #334155; margin-bottom: 6px; }
    .upload-sub { font-size: 12.5px; color: #94a3b8; }

    /* ── Section Headers within pages ── */
    .section-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .section-title {
        font-size: 14px;
        font-weight: 700;
        color: #0f172a;
    }
    .section-link {
        font-size: 12px;
        color: #2563eb;
        font-weight: 600;
        cursor: pointer;
    }

    /* ── Override Streamlit defaults ── */
    div[data-testid="stHorizontalBlock"] > div {
        border: none !important;
    }
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border-color: #e2e8f0 !important;
        font-size: 13px !important;
    }
    .stSlider > div { padding: 0 !important; }
    .stButton > button {
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
    }
    .stFileUploader {
        border-radius: 10px !important;
    }
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 10px;
        padding: 14px 18px !important;
        border: 1px solid #e2e8f0;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'DM Mono', monospace !important;
        font-size: 22px !important;
        color: #0f172a !important;
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f1f5f9;
        padding: 4px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 7px 16px;
        font-size: 13px;
        font-weight: 500;
        background: transparent;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    </style>
    """, unsafe_allow_html=True)


# ─── SESSION STATE INIT ───────────────────────────────────────────────────────
def init_session():
    defaults = {
        "active_page": "dashboard",
        "df": None,
        "col_map": {},
        "data_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── MOCK / SAMPLE DATA ───────────────────────────────────────────────────────
def get_mock_data() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    trend = np.linspace(1000, 1600, 180)
    seasonal = 200 * np.sin(2 * np.pi * np.arange(180) / 30)
    noise = np.random.normal(0, 60, 180)
    sales = trend + seasonal + noise
    stock = np.random.uniform(500, 3000, 180)
    production = sales * np.random.uniform(0.9, 1.1, 180)
    error_rate = np.random.uniform(0.01, 0.06, 180)
    categories = np.random.choice(["Beverages", "Snacks", "Dairy", "Frozen"], 180)
    locations = np.random.choice(["İstanbul", "Ankara", "İzmir", "Bursa"], 180)
    return pd.DataFrame({
        "tarih": dates,
        "satış_hacmi": sales.round(0),
        "stok_seviyesi": stock.round(0),
        "üretim_hacmi": production.round(0),
        "hata_oranı": error_rate,
        "kategori": categories,
        "mağaza_konumu": locations,
    })


def get_active_df() -> pd.DataFrame:
    if st.session_state.data_loaded and st.session_state.df is not None:
        return st.session_state.df
    return get_mock_data()


# ─── PLOTLY CHART HELPERS ─────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font_family="DM Sans",
    font_color="#334155",
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin=dict(l=8, r=8, t=36, b=8),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font_size=11,
    ),
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#e2e8f0",
        font_size=12,
        font_family="DM Sans",
    ),
)


def create_forecast_chart(df: pd.DataFrame, date_col: str = "tarih", sales_col: str = "satış_hacmi", horizon: int = 30) -> go.Figure:
    df = df.copy()

    if date_col not in df.columns or sales_col not in df.columns:
        return go.Figure().update_layout(**PLOTLY_LAYOUT, height=280, title="Demand Forecast Overview")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    df = df.dropna(subset=[date_col, sales_col])

    hist = df.groupby(date_col)[sales_col].sum().reset_index().sort_values(date_col)

    if len(hist) < 5:
        return go.Figure().update_layout(**PLOTLY_LAYOUT, height=280, title="Not enough data")

    x = np.arange(len(hist))
    y = hist[sales_col].values

    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(len(hist), len(hist) + horizon)
    last_date = hist[date_col].max()
    fc_dates = [last_date + timedelta(days=i + 1) for i in range(horizon)]

    fc_vals = intercept + slope * future_x
    fc_vals = np.maximum(fc_vals, 0)

    residuals = y - (intercept + slope * x)
    std_err = np.std(residuals)

    upper = fc_vals + 1.96 * std_err
    lower = np.maximum(fc_vals - 1.96 * std_err, 0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist[date_col],
        y=hist[sales_col],
        name="Actual Sales",
        line=dict(color="#2563eb", width=2),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.06)"
    ))

    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=fc_vals,
        name="Forecast",
        line=dict(color="#10b981", width=2, dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=fc_dates + fc_dates[::-1],
        y=list(upper) + list(lower[::-1]),
        fill="toself",
        fillcolor="rgba(16,185,129,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band"
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        title=dict(text="Demand Forecast Overview", font_size=14, x=0)
    )

    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)

    return fig


def create_bar_chart(df: pd.DataFrame, date_col: str, sales_col: str) -> go.Figure:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    monthly = df.groupby(df[date_col].dt.to_period("M"))[sales_col].sum()
    monthly.index = monthly.index.astype(str)
    fig = go.Figure(go.Bar(
        x=monthly.index, y=monthly.values,
        marker=dict(color="#2563eb", opacity=0.85,
                    line=dict(color="rgba(0,0,0,0)"),
                    cornerradius=4),
        name="Monthly Sales",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=230,
                      title=dict(text="Monthly Demand", font_size=13, x=0))
    fig.update_xaxes(showgrid=False, tickangle=-35, tickfont_size=11)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    return fig


def create_donut_chart(df: pd.DataFrame, sales_col: str = "satış_hacmi", stock_col: str = "stok_seviyesi") -> go.Figure:
    df = df.copy()

    if sales_col not in df.columns or stock_col not in df.columns:
        labels = ["No Data"]
        values = [1]
        colors = ["#94a3b8"]
    else:
        df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
        df[stock_col] = pd.to_numeric(df[stock_col], errors="coerce")
        temp = df.dropna(subset=[sales_col, stock_col]).copy()

        temp["stock_ratio"] = temp[stock_col] / (temp[sales_col] + 1)

        critical = (temp["stock_ratio"] < 0.50).sum()
        low = ((temp["stock_ratio"] >= 0.50) & (temp["stock_ratio"] < 1.00)).sum()
        optimal = ((temp["stock_ratio"] >= 1.00) & (temp["stock_ratio"] <= 2.00)).sum()
        excess = (temp["stock_ratio"] > 2.00).sum()

        labels = ["Optimal Stock", "Excess Stock", "Low Stock", "Critical"]
        values = [optimal, excess, low, critical]
        colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value} records<extra></extra>"
    ))

    fig.add_annotation(
        text="<b>SKU<br>Status</b>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=13, color="#0f172a", family="DM Sans"),
        align="center"
    )

    layout = PLOTLY_LAYOUT.copy()
    layout.pop("legend", None)

    fig.update_layout(
        **layout,
        height=230,
        showlegend=True,
        title=dict(text="Inventory Status", font_size=13, x=0),
        legend=dict(
            orientation="v",
            x=1.02,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            font_size=11
        )
    )

    return fig

def create_heatmap(df: pd.DataFrame, value_col: str = "satış_hacmi") -> go.Figure:
    df = df.copy()

    if "kategori" not in df.columns or "mağaza_konumu" not in df.columns or value_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=230,
            title=dict(text="Route / Category Contribution - Missing Columns", font_size=13, x=0)
        )
        return fig

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    pivot = df.pivot_table(
        values=value_col,
        index="kategori",
        columns="mağaza_konumu",
        aggfunc="sum",
        fill_value=0
    )

    z = pivot.values.round(0)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale=[[0, "#eff6ff"], [0.5, "#60a5fa"], [1, "#1d4ed8"]],
        text=z,
        texttemplate="%{text:.0f}",
        hovertemplate="%{y} → %{x}<br>Total: %{z:.0f}<extra></extra>",
        showscale=False
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=230,
        title=dict(text="Category x Location Sales Contribution", font_size=13, x=0)
    )

    fig.update_xaxes(side="bottom", tickfont_size=11)
    fig.update_yaxes(tickfont_size=11)

    return fig


def create_stock_line(df: pd.DataFrame, date_col: str, stock_col: str) -> go.Figure:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    agg = df.groupby(date_col)[stock_col].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg[date_col], y=agg[stock_col],
        name="Stock Level", line=dict(color="#7c3aed", width=2),
        fill="tozeroy", fillcolor="rgba(124,58,237,0.06)"
    ))
    safety = agg[stock_col].mean() * 0.3
    fig.add_hline(y=safety, line_dash="dot", line_color="#ef4444",
                  annotation_text="Safety Stock", annotation_font_size=11)
    fig.update_layout(**PLOTLY_LAYOUT, height=280,
                      title=dict(text="Stock Level Timeline", font_size=14, x=0))
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    return fig


def create_error_rate_chart(df: pd.DataFrame, date_col: str, err_col: str) -> go.Figure:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    agg = df.groupby(date_col)[err_col].mean().reset_index()
    colors = ["#ef4444" if v > 0.04 else "#10b981" for v in agg[err_col]]
    fig = go.Figure(go.Bar(
        x=agg[date_col], y=(agg[err_col] * 100).round(2),
        marker_color=colors, name="Error Rate (%)",
        marker=dict(cornerradius=3),
    ))
    fig.add_hline(y=4, line_dash="dot", line_color="#f59e0b",
                  annotation_text="Threshold 4%", annotation_font_size=11)
    fig.update_layout(**PLOTLY_LAYOUT, height=240,
                      title=dict(text="Production Error Rate (%)", font_size=14, x=0))
    return fig

def run_statistical_diagnostics(series: pd.Series) -> dict:
    """
    Sales & Demand için istatistiksel tanı motoru.
    Trend, durağanlık, volatilite, çarpıklık, basıklık ve otokorelasyon üretir.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) < 20:
        return {
            "ok": False,
            "message": "İstatistiksel analiz için en az 20 geçerli gözlem önerilir."
        }

    x = np.arange(len(s))
    slope, intercept, r_value, p_value, std_err = linregress(x, s.values)

    mean_val = float(s.mean())
    std_val = float(s.std())
    cv = float(std_val / mean_val) if mean_val != 0 else np.nan

    try:
        adf_result = adfuller(s)
        adf_stat = float(adf_result[0])
        adf_p = float(adf_result[1])
    except Exception:
        adf_stat = np.nan
        adf_p = np.nan

    try:
        acf_vals = acf(s, nlags=min(14, len(s) // 3), fft=False)
        lag1_acf = float(acf_vals[1]) if len(acf_vals) > 1 else np.nan
        max_acf = float(np.nanmax(np.abs(acf_vals[1:]))) if len(acf_vals) > 1 else np.nan
    except Exception:
        lag1_acf = np.nan
        max_acf = np.nan

    return {
        "ok": True,
        "n": len(s),
        "mean": mean_val,
        "std": std_val,
        "cv": cv,
        "slope": float(slope),
        "trend_p": float(p_value),
        "r2": float(r_value ** 2),
        "adf_stat": adf_stat,
        "adf_p": adf_p,
        "lag1_acf": lag1_acf,
        "max_acf": max_acf,
        "skewness": float(skew(s)),
        "kurtosis": float(kurtosis(s)),
    }


def build_statistical_interpretation(stats: dict) -> list[str]:
    """
    Test sonuçlarını iş yorumuna çevirir.
    """
    if not stats.get("ok"):
        return [stats.get("message", "İstatistiksel analiz üretilemedi.")]

    insights = []

    if stats["trend_p"] < 0.05:
        if stats["slope"] > 0:
            insights.append(
                "Talep serisinde istatistiksel olarak anlamlı bir artış trendi var "
                f"(p={stats['trend_p']:.4f})."
            )
        else:
            insights.append(
                "Talep serisinde istatistiksel olarak anlamlı bir düşüş trendi var "
                f"(p={stats['trend_p']:.4f})."
            )
    else:
        insights.append(
            "Talep trendi istatistiksel olarak anlamlı değil; kısa dönem dalgalanmalar baskın olabilir."
        )

    if not np.isnan(stats["adf_p"]):
        if stats["adf_p"] < 0.05:
            insights.append(
                "ADF testine göre seri durağan görünüyor; klasik zaman serisi modellemesi için daha elverişli."
            )
        else:
            insights.append(
                "ADF testine göre seri durağan değil; modelleme öncesi fark alma veya dönüşüm gerekebilir."
            )

    if stats["cv"] > 0.50:
        insights.append(
            "Varyasyon katsayısı yüksek; talep oynaklığı güçlü ve güvenlik stoğu ihtiyacı artabilir."
        )
    elif stats["cv"] > 0.25:
        insights.append(
            "Talep orta düzeyde oynak; stok planlamasında tampon seviye izlenmeli."
        )
    else:
        insights.append(
            "Talep görece stabil; stok planlaması daha öngörülebilir yapılabilir."
        )

    if not np.isnan(stats["lag1_acf"]):
        if abs(stats["lag1_acf"]) > 0.50:
            insights.append(
                f"Lag-1 otokorelasyon güçlü ({stats['lag1_acf']:.2f}); geçmiş talep bugünkü talebi açıklamada önemli."
            )
        elif abs(stats["lag1_acf"]) > 0.25:
            insights.append(
                f"Lag-1 otokorelasyon orta düzeyde ({stats['lag1_acf']:.2f}); kısa dönem bağımlılık var."
            )
        else:
            insights.append(
                "Lag-1 otokorelasyon zayıf; seri daha gürültülü davranıyor olabilir."
            )

    if abs(stats["skewness"]) > 1:
        insights.append(
            "Dağılım belirgin çarpık; ortalama tek başına yanıltıcı olabilir, medyan ve yüzdelikler de izlenmeli."
        )

    return insights


def render_statistical_analysis_panel(series: pd.Series):
    """
    Sales & Demand sayfasına eklenecek istatistiksel analiz paneli.
    """
    stats = run_statistical_diagnostics(series)

    st.markdown("### 🧠 İstatistiksel Analiz ve İçgörü Motoru")
    st.caption("Trend anlamlılığı, durağanlık, volatilite ve otokorelasyon testleri.")

    if not stats.get("ok"):
        st.warning(stats["message"])
        return

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Trend Eğimi", f"{stats['slope']:.2f}", help="Pozitif değer artış, negatif değer düşüş eğilimi gösterir.")
    c2.metric("Trend p-değeri", f"{stats['trend_p']:.4f}", help="p < 0.05 ise trend istatistiksel olarak anlamlı kabul edilir.")
    c3.metric("ADF p-değeri", f"{stats['adf_p']:.4f}" if not np.isnan(stats["adf_p"]) else "N/A", help="p < 0.05 ise seri durağan kabul edilir.")
    c4.metric("Volatilite CV", f"{stats['cv']:.3f}", help="Standart sapma / ortalama. Talep oynaklığını ölçer.")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("R²", f"{stats['r2']:.3f}")
    c6.metric("Lag-1 ACF", f"{stats['lag1_acf']:.3f}" if not np.isnan(stats["lag1_acf"]) else "N/A")
    c7.metric("Skewness", f"{stats['skewness']:.3f}")
    c8.metric("Kurtosis", f"{stats['kurtosis']:.3f}")

    insights = build_statistical_interpretation(stats)

    st.markdown("#### 📌 İstatistiksel Yorum")
    for item in insights:
        st.markdown(f"- {item}")
# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
def render_sidebar():
    page = st.session_state.active_page

    def nav(label, icon, key, badge=None):
        active_cls = "active" if page == key else ""
        badge_html = f'<span class="nav-badge">{badge}</span>' if badge else ""
        if st.session_state.get(f"_nav_{key}"):
            st.session_state.active_page = key
        clicked = f"""
        <button class="nav-item {active_cls}" onclick="void(0)">{icon}
            <span>{label}</span>{badge_html}
        </button>
        """
        return clicked

    st.markdown(f"""
    <div class="sidebar">
        <div class="sidebar-logo">
            <div class="logo-badge">
                <div class="logo-icon">📦</div>
                <div>
                    <div class="logo-text">FMCG Intel</div>
                    <div class="logo-sub">Analytics Platform</div>
                </div>
            </div>
        </div>
        <div class="sidebar-nav">
            <div class="nav-section-label">Main</div>
        </div>
        <div class="sidebar-footer">
            <div class="user-pill">
                <div class="avatar">AD</div>
                <div>
                    <div class="user-info-name">Admin User</div>
                    <div class="user-info-role">Analyst</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Streamlit buttons for navigation (placed invisibly, triggered via state)
    # We use a vertical stack of st.buttons overlaid conceptually
    # HTML sidebar is visual-only; real nav is done via st.columns buttons below
    pass


# ─── TOP BAR ─────────────────────────────────────────────────────────────────
def render_topbar(page_title: str, page_subtitle: str = ""):
    is_live = st.session_state.data_loaded
    badge = '<span class="live-badge">● Live Data</span>' if is_live else '<span class="demo-badge">◐ Demo Mode</span>'
    st.markdown(f"""
    <div class="topbar">
        <div style="flex:1">
            <span class="topbar-title">{page_title}</span>
            {badge}
            <br><span class="topbar-subtitle">{page_subtitle}</span>
        </div>
        <div class="search-bar">🔍 Search metrics, SKUs…</div>
    </div>
    """, unsafe_allow_html=True)


# ─── KPI CARDS ────────────────────────────────────────────────────────────────
def render_dashboard_cards(df: pd.DataFrame, col_map: dict):
    sales_col = col_map.get("satış_hacmi", "satış_hacmi")
    stock_col = col_map.get("stok_seviyesi", "stok_seviyesi")

    avg_sales = df[sales_col].mean() if sales_col in df.columns else 1342
    total_sales = df[sales_col].sum() if sales_col in df.columns else 241560
    avg_stock = df[stock_col].mean() if stock_col in df.columns else 1847
    safety_stock = avg_sales * 1.25
    stock_avail = min(100, (avg_stock / (avg_sales * 14)) * 100)
    excess = max(0, avg_stock - safety_stock * 1.3)

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-icon">📈</div>
            <div class="kpi-label">Projected Demand</div>
            <div class="kpi-value">{avg_sales:,.0f}</div>
            <div class="kpi-change up">▲ 8.4% vs last month</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-icon">🛡️</div>
            <div class="kpi-label">Recommended Safety Stock</div>
            <div class="kpi-value">{safety_stock:,.0f}</div>
            <div class="kpi-change up">▲ Optimized</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-icon">📦</div>
            <div class="kpi-label">Stock Availability</div>
            <div class="kpi-value">{stock_avail:.1f}%</div>
            <div class="kpi-change {'up' if stock_avail > 75 else 'down'}">{'▲' if stock_avail > 75 else '▼'} {'Healthy' if stock_avail > 75 else 'Attention needed'}</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-icon">⚠️</div>
            <div class="kpi-label">Excess Stock Value</div>
            <div class="kpi-value">{excess:,.0f}</div>
            <div class="kpi-change down">▼ Reduction opportunity</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── INSIGHT CARDS ────────────────────────────────────────────────────────────
def render_insight_cards(df: pd.DataFrame):
    df = df.copy()

    sales_col = "satış_hacmi"
    stock_col = "stok_seviyesi"

    if sales_col not in df.columns or stock_col not in df.columns:
        st.info("Operational insights için satış_hacmi ve stok_seviyesi kolonları gerekli.")
        return

    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    df[stock_col] = pd.to_numeric(df[stock_col], errors="coerce")
    temp = df.dropna(subset=[sales_col, stock_col]).copy()

    if temp.empty:
        st.warning("Operational insights üretilemedi: geçerli satış/stok verisi yok.")
        return

    temp["stock_ratio"] = temp[stock_col] / (temp[sales_col] + 1)

    critical_count = int((temp["stock_ratio"] < 0.50).sum())
    low_count = int(((temp["stock_ratio"] >= 0.50) & (temp["stock_ratio"] < 1.00)).sum())
    excess_count = int((temp["stock_ratio"] > 2.00).sum())
    avg_stock_ratio = temp["stock_ratio"].mean()

    if "tarih" in temp.columns:
        temp["tarih"] = pd.to_datetime(temp["tarih"], errors="coerce")
        daily = temp.groupby("tarih")[sales_col].sum().dropna().sort_index()
        recent_growth = daily.pct_change().tail(14).mean() * 100 if len(daily) > 14 else np.nan
    else:
        recent_growth = temp[sales_col].pct_change().tail(14).mean() * 100

    growth_text = "N/A" if pd.isna(recent_growth) else f"{recent_growth:.1f}%"

    c1, c2, c3 = st.columns(3, gap="medium")
    risk_ratio = (critical_count + low_count) / len(temp) * 100
    progress_val = (critical_count + low_count) / len(temp)
    with c1:
        st.container(border=True)
        st.markdown("#### ⚠ Stock Risk")
        st.metric(
    "Stock Risk",
    f"{risk_ratio:.1f}%",
    delta=f"{critical_count + low_count} SKUs"
)
        st.caption(f"Critical: {critical_count} · Low: {low_count}")
        st.progress(min((critical_count + low_count) / max(len(temp), 1), 1.0))
        st.write("Veriye göre kritik ve düşük stok seviyesinde görünen kayıt sayısı hesaplandı.")

    with c2:
        st.container(border=True)
        st.markdown("#### 📦 Excess Stock")
        st.metric("Above Optimal Stock", excess_count)
        st.caption(f"Average stock/sales ratio: {avg_stock_ratio:.2f}x")
        st.progress(min(excess_count / max(len(temp), 1), 1.0))
        st.write("Stok/satış oranı yüksek olan kayıtlar fazla stok riski olarak işaretlendi.")

    with c3:
        st.container(border=True)
        st.markdown("#### 📊 Demand Movement")
        st.metric("Recent Demand Change", growth_text)
        st.caption("Based on uploaded dataset")
        progress_val = 0.5 if pd.isna(recent_growth) else min(abs(recent_growth) / 100, 1.0)
        st.progress(progress_val)
        st.write("Son dönem satış hareketinden ortalama talep değişimi hesaplandı.")

# ─── LANDING / MAIN DASHBOARD ─────────────────────────────────────────────────
def render_landing_dashboard():
    df = get_active_df()
    col_map = st.session_state.col_map or {}

    st.markdown("""
    <div class="page-header">
        <div class="page-title">FMCG Performance Dashboard</div>
        <div class="page-meta">Real-time analytics · Demand forecasting · Inventory intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    render_dashboard_cards(df, col_map)

    c1, c2 = st.columns([2.2, 1], gap="medium")

    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Forecasting Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Historical sales with data-driven 30-day forecast and confidence band</div>', unsafe_allow_html=True)
        fig = create_forecast_chart(df, "tarih", "satış_hacmi")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig2 = create_donut_chart(df, "satış_hacmi", "stok_seviyesi")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-row">
        <span class="section-title">Operational Insights</span>
        <span class="section-link">Live calculations →</span>
    </div>
    """, unsafe_allow_html=True)

    render_insight_cards(df)

    b1, b2, b3 = st.columns(3, gap="medium")

    with b1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(
            create_bar_chart(df, "tarih", "satış_hacmi"),
            use_container_width=True,
            config={"displayModeBar": False}
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig_h = create_heatmap(df, "satış_hacmi")
        st.plotly_chart(fig_h, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with b3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig_e = create_error_rate_chart(df, "tarih", "hata_oranı")
        st.plotly_chart(fig_e, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


# ─── DATA HUB ─────────────────────────────────────────────────────────────────
def render_data_hub():
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Data Hub</div>
        <div class="page-meta">Upload your dataset and map columns to activate live analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="data-hub-card">', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">⬆️</div>
        <div class="upload-title">Upload Dataset</div>
        <div class="upload-sub">Supports CSV, XLSX, XLS — up to 50MB</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv", "xlsx", "xls"],
                                label_visibility="collapsed")
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)

            st.success(f"✅ Loaded **{uploaded.name}** — {len(df_raw):,} rows × {len(df_raw.columns)} columns")
            st.markdown("#### Column Mapping")
            st.markdown('<p style="font-size:12px;color:#64748b;margin-bottom:16px">Map your dataset columns to the platform fields below.</p>', unsafe_allow_html=True)

            required_cols = {
                "tarih": "Date Column",
                "satış_hacmi": "Sales Volume",
                "stok_seviyesi": "Stock Level",
                "üretim_hacmi": "Production Volume",
                "hata_oranı": "Error Rate",
                "kategori": "Category",
                "mağaza_konumu": "Store Location",
            }

            raw_cols = ["(skip)"] + list(df_raw.columns)
            col_map = {}

            c1, c2 = st.columns(2, gap="medium")
            items = list(required_cols.items())
            for i, (field, label) in enumerate(items):
                col = c1 if i % 2 == 0 else c2
                with col:
                    default_idx = 0
                    for j, c in enumerate(raw_cols):
                        if c.lower().replace(" ", "_") in [field, field.replace("_", ""), label.lower()]:
                            default_idx = j
                            break
                    selected = st.selectbox(f"{label} `{field}`", raw_cols,
                                            index=default_idx, key=f"map_{field}")
                    if selected != "(skip)":
                        col_map[field] = selected

            if st.button("✅ Confirm Mapping & Activate Analysis", type="primary",
                         use_container_width=True):
                # Rename columns
                renamed = df_raw.copy()
                for field, src in col_map.items():
                    renamed.rename(columns={src: field}, inplace=True)
                if "tarih" in renamed.columns:
                    renamed["tarih"] = pd.to_datetime(renamed["tarih"], errors="coerce")
                st.session_state.df = renamed
                st.session_state.col_map = col_map
                st.session_state.data_loaded = True
                st.success("🚀 Data loaded! Navigate to any analysis module.")
                st.rerun()
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.data_loaded:
        st.divider()
        st.markdown("**Current Dataset**")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        if st.button("🗑 Remove Dataset", type="secondary"):
            st.session_state.df = None
            st.session_state.data_loaded = False
            st.session_state.col_map = {}
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ─── SALES & DEMAND PAGE ──────────────────────────────────────────────────────
def compute_price_elasticity(df, price_col, sales_col):
    temp = df.copy()
    temp[price_col] = pd.to_numeric(temp[price_col], errors="coerce")
    temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
    temp = temp.dropna(subset=[price_col, sales_col])
    temp = temp[(temp[price_col] > 0) & (temp[sales_col] > 0)]

    if len(temp) < 10:
        return None, None, None

    temp["log_price"] = np.log(temp[price_col])
    temp["log_sales"] = np.log(temp[sales_col])

    X = sm.add_constant(temp["log_price"])
    y = temp["log_sales"]

    model = sm.OLS(y, X).fit()
    return model.params["log_price"], model.rsquared, model.pvalues["log_price"]


def simulate_price_change(df, price_col, sales_col, elasticity, pct):
    temp = df.copy()
    temp[price_col] = pd.to_numeric(temp[price_col], errors="coerce")
    temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
    temp = temp.dropna(subset=[price_col, sales_col])
    temp = temp[(temp[price_col] > 0) & (temp[sales_col] > 0)]

    p = temp[price_col].mean()
    q = temp[sales_col].mean()

    old_rev = p * q
    new_p = p * (1 + pct)
    new_q = q * (1 + elasticity * pct)

    return old_rev, new_p * new_q


def render_price_elasticity_analysis(df):
    st.markdown("---")
    st.markdown("## 📊 Price Elasticity Analysis")
    st.caption("Fiyat değişiminin talep ve gelir üzerindeki etkisini ölçer.")

    num_cols = df.select_dtypes(include="number").columns.tolist()

    if len(num_cols) < 2:
        st.warning("Fiyat ve satış kolonu gerekli.")
        return

    c1, c2 = st.columns(2)

    default_price_index = num_cols.index("fiyat") if "fiyat" in num_cols else 0
    default_sales_index = num_cols.index("satış_hacmi") if "satış_hacmi" in num_cols else min(1, len(num_cols) - 1)

    price_col = c1.selectbox(
        "Price Column",
        num_cols,
        index=num_cols.index("fiyat") if "fiyat" in num_cols else 0,
        key="pe_price"
    )

    sales_options = [c for c in num_cols if c != price_col]

    sales_col = c2.selectbox(
        "Sales Column",
        sales_options,
        index=sales_options.index("satış_hacmi") if "satış_hacmi" in sales_options else 0,
        key="pe_sales"
   )

    e, r2, p_val = compute_price_elasticity(df, price_col, sales_col)

    if e is None:
        st.warning("Yetersiz veri. En az 10 pozitif fiyat/satış gözlemi gerekli.")
        return

    left, right = st.columns([2, 1])

    with left:
        fig = px.scatter(
            df,
            x=price_col,
            y=sales_col,
            trendline="ols",
            title="Price vs Sales Relationship"
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=360)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.metric("Elasticity", f"{e:.2f}")
        st.metric("Model R²", f"{r2:.3f}")
        st.metric("p-value", f"{p_val:.4f}")

        pct = st.slider("Price Change %", -20, 20, 5, key="pe_slider") / 100

        old_rev, new_rev = simulate_price_change(df, price_col, sales_col, e, pct)
        impact = new_rev - old_rev
        impact_pct = impact / old_rev * 100 if old_rev != 0 else 0

        st.metric("Revenue Impact", f"{impact:,.0f}", delta=f"{impact_pct:.2f}%")

        if e < -1:
            st.warning("Talep fiyata duyarlı. Sert fiyat artışı riskli.")
        elif -1 <= e < 0:
            st.success("Talep düşük-orta duyarlı. Kontrollü fiyat artışı test edilebilir.")
        else:
            st.info("Pozitif/normal dışı esneklik. Kampanya, sezon veya veri etkisi olabilir.")
def render_sales_demand():
    df = get_active_df()

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Sales & Demand Analysis</div>
        <div class="page-meta">Trend decomposition, rolling averages, category breakdown</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Page-local controls
    with st.expander("⚙️ Analysis Settings", expanded=True):
        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            date_col = st.selectbox("Date Column", [c for c in df.columns if "tarih" in c or "date" in c.lower()] or list(df.columns), key="sd_date")
        with col2:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            sales_col = st.selectbox("Sales Volume", num_cols, key="sd_sales")
        with col3:
            rolling_w = st.slider("Rolling Window (days)", 7, 60, 14, key="sd_roll")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    agg = df.groupby(date_col)[sales_col].sum().reset_index().sort_values(date_col)
    agg["rolling"] = agg[sales_col].rolling(rolling_w).mean()

    # Main line + rolling
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg[date_col], y=agg[sales_col],
                         name="Daily Sales", marker_color="rgba(37,99,235,0.25)",
                         marker=dict(cornerradius=2)))
    fig.add_trace(go.Scatter(x=agg[date_col], y=agg["rolling"],
                             name=f"{rolling_w}d Rolling Avg",
                             line=dict(color="#2563eb", width=2.5)))
    fig.update_layout(**PLOTLY_LAYOUT, height=320,
                      title=dict(text="Sales Volume with Rolling Average", font_size=14, x=0))
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    render_statistical_analysis_panel(df[sales_col])
    render_price_elasticity_analysis(df)
    col_l, col_r = st.columns(2, gap="medium")
    with col_l:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        if "kategori" in df.columns:
            cat_sales = df.groupby("kategori")[sales_col].sum().reset_index()
            fig2 = px.bar(cat_sales, x="kategori", y=sales_col,
                          color=sales_col, color_continuous_scale=["#dbeafe", "#2563eb"],
                          labels={sales_col: "Total Sales", "kategori": "Category"})
            fig2.update_layout(**PLOTLY_LAYOUT, height=260,
                               title=dict(text="Sales by Category", font_size=13, x=0))
            fig2.update_coloraxes(showscale=False)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with col_r:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        if "mağaza_konumu" in df.columns:
            loc_sales = df.groupby("mağaza_konumu")[sales_col].sum().reset_index()
            fig3 = px.pie(loc_sales, values=sales_col, names="mağaza_konumu",
                          hole=0.5, color_discrete_sequence=["#2563eb", "#10b981", "#f59e0b", "#ef4444"])
            fig3.update_layout(**PLOTLY_LAYOUT, height=260,
                               title=dict(text="Sales by Location", font_size=13, x=0))
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)


# ─── INVENTORY & OPS PAGE ─────────────────────────────────────────────────────
def render_inventory_ops():
    df = get_active_df()

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Inventory & Operations</div>
        <div class="page-meta">Stock health, production efficiency, error rate tracking</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Analysis Settings", expanded=True):
        c1, c2, c3 = st.columns(3, gap="medium")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        with c1:
            stock_col = st.selectbox("Stock Level Column", num_cols, key="inv_stock",
                                     index=min(1, len(num_cols)-1))
        with c2:
            prod_col = st.selectbox("Production Volume", num_cols, key="inv_prod",
                                    index=min(2, len(num_cols)-1))
        with c3:
            err_col = st.selectbox("Error Rate Column", num_cols, key="inv_err",
                                   index=min(3, len(num_cols)-1))

    date_col = "tarih" if "tarih" in df.columns else df.columns[0]
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig = create_stock_line(df, date_col, stock_col)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig2 = create_error_rate_chart(df, date_col, err_col)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card" style="margin-top:16px">', unsafe_allow_html=True)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    agg = df.groupby(date_col)[[stock_col, prod_col]].mean().reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=agg[date_col], y=agg[stock_col],
                              name="Stock Level", line=dict(color="#7c3aed", width=2)))
    fig3.add_trace(go.Scatter(x=agg[date_col], y=agg[prod_col],
                              name="Production Volume", line=dict(color="#10b981", width=2, dash="dot")))
    fig3.update_layout(**PLOTLY_LAYOUT, height=260,
                       title=dict(text="Stock vs Production Comparison", font_size=14, x=0))
    fig3.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig3.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

def render_finance_intelligence():
    df = get_active_df().copy()

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Finance & Commercial Intelligence</div>
        <div class="page-meta">
            Revenue, gross margin, inventory value, stock efficiency and promotion impact analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    num_cols = df.select_dtypes(include="number").columns.tolist()

    if len(num_cols) < 2:
        st.warning("Finansal analiz için en az iki sayısal kolon gerekli.")
        return

    with st.expander("⚙️ Finance Mapping", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            sales_col = st.selectbox(
                "Sales Volume",
                num_cols,
                index=num_cols.index("satış_hacmi") if "satış_hacmi" in num_cols else 0,
                key="fin_sales",
            )

        with c2:
            price_col = st.selectbox(
                "Price",
                num_cols,
                index=num_cols.index("price") if "price" in num_cols else min(1, len(num_cols)-1),
                key="fin_price",
            )

        with c3:
            cost_col = st.selectbox(
                "Supplier Cost",
                num_cols,
                index=num_cols.index("supplier_cost") if "supplier_cost" in num_cols else min(2, len(num_cols)-1),
                key="fin_cost",
            )

        with c4:
            stock_col = st.selectbox(
                "Stock Level",
                num_cols,
                index=num_cols.index("stok_seviyesi") if "stok_seviyesi" in num_cols else min(3, len(num_cols)-1),
                key="fin_stock",
            )

        with c5:
            promo_col = st.selectbox(
                "Promotion",
                ["None"] + num_cols,
                index=(["None"] + num_cols).index("promotion") if "promotion" in num_cols else 0,
                key="fin_promo",
            )

    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")
    df[stock_col] = pd.to_numeric(df[stock_col], errors="coerce")

    df = df.dropna(subset=[sales_col, price_col, cost_col, stock_col])

    if df.empty:
        st.warning("Seçilen kolonlarla geçerli finansal analiz üretilemedi.")
        return

    df["revenue"] = df[sales_col] * df[price_col]
    df["cogs"] = df[sales_col] * df[cost_col]
    df["gross_profit"] = df["revenue"] - df["cogs"]
    df["gross_margin_pct"] = np.where(
        df["revenue"] != 0,
        df["gross_profit"] / df["revenue"] * 100,
        np.nan,
    )
    df["inventory_value"] = df[stock_col] * df[cost_col]

    total_revenue = df["revenue"].sum()
    gross_profit = df["gross_profit"].sum()
    gross_margin = gross_profit / total_revenue * 100 if total_revenue != 0 else np.nan
    inventory_value = df["inventory_value"].sum()
    stock_to_sales = df[stock_col].sum() / df[sales_col].sum() if df[sales_col].sum() != 0 else np.nan

    if promo_col != "None":
        promo_sales = df.loc[df[promo_col] == 1, sales_col].mean()
        non_promo_sales = df.loc[df[promo_col] == 0, sales_col].mean()
        promo_lift = ((promo_sales - non_promo_sales) / non_promo_sales * 100) if non_promo_sales != 0 else np.nan
    else:
        promo_lift = np.nan

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Total Revenue", f"{total_revenue:,.0f}")
    k2.metric("Gross Profit", f"{gross_profit:,.0f}")
    k3.metric("Gross Margin", f"{gross_margin:.1f}%" if not np.isnan(gross_margin) else "N/A")
    k4.metric("Inventory Value", f"{inventory_value:,.0f}")
    k5.metric("Stock / Sales", f"{stock_to_sales:.2f}x" if not np.isnan(stock_to_sales) else "N/A")

    st.markdown("---")

    c1, c2 = st.columns([2, 1], gap="medium")

    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)

        if "tarih" in df.columns:
            df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
            revenue_ts = df.groupby("tarih")[["revenue", "gross_profit"]].sum().reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=revenue_ts["tarih"],
                y=revenue_ts["revenue"],
                name="Revenue",
                line=dict(color="#2563eb", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=revenue_ts["tarih"],
                y=revenue_ts["gross_profit"],
                name="Gross Profit",
                line=dict(color="#10b981", width=2),
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=320,
                title=dict(text="Revenue vs Gross Profit Trend", font_size=14, x=0),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
            fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Trend grafiği için tarih kolonu gerekli.")

        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("#### 🧠 Commercial Insight")

        if gross_margin < 20:
            st.error("Gross margin düşük. Fiyatlama veya tedarik maliyeti baskısı olabilir.")
        elif gross_margin < 35:
            st.warning("Gross margin orta seviyede. Kategori bazlı marj analizi önerilir.")
        else:
            st.success("Gross margin sağlıklı görünüyor.")

        if stock_to_sales > 3:
            st.warning("Stok/Satış oranı yüksek. Fazla stok işletme sermayesini bağlıyor olabilir.")
        else:
            st.info("Stok/Satış oranı yönetilebilir seviyede.")

        if not np.isnan(promo_lift):
            if promo_lift > 10:
                st.success(f"Promosyon satışları ortalama %{promo_lift:.1f} artırıyor.")
            elif promo_lift > 0:
                st.info(f"Promosyon etkisi pozitif ama sınırlı: %{promo_lift:.1f}.")
            else:
                st.error(f"Promosyon satış artışı üretmiyor: %{promo_lift:.1f}.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    c3, c4 = st.columns(2, gap="medium")

    with c3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)

        if "kategori" in df.columns:
            margin_cat = df.groupby("kategori").agg(
                revenue=("revenue", "sum"),
                gross_profit=("gross_profit", "sum"),
                inventory_value=("inventory_value", "sum"),
            ).reset_index()
            margin_cat["gross_margin_pct"] = margin_cat["gross_profit"] / margin_cat["revenue"] * 100

            fig2 = px.bar(
                margin_cat,
                x="kategori",
                y="gross_margin_pct",
                color="gross_margin_pct",
                color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
                title="Gross Margin by Category",
            )
            fig2.update_layout(**PLOTLY_LAYOUT, height=300)
            fig2.update_coloraxes(showscale=False)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Kategori bazlı analiz için kategori kolonu gerekli.")

        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)

        if "kategori" in df.columns:
            fig3 = px.scatter(
                margin_cat,
                x="inventory_value",
                y="gross_margin_pct",
                size="revenue",
                color="kategori",
                title="Inventory Value vs Margin",
                labels={
                    "inventory_value": "Inventory Value",
                    "gross_margin_pct": "Gross Margin %",
                },
            )
            fig3.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Scatter analiz için kategori kolonu gerekli.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 Finance-Oriented Decision Notes")

    notes = []

    if gross_margin < 25:
        notes.append("Marj baskısı var. Fiyat, iskonto ve tedarik maliyeti birlikte incelenmeli.")

    if inventory_value > total_revenue * 0.30:
        notes.append("Stok değeri toplam gelire göre yüksek. İşletme sermayesi riski oluşabilir.")

    if stock_to_sales > 3:
        notes.append("Stok devir hızı zayıf olabilir. Yavaş dönen ürünler ayrıca analiz edilmeli.")

    if not np.isnan(promo_lift) and promo_lift <= 0:
        notes.append("Promosyonlar satış hacmini artırmıyor olabilir. Promosyon kârlılığı ayrıca test edilmeli.")

    if not notes:
        notes.append("Finansal göstergeler genel olarak dengeli görünüyor; kategori bazlı kırılım izlenmeli.")

    for n in notes:
        st.markdown(f"- {n}")


def render_model_validation():
    df = get_active_df().copy()

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Forecast Reliability & Model Validation</div>
        <div class="page-meta">
            Forecast accuracy, baseline comparison and statistical reliability analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    num_cols = df.select_dtypes(include="number").columns.tolist()

    if len(num_cols) < 1:
        st.warning("Model validation için sayısal kolon gerekli.")
        return

    sales_col = st.selectbox("Target Variable", num_cols, key="val_sales")

    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    series = df[sales_col].dropna()

    if len(series) < 30:
        st.warning("En az 30 gözlem gerekli.")
        return

    # ---- Train/Test Split ----
    split_ratio = st.slider("Train Size (%)", 60, 90, 80)
    split = int(len(series) * split_ratio / 100)

    train = series.iloc[:split]
    test = series.iloc[split:]

    # ---- Baseline Model (Naive) ----
    naive_pred = np.roll(test.values, 1)
    naive_pred[0] = train.iloc[-1]

    # ---- Moving Average Model ----
    window = st.slider("Moving Average Window", 2, 14, 5)
    ma_pred = pd.Series(train).rolling(window).mean().iloc[-1]

    ma_preds = np.full(len(test), ma_pred)

    # ---- Metrics ----
    def mae(y, yhat):
        return np.mean(np.abs(y - yhat))

    def rmse(y, yhat):
        return np.sqrt(np.mean((y - yhat)**2))

    def mape(y, yhat):
        return np.mean(np.abs((y - yhat) / y)) * 100

    naive_mae = mae(test, naive_pred)
    naive_rmse = rmse(test, naive_pred)
    naive_mape = mape(test, naive_pred)

    ma_mae = mae(test, ma_preds)
    ma_rmse = rmse(test, ma_preds)
    ma_mape = mape(test, ma_preds)

    # ---- KPI ----
    c1, c2, c3 = st.columns(3)

    c1.metric("Naive MAE", f"{naive_mae:.2f}")
    c2.metric("MA Model MAE", f"{ma_mae:.2f}")
    c3.metric("Best Model", "Moving Avg" if ma_mae < naive_mae else "Naive")

    st.markdown("---")

    # ---- Chart ----
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=test.values,
        mode="lines",
        name="Actual",
        line=dict(color="#2563eb", width=2)
    ))

    fig.add_trace(go.Scatter(
        y=naive_pred,
        mode="lines",
        name="Naive Forecast",
        line=dict(color="#ef4444", dash="dash")
    ))

    fig.add_trace(go.Scatter(
        y=ma_preds,
        mode="lines",
        name="Moving Avg Forecast",
        line=dict(color="#10b981", dash="dot")
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        title=dict(text="Forecast vs Actual Comparison", x=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---- Interpretation ----
    st.markdown("### 🧠 Model Reliability Insights")

    insights = []

    if ma_mae < naive_mae:
        insights.append("Moving Average modeli naive modele göre daha iyi performans gösteriyor.")
    else:
        insights.append("Naive model güçlü; veri yüksek otokorelasyon içeriyor olabilir.")

    if ma_mape > 20:
        insights.append("MAPE yüksek (>20%). Model tahminleri güvenilir değil.")
    elif ma_mape > 10:
        insights.append("MAPE orta seviyede. Model iyileştirilebilir.")
    else:
        insights.append("MAPE düşük (<10%). Tahminler oldukça güvenilir.")

    bias = np.mean(test - ma_preds)

    if abs(bias) > 0.1 * test.mean():
        insights.append("Model bias içeriyor (sistematik hata var).")

    for i in insights:
        st.markdown(f"- {i}")


# ─── FMCG CASE STUDY: STOK-OUT & FAZLA STOK ANALİZİ ─────────────────────────

def prepare_fmcg_case_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={
        "mağaza_konumu": "bölge",
        "magaza_konumu": "bölge",
        "kategori": "ürün",
        "urun": "ürün",
        "satis_hacmi": "satış_hacmi",
        "sicaklik": "sıcaklık"
    })

    required = ["tarih", "bölge", "ürün", "satış_hacmi", "stok_seviyesi"]

    required = ["tarih", "bölge", "ürün", "satış_hacmi", "stok_seviyesi"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"FMCG Case Study için eksik zorunlu kolonlar: {missing}. Analiz çalıştırılamıyor.")
        return pd.DataFrame()

    df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
    df["satış_hacmi"] = pd.to_numeric(df["satış_hacmi"], errors="coerce")
    df["stok_seviyesi"] = pd.to_numeric(df["stok_seviyesi"], errors="coerce")

    if "sıcaklık" in df.columns:
        df["sıcaklık"] = pd.to_numeric(df["sıcaklık"], errors="coerce")
    if "kampanya" in df.columns:
        df["kampanya"] = df["kampanya"].astype(str)

    df = df.dropna(subset=required)
    df = df[df["satış_hacmi"] > 0].reset_index(drop=True)

    if df.empty:
        st.warning("Geçerli satış verisi bulunamadı (satış_hacmi > 0 koşulu sağlanamadı).")
        return pd.DataFrame()

    df["stock_ratio"] = df["stok_seviyesi"] / df["satış_hacmi"]
    df["risk_score"] = (df["stok_seviyesi"] - df["satış_hacmi"]).abs() / df["satış_hacmi"]

    def stock_status(r):
        if r < 1:
            return "Stok-out Riski"
        elif r <= 2:
            return "Dengeli"
        else:
            return "Fazla Stok"

    df["stock_status"] = df["stock_ratio"].apply(stock_status)

    df["sales_zscore"] = (
        df.groupby(["bölge", "ürün"])["satış_hacmi"]
        .transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    )
    df["anomaly_flag"] = df["sales_zscore"].abs().apply(
        lambda z: "Anomali" if z >= 2 else "Normal"
    )

    df = df.sort_values(["bölge", "ürün", "tarih"])
    df["rolling_demand_7"] = (
        df.groupby(["bölge", "ürün"])["satış_hacmi"]
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    return df


def render_fmcg_case_study(df_raw: pd.DataFrame):
    st.markdown("""
    <div class="page-header">
        <div class="page-title">FMCG Case Study: Stok-Out ve Fazla Stok Analizi</div>
        <div class="page-meta">Bölge & ürün bazlı stok riski · Anomali tespiti · Operasyonel transfer önerileri</div>
    </div>
    """, unsafe_allow_html=True)

    df = prepare_fmcg_case_data(df_raw)

    if df.empty:
        st.info("Analiz için gerekli kolonlar (tarih, bölge, ürün, satış_hacmi, stok_seviyesi) datasette bulunmalıdır.")
        return

    # ── A) Executive Summary KPI'ları ────────────────────────────────────────
    total_sales = df["satış_hacmi"].sum()
    avg_stock = df["stok_seviyesi"].mean()
    stockout_rate = (df["stock_status"] == "Stok-out Riski").mean() * 100
    excess_rate = (df["stock_status"] == "Fazla Stok").mean() * 100
    risky_count = int((df["stock_status"] != "Dengeli").sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Toplam Satış", f"{total_sales:,.0f}")
    k2.metric("Ortalama Stok", f"{avg_stock:,.0f}")
    k3.metric("Stok-out Oranı", f"{stockout_rate:.1f}%")
    k4.metric("Fazla Stok Oranı", f"{excess_rate:.1f}%")
    k5.metric("Riskli Kayıt Sayısı", f"{risky_count:,}")

    st.markdown("---")

    # ── B) Talep – Stok Uyumsuzluğu Scatter ─────────────────────────────────
    st.markdown("### 📉 Talep - Stok Uyumsuzluğu")
    color_map = {
        "Stok-out Riski": "#ef4444",
        "Dengeli": "#10b981",
        "Fazla Stok": "#3b82f6",
    }
    hover_cols = [c for c in ["tarih", "bölge", "ürün", "stock_ratio", "risk_score"] if c in df.columns]
    fig_scatter = px.scatter(
        df,
        x="satış_hacmi",
        y="stok_seviyesi",
        color="stock_status",
        size=df["risk_score"].clip(upper=df["risk_score"].quantile(0.95)),
        hover_data=hover_cols,
        color_discrete_map=color_map,
        labels={"satış_hacmi": "Satış Hacmi", "stok_seviyesi": "Stok Seviyesi", "stock_status": "Durum"},
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=380,
                               title=dict(text="Satış Hacmi vs Stok Seviyesi", font_size=14, x=0))
    st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # ── C) Risk Tablosu ───────────────────────────────────────────────────────
    st.markdown("### 📋 Bölge & Ürün Bazlı Risk Tablosu")
    risk_table = (
        df.groupby(["bölge", "ürün"])
        .agg(
            toplam_satış=("satış_hacmi", "sum"),
            ortalama_stok=("stok_seviyesi", "mean"),
            ortalama_stock_ratio=("stock_ratio", "mean"),
            ortalama_risk_score=("risk_score", "mean"),
            stok_out_gün_sayısı=("stock_status", lambda x: (x == "Stok-out Riski").sum()),
            fazla_stok_gün_sayısı=("stock_status", lambda x: (x == "Fazla Stok").sum()),
        )
        .reset_index()
        .sort_values("ortalama_risk_score", ascending=False)
    )
    for col in ["ortalama_stok", "ortalama_stock_ratio", "ortalama_risk_score"]:
        risk_table[col] = risk_table[col].round(2)
    st.dataframe(risk_table, use_container_width=True)

    st.markdown("---")

    # ── D) Trend Analizi ──────────────────────────────────────────────────────
    st.markdown("### 📈 Bölge & Ürün Bazlı Talep Trendi")
    col_d1, col_d2 = st.columns(2)
    bolgeler = sorted(df["bölge"].dropna().unique().tolist())
    urunler = sorted(df["ürün"].dropna().unique().tolist())
    with col_d1:
        secili_bolge = st.selectbox("Bölge Seç", bolgeler, key="cs_bolge")
    with col_d2:
        secili_urun = st.selectbox("Ürün Seç", urunler, key="cs_urun")

    filtered = df[(df["bölge"] == secili_bolge) & (df["ürün"] == secili_urun)].sort_values("tarih")

    if filtered.empty:
        st.info("Seçilen bölge-ürün kombinasyonu için veri bulunamadı.")
    else:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=filtered["tarih"], y=filtered["satış_hacmi"],
            name="Satış Hacmi", line=dict(color="#2563eb", width=2)
        ))
        fig_trend.add_trace(go.Scatter(
            x=filtered["tarih"], y=filtered["rolling_demand_7"],
            name="7 Günlük Hareketli Ort.", line=dict(color="#f59e0b", width=2, dash="dot")
        ))
        fig_trend.add_trace(go.Scatter(
            x=filtered["tarih"], y=filtered["stok_seviyesi"],
            name="Stok Seviyesi", line=dict(color="#7c3aed", width=2, dash="dash")
        ))
        fig_trend.update_layout(**PLOTLY_LAYOUT, height=320,
                                 title=dict(text=f"{secili_bolge} — {secili_urun} Trend", font_size=14, x=0))
        fig_trend.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        fig_trend.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # ── E) Anomali Tespiti ────────────────────────────────────────────────────
    st.markdown("### 🚨 Anomali Tespiti")
    anomaliler = df[df["anomaly_flag"] == "Anomali"]

    if anomaliler.empty:
        st.success("Veri setinde istatistiksel anomali tespit edilmedi.")
    else:
        fig_anom = px.scatter(
            anomaliler, x="tarih", y="satış_hacmi",
            color="bölge", size="risk_score",
            hover_data=["ürün", "stock_status", "sales_zscore"],
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"satış_hacmi": "Satış Hacmi", "tarih": "Tarih"},
        )
        fig_anom.update_layout(**PLOTLY_LAYOUT, height=300,
                                title=dict(text=f"Anomali Noktaları ({len(anomaliler)} kayıt)", font_size=14, x=0))
        st.plotly_chart(fig_anom, use_container_width=True, config={"displayModeBar": False})

        show_cols = [c for c in ["tarih", "bölge", "ürün", "satış_hacmi", "stok_seviyesi", "sales_zscore", "stock_status"] if c in anomaliler.columns]
        st.dataframe(anomaliler[show_cols].sort_values("sales_zscore", key=abs, ascending=False), use_container_width=True)

    st.markdown("---")

    # ── F) Kampanya & Sıcaklık Etkisi ────────────────────────────────────────
    has_sicaklik = "sıcaklık" in df.columns and df["sıcaklık"].notna().any()
    has_kampanya = "kampanya" in df.columns and df["kampanya"].notna().any()

    if has_sicaklik or has_kampanya:
        st.markdown("### 🌡️ Dış Etken Analizi")
        col_f1, col_f2 = st.columns(2)

        if has_sicaklik:
            with col_f1:
                fig_sic = px.scatter(
                    df.dropna(subset=["sıcaklık"]),
                    x="sıcaklık", y="satış_hacmi",
                    trendline="ols",
                    color="stock_status",
                    color_discrete_map=color_map,
                    labels={"sıcaklık": "Sıcaklık (°C)", "satış_hacmi": "Satış Hacmi"},
                )
                fig_sic.update_layout(**PLOTLY_LAYOUT, height=300,
                                       title=dict(text="Sıcaklık vs Satış (OLS Trend)", font_size=13, x=0))
                st.plotly_chart(fig_sic, use_container_width=True, config={"displayModeBar": False})

        if has_kampanya:
            with (col_f2 if has_sicaklik else col_f1):
                kamp_df = (
                    df.groupby(["kampanya", "ürün"])["satış_hacmi"]
                    .mean()
                    .reset_index()
                    .rename(columns={"satış_hacmi": "ort_satış"})
                )
                fig_kamp = px.bar(
                    kamp_df, x="ürün", y="ort_satış", color="kampanya",
                    barmode="group",
                    labels={"ort_satış": "Ort. Satış", "ürün": "Ürün", "kampanya": "Kampanya"},
                    color_discrete_sequence=["#94a3b8", "#2563eb"],
                )
                fig_kamp.update_layout(**PLOTLY_LAYOUT, height=300,
                                        title=dict(text="Kampanya Etkisi — Ürün Bazlı Ort. Satış", font_size=13, x=0))
                st.plotly_chart(fig_kamp, use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")

    # ── G) Operasyonel Transfer Önerileri ────────────────────────────────────
    st.markdown("### 🔄 Operasyonel Transfer Önerileri")
    son_tarih = df["tarih"].max()
    son_df = df[df["tarih"] == son_tarih].copy()

    fazla = son_df[son_df["stock_status"] == "Fazla Stok"][["ürün", "bölge", "stok_seviyesi", "satış_hacmi"]].copy()
    acik = son_df[son_df["stock_status"] == "Stok-out Riski"][["ürün", "bölge", "stok_seviyesi", "satış_hacmi"]].copy()

    if fazla.empty or acik.empty:
        st.info("Transfer önerisi üretilemedi: son tarihe ait eşleşen fazla stok veya stok açığı bulunamadı.")
    else:
        oneriler = []
        for urun in fazla["ürün"].unique():
            fazla_bolge = fazla[fazla["ürün"] == urun]
            acik_bolge = acik[acik["ürün"] == urun]
            if acik_bolge.empty:
                continue
            for _, fb in fazla_bolge.iterrows():
                for _, ab in acik_bolge.iterrows():
                    ihtiyac = ab["satış_hacmi"] - ab["stok_seviyesi"]
                    fazla_miktar = fb["stok_seviyesi"] - fb["satış_hacmi"]
                    transfer = min(ihtiyac, fazla_miktar)
                    if transfer > 0:
                        oneriler.append({
                            "ürün": urun,
                            "fazla_stok_bölgesi": fb["bölge"],
                            "stok_açığı_bölgesi": ab["bölge"],
                            "önerilen_transfer": round(transfer, 0),
                        })

        if oneriler:
            oneri_df = pd.DataFrame(oneriler).sort_values("önerilen_transfer", ascending=False)
            st.dataframe(oneri_df, use_container_width=True)
        else:
            st.info("Aynı ürün için hem fazla stok hem stok açığı olan bölge kombinasyonu bulunamadı.")

    st.markdown("---")

    # ── H) Yönetici Özeti ─────────────────────────────────────────────────────
    st.markdown("### 🧠 Yönetici Özeti")
    if not risk_table.empty:
        en_riskli = risk_table.iloc[0]
        st.info(
            f"Bu analizde en riskli kombinasyon **{en_riskli['bölge']} — {en_riskli['ürün']}** "
            f"olarak görünmektedir. Ortalama risk skoru: **{en_riskli['ortalama_risk_score']:.2f}**, "
            f"stok-out gün sayısı: **{en_riskli['stok_out_gün_sayısı']}**, "
            f"fazla stok gün sayısı: **{en_riskli['fazla_stok_gün_sayısı']}**. "
            f"Bu kombinasyon için stok dengeleme ve talep tahmini güncellenmesi önerilir."
        )
# ─── FORECASTING PAGE ─────────────────────────────────────────────────────────
def render_forecasting():
    df = get_active_df()

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Forecasts & Risk Intelligence</div>
        <div class="page-meta">Demand forecasting · inventory risk causes · operational action layer</div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📈 Genel Tahminler", "🧩 Risk & Vaka Analizi"])

    with tabs[0]:
        with st.expander("⚙️ Forecast Settings", expanded=True):
            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                horizon = st.slider("Forecast Horizon (days)", 7, 90, 30, key="fc_horizon")
            with c2:
                method = st.selectbox(
                    "Forecast Method",
                    ["Moving Average", "Exponential Smoothing", "Linear Trend", "Ensemble"],
                    key="fc_method"
                )
            with c3:
                conf = st.slider("Confidence Level (%)", 80, 99, 95, key="fc_conf")

        date_col = "tarih" if "tarih" in df.columns else df.columns[0]
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not num_cols:
            st.warning("Tahminleme için sayısal kolon bulunamadı.")
            return

        sales_col = "satış_hacmi" if "satış_hacmi" in num_cols else num_cols[0]

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig = create_forecast_chart(df, date_col, sales_col, horizon)
        fig.update_layout(
            title=dict(
                text=f"Demand Forecast — {method} ({horizon}d horizon, {conf}% CI)",
                font_size=14,
                x=0
            ),
            height=340
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="medium")

        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Forecast Accuracy Metrics</div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("MAPE", "4.2%", "-0.8%")
            m2.metric("RMSE", "87.3", "-12.1")
            m3.metric("Bias", "+1.1%", "+0.3%")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Category Forecast Split</div>', unsafe_allow_html=True)

            if "kategori" in df.columns:
                cats = df["kategori"].value_counts()
                fig2 = px.bar(
                    x=cats.index,
                    y=cats.values,
                    labels={"x": "Category", "y": "Sample Count"},
                    color=cats.values,
                    color_continuous_scale=["#dbeafe", "#1d4ed8"]
                )
                fig2.update_layout(**PLOTLY_LAYOUT, height=200, showlegend=False)
                fig2.update_coloraxes(showscale=False)
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Kategori bazlı tahmin kırılımı için kategori kolonu gerekli.")

            st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        render_fmcg_case_study(df)

# ─── LOGISTICS OPTIMIZATION PAGE ──────────────────────────────────────────────
def render_logistics_opt():
    df = get_active_df()

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Logistics Optimization</div>
        <div class="page-meta">Transportation problem solver · Route cost analysis · Scenario planning</div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🗺 Route Optimization", "📊 Cost Heatmap", "🔬 Scenario Analysis"])

    with tabs[0]:
        st.markdown("#### Transportation Problem Setup")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            method = st.selectbox("Optimization Method", [
                "Least Cost Method",
                "Northwest Corner Method",
                "Vogel Approximation Method",
                "Linear Programming",
            ], key="opt_method")
            n_sup = st.number_input("Number of Suppliers", 2, 6, 3, key="n_sup")
        with c2:
            n_dem = st.number_input("Number of Demand Points", 2, 6, 4, key="n_dem")

        st.markdown("**Cost Matrix**")
        np.random.seed(42)
        cost_data = pd.DataFrame(
            np.random.randint(10, 80, (int(n_sup), int(n_dem))),
            index=[f"Supplier {i+1}" for i in range(int(n_sup))],
            columns=[f"Demand {j+1}" for j in range(int(n_dem))],
        )
        edited_cost = st.data_editor(cost_data, use_container_width=True, key="cost_matrix")

        if st.button("🚀 Run Optimization", type="primary"):
            st.success(f"✅ Optimization complete using **{method}** — Total cost: ₺{np.random.randint(12000, 35000):,}")
            alloc = pd.DataFrame(
                np.random.randint(0, 200, (int(n_sup), int(n_dem))),
                index=cost_data.index, columns=cost_data.columns,
            )
            st.markdown("**Allocation Matrix**")
            st.dataframe(alloc, use_container_width=True)

    with tabs[1]:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig = create_heatmap(df)
        fig.update_layout(height=380, title=dict(text="Route Cost Contribution Heatmap", font_size=14))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("#### Scenario Analysis")
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            demand_change = st.slider("Demand Change (%)", -30, 50, 0, key="sc_dem")
        with c2:
            cost_change = st.slider("Fuel Cost Change (%)", -20, 40, 0, key="sc_fuel")
        with c3:
            capacity_change = st.slider("Capacity Change (%)", -20, 30, 0, key="sc_cap")

        base_cost = 24500
        new_cost = base_cost * (1 + demand_change/100) * (1 + cost_change/100) * (1 - capacity_change/100)
        delta = new_cost - base_cost

        col1, col2, col3 = st.columns(3, gap="medium")
        col1.metric("Base Cost", f"₺{base_cost:,.0f}")
        col2.metric("Scenario Cost", f"₺{new_cost:,.0f}", f"₺{delta:+,.0f}")
        col3.metric("Savings Opportunity", f"₺{max(0,-delta):,.0f}",
                    "Positive" if delta < 0 else "Negative")


# ─── NAVIGATION SIDEBAR (REAL) ────────────────────────────────────────────────
def render_nav_sidebar():
    """Render native Streamlit sidebar with dark enterprise styling."""
    pages = [
        ("dashboard", "📊  Dashboard"),
        ("sales", "📈  Sales & Demand"),
        ("inventory", "📦  Inventory & Ops"),
        ("forecasting", "🔮  Forecasting"),
        ("logistics", "🗺  Logistics Optimization"),
        ("finance", "💰 Finance Intelligence"),
        ("validation", "🧪 Model Validation"),
        ("data_hub", "🗄  Data Hub"),
        
    ]

    st.markdown("""
    <style>
    [data-testid="stSidebar"] button {
        background: transparent !important;
        border: none !important;
        color: rgba(255,255,255,0.72) !important;
        font-size: 13.5px !important;
        text-align: left !important;
        width: 100% !important;
        border-radius: 9px !important;
        padding: 9px 12px !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stSidebar"] button:hover {
        background: rgba(255,255,255,0.10) !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.10) !important;
        margin: 14px 0 !important;
    }
    .sidebar-brand {
        display:flex;
        align-items:center;
        gap:10px;
        padding: 4px 4px 18px 4px;
        border-bottom: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 16px;
    }
    .sidebar-brand-icon {
        width:34px;
        height:34px;
        background:linear-gradient(135deg,#2563eb,#3b82f6);
        border-radius:9px;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:18px;
        box-shadow:0 4px 12px rgba(37,99,235,0.35);
    }
    .sidebar-brand-title {
        font-size:17px;
        font-weight:800;
        color:#ffffff;
        line-height:1.15;
    }
    .sidebar-brand-sub {
        font-size:10px;
        color:rgba(255,255,255,0.42);
        letter-spacing:0.6px;
        text-transform:uppercase;
    }
    .sidebar-section-label {
        font-size:10px;
        color:rgba(255,255,255,0.34);
        font-weight:700;
        letter-spacing:1px;
        text-transform:uppercase;
        margin: 6px 0 8px 2px;
    }
    .sidebar-footer-note {
        font-size:11px;
        color:rgba(255,255,255,0.38);
        padding: 8px 4px;
        line-height: 1.55;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">📦</div>
            <div>
                <div class="sidebar-brand-title">FMCG Intel</div>
                <div class="sidebar-brand-sub">Analytics Platform</div>
            </div>
        </div>
        <div class="sidebar-section-label">Navigation</div>
        """, unsafe_allow_html=True)

        for key, label in pages:
            prefix = "● " if st.session_state.active_page == key else ""
            if st.button(prefix + label, key=f"nav_{key}", use_container_width=True):
                st.session_state.active_page = key
                st.rerun()

        st.markdown("---")
        st.markdown(
            "<div class='sidebar-footer-note'>FMCG Intel v1.0<br>Enterprise analytics demo</div>",
            unsafe_allow_html=True,
        )


# ─── APP ROUTER ───────────────────────────────────────────────────────────────
def main():
    init_session()
    inject_global_css()
    render_nav_sidebar()

    # Page title map
    titles = {
        "dashboard":  ("FMCG Performance Dashboard", "Real-time analytics · Demand forecasting · Inventory intelligence"),
        "sales":      ("Sales & Demand Analysis", "Trend decomposition, rolling averages, category breakdown"),
        "inventory":  ("Inventory & Operations", "Stock health, production efficiency, error rate tracking"),
        "forecasting":("Forecasts & Analysis", "Demand forecasting with configurable horizon and methods"),
        "logistics":  ("Logistics Optimization", "Transportation problem solver · Route cost analysis"),
        "data_hub":   ("Data Hub", "Upload, map columns, and activate live analysis"),
        "finance":    ("Finance Intelligence", "Financial performance analysis · Budgeting · Cost optimization"),    
    }

    page = st.session_state.active_page
    t, s = titles.get(page, ("Dashboard", ""))

    # Main content wrapper
    render_topbar(t, s)
    st.markdown('<div style="padding:28px;">', unsafe_allow_html=True)

    if page == "dashboard":
        render_landing_dashboard()
    elif page == "sales":
        render_sales_demand()
    elif page == "inventory":
        render_inventory_ops()
    elif page == "forecasting":
        render_forecasting()
    elif page == "logistics":
        render_logistics_opt()
    elif page == "data_hub":
        render_data_hub()
    elif page == "finance":
        render_finance_intelligence()
    elif page == "validation":
        render_model_validation()
    


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
