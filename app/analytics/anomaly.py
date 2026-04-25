from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AnomalyResult:
    series_df: pd.DataFrame
    anomaly_count: int
    spike_count: int
    drop_count: int
    anomaly_ratio: float
    latest_status: str
    risk_level: str


def build_anomaly_detection(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    window: int = 28,
    z_thresh: float = 2.0,
) -> Optional[AnomalyResult]:
    if date_col not in df.columns or value_col not in df.columns:
        return None

    data = df[[date_col, value_col]].copy()
    data = data.dropna(subset=[date_col, value_col])
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[date_col, value_col])

    if data.empty or len(data) < window + 5:
        return None

    daily = (
        data.groupby(date_col, as_index=False)[value_col]
        .sum()
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    daily["rolling_mean"] = daily[value_col].rolling(window=window, min_periods=max(5, window // 2)).mean()
    daily["rolling_std"] = daily[value_col].rolling(window=window, min_periods=max(5, window // 2)).std()

    daily["rolling_std"] = daily["rolling_std"].replace(0, np.nan)

    daily["z_score"] = (daily[value_col] - daily["rolling_mean"]) / daily["rolling_std"]
    daily["z_score"] = daily["z_score"].replace([np.inf, -np.inf], np.nan).fillna(0)

    daily["is_anomaly"] = daily["z_score"].abs() >= z_thresh
    daily["anomaly_type"] = np.where(
        daily["z_score"] >= z_thresh,
        "spike",
        np.where(daily["z_score"] <= -z_thresh, "drop", "normal"),
    )

    anomaly_count = int(daily["is_anomaly"].sum())
    spike_count = int((daily["anomaly_type"] == "spike").sum())
    drop_count = int((daily["anomaly_type"] == "drop").sum())
    anomaly_ratio = anomaly_count / len(daily)

    latest_row = daily.iloc[-1]
    latest_status = latest_row["anomaly_type"]

    if anomaly_ratio >= 0.10:
        risk_level = "Yüksek"
    elif anomaly_ratio >= 0.05:
        risk_level = "Orta"
    else:
        risk_level = "Düşük"

    return AnomalyResult(
        series_df=daily,
        anomaly_count=anomaly_count,
        spike_count=spike_count,
        drop_count=drop_count,
        anomaly_ratio=anomaly_ratio,
        latest_status=latest_status,
        risk_level=risk_level,
    )

import pandas as pd
import numpy as np

def detect_zscore_anomalies(
    df: pd.DataFrame,
    value_col: str,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Z-score based anomaly detection
    """
    data = df.copy()

    mean = data[value_col].mean()
    std = data[value_col].std()

    data["z_score"] = (data[value_col] - mean) / std
    data["is_anomaly"] = data["z_score"].abs() > threshold

    return data

def build_anomaly_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    anomalies = df["is_anomaly"].sum()

    return {
        "total_points": total,
        "anomalies": int(anomalies),
        "anomaly_ratio": anomalies / total if total > 0 else 0,
    }