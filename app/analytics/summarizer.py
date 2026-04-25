"""
summarizer.py — Rule-based executive summary generator.

Converts structured analytics metrics into short, client-facing business language.
Fully deterministic — no LLM or external API required.

Output is a dict of section → sentence that the UI can render freely.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Individual summary builders
# ---------------------------------------------------------------------------


def summarize_data_quality(
    total_rows: int,
    total_cols: int,
    missing_ratio: float,
    duplicate_count: int,
) -> Tuple[str, str]:
    """
    Produce a quality label and a plain-English sentence.

    Returns:
        (label, sentence)
        label: 'strong' | 'moderate' | 'weak'
    """
    issues: list[str] = []

    if missing_ratio > 0.20:
        issues.append(f"{missing_ratio:.0%} of values are missing")
    elif missing_ratio > 0.05:
        issues.append(f"minor missing data ({missing_ratio:.0%})")

    dup_ratio = duplicate_count / total_rows if total_rows > 0 else 0
    if dup_ratio > 0.10:
        issues.append(f"{duplicate_count:,} duplicate rows detected")
    elif duplicate_count > 0:
        issues.append(f"{duplicate_count} duplicate rows (low impact)")

    if not issues:
        label = "strong"
        sentence = (
            f"Data quality is strong. "
            f"The dataset contains {total_rows:,} rows across {total_cols} columns "
            f"with minimal missing values and no significant duplicates."
        )
    elif len(issues) == 1 and missing_ratio <= 0.20:
        label = "moderate"
        sentence = (
            f"Data quality is moderate. "
            f"The dataset has {total_rows:,} rows; {'; '.join(issues)}. "
            "Consider cleaning before production use."
        )
    else:
        label = "weak"
        sentence = (
            f"Data quality is weak. "
            f"Issues found: {'; '.join(issues)}. "
            "Data cleaning is strongly recommended before analysis."
        )

    return label, sentence


def summarize_trend(
    trend_direction: str,
    value_col: str,
    recent_mean: float,
    prior_mean: float,
) -> str:
    """
    Produce a trend summary sentence.

    Args:
        trend_direction: 'upward' | 'downward' | 'flat'
        value_col:       Name of the metric being analysed.
        recent_mean:     Mean of the most recent period.
        prior_mean:      Mean of the prior period (for comparison).

    Returns:
        A single plain-English summary sentence.
    """
    pct = 0.0
    if prior_mean and prior_mean != 0:
        pct = ((recent_mean - prior_mean) / abs(prior_mean)) * 100

    label_map = {
        "upward": "trending upward",
        "downward": "trending downward",
        "flat": "relatively stable",
    }
    direction_label = label_map.get(trend_direction, "stable")

    col_display = value_col.replace("_", " ").title()

    if trend_direction == "upward":
        return (
            f"{col_display} is {direction_label}, "
            f"up approximately {abs(pct):.1f}% compared to the prior period. "
            "Momentum appears positive."
        )
    elif trend_direction == "downward":
        return (
            f"{col_display} is {direction_label}, "
            f"down approximately {abs(pct):.1f}% compared to the prior period. "
            "Monitor closely for continued decline."
        )
    else:
        return (
            f"{col_display} is {direction_label} with less than {abs(pct):.1f}% "
            "change versus the prior period. No significant movement detected."
        )


def summarize_volatility(volatility_label: str, value_col: str, cv: float) -> str:
    """
    Produce a volatility summary sentence.

    Args:
        volatility_label: 'low' | 'moderate' | 'high' | 'unknown'
        value_col:        Metric name.
        cv:               Coefficient of variation.

    Returns:
        A single plain-English summary sentence.
    """
    col_display = value_col.replace("_", " ").title()

    messages = {
        "low": (
            f"Variability in {col_display} is low (CV={cv:.2f}), "
            "suggesting stable and predictable behaviour."
        ),
        "moderate": (
            f"Variability in {col_display} is moderate (CV={cv:.2f}). "
            "Some fluctuation is present but within a manageable range."
        ),
        "high": (
            f"Variability in {col_display} is high (CV={cv:.2f}). "
            "Significant fluctuations detected — investigate root causes."
        ),
        "unknown": (
            f"Variability in {col_display} could not be determined "
            "from the available data."
        ),
    }
    return messages.get(volatility_label, messages["unknown"])


def summarize_forecast(
    method: str,
    horizon: int,
    forecast_mean: float,
    current_mean: float,
    value_col: str,
) -> str:
    """
    Produce a plain-English forecast summary.

    Args:
        method:        Forecast method name (for display).
        horizon:       Forecast horizon in periods.
        forecast_mean: Mean of forecast values.
        current_mean:  Mean of recent actual values (last 14 periods).
        value_col:     Metric name.

    Returns:
        A single plain-English summary sentence.
    """
    col_display = value_col.replace("_", " ").title()
    method_label = method.replace("_", " ").title()

    if current_mean == 0:
        direction = "remain near current levels"
    elif forecast_mean > current_mean * 1.02:
        pct = ((forecast_mean - current_mean) / abs(current_mean)) * 100
        direction = f"increase by approximately {pct:.1f}%"
    elif forecast_mean < current_mean * 0.98:
        pct = ((current_mean - forecast_mean) / abs(current_mean)) * 100
        direction = f"decrease by approximately {pct:.1f}%"
    else:
        direction = "remain broadly stable"

    return (
        f"The {method_label} forecast over the next {horizon} periods suggests "
        f"{col_display} will {direction}. "
        "Note: this is an indicative simple forecast only and should not replace "
        "professional analysis."
    )


# ---------------------------------------------------------------------------
# Unified summary generator
# ---------------------------------------------------------------------------


def generate_executive_summary(
    total_rows: int,
    total_cols: int,
    missing_ratio: float,
    duplicate_count: int,
    trend_direction: str,
    volatility_label: str,
    volatility_cv: float,
    value_col: str,
    recent_mean: float,
    prior_mean: float,
    forecast_method: str,
    forecast_horizon: int,
    forecast_mean: float,
) -> Dict[str, Any]:
    """
    Generate a complete executive summary as a structured dict.

    Args: (see individual helpers above for parameter descriptions)

    Returns:
        {
            "quality_label":      str,
            "quality_sentence":   str,
            "trend_sentence":     str,
            "volatility_sentence":str,
            "forecast_sentence":  str,
            "overall_verdict":    str,  ← one-liner for the top of the summary card
        }
    """
    quality_label, quality_sentence = summarize_data_quality(
        total_rows, total_cols, missing_ratio, duplicate_count
    )
    trend_sentence = summarize_trend(
        trend_direction, value_col, recent_mean, prior_mean
    )
    volatility_sentence = summarize_volatility(
        volatility_label, value_col, volatility_cv
    )
    forecast_sentence = summarize_forecast(
        forecast_method, forecast_horizon, forecast_mean, recent_mean, value_col
    )

    # Overall verdict (top-level card)
    quality_emoji = {"strong": "✅", "moderate": "⚠️", "weak": "❌"}.get(quality_label, "")
    trend_emoji = {"upward": "📈", "downward": "📉", "flat": "➡️"}.get(trend_direction, "")
    volatility_emoji = {"low": "🟢", "moderate": "🟡", "high": "🔴"}.get(volatility_label, "")

    col_display = value_col.replace("_", " ").title()
    overall_verdict = (
        f"{quality_emoji} Data quality is **{quality_label}**.  "
        f"{trend_emoji} {col_display} is **{trend_direction}**.  "
        f"{volatility_emoji} Volatility is **{volatility_label}**."
    )

    return {
        "quality_label": quality_label,
        "quality_sentence": quality_sentence,
        "trend_sentence": trend_sentence,
        "volatility_sentence": volatility_sentence,
        "forecast_sentence": forecast_sentence,
        "overall_verdict": overall_verdict,
    }
