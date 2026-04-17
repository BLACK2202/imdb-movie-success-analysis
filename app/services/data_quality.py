from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isnull().sum().values,
            "missing_pct": (df.isnull().sum().values / len(df) * 100).round(2) if len(df) else 0,
        }
    )
    return report.sort_values(["missing_pct", "missing_count"], ascending=False).reset_index(drop=True)


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    rows: list[dict[str, Any]] = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        rows.append(
            {
                "column": col,
                "outlier_count": int(len(outliers)),
                "outlier_pct": round(float(len(outliers) / len(series) * 100), 2),
                "lower_bound": round(float(lower), 2),
                "upper_bound": round(float(upper), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("outlier_pct", ascending=False).reset_index(drop=True)


def suspicious_patterns(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if "averageRating" in df.columns:
        zero_ratings = int((df["averageRating"] == 0).sum())
        if zero_ratings:
            issues.append(f"{zero_ratings:,} titles have a zero rating.")
        extreme_ratings = int((df["averageRating"] > 9.5).sum())
        if extreme_ratings:
            issues.append(f"{extreme_ratings:,} titles have ratings above 9.5.")
    if "numVotes" in df.columns:
        single_vote = int((df["numVotes"] == 1).sum())
        if single_vote:
            issues.append(f"{single_vote:,} titles have only one vote.")
    if "startYear" in df.columns:
        future_years = int((df["startYear"] > 2026).sum())
        if future_years:
            issues.append(f"{future_years:,} titles have future release years beyond 2026.")
        very_old = int((df["startYear"] < 1900).sum())
        if very_old:
            issues.append(f"{very_old:,} titles are dated before 1900.")
    if "genres" in df.columns:
        missing_genres = int(df["genres"].isnull().sum())
        if missing_genres:
            issues.append(f"{missing_genres:,} titles are missing genre values.")
    return issues


def quality_assessment(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "completeness_pct": 0.0,
            "outlier_ratio_pct": 0.0,
            "overall_quality": "NO DATA",
            "issues": ["No records available for quality analysis."],
        }
    miss = missing_values_report(df)
    outliers = outlier_report(df)
    completeness = (1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))) * 100
    outlier_ratio = 0.0 if outliers.empty else (outliers["outlier_count"].sum() / (len(df) * len(outliers))) * 100
    if completeness > 95 and outlier_ratio < 5:
        quality = "EXCELLENT"
    elif completeness > 90 and outlier_ratio < 10:
        quality = "GOOD"
    elif completeness > 80 and outlier_ratio < 15:
        quality = "ACCEPTABLE"
    else:
        quality = "NEEDS ATTENTION"
    recommendations = []
    issues = suspicious_patterns(df)
    if any("one vote" in issue for issue in issues):
        recommendations.append("Consider filtering out titles with very low vote counts for more reliable ratings analysis.")
    if any("future release years" in issue for issue in issues):
        recommendations.append("Validate rows with implausible release years.")
    if any("missing genre" in issue for issue in issues):
        recommendations.append("Fill or exclude missing genres before genre-specific analysis.")
    return {
        "completeness_pct": round(float(completeness), 2),
        "outlier_ratio_pct": round(float(outlier_ratio), 2),
        "overall_quality": quality,
        "issues": issues,
        "recommendations": recommendations or ["The dataset looks clean under the current checks."],
        "missing_values": miss.to_dict(orient="records"),
        "outliers": outliers.to_dict(orient="records"),
    }


def distribution_summary(df: pd.DataFrame, column: str) -> dict[str, float]:
    series = df[column].dropna()
    return {
        "mean": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "skewness": round(float(series.skew()), 2),
        "kurtosis": round(float(series.kurtosis()), 2),
    }
