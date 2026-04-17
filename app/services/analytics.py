from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def summary_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "total_titles": 0,
            "average_rating": 0.0,
            "median_rating": 0.0,
            "total_votes": 0,
            "content_types": 0,
            "median_runtime": 0.0,
        }
    return {
        "total_titles": int(len(df)),
        "average_rating": round(float(df["averageRating"].mean()), 3),
        "median_rating": round(float(df["averageRating"].median()), 3),
        "total_votes": int(df["numVotes"].sum()),
        "content_types": int(df["titleType"].nunique()),
        "median_runtime": round(float(df["runtimeMinutes"].median()), 1),
    }


def rating_statistics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "mean": 0.0,
            "median": 0.0,
            "mode": 0.0,
            "std_dev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }
    mode_series = df["averageRating"].mode()
    return {
        "mean": round(float(df["averageRating"].mean()), 2),
        "median": round(float(df["averageRating"].median()), 2),
        "mode": round(float(mode_series.iloc[0]), 2) if not mode_series.empty else 0.0,
        "std_dev": round(float(df["averageRating"].std()), 2),
        "min": round(float(df["averageRating"].min()), 2),
        "max": round(float(df["averageRating"].max()), 2),
        "p25": round(float(df["averageRating"].quantile(0.25)), 2),
        "p50": round(float(df["averageRating"].quantile(0.50)), 2),
        "p75": round(float(df["averageRating"].quantile(0.75)), 2),
        "p90": round(float(df["averageRating"].quantile(0.90)), 2),
    }


def genre_breakdown(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    exploded = df.assign(genre=df["genres"].str.split(",")).explode("genre")
    exploded["genre"] = exploded["genre"].fillna("Unknown")
    summary = (
        exploded.groupby("genre")
        .agg(
            titles=("tconst", "count"),
            average_rating=("averageRating", "mean"),
            total_votes=("numVotes", "sum"),
            median_runtime=("runtimeMinutes", "median"),
        )
        .reset_index()
        .sort_values(["titles", "average_rating"], ascending=[False, False])
    )
    return summary.head(top_n)


def yearly_trends(df: pd.DataFrame) -> pd.DataFrame:
    trends = (
        df.groupby("startYear")
        .agg(
            titles=("tconst", "count"),
            average_rating=("averageRating", "mean"),
            median_votes=("numVotes", "median"),
        )
        .reset_index()
        .sort_values("startYear")
    )
    return trends


def decade_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    decade_df = df.copy()
    decade_df["decade"] = (decade_df["startYear"] // 10) * 10
    return (
        decade_df.groupby("decade")
        .agg(
            average_rating=("averageRating", "mean"),
            title_count=("tconst", "count"),
            average_votes=("numVotes", "mean"),
        )
        .reset_index()
        .sort_values("decade")
    )


def rating_vote_correlation(df: pd.DataFrame) -> float:
    if df.empty or df["numVotes"].nunique() <= 1:
        return 0.0
    return float(df["averageRating"].corr(np.log1p(df["numVotes"])))


def recommendation_insights(df: pd.DataFrame, min_votes: int = 20_000, top_n: int = 10) -> pd.DataFrame:
    curated = df[(df["numVotes"] >= min_votes) & (df["averageRating"] >= 7.5)].copy()
    curated["momentumScore"] = curated["averageRating"] * np.log1p(curated["numVotes"])
    return (
        curated.sort_values(["momentumScore", "averageRating"], ascending=False)
        .loc[:, ["tconst", "primaryTitle", "titleType", "startYear", "genres", "averageRating", "numVotes"]]
        .head(top_n)
        .reset_index(drop=True)
    )


def content_type_comparison(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("titleType")
        .agg(
            titles=("tconst", "count"),
            average_rating=("averageRating", "mean"),
            median_votes=("numVotes", "median"),
            total_votes=("numVotes", "sum"),
        )
        .reset_index()
        .sort_values("average_rating", ascending=False)
    )


def top_titles(df: pd.DataFrame, *, sort_by: str, limit: int = 10, min_votes: int | None = None) -> pd.DataFrame:
    target = df.copy()
    if min_votes is not None:
        target = target[target["numVotes"] >= min_votes]
    columns = ["primaryTitle", "averageRating", "numVotes", "titleType"]
    return target.nlargest(limit, sort_by)[columns].reset_index(drop=True)


def key_findings(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No records match the current filters, so there are no findings to summarize yet."]

    findings: list[str] = []
    findings.append(
        f"The current slice contains {len(df):,} titles with an average rating of {df['averageRating'].mean():.2f}."
    )
    top_type = (
        df.groupby("titleType")["averageRating"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    findings.append(f"{top_type} is the strongest-performing content type by average rating in the active filter context.")
    top_genre_df = genre_breakdown(df, top_n=50)
    eligible_genres = top_genre_df[top_genre_df["titles"] >= max(25, min(100, len(df) // 1000 or 25))]
    if not eligible_genres.empty:
        top_genre = eligible_genres.sort_values("average_rating", ascending=False).iloc[0]
        findings.append(
            f"{top_genre['genre']} leads genre quality with an average rating of {top_genre['average_rating']:.2f} "
            f"across {int(top_genre['titles'])} titles."
        )
    corr = rating_vote_correlation(df)
    findings.append(
        f"The rating-vote correlation is {corr:.3f}, suggesting {'a noticeable alignment' if abs(corr) > 0.3 else 'only a mild relationship'} between popularity and quality."
    )
    decade_df = decade_breakdown(df)
    if not decade_df.empty:
        best_decade = decade_df.sort_values("average_rating", ascending=False).iloc[0]
        findings.append(
            f"The {int(best_decade['decade'])}s currently stand out as the highest-rated decade at {best_decade['average_rating']:.2f}."
        )
    top_title = df.loc[df["averageRating"].idxmax()]
    findings.append(
        f"The highest-rated title in the filtered view is {top_title['primaryTitle']} ({top_title['averageRating']:.1f}) with {int(top_title['numVotes']):,} votes."
    )
    most_voted = df.loc[df["numVotes"].idxmax()]
    findings.append(
        f"The audience footprint is anchored by {most_voted['primaryTitle']}, which has the largest vote count at {int(most_voted['numVotes']):,}."
    )
    runtime_median = df["runtimeMinutes"].median()
    findings.append(
        f"The median runtime is {runtime_median:.0f} minutes, which provides a useful baseline for comparing long-form and short-form content."
    )
    return findings[:8]
