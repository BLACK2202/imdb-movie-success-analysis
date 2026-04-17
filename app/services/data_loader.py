from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config.settings import settings

BASICS_COLS = [
    "tconst",
    "titleType",
    "primaryTitle",
    "originalTitle",
    "isAdult",
    "startYear",
    "runtimeMinutes",
    "genres",
]
RATINGS_COLS = ["tconst", "averageRating", "numVotes"]


def resolve_data_path(filename: str) -> Path:
    """Resolve a raw IMDb file from common project locations."""

    candidates = [
        settings.raw_data_dir / filename,
        settings.project_root / filename,
        settings.data_dir / filename,
        settings.project_root / "notebooks" / filename,
    ]
    gz_candidates = [Path(f"{path}.gz") for path in candidates]
    for candidate in [*candidates, *gz_candidates]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {filename}. Place it in {settings.raw_data_dir} or the project root."
    )


def read_tsv(path: Path, usecols: list[str], chunksize: int | None = None):
    return pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
        na_values=["\\N"],
        keep_default_na=True,
        on_bad_lines="skip",
    )


def _load_filtered_basics(chunksize: int = settings.default_chunksize) -> pd.DataFrame:
    basics_path = resolve_data_path(settings.basics_filename)
    chunks: list[pd.DataFrame] = []
    reader = read_tsv(basics_path, BASICS_COLS, chunksize=chunksize)
    for chunk in reader:
        filtered = chunk[chunk["titleType"].isin(settings.target_title_types)].copy()
        if filtered.empty:
            continue
        filtered["startYear"] = pd.to_numeric(filtered["startYear"], errors="coerce")
        filtered["runtimeMinutes"] = pd.to_numeric(filtered["runtimeMinutes"], errors="coerce")
        filtered["isAdult"] = pd.to_numeric(filtered["isAdult"], errors="coerce").fillna(0).astype(int)
        chunks.append(filtered)
    if not chunks:
        return pd.DataFrame(columns=BASICS_COLS)
    return pd.concat(chunks, ignore_index=True)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["startYear"] = pd.to_numeric(cleaned["startYear"], errors="coerce")
    cleaned["runtimeMinutes"] = pd.to_numeric(cleaned["runtimeMinutes"], errors="coerce")
    cleaned["averageRating"] = pd.to_numeric(cleaned["averageRating"], errors="coerce")
    cleaned["numVotes"] = pd.to_numeric(cleaned["numVotes"], errors="coerce")

    cleaned = cleaned.dropna(subset=["startYear", "averageRating", "numVotes"]).copy()
    cleaned["startYear"] = cleaned["startYear"].astype(int)
    cleaned["numVotes"] = cleaned["numVotes"].astype(int)
    cleaned["runtimeMinutes"] = cleaned["runtimeMinutes"].fillna(cleaned["runtimeMinutes"].median())
    cleaned["titleType"] = cleaned["titleType"].fillna("unknown")
    cleaned["genres"] = cleaned["genres"].fillna("Unknown")
    cleaned["primaryTitle"] = cleaned["primaryTitle"].fillna("Unknown Title")
    cleaned["originalTitle"] = cleaned["originalTitle"].fillna(cleaned["primaryTitle"])
    cleaned["isAdult"] = cleaned["isAdult"].fillna(0).astype(int)
    cleaned["logVotes"] = np.log1p(cleaned["numVotes"])
    cleaned["success_score"] = cleaned["averageRating"] * np.log10(cleaned["numVotes"] + 1)
    cleaned["decade"] = (cleaned["startYear"] // 10) * 10
    cleaned["mainGenre"] = cleaned["genres"].astype(str).str.split(",").str[0].fillna("Unknown")
    cleaned["ratingBucket"] = pd.cut(
        cleaned["averageRating"],
        bins=[0, 5, 6, 7, 8, 10],
        labels=["Low", "Below Avg", "Average", "Strong", "Excellent"],
        include_lowest=True,
    ).astype(str)
    cleaned["voteBucket"] = pd.cut(
        cleaned["numVotes"],
        bins=[0, 1_000, 10_000, 100_000, 1_000_000, np.inf],
        labels=["0-1k", "1k-10k", "10k-100k", "100k-1M", "1M+"],
        include_lowest=True,
    ).astype(str)
    return cleaned.reset_index(drop=True)


def ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill derived columns for older cached datasets."""

    enriched = df.copy()
    if "logVotes" not in enriched.columns and "numVotes" in enriched.columns:
        enriched["logVotes"] = np.log1p(pd.to_numeric(enriched["numVotes"], errors="coerce").fillna(0))
    if "success_score" not in enriched.columns and {"averageRating", "numVotes"}.issubset(enriched.columns):
        ratings = pd.to_numeric(enriched["averageRating"], errors="coerce").fillna(0)
        votes = pd.to_numeric(enriched["numVotes"], errors="coerce").fillna(0)
        enriched["success_score"] = ratings * np.log10(votes + 1)
    if "decade" not in enriched.columns and "startYear" in enriched.columns:
        years = pd.to_numeric(enriched["startYear"], errors="coerce").fillna(0).astype(int)
        enriched["decade"] = (years // 10) * 10
    if "mainGenre" not in enriched.columns and "genres" in enriched.columns:
        enriched["mainGenre"] = enriched["genres"].astype(str).str.split(",").str[0].fillna("Unknown")
    if "ratingBucket" not in enriched.columns and "averageRating" in enriched.columns:
        enriched["ratingBucket"] = pd.cut(
            pd.to_numeric(enriched["averageRating"], errors="coerce").fillna(0),
            bins=[0, 5, 6, 7, 8, 10],
            labels=["Low", "Below Avg", "Average", "Strong", "Excellent"],
            include_lowest=True,
        ).astype(str)
    if "voteBucket" not in enriched.columns and "numVotes" in enriched.columns:
        enriched["voteBucket"] = pd.cut(
            pd.to_numeric(enriched["numVotes"], errors="coerce").fillna(0),
            bins=[0, 1_000, 10_000, 100_000, 1_000_000, np.inf],
            labels=["0-1k", "1k-10k", "10k-100k", "100k-1M", "1M+"],
            include_lowest=True,
        ).astype(str)
    return enriched


def prepare_dataset(chunksize: int = settings.default_chunksize, force: bool = False) -> pd.DataFrame:
    """Create the cleaned, analysis-ready dataset and persist it."""

    settings.ensure_directories()
    if not force:
        cached = load_prepared_dataset()
        if cached is not None:
            return cached

    basics = _load_filtered_basics(chunksize=chunksize)
    ratings_path = resolve_data_path(settings.ratings_filename)
    ratings = read_tsv(ratings_path, RATINGS_COLS, chunksize=None)
    merged = basics.merge(ratings, on="tconst", how="inner")
    cleaned = clean_dataset(merged)
    persist_prepared_dataset(cleaned)
    load_dataset.cache_clear()
    return cleaned


def persist_prepared_dataset(df: pd.DataFrame) -> Path:
    metadata_path = settings.processed_data_dir / "dataset_metadata.json"
    try:
        df.to_parquet(settings.prepared_parquet_path, index=False)
        persisted_path = settings.prepared_parquet_path
    except Exception:
        df.to_pickle(settings.prepared_pickle_path)
        persisted_path = settings.prepared_pickle_path

    metadata = {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "path": str(persisted_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return persisted_path


def load_prepared_dataset() -> pd.DataFrame | None:
    if settings.prepared_parquet_path.exists():
        try:
            return ensure_derived_columns(pd.read_parquet(settings.prepared_parquet_path))
        except Exception:
            pass
    if settings.prepared_pickle_path.exists():
        return ensure_derived_columns(pd.read_pickle(settings.prepared_pickle_path))
    return None


@lru_cache(maxsize=1)
def load_dataset(force_refresh: bool = False) -> pd.DataFrame:
    if force_refresh:
        return prepare_dataset(force=True)
    prepared = load_prepared_dataset()
    if prepared is not None:
        return prepared
    return prepare_dataset(force=True)


def apply_filters(
    df: pd.DataFrame,
    year_range: tuple[int, int] | None = None,
    min_rating: float | None = None,
    min_votes: int | None = None,
    title_types: list[str] | None = None,
    genres: list[str] | None = None,
    search: str | None = None,
) -> pd.DataFrame:
    filtered = df.copy()
    if year_range:
        filtered = filtered[
            (filtered["startYear"] >= year_range[0]) & (filtered["startYear"] <= year_range[1])
        ]
    if min_rating is not None:
        filtered = filtered[filtered["averageRating"] >= min_rating]
    if min_votes is not None:
        filtered = filtered[filtered["numVotes"] >= min_votes]
    if title_types:
        filtered = filtered[filtered["titleType"].isin(title_types)]
    if genres:
        genre_pattern = "|".join(map(str, genres))
        filtered = filtered[filtered["genres"].str.contains(genre_pattern, case=False, na=False)]
    if search:
        mask = (
            filtered["primaryTitle"].str.contains(search, case=False, na=False)
            | filtered["originalTitle"].str.contains(search, case=False, na=False)
            | filtered["tconst"].str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]
    return filtered.reset_index(drop=True)


def dataframe_to_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    target = df.head(limit) if limit else df
    return target.replace({np.nan: None}).to_dict(orient="records")


def prepare_live_rows(new_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["tconst", "titleType", "startYear", "genres", "primaryTitle", "averageRating", "numVotes"]
    missing_cols = [col for col in required_cols if col not in new_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    live_df = new_df[required_cols].copy()
    live_df["tconst"] = live_df["tconst"].astype(str).str.strip()
    live_df["titleType"] = live_df["titleType"].astype(str).str.strip()
    live_df["primaryTitle"] = live_df["primaryTitle"].astype(str).str.strip()
    live_df["genres"] = live_df["genres"].fillna("Unknown").astype(str)
    live_df["startYear"] = pd.to_numeric(live_df["startYear"], errors="coerce")
    live_df["averageRating"] = pd.to_numeric(live_df["averageRating"], errors="coerce")
    live_df["numVotes"] = pd.to_numeric(live_df["numVotes"], errors="coerce")
    live_df = live_df.dropna(subset=["tconst", "titleType", "primaryTitle", "startYear", "averageRating", "numVotes"])
    live_df = live_df[live_df["numVotes"] >= 0].copy()
    live_df["startYear"] = live_df["startYear"].astype(int)
    live_df["numVotes"] = live_df["numVotes"].astype(int)
    live_df["runtimeMinutes"] = np.nan
    live_df["originalTitle"] = live_df["primaryTitle"]
    live_df["isAdult"] = 0
    return clean_dataset(live_df)
