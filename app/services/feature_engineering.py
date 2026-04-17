from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SuccessDefinition:
    rating_threshold: float = 7.0
    votes_threshold: int = 25_000


def build_success_frame(df: pd.DataFrame, definition: SuccessDefinition) -> pd.DataFrame:
    modeling_df = df.copy()
    modeling_df["success"] = (
        (modeling_df["averageRating"] >= definition.rating_threshold)
        & (modeling_df["numVotes"] >= definition.votes_threshold)
    ).astype(int)
    modeling_df["genres"] = modeling_df["genres"].fillna("Unknown")
    modeling_df["titleType"] = modeling_df["titleType"].fillna("unknown")
    modeling_df["runtimeMinutes"] = modeling_df["runtimeMinutes"].fillna(
        modeling_df["runtimeMinutes"].median()
    )
    return modeling_df.reset_index(drop=True)
