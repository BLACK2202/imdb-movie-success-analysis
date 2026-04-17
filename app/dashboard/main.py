from __future__ import annotations

import pandas as pd
import streamlit as st

from app.dashboard.components.filters import render_sidebar_filters
from app.dashboard.components.layout import apply_theme, render_callout, render_hero, render_kpi
from app.dashboard.components.modeling import render_modeling
from app.dashboard.components.sections import (
    render_content_types,
    render_data_quality,
    render_explorer,
    render_genres,
    render_overview,
    render_pairwise_insight,
    render_popularity_quality,
    render_trends,
)
from app.services.analytics import summary_metrics
from app.services.data_loader import apply_filters, load_dataset


@st.cache_data(show_spinner=True)
def get_dashboard_dataset():
    return load_dataset()


def run_dashboard() -> None:
    st.set_page_config(
        page_title="IMDb Movie Success Analysis",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    render_hero(
        "IMDb Movie Success Intelligence Hub",
        "A modular analytics dashboard with reusable backend services, robust model comparison, "
        "and an end-to-end pipeline for exploring what drives movie success on IMDb.",
    )

    df = get_dashboard_dataset()
    if "live_rows" not in st.session_state:
        st.session_state.live_rows = pd.DataFrame()
    if not st.session_state.live_rows.empty:
        df = pd.concat([df, st.session_state.live_rows], ignore_index=True)

    filters = render_sidebar_filters(df)
    filtered_df = apply_filters(
        df,
        year_range=filters["year_range"],
        min_rating=filters["min_rating"],
        min_votes=filters["min_votes"],
        title_types=filters["selected_types"],
        genres=filters["selected_genres"],
        search=filters["search_term"],
    )
    if filters["highlight_title"]:
        filtered_df["highlight_state"] = "Normal"
        filtered_df.loc[
            filtered_df["primaryTitle"].str.contains(filters["highlight_title"], case=False, na=False),
            "highlight_state",
        ] = "Highlighted"

    metrics = summary_metrics(filtered_df)
    kpi_columns = st.columns(5)
    kpi_data = [
        ("Titles", f"{metrics['total_titles']:,}", "Records in the current analytical slice"),
        ("Avg Rating", f"{metrics['average_rating']:.2f}", "Mean IMDb score after filtering"),
        ("Median Rating", f"{metrics['median_rating']:.2f}", "Central tendency across selected titles"),
        ("Votes", f"{metrics['total_votes']:,}", "Audience footprint represented here"),
        ("Median Runtime", f"{metrics['median_runtime']:.0f} min", "Typical duration in the filtered view"),
    ]
    for column, (label, value, caption) in zip(kpi_columns, kpi_data):
        with column:
            render_kpi(label, value, caption)

    summary_col1, summary_col2 = st.columns([1.2, 1])
    with summary_col1:
        render_callout(
            "Executive Summary",
            f"Preset: {filters['preset_name']}. The current dashboard slice spans {filters['year_range'][0]} to {filters['year_range'][1]} "
            f"with at least {filters['min_votes']:,} votes and a minimum rating threshold of {filters['min_rating']:.1f}.",
        )
    with summary_col2:
        render_callout(
            "Model Context",
            f"The modeling lab is set to {filters['model_mode'].title()} mode with success defined as rating >= {filters['success_rating']:.1f} "
            f"and votes >= {filters['success_votes']:,}.",
        )

    tabs = st.tabs(
        [
            "📊 Overview",
            "🎭 Content Types",
            "⭐ Popularity vs Quality",
            "🕸️ Pairwise Insight",
            "🎬 Genre Intelligence",
            "📅 Trends",
            "🤖 Model Lab",
            "📋 Explorer",
            "🧪 Data Quality",
        ]
    )

    with tabs[0]:
        render_overview(filtered_df)
    with tabs[1]:
        render_content_types(filtered_df)
    with tabs[2]:
        render_popularity_quality(filtered_df)
    with tabs[3]:
        render_pairwise_insight(filtered_df)
    with tabs[4]:
        render_genres(filtered_df)
    with tabs[5]:
        render_trends(filtered_df)
    with tabs[6]:
        render_modeling(filtered_df, filters)
    with tabs[7]:
        render_explorer(filtered_df)
    with tabs[8]:
        render_data_quality(filtered_df)


if __name__ == "__main__":
    run_dashboard()
