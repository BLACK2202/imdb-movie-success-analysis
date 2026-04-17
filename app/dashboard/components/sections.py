from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.dashboard.components.layout import render_callout, render_mini_stats, render_section_intro
from app.services.analytics import (
    content_type_comparison,
    decade_breakdown,
    genre_breakdown,
    key_findings,
    rating_vote_correlation,
    rating_statistics,
    recommendation_insights,
    top_titles,
    yearly_trends,
)
from app.services.data_quality import distribution_summary, quality_assessment


def style_figure(fig, height: int, legend: bool = True):
    fig.update_layout(
        height=height,
        showlegend=legend,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        font=dict(color="#e5edf7"),
        title_font=dict(size=20, color="#f8fafc"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        zeroline=False,
        linecolor="rgba(148,163,184,0.16)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        zeroline=False,
        linecolor="rgba(148,163,184,0.16)",
    )
    return fig


def render_overview(filtered_df: pd.DataFrame) -> None:
    render_section_intro(
        "Overview",
        "Rating Distribution",
        "This opening view frames the shape of IMDb ratings before we drill into type, popularity, "
        "genre, and model behavior.",
        icon="📊",
    )
    stats_line = distribution_summary(filtered_df, "averageRating")
    render_callout(
        "Distribution Summary",
        f"Mean {stats_line['mean']:.2f}, median {stats_line['median']:.2f}, skewness {stats_line['skewness']:.2f}, kurtosis {stats_line['kurtosis']:.2f}.",
    )
    st.subheader("Key Findings")
    for finding in key_findings(filtered_df)[:6]:
        render_callout("Insight", finding)

    col1, col2 = st.columns([1.4, 1])
    with col1:
        hist = px.histogram(
            filtered_df,
            x="averageRating",
            nbins=30,
            title="Distribution of IMDb Ratings",
            color_discrete_sequence=["#3b82f6"],
        )
        hist.update_layout(bargap=0.05)
        style_figure(hist, 430, legend=False)
        if not filtered_df.empty:
            peak_bin = filtered_df["averageRating"].round(1).value_counts().sort_index().idxmax()
            hist.add_annotation(
                x=peak_bin,
                y=max(filtered_df["averageRating"].round(1).value_counts().tolist()),
                text=f"Peak around {peak_bin:.1f}",
                showarrow=True,
                arrowcolor="#f59e0b",
                font=dict(color="#f8fafc"),
            )
        st.plotly_chart(hist, use_container_width=True)

    with col2:
        stats = rating_statistics(filtered_df)
        render_mini_stats(
            [
                ("Mean", f"{stats['mean']:.2f}"),
                ("Median", f"{stats['median']:.2f}"),
                ("Mode", f"{stats['mode']:.2f}"),
                ("Std Dev", f"{stats['std_dev']:.2f}"),
                ("25th %", f"{stats['p25']:.2f}"),
                ("75th %", f"{stats['p75']:.2f}"),
            ]
        )
        render_callout(
            "Reading The Shape",
            f"Ratings span from {stats['min']:.2f} to {stats['max']:.2f}, with the 90th percentile at "
            f"{stats['p90']:.2f}. That gives us a quick benchmark for what counts as exceptional.",
        )


def render_content_types(filtered_df: pd.DataFrame) -> None:
    render_section_intro(
        "Formats",
        "Rating Distribution by Content Type",
        "This tab keeps the original content-type comparison intact, but presents it with clearer hierarchy "
        "and cleaner visual balance.",
        icon="🎭",
    )

    content_df = content_type_comparison(filtered_df)
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.box(
            filtered_df,
            x="titleType",
            y="averageRating",
            color="titleType",
            title="Rating Distribution by Content Type",
            labels={"titleType": "Content Type", "averageRating": "Average Rating"},
        )
        style_figure(fig, 500, legend=False)
        fig.update_xaxes(tickangle=45)
        if not content_df.empty:
            best = content_df.iloc[0]
            fig.add_annotation(
                x=best["titleType"],
                y=best["average_rating"],
                text="Best avg rating",
                showarrow=True,
                arrowcolor="#f59e0b",
                font=dict(color="#f8fafc"),
            )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        render_callout(
            "Content Type Lens",
            "Use this section to compare whether long-form series, feature films, and other formats differ "
            "in rating consistency and audience footprint.",
        )
        display_df = content_df.rename(
            columns={
                "average_rating": "Avg Rating",
                "total_votes": "Total Votes",
                "titles": "Count",
                "median_votes": "Median Votes",
            }
        ).set_index("titleType")
        st.dataframe(display_df.round(2), use_container_width=True)

        pie = px.pie(
            filtered_df,
            names="titleType",
            title="Distribution of Content Types",
            hole=0.52,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        style_figure(pie, 300, legend=True)
        st.plotly_chart(pie, use_container_width=True)


def render_popularity_quality(filtered_df: pd.DataFrame) -> None:
    render_section_intro(
        "Signals",
        "Popularity vs. Quality Analysis",
        "This view keeps the core popularity-quality relationship from the original dashboard while making "
        "the message and supporting tables easier to read at a glance.",
        icon="⭐",
    )

    scatter = px.scatter(
        filtered_df.sample(min(10000, len(filtered_df)), random_state=42),
        x="numVotes",
        y="averageRating",
        color="highlight_state" if "highlight_state" in filtered_df.columns else "titleType",
        opacity=0.6,
        hover_name="primaryTitle",
        title=f"Popularity (Votes) vs. Quality (Rating) - Sample of {min(10000, len(filtered_df)):,} titles",
        log_x=True,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    style_figure(scatter, 600, legend=True)
    st.plotly_chart(scatter, use_container_width=True)

    correlation = rating_vote_correlation(filtered_df)
    render_callout("Correlation Analysis", f"Correlation between votes and rating: {correlation:.3f}")
    if correlation > 0.3:
        st.success("Positive correlation: more popular titles tend to have higher ratings.")
    elif correlation < -0.3:
        st.error("Negative correlation: more popular titles tend to have lower ratings.")
    else:
        st.info("Weak correlation: popularity and rating are relatively independent.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Most Popular Titles")
        st.dataframe(top_titles(filtered_df, sort_by="numVotes"), use_container_width=True)
    with col2:
        st.subheader("Highest Rated Titles (min 1000 votes)")
        st.dataframe(
            top_titles(filtered_df, sort_by="averageRating", min_votes=1000),
            use_container_width=True,
        )

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Ultimate Winners (Success Index)")
        st.dataframe(
            filtered_df.nlargest(10, "success_score")[["primaryTitle", "averageRating", "numVotes", "success_score"]],
            use_container_width=True,
        )
    with col4:
        st.subheader("Hidden Gems")
        gems = filtered_df[(filtered_df["averageRating"] >= 8.5) & (filtered_df["numVotes"] < 5000)]
        st.dataframe(gems[["primaryTitle", "averageRating", "numVotes"]].head(10), use_container_width=True)


def render_genres(filtered_df: pd.DataFrame) -> None:
    render_section_intro(
        "Genres",
        "Genre Performance Analysis",
        "This tab preserves the original genre ranking and comparison views, but gives them a more curated, "
        "editorial look for presentation and portfolio use.",
        icon="🎬",
    )
    min_titles = st.slider("Minimum titles per genre", 10, 1000, 100)
    top_n = st.slider("Number of top genres to display", 5, 30, 20)
    genre_df = genre_breakdown(filtered_df, top_n=200)
    genre_df = genre_df[genre_df["titles"] >= min_titles].sort_values("average_rating", ascending=False)
    col1, col2 = st.columns([1, 1])
    with col1:
        heat = px.bar(
            genre_df.head(top_n).sort_values("average_rating", ascending=True),
            x="average_rating",
            y="genre",
            orientation="h",
            title=f"Top {top_n} Genres by Average Rating (min. {min_titles} titles)",
            labels={"average_rating": "Average Rating", "genre": "Genre"},
            color="average_rating",
            color_continuous_scale="Viridis",
        )
        style_figure(heat, 600, legend=False)
        if not genre_df.empty:
            top_row = genre_df.iloc[0]
            heat.add_annotation(
                x=top_row["average_rating"],
                y=top_row["genre"],
                text="Top genre",
                showarrow=True,
                arrowcolor="#f59e0b",
                font=dict(color="#f8fafc"),
            )
        st.plotly_chart(heat, use_container_width=True)

    with col2:
        render_callout(
            "Genre Lens",
            "Balance rating quality with production volume. High-rated genres with tiny sample sizes can be "
            "less reliable than consistently strong large categories.",
        )
        display_df = genre_df.head(10)[["genre", "average_rating", "titles", "total_votes"]].copy()
        display_df.columns = ["Genre", "Avg Rating", "Count", "Total Votes"]
        st.dataframe(display_df.round(2), use_container_width=True)

        st.subheader("Most Produced Genres")
        produced_df = genre_df.nlargest(10, "titles")[["genre", "titles", "average_rating"]].copy()
        produced_df.columns = ["Genre", "Count", "Avg Rating"]
        st.dataframe(produced_df.round(2), use_container_width=True)

    st.subheader("Genre Comparison Matrix")
    selected_genres = st.multiselect(
        "Select genres to compare",
        options=genre_df["genre"].tolist(),
        default=genre_df.head(5)["genre"].tolist(),
    )

    if selected_genres:
        comparison_df = genre_df[genre_df["genre"].isin(selected_genres)]
        bubble = px.scatter(
            comparison_df,
            x="titles",
            y="average_rating",
            size="total_votes",
            color="average_rating",
            text="genre",
            title="Genre Performance Matrix",
            color_continuous_scale="Viridis",
        )
        bubble.update_traces(textposition="top center")
        bubble.update_layout(xaxis_title="Number of Titles", yaxis_title="Average Rating")
        style_figure(bubble, 520, legend=False)
        st.plotly_chart(bubble, use_container_width=True)

    st.dataframe(genre_df.round(2), use_container_width=True, height=320)


def render_trends(filtered_df: pd.DataFrame) -> None:
    render_section_intro(
        "Time",
        "Success Over Time",
        "This section keeps the original temporal analysis but presents the long-term trend story in a more "
        "polished way, with the decade table and recommendation-style insights still available below.",
        icon="📅",
    )
    trend_df = yearly_trends(filtered_df)
    line = go.Figure()
    line.add_trace(
        go.Scatter(
            x=trend_df["startYear"],
            y=trend_df["average_rating"],
            mode="lines+markers",
            name="Average rating",
            line=dict(color="#a855f7", width=2),
            marker=dict(size=4),
        )
    )
    line.add_trace(
        go.Scatter(
            x=trend_df["startYear"],
            y=trend_df["titles"],
            mode="lines",
            name="Title count",
            line=dict(color="#f59e0b", width=2),
            fill="tozeroy",
            yaxis="y2",
        )
    )
    line.update_layout(
        title="Average Rating Over Time and Titles Released Per Year",
        yaxis=dict(title="Average rating"),
        yaxis2=dict(title="Titles released", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    style_figure(line, 800, legend=True)
    if not trend_df.empty:
        best_year = trend_df.sort_values("average_rating", ascending=False).iloc[0]
        line.add_annotation(
            x=best_year["startYear"],
            y=best_year["average_rating"],
            text=f"Peak rating year: {int(best_year['startYear'])}",
            showarrow=True,
            arrowcolor="#f59e0b",
            font=dict(color="#f8fafc"),
        )
    st.plotly_chart(line, use_container_width=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Analysis by Decade")
        st.dataframe(decade_breakdown(filtered_df).round(2), use_container_width=True)
    with col2:
        render_callout(
            "Temporal Reading",
            "Use the top line to judge quality drift and the lower fill to see whether content growth changed "
            "faster than audience-perceived quality.",
        )

    recommendations = recommendation_insights(filtered_df)
    st.subheader("High-Momentum Recommendation-Like Picks")
    st.dataframe(recommendations, use_container_width=True, height=300)


def render_pairwise_insight(filtered_df: pd.DataFrame) -> None:
    render_section_intro(
        "Relationships",
        "Pairwise Insight",
        "This interactive scatter matrix brings back the pairwise relationship view from the exploratory dashboard and lets you inspect how year, rating, votes, and success score move together.",
        icon="🕸️",
    )
    matrix_sample = filtered_df.sample(n=min(2000, len(filtered_df)), random_state=42)
    fig_matrix = px.scatter_matrix(
        matrix_sample,
        dimensions=["startYear", "averageRating", "numVotes", "success_score"],
        color="titleType",
        title="Interactive Pair Plots",
        hover_data=["primaryTitle"],
        opacity=0.4,
    )
    fig_matrix.update_layout(height=800)
    st.plotly_chart(fig_matrix, use_container_width=True)
    st.info("Hint: select a content type in the legend to focus the matrix on one format.")


def render_data_quality(filtered_df: pd.DataFrame) -> None:
    render_section_intro(
        "Quality",
        "Data Quality Analysis",
        "This tab turns the standalone data quality script into a reusable dashboard feature, covering missingness, outliers, suspicious patterns, and a simple overall quality score.",
        icon="🧪",
    )
    quality = quality_assessment(filtered_df)
    render_mini_stats(
        [
            ("Completeness", f"{quality['completeness_pct']:.2f}%"),
            ("Outlier Ratio", f"{quality['outlier_ratio_pct']:.2f}%"),
            ("Quality", quality["overall_quality"]),
            ("Rows", f"{len(filtered_df):,}"),
        ]
    )
    for issue in quality["issues"][:5] or ["No major suspicious patterns detected."]:
        render_callout("Quality Signal", issue)
    for recommendation in quality["recommendations"][:4]:
        render_callout("Recommendation", recommendation)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Missing Values")
        st.dataframe(pd.DataFrame(quality["missing_values"]), use_container_width=True, height=280)
    with col2:
        st.subheader("Outlier Report")
        st.dataframe(pd.DataFrame(quality["outliers"]), use_container_width=True, height=280)


def render_explorer(filtered_df: pd.DataFrame) -> None:
    render_section_intro(
        "Explorer",
        "Data Explorer",
        "The explorer keeps the original browsing and export behavior, but sits inside the same visual system "
        "as the analysis tabs so the project feels complete end to end.",
        icon="📋",
    )
    show_rows = st.slider("Rows to display", 10, 500, 100)
    sort_by = st.selectbox(
        "Sort by",
        ["averageRating", "numVotes", "startYear", "runtimeMinutes", "primaryTitle"],
    )
    sort_order = st.radio("Sort order", ["Descending", "Ascending"], horizontal=True)
    display_df = filtered_df.sort_values(sort_by, ascending=sort_order == "Ascending").head(show_rows)
    st.dataframe(display_df, use_container_width=True, height=520)
    st.download_button(
        "Download filtered data as CSV",
        filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="imdb_filtered_data.csv",
        mime="text/csv",
    )
    render_callout(
        "Export",
        "Use the current sidebar filters and sorting controls to curate a slice of the IMDb data, then export "
        "it for reporting or offline analysis.",
    )
    st.subheader("Data Summary")
    st.write(filtered_df.describe(include="all"))
