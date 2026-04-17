from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.data_loader import prepare_live_rows


DEFAULT_PRESETS = {
    "Balanced View": {
        "min_rating": 0.0,
        "min_votes": 500,
        "success_rating": 7.0,
        "success_votes": 25000,
        "model_mode": "Fast",
    },
    "Prestige Titles": {
        "min_rating": 7.0,
        "min_votes": 10000,
        "success_rating": 7.5,
        "success_votes": 50000,
        "model_mode": "Full",
    },
    "Audience Favorites": {
        "min_rating": 6.5,
        "min_votes": 50000,
        "success_rating": 7.2,
        "success_votes": 100000,
        "model_mode": "Fast",
    },
}


def render_sidebar_filters(df):
    with st.sidebar:
        st.header("Control Room")
        st.caption("Slice the dataset, adjust the success definition, and export what matters.")
        preset_names = ["Custom", *DEFAULT_PRESETS.keys(), *st.session_state.get("saved_presets", {}).keys()]
        selected_preset = st.selectbox("Saved filter preset", preset_names)
        preset = DEFAULT_PRESETS.get(selected_preset) or st.session_state.get("saved_presets", {}).get(selected_preset, {})

        year_bounds = (int(df["startYear"].min()), int(df["startYear"].max()))
        year_range = st.slider("Release year range", year_bounds[0], year_bounds[1], year_bounds)
        min_rating = st.slider("Minimum IMDb rating", 0.0, 10.0, float(preset.get("min_rating", 0.0)), 0.1)
        min_votes = st.number_input("Minimum votes", min_value=0, value=int(preset.get("min_votes", 500)), step=500)
        selected_types = st.multiselect(
            "Title types",
            options=sorted(df["titleType"].dropna().unique().tolist()),
            default=sorted(df["titleType"].dropna().unique().tolist()),
        )
        genre_options = sorted(
            {
                genre
                for value in df["genres"].dropna().astype(str)
                for genre in value.split(",")
                if genre
            }
        )
        selected_genres = st.multiselect("Focus genres", options=genre_options, default=[])
        search_term = st.text_input("Movie search", placeholder="Search title or IMDb id")
        highlight_title = st.text_input("Highlight title", placeholder="e.g. The Shawshank Redemption")

        st.markdown("---")
        st.subheader("Live Data Ingestion")
        uploaded_file = st.file_uploader(
            "Upload new rows (CSV or TSV)",
            type=["csv", "tsv", "txt"],
            help="Required columns: tconst, titleType, startYear, genres, primaryTitle, averageRating, numVotes",
        )

        if uploaded_file is not None:
            try:
                separator = "\t" if uploaded_file.name.lower().endswith((".tsv", ".txt")) else ","
                incoming_df = pd.read_csv(uploaded_file, sep=separator)
                st.caption(f"Detected {len(incoming_df):,} rows in upload")
                st.dataframe(incoming_df.head(3), use_container_width=True)

                if st.button("Add Uploaded Rows", use_container_width=True):
                    prepared_rows = prepare_live_rows(incoming_df)
                    existing_ids = set(df["tconst"].astype(str))
                    if "live_rows" in st.session_state and not st.session_state.live_rows.empty:
                        existing_ids |= set(st.session_state.live_rows["tconst"].astype(str))
                    prepared_rows = prepared_rows[~prepared_rows["tconst"].astype(str).isin(existing_ids)]
                    if prepared_rows.empty:
                        st.warning("No new unique rows were added.")
                    else:
                        st.session_state.live_rows = pd.concat(
                            [st.session_state.get("live_rows", pd.DataFrame()), prepared_rows],
                            ignore_index=True,
                        )
                        st.success(f"Added {len(prepared_rows):,} new rows to the live dashboard.")
                        st.rerun()
            except Exception as exc:
                st.error(f"Upload could not be processed: {exc}")

        with st.expander("Add One Title Manually"):
            with st.form("manual_live_row", clear_on_submit=True):
                m_tconst = st.text_input("tconst", value="tt_new_001")
                m_primary = st.text_input("primaryTitle", value="New Title")
                m_type = st.selectbox("titleType", ["movie", "tvSeries", "tvMiniSeries", "tvMovie", "tvSpecial", "video"])
                m_year = st.number_input("startYear", min_value=1900, max_value=2035, value=2025)
                m_genres = st.text_input("genres", value="Drama")
                m_rating = st.slider("averageRating", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
                m_votes = st.number_input("numVotes", min_value=0, value=100)
                add_manual = st.form_submit_button("Add Manual Row")

            if add_manual:
                try:
                    manual_df = pd.DataFrame(
                        [
                            {
                                "tconst": m_tconst,
                                "titleType": m_type,
                                "startYear": m_year,
                                "genres": m_genres,
                                "primaryTitle": m_primary,
                                "averageRating": m_rating,
                                "numVotes": m_votes,
                            }
                        ]
                    )
                    prepared_manual = prepare_live_rows(manual_df)
                    existing_ids = set(df["tconst"].astype(str))
                    if "live_rows" in st.session_state and not st.session_state.live_rows.empty:
                        existing_ids |= set(st.session_state.live_rows["tconst"].astype(str))
                    prepared_manual = prepared_manual[~prepared_manual["tconst"].astype(str).isin(existing_ids)]
                    if prepared_manual.empty:
                        st.warning("This tconst already exists or the row is invalid.")
                    else:
                        st.session_state.live_rows = pd.concat(
                            [st.session_state.get("live_rows", pd.DataFrame()), prepared_manual],
                            ignore_index=True,
                        )
                        st.success("Manual row added to the live dashboard.")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Manual row could not be added: {exc}")

        if "live_rows" in st.session_state and not st.session_state.live_rows.empty:
            st.info(f"Live rows currently active: {len(st.session_state.live_rows):,}")
            if st.button("Clear Live Rows", use_container_width=True):
                st.session_state.live_rows = pd.DataFrame()
                st.rerun()

        st.markdown("---")
        st.subheader("Model Settings")
        model_mode = st.radio("Model mode", ["Fast", "Full"], index=0 if preset.get("model_mode", "Fast") == "Fast" else 1, horizontal=True)
        success_rating = st.slider("Success rating threshold", 5.0, 9.5, float(preset.get("success_rating", 7.0)), 0.1)
        success_votes = st.number_input(
            "Success votes threshold",
            min_value=1_000,
            value=int(preset.get("success_votes", 25_000)),
            step=1_000,
        )
        random_seed = st.number_input("Random seed", min_value=0, max_value=9_999, value=42, step=1)
        new_preset_name = st.text_input("Save current preset as", placeholder="e.g. Awards Season")
        if st.button("Save preset", use_container_width=True) and new_preset_name.strip():
            saved = st.session_state.get("saved_presets", {})
            saved[new_preset_name.strip()] = {
                "min_rating": float(min_rating),
                "min_votes": int(min_votes),
                "success_rating": float(success_rating),
                "success_votes": int(success_votes),
                "model_mode": model_mode,
            }
            st.session_state["saved_presets"] = saved
            st.success(f"Saved preset: {new_preset_name.strip()}")

    return {
        "year_range": year_range,
        "min_rating": min_rating,
        "min_votes": int(min_votes),
        "selected_types": selected_types,
        "selected_genres": selected_genres,
        "search_term": search_term,
        "highlight_title": highlight_title,
        "success_rating": success_rating,
        "success_votes": int(success_votes),
        "random_seed": int(random_seed),
        "model_mode": model_mode.lower(),
        "preset_name": selected_preset,
    }
