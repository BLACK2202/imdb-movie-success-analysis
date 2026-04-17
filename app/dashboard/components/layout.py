from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.config.settings import settings


def apply_theme() -> None:
    css_path = settings.assets_dir / "dashboard.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(eyebrow: str, title: str, copy: str, icon: str = "") -> None:
    title_text = f"{icon} {title}" if icon else title
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-eyebrow">{eyebrow}</div>
            <div class="section-title">{title_text}</div>
            <div class="section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_callout(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="callout-card">
            <div class="callout-title">{title}</div>
            <div class="callout-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mini_stats(items: list[tuple[str, str]]) -> None:
    for index in range(0, len(items), 2):
        cols = st.columns(2)
        row_items = items[index : index + 2]
        for col, (label, value) in zip(cols, row_items):
            with col:
                st.markdown(
                    f"""
                    <div class="mini-stat">
                        <div class="mini-stat-label">{label}</div>
                        <div class="mini-stat-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
