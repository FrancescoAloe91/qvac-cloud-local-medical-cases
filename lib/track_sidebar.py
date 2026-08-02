"""Shared left-rail shell for Comprehension home and Structured page.

Order: Tracks → (caller OpenRouter/QVAC blocks) → Guides → protocol chip.
Structured may append History / STOP; both pin Run clock last.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

import streamlit as st

from lib.guide_overlays import sidebar_guides_block_html


def render_tracks_block(*, active: str = "comprehension") -> None:
    """Track links — Comprehension = home/default; Structured = optional secondary.

    Compact single-row pills (yellow / orange) styled in ``dashboard.css``.
    ``data-active`` drives the thick active border via portal JS.
    """
    st.markdown("**Tracks**")
    active_norm = "structured" if active == "structured" else "comprehension"
    st.markdown(
        f'<div id="qvac-track-active" data-active="{html.escape(active_norm)}" hidden></div>',
        unsafe_allow_html=True,
    )
    # Short labels so pills stay one line in the ~220–280px sidebar.
    st.page_link("app.py", label="Comprehension", icon="🏠")
    st.page_link(
        "pages/structured_graded.py",
        label="Structured A1–A5",
        icon="📋",
        help="Secondary rigid slot Q&A track · separate History / Rebuild · never pool KPIs",
    )
    if active_norm == "comprehension":
        st.caption(
            "Default track · discursive free-form. "
            "Structured is optional · KPIs never pool across tracks."
        )
        legacy = Path("pages/comprehension_redirect.py")
        if legacy.is_file():
            st.caption("Legacy redirect page kept for old bookmarks — prefer Home.")
    else:
        st.caption(
            "This page = optional Structured A1–A5 · rigid slots · graded History. "
            "Prefer Comprehension home for free-form comparisons."
        )


def render_guides_and_protocol(
    *,
    protocol_id: Optional[str] = None,
    extra_caption: Optional[str] = None,
) -> None:
    st.markdown(sidebar_guides_block_html(), unsafe_allow_html=True)
    if protocol_id:
        st.code(str(protocol_id), language=None)
        st.caption("Wire protocol id for this track (History / Rebuild isolation).")
    if extra_caption:
        st.caption(extra_caption)
