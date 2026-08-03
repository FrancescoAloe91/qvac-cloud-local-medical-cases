"""Shared left-rail shell for Comprehension home and Structured page.

Order: Tracks (Comprehension only) → (caller OpenRouter/QVAC blocks) → Guides →
protocol chip → Advanced (muted Structured link).
Structured may append History / STOP; both pin Run clock last.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

import streamlit as st

from lib.guide_overlays import sidebar_guides_block_html
from lib.i18n import DEFAULT_LANG

# Nav label for the optional Structured page — keep quiet / non-primary.
STRUCTURED_NAV_LABEL = "Structured (legacy / advanced)"


def render_tracks_block(*, active: str = "comprehension") -> None:
    """Primary track link — Comprehension = home/default.

    Structured lives lower via ``render_advanced_track_link`` so the left rail
    reads as a single public track. Compact Comprehension pill styled in
    ``dashboard.css``. ``data-active`` drives the thick active border via portal JS.
    """
    st.markdown("**Tracks**")
    active_norm = "structured" if active == "structured" else "comprehension"
    st.markdown(
        f'<div id="qvac-track-active" data-active="{html.escape(active_norm)}" hidden></div>',
        unsafe_allow_html=True,
    )
    st.page_link("app.py", label="Comprehension", icon="🏠")
    if active_norm == "comprehension":
        st.caption(
            "Default track · free-form answers. "
            "Scores never mix with the optional Structured track below."
        )
        legacy = Path("pages/comprehension_redirect.py")
        if legacy.is_file():
            st.caption("Old bookmark redirect kept — prefer this Home page.")
    else:
        st.caption(
            "You are on the optional Structured track. "
            "Prefer Comprehension home for free-form comparisons."
        )


def render_advanced_track_link(*, active: str = "comprehension") -> None:
    """Muted Structured entry — lower in the rail, not a primary peer of Home."""
    active_norm = "structured" if active == "structured" else "comprehension"
    st.caption("Advanced")
    st.page_link(
        "pages/structured_graded.py",
        label=STRUCTURED_NAV_LABEL,
        help=(
            "Optional legacy / advanced fixed-slot Q&A · separate History · "
            "KPIs never pool with Comprehension"
        ),
    )
    if active_norm == "structured":
        st.caption(
            "Optional / advanced · fixed answer slots · "
            "do not pool KPIs with Comprehension."
        )
    else:
        st.caption("Optional · KPIs never pool with Comprehension.")


def render_guides_and_protocol(
    *,
    protocol_id: Optional[str] = None,
    extra_caption: Optional[str] = None,
    lang: Optional[str] = None,
    active_track: Optional[str] = None,
) -> None:
    st.markdown(
        sidebar_guides_block_html(lang=lang or DEFAULT_LANG),
        unsafe_allow_html=True,
    )
    if protocol_id:
        st.code(str(protocol_id), language=None)
        st.caption(
            "Technical track id (keeps History / Rebuild separate between tracks)."
        )
    if extra_caption:
        st.caption(extra_caption)
    if active_track is not None:
        render_advanced_track_link(active=active_track)
