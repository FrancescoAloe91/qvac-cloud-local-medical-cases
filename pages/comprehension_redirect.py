"""Legacy Comprehension URL — redirects to Home when idle.

Kept so mid-flight Multi sessions that still have an old bookmark open are not
hard-killed. When no Comprehension run is active, send users to ``app.py``.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Comprehension · redirect",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_busy = bool(
    st.session_state.get("beta_running")
    or st.session_state.get("beta_confirmed_run")
    or st.session_state.get("beta_pending_run")
)

st.markdown("### Comprehension is Home")
st.caption(
    "This legacy URL is not a second UI. "
    "Protocol id = `comprehension-v1` (older History may still say beta-* and still pools)."
)

if _busy:
    st.warning(
        "A Comprehension run is still marked active in this session. "
        "Finish or Stop it on **Home**, then reload — this page will not "
        "interrupt an in-flight collect."
    )
    if st.button("Open Comprehension home", type="primary", use_container_width=True):
        st.switch_page("app.py")
else:
    st.info("Redirecting to Comprehension home…")
    st.switch_page("app.py")
