"""Deployment context without visitor tracking or IP-based identity."""

from __future__ import annotations

import os
from pathlib import Path

from lib.runtime_env import is_streamlit_cloud


def streamlit_home_page() -> str:
    """Main-script basename for ``st.switch_page`` / ``st.page_link``.

    Community Cloud defaults to ``streamlit_app.py``; local docs use ``app.py``.
    Both sit at repo root next to ``pages/``, so multipage discovery stays valid.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx and getattr(ctx, "main_script_path", None):
            name = Path(ctx.main_script_path).name
            if name:
                return name
    except Exception:
        pass
    return "app.py"


def is_local_install() -> bool:
    return not is_streamlit_cloud()


def capture_and_strip_openrouter_env() -> str:
    """Return server OPENROUTER key for local prefill; strip it on Cloud.

    On Streamlit Cloud, always clear process-wide ``OPENROUTER_API_KEY`` so a
    host/shared secret cannot be spent via ``openrouter_api_key()`` fallback
    when the visitor session key is empty. Local ``.env`` remains developer-only.
    """
    server = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if is_streamlit_cloud():
        os.environ.pop("OPENROUTER_API_KEY", None)
        return ""
    return server