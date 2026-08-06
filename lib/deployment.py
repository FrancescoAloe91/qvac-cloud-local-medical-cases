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


def is_hosted_byok_required() -> bool:
    """True when visitor BYOK is mandatory (no host OPENROUTER_API_KEY fallback).

    Covers Streamlit Community Cloud plus public/shared hosts (Render, explicit
    ``HOSTED_BYOK=1``, Streamlit bound to ``0.0.0.0`` / ``::``). Local loopback
    installs keep developer ``.env`` prefill.
    """
    if is_streamlit_cloud():
        return True
    flag = (os.environ.get("HOSTED_BYOK") or "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    if os.environ.get("RENDER") or os.environ.get("RENDER_SERVICE_ID"):
        return True
    if os.environ.get("DYNO"):  # Heroku-style
        return True
    addr = (os.environ.get("STREAMLIT_SERVER_ADDRESS") or "").strip().lower()
    if addr in {"0.0.0.0", "::", "[::]", "*"}:
        return True
    return False


def is_local_install() -> bool:
    """True for a private developer machine (loopback), not a public host."""
    return not is_hosted_byok_required()


def capture_and_strip_openrouter_env() -> str:
    """Return server OPENROUTER key for local prefill; strip it on hosted BYOK.

    On Streamlit Cloud / Render / any public bind (``HOSTED_BYOK``), always clear
    process-wide ``OPENROUTER_API_KEY`` so a host/shared secret cannot be spent
    via ``openrouter_api_key()`` fallback when the visitor session key is empty.
    Local loopback ``.env`` remains developer-only.
    """
    server = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if is_hosted_byok_required():
        os.environ.pop("OPENROUTER_API_KEY", None)
        return ""
    return server