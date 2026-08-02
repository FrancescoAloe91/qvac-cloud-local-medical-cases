"""Deployment context without visitor tracking or IP-based identity."""

from __future__ import annotations

import os
from pathlib import Path


def is_streamlit_cloud() -> bool:
    return bool(
        os.environ.get("STREAMLIT_SHARING_MODE")
        or os.environ.get("STREAMLIT_CLOUD")
        or Path("/mount/src").is_dir()
    )


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