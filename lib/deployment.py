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
