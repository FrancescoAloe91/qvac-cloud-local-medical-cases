"""Runtime environment detection (local vs Streamlit Community Cloud)."""

from __future__ import annotations

import os
from pathlib import Path

LIVE_DEMO_URL = "https://qvac-cloud-local-medical-cases.streamlit.app"
GITHUB_REPO_URL = "https://github.com/frankys91/qvac-cloud-local-medical-cases"


def is_streamlit_cloud() -> bool:
    """True when running on Streamlit Community Cloud (no local Ollama).

    Single source of truth for Cloud heuristics. ``lib.deployment`` re-exports
    this so callers agree on STREAMLIT_* env vars, ``*.streamlit.app`` host, and
    the ``/mount/src`` Community Cloud mount.
    """
    if os.environ.get("STREAMLIT_RUNTIME_ENVIRONMENT", "").lower() == "cloud":
        return True
    if os.environ.get("STREAMLIT_CLOUD"):
        return True
    sharing = (os.environ.get("STREAMLIT_SHARING_MODE") or "").strip()
    if sharing:
        return True
    host_blob = " ".join(
        str(os.environ.get(key, ""))
        for key in (
            "HOSTNAME",
            "STREAMLIT_SERVER_URL",
            "STREAMLIT_SERVER_ADDRESS",
        )
    ).lower()
    if "streamlit.app" in host_blob:
        return True
    # Streamlit Cloud mounts the repo at /mount/src/<repo>
    return Path("/mount/src").is_dir()
