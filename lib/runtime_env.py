"""Runtime environment detection (local vs Streamlit Community Cloud)."""

from __future__ import annotations

import os
from pathlib import Path

LIVE_DEMO_URL = (
    "https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app"
)
GITHUB_REPO_URL = "https://github.com/FrancescoAloe91/qvac-cloud-local-medical-cases"


def is_streamlit_cloud() -> bool:
    """True when running on Streamlit Community Cloud (no local Ollama)."""
    if os.environ.get("STREAMLIT_RUNTIME_ENVIRONMENT", "").lower() == "cloud":
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
    if os.environ.get("STREAMLIT_SHARING_MODE", "").lower() in ("true", "1", "yes"):
        return True
    # Streamlit Cloud mounts the repo at /mount/src/<repo>
    return Path("/mount/src").is_dir()
