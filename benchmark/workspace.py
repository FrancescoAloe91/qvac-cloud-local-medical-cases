"""Per-API-key private artifact workspaces (privacy login for case history).

Visitors who paste the same OpenRouter key see the same History / Rebuild mean.
Different keys never share runs — especially important for Custom Case gold text.

The raw API key is never written to disk; only a SHA-256 fingerprint is used
as a directory name under artifacts/owners/<fingerprint>/.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from benchmark.config import ARTIFACTS_DIR, is_usable_openrouter_key

OWNERS_DIRNAME = "owners"
# QVAC-only / no usable key — local machine only; never used to list cloud peers.
LOCAL_NO_KEY_ID = "_local_no_key"


def normalize_api_key(key: str | None) -> str:
    return (key or "").strip()


def owner_fingerprint(key: str | None) -> str:
    """Stable short fingerprint for a usable OpenRouter key (empty if invalid)."""
    k = normalize_api_key(key)
    if not is_usable_openrouter_key(k):
        return ""
    return hashlib.sha256(k.encode("utf-8")).hexdigest()[:24]


def current_api_key() -> str:
    """Prefer session-injected env (Streamlit BYOK), else process env."""
    return normalize_api_key(os.environ.get("OPENROUTER_API_KEY"))


def owner_id_for_current_key(key: str | None = None) -> str:
    """Workspace id for the active key, or LOCAL_NO_KEY_ID when none."""
    fp = owner_fingerprint(key if key is not None else current_api_key())
    return fp or LOCAL_NO_KEY_ID


def scoped_artifacts_dir(key: str | None = None) -> Path:
    """
    Directory for this visitor's run JSON.

    With a usable key → artifacts/owners/<sha256[:24]>/
    Without → artifacts/owners/_local_no_key/ (local QVAC rehearsal only)
    """
    k = normalize_api_key(key) if key is not None else current_api_key()
    fp = owner_fingerprint(k)
    oid = fp or LOCAL_NO_KEY_ID
    path = ARTIFACTS_DIR / OWNERS_DIRNAME / oid
    path.mkdir(parents=True, exist_ok=True)
    return path


def short_owner_label(key: str | None = None) -> str:
    """UI hint: key fingerprint prefix (never the raw key)."""
    k = normalize_api_key(key) if key is not None else current_api_key()
    fp = owner_fingerprint(k)
    if not fp:
        return "local (no API key)"
    return f"key …{fp[-6:]}"


def assert_path_in_workspace(path: Path, workspace: Optional[Path] = None) -> bool:
    """True if path resolves inside the caller's private workspace (path traversal guard)."""
    ws = (workspace or scoped_artifacts_dir()).resolve()
    try:
        path.resolve().relative_to(ws)
        return True
    except (ValueError, OSError):
        return False


def claim_unscoped_root_artifacts(dest: Path | None = None) -> int:
    """
    Move leftover ``artifacts/*.json`` (outside ``owners/``) into ``dest``.

    Needed after the per-key layout shipped: older Custom Case / Demo runs still
    sat in the root folder while History looked only under ``owners/<fp>/``.
    On Streamlit Cloud the root is normally empty (gitignored).
    """
    if not ARTIFACTS_DIR.is_dir():
        return 0
    dest = dest or scoped_artifacts_dir()
    dest.mkdir(parents=True, exist_ok=True)
    root_files = [
        p
        for p in ARTIFACTS_DIR.glob("*.json")
        if p.is_file() and "-summary-" not in p.name
    ]
    if not root_files:
        return 0
    moved = 0
    for p in root_files:
        target = dest / p.name
        if target.exists():
            continue
        try:
            p.rename(target)
            moved += 1
        except OSError:
            continue
    return moved


def maybe_claim_legacy_root_artifacts() -> int:
    """Automatic ownership claims are intentionally disabled.

    Legacy files have no trustworthy owner metadata. They remain local/read-only
    until an explicit migration tool is run by the operator.
    """
    return 0