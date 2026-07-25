"""Remember OpenRouter keys per visitor IP (not a shared Streamlit Secret).

- Same IP → prefill the key field after Save
- Different IP → empty field (cannot spend someone else's credits)
- Raw keys live only in a gitignored vault file; lookup key is sha256(IP)

On Streamlit Community Cloud, ``st.context.ip_address`` is often None behind
the proxy — we fall back to ``X-Forwarded-For`` / ``X-Real-IP`` when present.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

VAULT_FILE = Path(__file__).resolve().parent.parent / ".ip_key_vault.json"


def is_streamlit_cloud() -> bool:
    """True when running on Streamlit Community Cloud (shared multi-visitor host)."""
    if os.environ.get("STREAMLIT_SHARING_MODE"):
        return True
    if os.environ.get("STREAMLIT_CLOUD"):
        return True
    # Common mount used by Streamlit Cloud repos
    if Path("/mount/src").is_dir():
        return True
    return False


def _headers_get(headers: Any, name: str) -> str:
    if headers is None:
        return ""
    try:
        if hasattr(headers, "get"):
            val = headers.get(name) or headers.get(name.lower()) or headers.get(name.title())
            if val:
                return str(val).strip()
        if hasattr(headers, "get_all"):
            vals = headers.get_all(name) or headers.get_all(name.lower())
            if vals:
                return str(vals[0]).strip()
    except Exception:
        return ""
    return ""


def client_ip() -> str:
    """Best-effort visitor IP (may be empty behind some proxies)."""
    try:
        import streamlit as st

        ip = getattr(st.context, "ip_address", None)
        if ip:
            return str(ip).strip()
        headers = getattr(st.context, "headers", None)
        xff = _headers_get(headers, "X-Forwarded-For")
        if xff:
            return xff.split(",")[0].strip()
        real = _headers_get(headers, "X-Real-IP")
        if real:
            return real.split(",")[0].strip()
    except Exception:
        pass
    return ""


def client_identity() -> str:
    """
    Stable id for the vault.

    - Real IP → hash that IP
    - Localhost / no IP on a local install → ``local``
    - No IP on Streamlit Cloud → empty (do not share a global remembered key)
    """
    ip = client_ip()
    if ip and ip not in ("127.0.0.1", "::1", "localhost"):
        return "ip:" + hashlib.sha256(ip.encode("utf-8")).hexdigest()[:32]
    if is_streamlit_cloud():
        return ""
    return "local"


def _load_vault() -> Dict[str, Any]:
    if not VAULT_FILE.exists():
        return {"by_id": {}}
    try:
        data = json.loads(VAULT_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("by_id"), dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {"by_id": {}}


def _save_vault(data: Dict[str, Any]) -> None:
    try:
        VAULT_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def load_key_for_client() -> str:
    """Return saved OpenRouter key for this IP identity, or ''."""
    cid = client_identity()
    if not cid:
        return ""
    entry = _load_vault().get("by_id", {}).get(cid) or {}
    key = (entry.get("key") or "").strip()
    return key


def save_key_for_client(key: str) -> bool:
    """Persist key for this visitor IP. Returns False if identity unknown."""
    cid = client_identity()
    if not cid:
        return False
    key = (key or "").strip()
    if not key:
        return False
    data = _load_vault()
    data.setdefault("by_id", {})[cid] = {
        "key": key,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "ip_hint": _ip_hint(client_ip()),
    }
    _save_vault(data)
    return True


def clear_key_for_client() -> None:
    cid = client_identity()
    if not cid:
        return
    data = _load_vault()
    data.get("by_id", {}).pop(cid, None)
    _save_vault(data)


def _ip_hint(ip: str) -> str:
    """Partial IP for debugging (never full address in UI)."""
    ip = (ip or "").strip()
    if not ip:
        return "local"
    if "." in ip:
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.*.*"
    if ":" in ip:
        return ip.split(":")[0] + ":…"
    return "…"


def identity_caption() -> str:
    cid = client_identity()
    if cid == "local":
        return "Remembered on this computer (local)"
    if cid.startswith("ip:"):
        return f"Remembered for this network IP ({_ip_hint(client_ip())})"
    return "IP not detected — key will not be remembered across refresh on this host"
