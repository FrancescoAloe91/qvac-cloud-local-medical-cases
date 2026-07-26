"""Editable cloud model version labels (shown on cards, charts, exports)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

from lib.tiers import CLOUD_KEYS, MODEL_CONFIG

_TIERS_PATH = Path(__file__).resolve().parent.parent / "data" / "cloud_tiers.json"

DEFAULT_TIER_LABELS: Dict[str, str] = {
    "chatgpt": "5.5 Instant",
    "claude": "Sonnet 5 · Extra",
    "gemini": "3.5 Flash · extended thinking",
    "qvac": "",
}

_LEGACY_CHATGPT_BOILERPLATE = re.compile(
    r"^(?:Tier gratuito predefinito|Free tier default|Default free tier)"
    r"(?:\s*\([^)]*\))?$",
    re.IGNORECASE,
)


def _strip_parentheticals(text: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", text).strip()


def _strip_site_suffix(text: str, site: str) -> str:
    text = re.sub(rf"\s*[—–-]\s*{re.escape(site)}.*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(
        r"\s*[—–-]\s*(?:nuova chat|new chat)(?:\s+predefinita)?.*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text


def _sanitize_cloud_tier_label(model_key: str, text: str) -> str:
    """Remove site/new-chat boilerplate from editable cloud version labels."""
    cleaned = _single_line(str(text or "").strip())
    if not cleaned:
        return DEFAULT_TIER_LABELS.get(model_key, "") if model_key in CLOUD_KEYS else ""

    if model_key == "chatgpt":
        cleaned = _strip_parentheticals(cleaned)
        cleaned = _strip_site_suffix(cleaned, "chatgpt.com")
        if _LEGACY_CHATGPT_BOILERPLATE.match(cleaned) or not cleaned:
            cleaned = DEFAULT_TIER_LABELS["chatgpt"]
    elif model_key == "claude":
        cleaned = _strip_parentheticals(cleaned)
        cleaned = _strip_site_suffix(cleaned, "claude.ai")
        cleaned = re.sub(r"(\bSonnet\s*5?\s*·\s*)Medium\b", r"\1Extra", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(\bthinking\s+)Medium\b", r"\1Extra", cleaned, flags=re.IGNORECASE)
        if re.search(r"·\s*Medium\b", cleaned, re.IGNORECASE):
            cleaned = re.sub(r"·\s*Medium\b", "· Extra", cleaned, flags=re.IGNORECASE)
        if re.match(r"^Tier gratuito predefinito$", cleaned, re.IGNORECASE) or re.match(
            r"^Default free tier$", cleaned, re.IGNORECASE
        ):
            cleaned = DEFAULT_TIER_LABELS.get(model_key, cleaned)
    elif model_key == "gemini":
        cleaned = _strip_parentheticals(cleaned)
        cleaned = _strip_site_suffix(cleaned, "gemini.google.com")
        if re.match(r"^Tier gratuito predefinito$", cleaned, re.IGNORECASE) or re.match(
            r"^Default free tier$", cleaned, re.IGNORECASE
        ):
            cleaned = DEFAULT_TIER_LABELS.get(model_key, cleaned)

    return cleaned


def normalize_tier_labels_dict(labels: Dict[str, str]) -> Dict[str, str]:
    """Clean stored/session tier labels without touching benchmark snapshots."""
    out = dict(labels)
    for key in ("chatgpt", "claude", "gemini"):
        if key in out:
            out[key] = _sanitize_cloud_tier_label(key, out.get(key, ""))
    return out


def _strip_ollama_segments(text: str) -> str:
    parts = [p.strip() for p in text.split("·") if p.strip()]
    kept = [p for p in parts if "ollama" not in p.lower()]
    return " · ".join(kept) if kept else text.strip()


def _single_line(text: str) -> str:
    return " ".join(str(text).split())


def table_version_label(model_key: str, labels: Optional[Dict[str, str]] = None) -> str:
    """Clean one-line version string for KPI tables (no site notes, no Ollama tag)."""
    raw = tier_label(model_key, labels)
    if not raw:
        return "—"
    if model_key == "qvac":
        raw = _strip_ollama_segments(raw)
    return _single_line(raw)


def _qvac_runtime_label() -> str:
    """Best-effort local runtime label; safe when Ollama is absent (Streamlit Cloud)."""
    try:
        from lib import medpsy

        return medpsy.runtime_tier_label()
    except Exception:
        return "MedPsy-4B · on-device (local setup)"


def effective_tier_labels(labels: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Merge saved cloud labels with live QVAC runtime info from Ollama/MedPsy setup."""
    data = normalize_tier_labels_dict(dict(labels or load_tier_labels()))
    data["qvac"] = _qvac_runtime_label()
    return data


def _blank_defaults() -> Dict[str, str]:
    return {k: "" for k in DEFAULT_TIER_LABELS}


def _can_persist_tier_labels() -> bool:
    """Streamlit Cloud mounts the repo read-only — never write there."""
    try:
        from lib.runtime_env import is_streamlit_cloud

        if is_streamlit_cloud():
            return False
    except Exception:
        pass
    return True


def _persist_tier_labels_file(labels: Dict[str, str]) -> None:
    if not _can_persist_tier_labels():
        return
    try:
        _TIERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chatgpt": _sanitize_cloud_tier_label(
                "chatgpt", labels.get("chatgpt") or DEFAULT_TIER_LABELS.get("chatgpt", "")
            ),
            "claude": _sanitize_cloud_tier_label(
                "claude", labels.get("claude") or DEFAULT_TIER_LABELS.get("claude", "")
            ),
            "gemini": _sanitize_cloud_tier_label(
                "gemini", labels.get("gemini") or DEFAULT_TIER_LABELS.get("gemini", "")
            ),
            "qvac": _qvac_runtime_label(),
        }
        _TIERS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError:
        # Read-only or missing write perms (cloud / locked checkout).
        return


def load_tier_labels() -> Dict[str, str]:
    labels = dict(DEFAULT_TIER_LABELS)
    raw_from_file = None
    if _TIERS_PATH.is_file():
        try:
            raw_from_file = json.loads(_TIERS_PATH.read_text(encoding="utf-8"))
            if isinstance(raw_from_file, dict):
                for key in ("chatgpt", "claude", "gemini"):
                    val = raw_from_file.get(key)
                    if isinstance(val, str) and val.strip():
                        labels[key] = val.strip()
        except (json.JSONDecodeError, OSError):
            pass
    labels = normalize_tier_labels_dict(labels)
    labels["qvac"] = _qvac_runtime_label()
    if isinstance(raw_from_file, dict):
        dirty = False
        for key in ("chatgpt", "claude", "gemini"):
            val = raw_from_file.get(key)
            if isinstance(val, str) and _sanitize_cloud_tier_label(key, val) != val.strip():
                dirty = True
                break
        if dirty:
            _persist_tier_labels_file(labels)
    return labels


def save_tier_labels(labels: Dict[str, str]) -> None:
    cleaned = normalize_tier_labels_dict(dict(labels))
    _persist_tier_labels_file(cleaned)


def tier_label(model_key: str, labels: Optional[Dict[str, str]] = None) -> str:
    data = effective_tier_labels(labels)
    if model_key == "qvac":
        return data.get("qvac", "")
    if model_key == "qvac_1_7b":
        return "MedPsy-1.7B Q4 · on-device"
    if model_key == "qvac_4b_q8":
        return "MedPsy-4B Q8 · on-device"
    raw = (data.get(model_key) or DEFAULT_TIER_LABELS.get(model_key, "")).strip()
    if model_key in CLOUD_KEYS:
        return _sanitize_cloud_tier_label(model_key, raw)
    return raw


def display_model_name(model_key: str, labels: Optional[Dict[str, str]] = None) -> str:
    base = MODEL_CONFIG.get(model_key, {}).get("name", model_key)
    tier = tier_label(model_key, labels)
    return f"{base} · {tier}" if tier else base


def short_chart_label(model_key: str, labels: Optional[Dict[str, str]] = None) -> str:
    """Full model + tier label for charts (no truncation)."""
    return display_model_name(model_key, labels)
