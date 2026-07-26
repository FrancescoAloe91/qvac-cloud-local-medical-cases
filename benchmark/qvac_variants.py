"""On-device MedPsy GGUF variants for optional 3×QVAC compare mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = _REPO / "models"

# Default single-slot key stays "qvac" (4B Q4) for History compatibility.
QVAC_VARIANT_SPECS: List[Dict[str, Any]] = [
    {
        "key": "qvac_1_7b",
        "label": "MedPsy 1.7B",
        "display_label": "QVAC · MedPsy-1.7B Q4 · on-device",
        "vendor": "Tether QVAC",
        "site": "local (QVAC SDK)",
        "color": "#34d399",
        "provider": "qvac",
        "model": "medpsy-1.7b-q4",
        "gguf": "medpsy-1.7b-q4_k_m-imat.gguf",
    },
    {
        "key": "qvac",
        "label": "MedPsy 4B Q4",
        "display_label": "QVAC · MedPsy-4B Q4 · on-device",
        "vendor": "Tether QVAC",
        "site": "local (QVAC SDK)",
        "color": "#00d09c",
        "provider": "qvac",
        "model": "medpsy-4b-q4",
        "gguf": "medpsy-4b-q4_k_m-imat.gguf",
    },
    {
        "key": "qvac_4b_q8",
        "label": "MedPsy 4B Q8",
        "display_label": "QVAC · MedPsy-4B Q8 · on-device",
        "vendor": "Tether QVAC",
        "site": "local (QVAC SDK)",
        "color": "#2dd4bf",
        "provider": "qvac",
        "model": "medpsy-4b-q8",
        "gguf": "medpsy-4b-q8_0.gguf",
    },
]


def gguf_path(filename: str) -> Path:
    return MODELS_DIR / filename


def variant_candidates(*, triple: bool) -> List[Dict[str, Any]]:
    """Return 1 (default 4B Q4) or all 3 QVAC candidate dicts with absolute gguf_path."""
    specs = QVAC_VARIANT_SPECS if triple else [s for s in QVAC_VARIANT_SPECS if s["key"] == "qvac"]
    out: List[Dict[str, Any]] = []
    for s in specs:
        row = dict(s)
        path = gguf_path(str(s["gguf"]))
        row["gguf_path"] = str(path.resolve()) if path.is_file() else str(path)
        row["gguf_ready"] = path.is_file() and path.stat().st_size > 1_000_000
        out.append(row)
    return out


def merge_roster(
    cloud_candidates: List[Dict[str, Any]],
    *,
    triple_qvac: bool,
    include_qvac: bool,
) -> List[Dict[str, Any]]:
    """Cloud + optional QVAC slot(s). Drops YAML qvac entries to avoid duplicates."""
    cloud = [c for c in cloud_candidates if c.get("provider") != "qvac"]
    if not include_qvac:
        return cloud
    return cloud + variant_candidates(triple=triple_qvac)


def is_qvac_key(key: str) -> bool:
    k = (key or "").lower()
    return k == "qvac" or k.startswith("qvac_")


def panel_rows_for_roster(roster: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Layout: 7 → 3+3+1 · 9 → 3×3 · 6 cloud-only → 3+3 · else one row."""
    n = len(roster)
    if n >= 9:
        return [roster[0:3], roster[3:6], roster[6:9]]
    if n == 7:
        return [roster[0:3], roster[3:6], roster[6:7]]
    if n == 6:
        return [roster[0:3], roster[3:6]]
    if n:
        return [roster]
    return [[]]
