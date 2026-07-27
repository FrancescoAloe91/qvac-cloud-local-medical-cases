"""On-device GGUF variants: MedPsy (QVAC) + Band B open local peers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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
        "band": "qvac",
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
        "band": "qvac",
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
        "band": "qvac",
    },
]

# Band B — open small Q4 GGUFs, same sidecar as MedPsy (real on-device privacy).
# Gemma-2-2B-IT replaces Qwen: Qwen 3B→zeros / 1.5B→CJK garbage under this SDK path.
LOCAL_PEER_SPECS: List[Dict[str, Any]] = [
    {
        "key": "local_gemma",
        "label": "Gemma 2 2B",
        "display_label": "Local · Gemma-2-2B-IT Q4 · on-device",
        "vendor": "Google (open)",
        "site": "local (QVAC SDK)",
        "color": "#a855f7",
        "provider": "qvac",
        "model": "gemma-2-2b-it-q4",
        "gguf": "gemma-2-2b-it-Q4_K_M.gguf",
        "band": "local_peer",
    },
    {
        "key": "local_llama",
        "label": "Llama 3.2 3B",
        "display_label": "Local · Llama-3.2-3B Instruct Q4 · on-device",
        "vendor": "Meta (open)",
        "site": "local (QVAC SDK)",
        "color": "#3b82f6",
        "provider": "qvac",
        "model": "llama-3.2-3b-instruct-q4",
        "gguf": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "band": "local_peer",
    },
    {
        "key": "local_phi",
        "label": "Phi-3.5 mini",
        "display_label": "Local · Phi-3.5-mini Instruct Q4 · on-device",
        "vendor": "Microsoft (open)",
        "site": "local (QVAC SDK)",
        "color": "#0ea5e9",
        "provider": "qvac",
        "model": "phi-3.5-mini-instruct-q4",
        "gguf": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "band": "local_peer",
    },
]


def gguf_path(filename: str) -> Path:
    return MODELS_DIR / filename


def _with_gguf(spec: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(spec)
    path = gguf_path(str(spec["gguf"]))
    row["gguf_path"] = str(path.resolve()) if path.is_file() else str(path)
    row["gguf_ready"] = path.is_file() and path.stat().st_size > 1_000_000
    return row


def variant_candidates(*, triple: bool) -> List[Dict[str, Any]]:
    """Return 1 (default 4B Q4) or all 3 MedPsy candidate dicts with absolute gguf_path."""
    specs = QVAC_VARIANT_SPECS if triple else [s for s in QVAC_VARIANT_SPECS if s["key"] == "qvac"]
    return [_with_gguf(s) for s in specs]


def local_peer_candidates() -> List[Dict[str, Any]]:
    """Band B open local GGUFs (always all three when included)."""
    return [_with_gguf(s) for s in LOCAL_PEER_SPECS]


def local_only_roster() -> List[Dict[str, Any]]:
    """Fair on-device compare: 3 open peers + 3 MedPsy quants (always all six)."""
    return local_peer_candidates() + variant_candidates(triple=True)


def merge_roster(
    cloud_candidates: List[Dict[str, Any]],
    *,
    triple_qvac: bool,
    include_qvac: bool,
    include_local_peers: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Band A cloud + Band B local peers + optional MedPsy slot(s).

    Drops YAML qvac / free_light / local_peer rows to avoid duplicates.
    ``include_local_peers`` defaults to the same gate as ``include_qvac``
    (sidecar up = all on-device slots available).
    """
    if include_local_peers is None:
        include_local_peers = include_qvac
    # Band A only from YAML. Accept current API rows and archived free_web rows.
    cloud = [
        c
        for c in cloud_candidates
        if c.get("provider") == "openrouter"
        and (c.get("band") or "free_web") in {"api", "free_web"}
    ]
    peers = local_peer_candidates() if include_local_peers else []
    medpsy = variant_candidates(triple=triple_qvac) if include_qvac else []
    return cloud + peers + medpsy


def is_qvac_key(key: str) -> bool:
    """MedPsy / QVAC brand slots only (not Band B local peers)."""
    k = (key or "").lower()
    return k == "qvac" or k.startswith("qvac_")


def is_local_peer_key(key: str) -> bool:
    k = (key or "").lower()
    return k.startswith("local_")


def is_on_device_key(key: str) -> bool:
    return is_qvac_key(key) or is_local_peer_key(key)


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
