"""On-device GGUF variants: MedPsy (QVAC) + Band B open peers + medical-local peers."""

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

# Band medical_local — medical-specialized open Q4 GGUFs (not Band B generics).
MEDICAL_PEER_SPECS: List[Dict[str, Any]] = [
    {
        "key": "local_medgemma",
        "label": "MedGemma 4B IT",
        "display_label": "Medical · MedGemma-4B-IT Q4 · on-device",
        "vendor": "Google / Unsloth (open)",
        "site": "local (QVAC SDK)",
        "color": "#f59e0b",
        "provider": "qvac",
        "model": "medgemma-4b-it-q4",
        "gguf": "medgemma-4b-it-Q4_K_M.gguf",
        "band": "medical_local",
    },
    {
        "key": "local_med42",
        "label": "Med42 8B",
        "display_label": "Medical · Med42-8B Q4 · on-device",
        "vendor": "M42 Health",
        "site": "local (QVAC SDK)",
        "color": "#ef4444",
        "provider": "qvac",
        "model": "med42-8b-q4",
        "gguf": "Llama3-Med42-8B.Q4_K_M.gguf",
        "band": "medical_local",
    },
    {
        "key": "local_openbiollm",
        "label": "OpenBioLLM 8B",
        "display_label": "Medical · Llama3-OpenBioLLM-8B Q4 · on-device",
        "vendor": "Aaditya / QuantFactory (open)",
        "site": "local (QVAC SDK)",
        "color": "#ec4899",
        "provider": "qvac",
        "model": "llama3-openbiollm-8b-q4",
        "gguf": "Llama3-OpenBioLLM-8B.Q4_K_M.gguf",
        # QuantFactory GGUF omits tokenizer.chat_template; Med42 embeds the same
        # Llama-3 Instruct Jinja. Equalize role packing via local_chat_messages.
        "chat_format": "llama3",
        "band": "medical_local",
    },
]

_LOCAL_PEER_KEYS = frozenset(str(s["key"]) for s in LOCAL_PEER_SPECS)
_MEDICAL_PEER_KEYS = frozenset(str(s["key"]) for s in MEDICAL_PEER_SPECS)


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


def medical_peer_candidates() -> List[Dict[str, Any]]:
    """Band medical_local specialized GGUFs (always all three when included)."""
    return [_with_gguf(s) for s in MEDICAL_PEER_SPECS]


def medical_peers_ready() -> bool:
    """True when all three medical-local GGUFs are present and non-trivial."""
    return all(c.get("gguf_ready") for c in medical_peer_candidates())


def local_only_roster() -> List[Dict[str, Any]]:
    """Fair on-device compare: 3 open peers + 3 MedPsy quants (always all six)."""
    return local_peer_candidates() + variant_candidates(triple=True)


def local_medical_only_roster() -> List[Dict[str, Any]]:
    """Medical-specialized on-device only: 3 MedPsy + 3 medical_local (six slots)."""
    return medical_peer_candidates() + variant_candidates(triple=True)


def merge_roster(
    cloud_candidates: List[Dict[str, Any]],
    *,
    triple_qvac: bool,
    include_qvac: bool,
    include_local_peers: Optional[bool] = None,
    include_medical_peers: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Band A cloud + Band B generics + medical_local + optional MedPsy slot(s).

    Drops YAML qvac / free_light / local_peer / medical_local rows to avoid duplicates.
    ``include_local_peers`` defaults to the same gate as ``include_qvac``
    (sidecar up = generic on-device slots available).
    ``include_medical_peers`` defaults False unless explicitly requested (UI
    turns it on when GGUFs are ready).
    Cap: ≤12 = 3 cloud + 3 generic + 3 medical + 3 MedPsy.
    """
    if include_local_peers is None:
        include_local_peers = include_qvac
    if include_medical_peers is None:
        include_medical_peers = False
    # Band A only from YAML. Accept current API rows and archived free_web rows.
    cloud = [
        c
        for c in cloud_candidates
        if c.get("provider") == "openrouter"
        and (c.get("band") or "free_web") in {"api", "free_web"}
    ]
    peers = local_peer_candidates() if include_local_peers else []
    medical = medical_peer_candidates() if include_medical_peers else []
    medpsy = variant_candidates(triple=triple_qvac) if include_qvac else []
    return cloud + peers + medical + medpsy


def is_qvac_key(key: str) -> bool:
    """MedPsy / QVAC brand slots only (not Band B or medical_local peers)."""
    k = (key or "").lower()
    return k == "qvac" or k.startswith("qvac_")


def is_local_peer_key(key: str) -> bool:
    """Band B generic open peers only (not medical_local)."""
    return (key or "") in _LOCAL_PEER_KEYS


def is_medical_peer_key(key: str) -> bool:
    return (key or "") in _MEDICAL_PEER_KEYS


def is_on_device_key(key: str) -> bool:
    return is_qvac_key(key) or is_local_peer_key(key) or is_medical_peer_key(key)


def panel_rows_for_roster(roster: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Layout: 12 → 3×4 · 9 → 3×3 · 7 → 3+3+1 · 6 → 3+3 · else one row."""
    n = len(roster)
    if n >= 12:
        return [roster[0:3], roster[3:6], roster[6:9], roster[9:12]]
    if n >= 9:
        return [roster[0:3], roster[3:6], roster[6:9]]
    if n == 7:
        return [roster[0:3], roster[3:6], roster[6:7]]
    if n == 6:
        return [roster[0:3], roster[3:6]]
    if n:
        return [roster]
    return [[]]
