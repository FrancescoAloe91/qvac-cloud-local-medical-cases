"""Local UI preferences that survive browser refresh (not secrets).

Stored under the project root as ``.ui_prefs.json`` (gitignored). Never put
API keys here — only non-sensitive acknowledgments / toggles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
PREFS_FILE = ROOT / ".ui_prefs.json"

_QVAC_SDK_ACK = "qvac_sdk_ack"


def _load_raw() -> Dict[str, Any]:
    if not PREFS_FILE.is_file():
        return {}
    try:
        data = json.loads(PREFS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_raw(data: Dict[str, Any]) -> None:
    try:
        PREFS_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def load_qvac_sdk_ack() -> bool:
    """True when the user already confirmed the QVAC SDK / MedPsy status dialog."""
    return bool(_load_raw().get(_QVAC_SDK_ACK))


def save_qvac_sdk_ack(acked: bool = True) -> None:
    """Remember QVAC SDK status acknowledgment across reloads (local install)."""
    data = _load_raw()
    data[_QVAC_SDK_ACK] = bool(acked)
    _save_raw(data)
