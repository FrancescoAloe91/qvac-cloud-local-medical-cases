"""Load Beta comprehension case pack (stem + undivided reference prose)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from benchmark.beta_protocol import CASE_ID, PACK_FILENAME, PROTOCOL_ID, SCORING_VERSION
from benchmark.case_slots import DEFAULT_CASES_DIR
from benchmark.gold import confirmed_gold, try_extract_qna_sections

BETA_PACK_PATH = DEFAULT_CASES_DIR / PACK_FILENAME


def is_beta_artifact(art: Any) -> bool:
    """True when an artifact belongs to the Beta comprehension track."""
    if art is None:
        return False
    case_id = str(getattr(art, "case_id", "") or "")
    scoring = str(getattr(art, "scoring_version", "") or "")
    if case_id == CASE_ID or scoring == SCORING_VERSION:
        return True
    mc = getattr(art, "models_config", None) or {}
    if isinstance(mc, Mapping) and mc.get("beta_case_slot") is not None:
        return True
    return False


def beta_case_slot_of(art: Any) -> Optional[int]:
    """Return 1-based Beta pack slot from models_config, if present."""
    mc = getattr(art, "models_config", None) or {}
    if not isinstance(mc, Mapping):
        return None
    raw = mc.get("beta_case_slot")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def count_beta_runs_by_slot(
    artifacts: Iterable[Any],
) -> Dict[int, int]:
    """Count persisted Beta artifacts per pack case slot (incl. Multi×all rounds)."""
    counts: Counter[int] = Counter()
    for art in artifacts:
        if not is_beta_artifact(art):
            continue
        slot = beta_case_slot_of(art)
        if slot is None:
            continue
        counts[slot] += 1
    return dict(counts)


def load_beta_pack(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else BETA_PACK_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Beta pack must be a JSON object")
    slots = data.get("slots") or {}
    if not isinstance(slots, dict) or not slots:
        raise ValueError("Beta pack missing slots")
    return data


def list_beta_slots(
    pack: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    data: Union[Mapping[str, Any], Dict[str, Any]]
    data = pack if pack is not None else load_beta_pack()
    out: List[Dict[str, Any]] = []
    slots = data.get("slots") or {}
    for key in sorted(slots.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
        entry = slots[key] or {}
        stem = str(entry.get("stem") or "").strip()
        prose = str(entry.get("reference_prose") or "").strip()
        gold_raw = str(entry.get("gold_raw") or "").strip()
        if not stem or not prose:
            continue
        out.append(
            {
                "slot": int(key) if str(key).isdigit() else key,
                "title": str(entry.get("title") or f"Case {key}"),
                "stem": stem,
                "reference_prose": prose,
                "gold_raw": gold_raw,
                "protocol_id": str(data.get("protocol_id") or PROTOCOL_ID),
            }
        )
    return out


def auto_freeze_beta_slot(case_entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Force-confirm a pack slot for Multi×all-cases (no manual UI checkbox)."""
    gold_raw = str(case_entry.get("gold_raw") or "").strip()
    sections = try_extract_qna_sections(gold_raw) if gold_raw else None
    if sections is None:
        raise ValueError(
            f"Case {case_entry.get('slot')}: pack gold_raw scaffold missing"
        )
    gold = confirmed_gold(
        raw_text=gold_raw,
        sections=sections,
        extraction_model="beta-pack-local-qna-scaffold-auto",
    )
    payload = gold.model_dump()
    payload.update(
        {
            "beta_reference_prose": str(case_entry.get("reference_prose") or "").strip(),
            "beta_stem": str(case_entry.get("stem") or "").strip(),
            "protocol_id": PROTOCOL_ID,
            "scoring_version": SCORING_VERSION,
            "case_slot": case_entry.get("slot"),
            "case_title": case_entry.get("title") or f"Case {case_entry.get('slot')}",
            "auto_confirmed": True,
        }
    )
    return payload
