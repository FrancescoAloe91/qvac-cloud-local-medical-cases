"""Load Comprehension case pack (stem + narrative prose + curated gold_raw)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from benchmark.beta_protocol import (
    CASE_ID,
    CASE_ID_ALIASES,
    LEGACY_PACK_FILENAME,
    PACK_FILENAME,
    PROTOCOL_ID,
    SCORING_VERSION,
    SCORING_VERSION_ALIASES,
)
from benchmark.case_slots import DEFAULT_CASES_DIR
from benchmark.gold import confirmed_gold, try_extract_qna_sections

BETA_PACK_PATH = DEFAULT_CASES_DIR / PACK_FILENAME
_LEGACY_PACK_PATH = DEFAULT_CASES_DIR / LEGACY_PACK_FILENAME
# Soft cap for session custom Comprehension cases (pack slots stay untouched).
SOFT_MAX_BETA_SLOTS = 50


def is_beta_artifact(art: Any) -> bool:
    """True when an artifact belongs to the Comprehension track."""
    if art is None:
        return False
    case_id = str(getattr(art, "case_id", "") or "")
    scoring = str(getattr(art, "scoring_version", "") or "")
    if case_id in CASE_ID_ALIASES or scoring in SCORING_VERSION_ALIASES:
        return True
    mc = getattr(art, "models_config", None) or {}
    if isinstance(mc, Mapping) and (
        mc.get("comprehension_case_slot") is not None
        or mc.get("beta_case_slot") is not None
    ):
        return True
    return False


def beta_case_slot_of(art: Any) -> Optional[int]:
    """Return 1-based Comprehension pack slot from models_config, if present."""
    mc = getattr(art, "models_config", None) or {}
    if not isinstance(mc, Mapping):
        return None
    raw = mc.get("comprehension_case_slot")
    if raw is None:
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
    """Count persisted Comprehension artifacts per pack case slot."""
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
    if path is not None:
        p = Path(path)
    elif BETA_PACK_PATH.is_file():
        p = BETA_PACK_PATH
    elif _LEGACY_PACK_PATH.is_file():
        p = _LEGACY_PACK_PATH
    else:
        p = BETA_PACK_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Comprehension pack must be a JSON object")
    slots = data.get("slots") or {}
    if not isinstance(slots, dict) or not slots:
        raise ValueError("Comprehension pack missing slots")
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


def synthetic_gold_raw_from_prose(prose: str) -> str:
    """Build a minimal Q1–A5 scaffold from undivided reference prose (custom cases).

    Each A# gets a unique section-tagged quote so local QnA extract validation
    (no duplicate/overlapping source_quote) still accepts the scaffold.
    """
    body = (prose or "").strip()
    if not body:
        raise ValueError("reference prose is empty")
    # Unique verbatim quotes per section (validator rejects duplicates).
    a1 = f"[diagnosis] {body}"
    a2 = f"[tests] {body}"
    a3 = f"[urgency] {body}"
    a4 = f"[safety] {body}"
    a5 = f"[plan] {body}"
    return (
        "Q1 [diagnosis]: What is the most likely primary diagnosis? Rank the top differential.\n"
        f"A1: {a1}\n\n"
        "Q2 [tests]: Which tests should be ordered next?\n"
        f"A2: {a2}\n\n"
        "Q3 [urgency]: Urgency level (critical / high / moderate / low) and red flags.\n"
        f"A3: {a3}\n\n"
        "Q4 [safety]: Critical contraindications or safety traps.\n"
        f"A4: {a4}\n\n"
        "Q5 [plan]: Outline the initial management plan.\n"
        f"A5: {a5}"
    )


def empty_beta_custom_slot(slot: int) -> Dict[str, Any]:
    """Empty custom Comprehension case (session/workspace draft — not pack JSON)."""
    return {
        "slot": int(slot),
        "title": f"Custom case {int(slot)}",
        "stem": "",
        "reference_prose": "",
        "gold_raw": "",
        "protocol_id": PROTOCOL_ID,
        "custom": True,
    }


def merge_beta_slots(
    pack_slots: List[Dict[str, Any]],
    custom_drafts: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Pack slots first, then custom drafts (higher slot indexes)."""
    by_slot: Dict[int, Dict[str, Any]] = {}
    for entry in pack_slots or []:
        try:
            sid = int(entry["slot"])
        except (KeyError, TypeError, ValueError):
            continue
        row = dict(entry)
        row["custom"] = False
        by_slot[sid] = row
    for raw_sid, draft in (custom_drafts or {}).items():
        try:
            sid = int(raw_sid)
        except (TypeError, ValueError):
            continue
        if sid in by_slot and not by_slot[sid].get("custom"):
            # Never overwrite curated pack cases 1…K.
            continue
        if not (1 <= sid <= SOFT_MAX_BETA_SLOTS):
            continue
        row = empty_beta_custom_slot(sid)
        if isinstance(draft, Mapping):
            for key in ("title", "stem", "reference_prose", "gold_raw"):
                if draft.get(key) is not None:
                    row[key] = str(draft.get(key) or "")
        row["custom"] = True
        by_slot[sid] = row
    return [by_slot[k] for k in sorted(by_slot)]


def open_new_beta_case_slot(
    slots: List[Dict[str, Any]],
    *,
    custom_drafts: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> tuple[int, Dict[int, Dict[str, Any]]]:
    """Open next empty custom slot after the pack (or reuse an empty custom).

    Returns ``(slot_index, updated_custom_drafts)``. Raises ValueError at soft max.
    Pack slots are never mutated.
    """
    drafts: Dict[int, Dict[str, Any]] = {
        int(k): dict(v) for k, v in (custom_drafts or {}).items()
    }
    # Prefer an existing empty custom draft.
    for sid in sorted(drafts):
        d = drafts[sid]
        if not str(d.get("stem") or "").strip() and not str(
            d.get("reference_prose") or ""
        ).strip():
            return sid, drafts
    max_pack = max((int(s["slot"]) for s in slots if not s.get("custom")), default=0)
    max_any = max(
        [max_pack]
        + [int(s["slot"]) for s in slots]
        + list(drafts.keys()),
        default=0,
    )
    new_idx = max(max_pack, max_any) + 1
    if new_idx <= max_pack:
        new_idx = max_pack + 1
    if new_idx > SOFT_MAX_BETA_SLOTS:
        raise ValueError(f"Comprehension case soft max is {SOFT_MAX_BETA_SLOTS}")
    drafts[new_idx] = empty_beta_custom_slot(new_idx)
    return new_idx, drafts


def resolve_beta_gold_raw(case_entry: Mapping[str, Any]) -> str:
    """Pack gold_raw, or synthesize from reference prose for custom cases."""
    gold_raw = str(case_entry.get("gold_raw") or "").strip()
    if gold_raw and try_extract_qna_sections(gold_raw) is not None:
        return gold_raw
    prose = str(case_entry.get("reference_prose") or "").strip()
    return synthetic_gold_raw_from_prose(prose)


def auto_freeze_beta_slot(case_entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Force-confirm a pack slot for Multi×all-cases (no manual UI checkbox)."""
    gold_raw = resolve_beta_gold_raw(case_entry)
    sections = try_extract_qna_sections(gold_raw)
    if sections is None:
        raise ValueError(
            f"Case {case_entry.get('slot')}: pack gold_raw scaffold missing"
        )
    gold = confirmed_gold(
        raw_text=gold_raw,
        sections=sections,
        extraction_model=(
            "comprehension-custom-prose-scaffold-auto"
            if case_entry.get("custom")
            else "comprehension-pack-local-qna-scaffold-auto"
        ),
    )
    payload = gold.model_dump()
    payload.update(
        {
            "reference_prose": str(case_entry.get("reference_prose") or "").strip(),
            "beta_reference_prose": str(case_entry.get("reference_prose") or "").strip(),
            "stem": str(case_entry.get("stem") or "").strip(),
            "beta_stem": str(case_entry.get("stem") or "").strip(),
            "protocol_id": PROTOCOL_ID,
            "scoring_version": SCORING_VERSION,
            "case_slot": case_entry.get("slot"),
            "case_title": case_entry.get("title") or f"Case {case_entry.get('slot')}",
            "auto_confirmed": True,
            "custom_case": bool(case_entry.get("custom")),
        }
    )
    return payload
