"""Load Comprehension case pack (stem + narrative prose + curated gold_raw)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from benchmark.beta_protocol import (
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
# Owner-workspace persistence (API-key fingerprint dir) — not pack JSON.
CUSTOM_DRAFTS_FILENAME = "comprehension_custom_drafts.json"


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


def delete_beta_custom_slot(
    slot: int,
    *,
    pack_slots: List[Dict[str, Any]],
    custom_drafts: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Remove a session custom draft. Never deletes pack Case 1…K. Gaps kept."""
    sid = int(slot)
    pack_ids = set()
    for entry in pack_slots or []:
        try:
            pack_ids.add(int(entry["slot"]))
        except (KeyError, TypeError, ValueError):
            continue
    if sid in pack_ids:
        raise ValueError(f"Cannot delete pack Case {sid}")
    drafts: Dict[int, Dict[str, Any]] = {
        int(k): dict(v) for k, v in (custom_drafts or {}).items()
    }
    drafts.pop(sid, None)
    return drafts


def custom_slots_for_multi_all(
    slots: List[Dict[str, Any]],
    locked_custom_slots: Optional[Iterable[Any]] = None,
) -> List[Dict[str, Any]]:
    """Custom rows eligible for Multi×all: Lock-marked + non-empty stem/prose."""
    locked = set()
    for raw in locked_custom_slots or []:
        try:
            locked.add(int(raw))
        except (TypeError, ValueError):
            continue
    out: List[Dict[str, Any]] = []
    for entry in slots or []:
        if not entry.get("custom"):
            continue
        try:
            sid = int(entry["slot"])
        except (KeyError, TypeError, ValueError):
            continue
        if sid not in locked:
            continue
        if not str(entry.get("stem") or "").strip():
            continue
        if not str(entry.get("reference_prose") or "").strip():
            continue
        out.append(dict(entry))
    return out


def delete_beta_artifacts_for_slot(workspace: Path, slot: int) -> int:
    """Delete Comprehension run JSON files stamped with ``slot``. Returns count."""
    from benchmark.report import list_run_artifacts, load_artifact

    sid = int(slot)
    removed = 0
    root = Path(workspace)
    if not root.is_dir():
        return 0
    for path in list_run_artifacts(root):
        try:
            art = load_artifact(path)
        except Exception:
            continue
        if beta_case_slot_of(art) != sid:
            continue
        try:
            path.unlink(missing_ok=True)
            removed += 1
        except OSError:
            continue
    return removed


def _parse_persisted_custom_drafts(
    raw_drafts: Any,
) -> Dict[int, Dict[str, Any]]:
    if not isinstance(raw_drafts, dict):
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for key, val in raw_drafts.items():
        try:
            sid = int(key)
        except (TypeError, ValueError):
            continue
        if not (1 <= sid <= SOFT_MAX_BETA_SLOTS) or not isinstance(val, Mapping):
            continue
        row = empty_beta_custom_slot(sid)
        for field in ("title", "stem", "reference_prose", "gold_raw"):
            if val.get(field) is not None:
                row[field] = str(val.get(field) or "")
        # Keep empty shells too (user opened New case) so slot buttons return.
        out[sid] = row
    return out


def load_beta_custom_state(
    workspace: Path,
) -> tuple[Dict[int, Dict[str, Any]], List[int]]:
    """Load custom drafts + Lock marks from owner workspace (API-key scoped)."""
    path = Path(workspace) / CUSTOM_DRAFTS_FILENAME
    if not path.is_file():
        return {}, []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, []
    if not isinstance(raw, dict):
        return {}, []
    drafts = _parse_persisted_custom_drafts(raw.get("drafts"))
    locked: List[int] = []
    for item in raw.get("locked_slots") or []:
        try:
            sid = int(item)
        except (TypeError, ValueError):
            continue
        if sid in drafts and sid not in locked:
            locked.append(sid)
    return drafts, locked


def save_beta_custom_state(
    workspace: Path,
    custom_drafts: Optional[Mapping[int, Mapping[str, Any]]] = None,
    locked_custom_slots: Optional[Iterable[Any]] = None,
) -> None:
    """Persist custom drafts + Lock marks under the owner workspace."""
    drafts = _parse_persisted_custom_drafts(
        {int(k): v for k, v in (custom_drafts or {}).items()}
    )
    locked: List[int] = []
    for item in locked_custom_slots or []:
        try:
            sid = int(item)
        except (TypeError, ValueError):
            continue
        if sid in drafts and sid not in locked:
            locked.append(sid)
    payload = {
        "version": 1,
        "drafts": {
            str(sid): {
                "title": drafts[sid].get("title") or f"Custom case {sid}",
                "stem": str(drafts[sid].get("stem") or ""),
                "reference_prose": str(drafts[sid].get("reference_prose") or ""),
                "gold_raw": str(drafts[sid].get("gold_raw") or ""),
            }
            for sid in sorted(drafts)
        },
        "locked_slots": locked,
    }
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    path = root / CUSTOM_DRAFTS_FILENAME
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_beta_gold_raw(case_entry: Mapping[str, Any]) -> str:
    """Pack gold_raw, or synthesize from reference prose for custom cases."""
    gold_raw = str(case_entry.get("gold_raw") or "").strip()
    if gold_raw and try_extract_qna_sections(gold_raw) is not None:
        return gold_raw
    prose = str(case_entry.get("reference_prose") or "").strip()
    return synthetic_gold_raw_from_prose(prose)


def _section_body_for_photocopy_check(section: Any) -> str:
    """Normalize one gold section body (strip [tag] prefixes)."""
    summary = ""
    claims_text = ""
    if hasattr(section, "summary"):
        summary = str(getattr(section, "summary", "") or "")
        claims = list(getattr(section, "claims", None) or [])
        claims_text = " ".join(str(getattr(c, "text", "") or "") for c in claims)
    elif isinstance(section, Mapping):
        summary = str(section.get("summary") or "")
        claims = section.get("claims") or []
        if isinstance(claims, list):
            claims_text = " ".join(
                str((c.get("text") if isinstance(c, Mapping) else getattr(c, "text", "")) or "")
                for c in claims
            )
    raw = f"{summary} {claims_text}".strip()
    for tag in ("[diagnosis]", "[tests]", "[urgency]", "[safety]", "[plan]"):
        raw = raw.replace(tag, " ")
    return " ".join(raw.lower().split())


def is_photocopy_custom_gold(case_entry: Mapping[str, Any]) -> bool:
    """True when custom gold is the undivided-prose scaffold (same body × 5).

    Pack cases with curated distinct Q1–A5 return False. Used to warn and to
    keep Multi×all public plan on pack + non-photocopy customs only.
    """
    if not case_entry.get("custom"):
        return False
    try:
        raw = resolve_beta_gold_raw(case_entry)
    except ValueError:
        return True
    sections = try_extract_qna_sections(raw)
    if not sections:
        return True
    bodies = []
    for key in ("diagnosis", "tests", "urgency", "safety", "plan"):
        sec = sections.get(key) if hasattr(sections, "get") else None
        if sec is None and isinstance(sections, Mapping):
            sec = sections.get(key)
        body = _section_body_for_photocopy_check(sec) if sec is not None else ""
        if body:
            bodies.append(body)
    if len(bodies) < 5:
        return True
    return len(set(bodies)) <= 1


def custom_slots_ready_for_multi_all(
    slots: List[Dict[str, Any]],
    locked_custom_slots: Optional[Iterable[Any]] = None,
) -> List[Dict[str, Any]]:
    """Lock-ed customs with non-empty text and non-photocopy gold only."""
    out: List[Dict[str, Any]] = []
    for entry in custom_slots_for_multi_all(slots, locked_custom_slots):
        if is_photocopy_custom_gold(entry):
            continue
        out.append(entry)
    return out


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
