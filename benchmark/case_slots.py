"""Per-owner clinical case slots mapped onto History artifacts.

Base slots Case 1–5 are always present. **New case** opens the next empty
base slot, or grows the bar to Case 6, 7, … when 1–5 are filled. Bindings are
sticky slot index → stem_key (normalized stem hash).

Stem text + confirmed gold are resolved from run artifacts when present.
Empty base slots may also be seeded from a shipped default case pack
(``benchmark/default_cases/``) into owner-scoped sticky drafts — stem + freeform
gold Q&A for Prepare/Confirm — without rewriting History artifacts.

History ownership remains ``artifacts/owners/<fingerprint>/`` (API key / account).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.gold import _normalized, load_confirmed_gold
from benchmark.schema import RunArtifact

BASE_CASE_SLOTS = 5
# Practical safety rail (not a product “max 5” cap). New case grows until this.
SOFT_MAX_CASE_SLOTS = 50
# Backward-compatible alias: base bar size (Case 1–5 always shown).
MAX_CASE_SLOTS = BASE_CASE_SLOTS
SLOTS_FILENAME = "case_slots.json"
DRAFTS_FILENAME = "case_slot_drafts.json"
DEFAULT_CASES_DIR = Path(__file__).resolve().parent / "default_cases"
DEFAULT_PACK_FILENAME = "abc_emergency.json"


def slot_indexes(slot_count: int) -> tuple[int, ...]:
    n = max(BASE_CASE_SLOTS, int(slot_count or BASE_CASE_SLOTS))
    n = min(n, SOFT_MAX_CASE_SLOTS)
    return tuple(range(1, n + 1))


def stem_key(case_stem: str) -> str:
    """Stable short fingerprint for a clinical stem (NFC + whitespace + casefold)."""
    return hashlib.sha256(_normalized(case_stem).encode("utf-8")).hexdigest()[:24]


def artifact_stem(art: RunArtifact) -> str:
    return str((art.models_config or {}).get("case_stem") or "").strip()


def artifact_gold_json(art: RunArtifact) -> str:
    return str((art.models_config or {}).get("gold_reference") or "").strip()


def artifact_stem_key(art: RunArtifact) -> str:
    stem = artifact_stem(art)
    return stem_key(stem) if stem else ""


def _finished_at(art: RunArtifact) -> str:
    return str(art.finished_at or art.started_at or "")


@dataclass
class StemFamily:
    stem_key: str
    stem: str
    gold_reference: str
    cohort_id: str
    run_ids: list[str]
    run_count: int
    latest_finished_at: str
    has_protocol_cohort: bool


@dataclass
class CaseSlot:
    index: int
    label: str
    stem_key: str = ""
    stem: str = ""
    gold_reference: str = ""
    gold_raw: str = ""
    cohort_id: str = ""
    run_ids: list[str] | None = None
    run_count: int = 0
    latest_finished_at: str = ""

    @property
    def filled(self) -> bool:
        # Bound stem_key counts even before the first History artifact exists
        # (Confirm binds; collect writes the stem into models_config later).
        # Draft-only slots (stem text, no hash yet) also count as filled.
        return bool(self.stem_key or self.stem)

    @property
    def has_confirmed_gold(self) -> bool:
        return bool(self.gold_reference)

    @property
    def has_draft_gold(self) -> bool:
        return bool(self.gold_raw)


def empty_slot(index: int) -> CaseSlot:
    return CaseSlot(index=index, label=f"Case {index}", run_ids=[])


def discover_stem_families(
    artifacts: Sequence[RunArtifact],
) -> list[StemFamily]:
    """Group artifacts by normalized stem; newest gold/cohort preferred per family."""
    buckets: dict[str, dict[str, Any]] = {}
    for art in artifacts:
        stem = artifact_stem(art)
        if not stem:
            continue
        key = stem_key(stem)
        gold = artifact_gold_json(art)
        finished = _finished_at(art)
        cohort = str(art.cohort_id or "").strip()
        bucket = buckets.get(key)
        if bucket is None:
            buckets[key] = {
                "stem_key": key,
                "stem": stem,
                "gold_reference": gold,
                "cohort_id": cohort,
                "run_ids": [art.run_id],
                "run_count": 1,
                "latest_finished_at": finished,
                "has_protocol_cohort": bool(cohort and gold),
            }
            continue
        bucket["run_count"] = int(bucket["run_count"]) + 1
        bucket["run_ids"].append(art.run_id)
        if finished >= str(bucket.get("latest_finished_at") or ""):
            bucket["latest_finished_at"] = finished
            bucket["stem"] = stem
            if gold:
                bucket["gold_reference"] = gold
            if cohort:
                bucket["cohort_id"] = cohort
        elif gold and not bucket.get("gold_reference"):
            bucket["gold_reference"] = gold
        if cohort and gold:
            bucket["has_protocol_cohort"] = True
            if not bucket.get("cohort_id"):
                bucket["cohort_id"] = cohort

    families = [
        StemFamily(
            stem_key=str(b["stem_key"]),
            stem=str(b["stem"]),
            gold_reference=str(b.get("gold_reference") or ""),
            cohort_id=str(b.get("cohort_id") or ""),
            run_ids=list(b.get("run_ids") or []),
            run_count=int(b.get("run_count") or 0),
            latest_finished_at=str(b.get("latest_finished_at") or ""),
            has_protocol_cohort=bool(b.get("has_protocol_cohort")),
        )
        for b in buckets.values()
    ]
    # Protocol-valid first, then recency — migration prefers real confirmed cohorts.
    families.sort(
        key=lambda f: (
            1 if f.has_protocol_cohort else 0,
            f.latest_finished_at,
        ),
        reverse=True,
    )
    return families


def _parse_bindings(raw_bindings: Any) -> dict[int, str]:
    if not isinstance(raw_bindings, dict):
        return {}
    out: dict[int, str] = {}
    for k, v in raw_bindings.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if idx < 1 or idx > SOFT_MAX_CASE_SLOTS:
            continue
        key = str(v or "").strip()
        if key:
            out[idx] = key
    return out


def _read_slot_state_raw(workspace: Path) -> dict[str, Any]:
    path = Path(workspace) / SLOTS_FILENAME
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def load_slot_state(workspace: Path) -> tuple[dict[int, str], int]:
    """Load sticky bindings + how many Case buttons exist ( ≥ base 5 )."""
    raw = _read_slot_state_raw(workspace)
    if not raw:
        return {}, BASE_CASE_SLOTS
    bindings = _parse_bindings(raw.get("bindings"))
    stored = raw.get("slot_count")
    try:
        slot_count = int(stored) if stored is not None else BASE_CASE_SLOTS
    except (TypeError, ValueError):
        slot_count = BASE_CASE_SLOTS
    max_bound = max(bindings.keys(), default=0)
    slot_count = max(BASE_CASE_SLOTS, slot_count, max_bound)
    slot_count = min(slot_count, SOFT_MAX_CASE_SLOTS)
    return bindings, slot_count


def load_pack_revision(workspace: Path) -> int:
    """Owner workspace's last applied default-pack revision (0 = never)."""
    raw = _read_slot_state_raw(workspace)
    try:
        return max(0, int(raw.get("pack_revision") or 0))
    except (TypeError, ValueError):
        return 0


def load_bindings(workspace: Path) -> dict[int, str]:
    """Load sticky slot→stem_key bindings (hashes only; no clinical plaintext)."""
    bindings, _ = load_slot_state(workspace)
    return bindings


def save_bindings(
    workspace: Path,
    bindings: Mapping[int, str],
    *,
    slot_count: int | None = None,
    pack_revision: int | None = None,
) -> None:
    """Persist sticky bindings (stem hashes only) + slot bar size."""
    path = Path(workspace) / SLOTS_FILENAME
    clean = {
        int(i): str(k).strip()
        for i, k in bindings.items()
        if 1 <= int(i) <= SOFT_MAX_CASE_SLOTS and str(k or "").strip()
    }
    max_bound = max(clean.keys(), default=0)
    count = max(
        BASE_CASE_SLOTS,
        int(slot_count or BASE_CASE_SLOTS),
        max_bound,
    )
    count = min(count, SOFT_MAX_CASE_SLOTS)
    # Preserve prior pack_revision unless caller supplies a new one.
    prior_rev = load_pack_revision(workspace)
    rev = prior_rev if pack_revision is None else max(0, int(pack_revision))
    payload = {
        "version": 2,
        "slot_count": count,
        "pack_revision": rev,
        "bindings": {str(i): clean[i] for i in sorted(clean)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_drafts(raw_drafts: Any) -> dict[int, dict[str, str]]:
    if not isinstance(raw_drafts, dict):
        return {}
    out: dict[int, dict[str, str]] = {}
    for k, v in raw_drafts.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if idx < 1 or idx > SOFT_MAX_CASE_SLOTS or not isinstance(v, Mapping):
            continue
        stem = str(v.get("stem") or "").strip()
        gold_raw = str(v.get("gold_raw") or "").strip()
        key = str(v.get("stem_key") or "").strip()
        if not stem and not gold_raw and not key:
            continue
        if stem and not key:
            key = stem_key(stem)
        out[idx] = {"stem": stem, "gold_raw": gold_raw, "stem_key": key}
    return out


def load_drafts(workspace: Path) -> dict[int, dict[str, str]]:
    """Load owner-scoped stem + freeform gold drafts (for empty / pre-run slots)."""
    path = Path(workspace) / DRAFTS_FILENAME
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return _parse_drafts(raw.get("drafts"))


def save_drafts(workspace: Path, drafts: Mapping[int, Mapping[str, str]]) -> None:
    """Persist owner-scoped drafts (clinical plaintext — local plaintext store only)."""
    path = Path(workspace) / DRAFTS_FILENAME
    clean = _parse_drafts(dict(drafts))
    payload = {
        "version": 1,
        "drafts": {
            str(i): {
                "stem": clean[i]["stem"],
                "gold_raw": clean[i]["gold_raw"],
                "stem_key": clean[i]["stem_key"],
            }
            for i in sorted(clean)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_default_pack_document(path: Path | None = None) -> dict[str, Any]:
    pack_path = Path(path) if path is not None else DEFAULT_CASES_DIR / DEFAULT_PACK_FILENAME
    if not pack_path.is_file():
        return {}
    try:
        raw = json.loads(pack_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def load_default_pack_meta(
    path: Path | None = None,
) -> tuple[int, frozenset[int]]:
    """Return ``(revision, force_seed_slots)`` from the shipped default pack."""
    raw = _load_default_pack_document(path)
    try:
        revision = max(0, int(raw.get("revision") or 1))
    except (TypeError, ValueError):
        revision = 1
    force: set[int] = set()
    for item in raw.get("force_seed_slots") or []:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= SOFT_MAX_CASE_SLOTS:
            force.add(idx)
    return revision, frozenset(force)


def load_default_pack(
    path: Path | None = None,
) -> dict[int, dict[str, str]]:
    """Load shipped default case pack (slot index → stem + gold_raw)."""
    raw = _load_default_pack_document(path)
    slots = raw.get("slots") if raw else None
    if not isinstance(slots, dict):
        return {}
    out: dict[int, dict[str, str]] = {}
    for k, v in slots.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if idx < 1 or idx > SOFT_MAX_CASE_SLOTS or not isinstance(v, Mapping):
            continue
        stem = str(v.get("stem") or "").strip()
        gold_raw = str(v.get("gold_raw") or "").strip()
        if not stem:
            continue
        out[idx] = {
            "stem": stem,
            "gold_raw": gold_raw,
            "stem_key": stem_key(stem),
            "title": str(v.get("title") or "").strip(),
        }
    return out


def apply_default_pack_to_empty_slots(
    bindings: Mapping[int, str],
    drafts: Mapping[int, Mapping[str, str]],
    *,
    pack: Mapping[int, Mapping[str, str]] | None = None,
) -> tuple[dict[int, str], dict[int, dict[str, str]]]:
    """Seed empty slots from the shipped pack without touching filled History slots.

    Only applies when the slot has no binding and no existing draft. Never
    overwrites Case slots that already have sticky History bindings.
    """
    pack_data = dict(pack) if pack is not None else load_default_pack()
    out_bindings = dict(bindings)
    out_drafts = _parse_drafts(dict(drafts))
    used_keys = {str(k).strip() for k in out_bindings.values() if str(k or "").strip()}
    for idx, entry in sorted(pack_data.items()):
        if idx in out_bindings:
            continue
        if idx in out_drafts and (
            out_drafts[idx].get("stem") or out_drafts[idx].get("gold_raw")
        ):
            # Keep user/owner draft; still ensure binding if missing.
            key = str(out_drafts[idx].get("stem_key") or "").strip()
            if not key and out_drafts[idx].get("stem"):
                key = stem_key(out_drafts[idx]["stem"])
                out_drafts[idx]["stem_key"] = key
            if key and key not in used_keys:
                out_bindings[idx] = key
                used_keys.add(key)
            continue
        stem = str(entry.get("stem") or "").strip()
        if not stem:
            continue
        key = str(entry.get("stem_key") or stem_key(stem)).strip()
        if not key or key in used_keys:
            continue
        out_bindings[idx] = key
        used_keys.add(key)
        out_drafts[idx] = {
            "stem": stem,
            "gold_raw": str(entry.get("gold_raw") or "").strip(),
            "stem_key": key,
        }
    return out_bindings, out_drafts


def force_seed_pack_slots(
    bindings: Mapping[int, str],
    drafts: Mapping[int, Mapping[str, str]],
    *,
    slot_indexes_to_seed: Iterable[int],
    pack: Mapping[int, Mapping[str, str]] | None = None,
) -> tuple[dict[int, str], dict[int, dict[str, str]]]:
    """Overwrite listed slots with pack stem + freeform gold (History artifacts kept).

    Used for one-time pack revision migrations (e.g. replace Case 2 arthritis/SLE
    with anaphylaxis). Does not delete old-stem artifacts; they become unslotted.
    """
    pack_data = dict(pack) if pack is not None else load_default_pack()
    out_bindings = dict(bindings)
    out_drafts = _parse_drafts(dict(drafts))
    for raw_idx in slot_indexes_to_seed:
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        entry = pack_data.get(idx)
        if not entry:
            continue
        stem = str(entry.get("stem") or "").strip()
        if not stem:
            continue
        key = str(entry.get("stem_key") or stem_key(stem)).strip()
        if not key:
            continue
        # Drop any other slot that already owns this pack stem.
        for other, existing in list(out_bindings.items()):
            if existing == key and other != idx:
                del out_bindings[other]
                out_drafts.pop(other, None)
        out_bindings[idx] = key
        out_drafts[idx] = {
            "stem": stem,
            "gold_raw": str(entry.get("gold_raw") or "").strip(),
            "stem_key": key,
        }
    return out_bindings, out_drafts


def migrate_bindings(
    artifacts: Sequence[RunArtifact],
    existing: Mapping[int, str] | None = None,
    *,
    slot_count: int = BASE_CASE_SLOTS,
) -> dict[int, str]:
    """Keep sticky bindings; auto-fill only empty base slots 1–5.

    Unassigned protocol-valid families fill empty **base** slots by recency.
    Slots 6+ are never auto-migrated — only created via New case / Confirm.
    """
    families = discover_stem_families(artifacts)
    by_key = {f.stem_key: f for f in families}
    bindings: dict[int, str] = {}
    used: set[str] = set()
    indexes = slot_indexes(slot_count)

    for idx in indexes:
        key = str((existing or {}).get(idx) or "").strip()
        if key and key in by_key and key not in used:
            bindings[idx] = key
            used.add(key)
        elif key and key not in by_key:
            # Keep sticky hash even if History not loaded yet (session-only stem).
            bindings[idx] = key
            used.add(key)

    # Auto-migrate only protocol-valid families into empty **base** slots.
    base_indexes = slot_indexes(BASE_CASE_SLOTS)
    for fam in families:
        if not fam.has_protocol_cohort:
            continue
        if fam.stem_key in used:
            continue
        empty = next((i for i in base_indexes if i not in bindings), None)
        if empty is None:
            break
        bindings[empty] = fam.stem_key
        used.add(fam.stem_key)
    return bindings


def resolve_slots(
    artifacts: Sequence[RunArtifact],
    bindings: Mapping[int, str],
    *,
    slot_count: int = BASE_CASE_SLOTS,
    drafts: Mapping[int, Mapping[str, str]] | None = None,
) -> list[CaseSlot]:
    """Resolve slot views (stem + gold + runs) from bindings + artifacts + drafts."""
    by_key = {f.stem_key: f for f in discover_stem_families(artifacts)}
    draft_map = _parse_drafts(dict(drafts or {}))
    slots: list[CaseSlot] = []
    for idx in slot_indexes(slot_count):
        key = str(bindings.get(idx) or "").strip()
        draft = draft_map.get(idx) or {}
        fam = by_key.get(key) if key else None
        if not fam:
            stem = str(draft.get("stem") or "").strip()
            gold_raw = str(draft.get("gold_raw") or "").strip()
            if not key and stem:
                key = str(draft.get("stem_key") or stem_key(stem)).strip()
            if key or stem:
                # Sticky binding and/or draft without a History artifact yet.
                slots.append(
                    CaseSlot(
                        index=idx,
                        label=f"Case {idx}",
                        stem_key=key,
                        stem=stem,
                        gold_raw=gold_raw,
                        run_ids=[],
                    )
                )
            else:
                slots.append(empty_slot(idx))
            continue
        slots.append(
            CaseSlot(
                index=idx,
                label=f"Case {idx}",
                stem_key=fam.stem_key,
                stem=fam.stem,
                gold_reference=fam.gold_reference,
                # Keep freeform draft only when confirmed gold JSON is absent.
                gold_raw=(
                    ""
                    if fam.gold_reference
                    else str(draft.get("gold_raw") or "").strip()
                ),
                cohort_id=fam.cohort_id,
                run_ids=list(fam.run_ids),
                run_count=fam.run_count,
                latest_finished_at=fam.latest_finished_at,
            )
        )
    return slots


def ensure_owner_slots(
    workspace: Path,
    artifacts: Sequence[RunArtifact],
    *,
    session_bindings: Mapping[int, str] | None = None,
    session_slot_count: int | None = None,
    session_drafts: Mapping[int, Mapping[str, str]] | None = None,
    persist: bool = True,
    apply_defaults: bool = True,
) -> tuple[list[CaseSlot], dict[int, str], int, dict[int, dict[str, str]]]:
    """Load/migrate sticky bindings, seed empty defaults, resolve Case 1–N.

    Returns ``(slots, bindings, slot_count, drafts)``. Defaults fill only empty
    slots, except ``force_seed_slots`` on a pack revision bump (Case 2 anaphylaxis
    migration). Case 1/3/4/5 History bindings stay unless listed in force_seed.
    """
    disk, disk_count = load_slot_state(workspace)
    disk_pack_rev = load_pack_revision(workspace)
    merged: dict[int, str] = dict(disk)
    for idx, key in (session_bindings or {}).items():
        try:
            i = int(idx)
        except (TypeError, ValueError):
            continue
        k = str(key or "").strip()
        if 1 <= i <= SOFT_MAX_CASE_SLOTS and k:
            merged[i] = k
    drafts = load_drafts(workspace)
    for idx, draft in _parse_drafts(dict(session_drafts or {})).items():
        drafts[idx] = draft
    max_bound = max(merged.keys(), default=0)
    slot_count = max(
        BASE_CASE_SLOTS,
        disk_count,
        int(session_slot_count or 0),
        max_bound,
    )
    slot_count = min(slot_count, SOFT_MAX_CASE_SLOTS)
    bindings = migrate_bindings(artifacts, merged, slot_count=slot_count)
    pack_rev_to_store = disk_pack_rev
    if apply_defaults:
        pack_data = load_default_pack()
        pack_rev, force_slots = load_default_pack_meta()
        bindings, drafts = apply_default_pack_to_empty_slots(
            bindings, drafts, pack=pack_data
        )
        if force_slots and pack_rev > disk_pack_rev:
            bindings, drafts = force_seed_pack_slots(
                bindings,
                drafts,
                slot_indexes_to_seed=force_slots,
                pack=pack_data,
            )
            pack_rev_to_store = pack_rev
        elif pack_rev_to_store < pack_rev and not force_slots:
            # Pack advanced with empty-slot-only seeds; record revision.
            pack_rev_to_store = pack_rev
        max_bound = max(bindings.keys(), default=0)
        slot_count = max(slot_count, max_bound, BASE_CASE_SLOTS)
        slot_count = min(slot_count, SOFT_MAX_CASE_SLOTS)
    if persist:
        try:
            save_bindings(
                workspace,
                bindings,
                slot_count=slot_count,
                pack_revision=pack_rev_to_store,
            )
        except OSError:
            pass
        try:
            save_drafts(workspace, drafts)
        except OSError:
            pass
    slots = resolve_slots(
        artifacts, bindings, slot_count=slot_count, drafts=drafts
    )
    return slots, bindings, slot_count, drafts


def next_empty_slot(slots: Sequence[CaseSlot]) -> int | None:
    for slot in slots:
        if not slot.filled:
            return slot.index
    return None


def open_new_case_slot(
    slots: Sequence[CaseSlot],
    *,
    slot_count: int,
) -> tuple[int, int]:
    """Pick next empty Case, or grow the bar (Case 6, 7, …).

    Returns ``(slot_index, new_slot_count)``. Raises ValueError at soft max.
    """
    empty = next_empty_slot(slots)
    current = max(
        BASE_CASE_SLOTS,
        int(slot_count or BASE_CASE_SLOTS),
        max((s.index for s in slots), default=0),
    )
    if empty is not None:
        return empty, current
    new_idx = current + 1
    if new_idx > SOFT_MAX_CASE_SLOTS:
        raise ValueError(f"case slot soft max is {SOFT_MAX_CASE_SLOTS}")
    return new_idx, new_idx


def bind_stem_to_slot(
    bindings: Mapping[int, str],
    *,
    slot_index: int,
    case_stem: str,
) -> dict[int, str]:
    """Bind (or re-bind) a slot to a clinical stem (indexes 1…soft max)."""
    if not (1 <= int(slot_index) <= SOFT_MAX_CASE_SLOTS):
        raise ValueError(f"slot_index must be 1–{SOFT_MAX_CASE_SLOTS}")
    key = stem_key(case_stem)
    if not key:
        raise ValueError("case stem is empty")
    out = dict(bindings)
    # If this stem already owns another slot, keep that slot (no silent swap).
    for idx, existing in list(out.items()):
        if existing == key and idx != slot_index:
            return out
    out[int(slot_index)] = key
    return out


def slot_for_stem_key(
    bindings: Mapping[int, str], stem_key_value: str
) -> int | None:
    key = str(stem_key_value or "").strip()
    if not key:
        return None
    for idx in sorted(int(i) for i in bindings.keys()):
        if bindings.get(idx) == key:
            return idx
    return None


def slot_label_for_artifact(
    art: RunArtifact, bindings: Mapping[int, str]
) -> str:
    idx = slot_for_stem_key(bindings, artifact_stem_key(art))
    if idx is None:
        return "Unslotted"
    return f"Case {idx}"


def count_distinct_stem_keys(artifacts: Iterable[RunArtifact]) -> int:
    keys = {artifact_stem_key(a) for a in artifacts if artifact_stem_key(a)}
    return len(keys)


def filter_artifacts_for_slot(
    artifacts: Sequence[RunArtifact],
    slot: CaseSlot,
) -> list[RunArtifact]:
    """Same-case cohort history for one slot (by stem_key)."""
    if not slot.stem_key:
        return []
    return [a for a in artifacts if artifact_stem_key(a) == slot.stem_key]


def validate_gold_for_restore(gold_reference: str) -> bool:
    if not (gold_reference or "").strip():
        return False
    try:
        load_confirmed_gold(gold_reference)
        return True
    except Exception:  # noqa: BLE001 — invalid gold JSON must not break slot UI
        return False
