"""Per-owner clinical case slots (1–5) mapped onto History artifacts.

Slots are sticky bindings from slot index → stem_key (normalized stem hash).
Stem text + confirmed gold are always resolved from existing run artifacts —
this module does not invent clinical content and does not rewrite artifacts.

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

MAX_CASE_SLOTS = 5
SLOTS_FILENAME = "case_slots.json"
SLOT_INDEXES = tuple(range(1, MAX_CASE_SLOTS + 1))


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
    cohort_id: str = ""
    run_ids: list[str] | None = None
    run_count: int = 0
    latest_finished_at: str = ""

    @property
    def filled(self) -> bool:
        return bool(self.stem_key and self.stem)

    @property
    def has_confirmed_gold(self) -> bool:
        return bool(self.gold_reference)


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


def load_bindings(workspace: Path) -> dict[int, str]:
    """Load sticky slot→stem_key bindings (hashes only; no clinical plaintext)."""
    path = Path(workspace) / SLOTS_FILENAME
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    bindings = raw.get("bindings") if isinstance(raw, dict) else None
    if not isinstance(bindings, dict):
        return {}
    out: dict[int, str] = {}
    for k, v in bindings.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if idx not in SLOT_INDEXES:
            continue
        key = str(v or "").strip()
        if key:
            out[idx] = key
    return out


def save_bindings(workspace: Path, bindings: Mapping[int, str]) -> None:
    """Persist sticky bindings (stem hashes only)."""
    path = Path(workspace) / SLOTS_FILENAME
    payload = {
        "version": 1,
        "bindings": {str(i): bindings[i] for i in SLOT_INDEXES if bindings.get(i)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def migrate_bindings(
    artifacts: Sequence[RunArtifact],
    existing: Mapping[int, str] | None = None,
) -> dict[int, str]:
    """Map up to 5 distinct stems into slots 1–5 without moving existing bindings.

    Unassigned protocol-valid families fill empty slots by recency. Families
    without cohort+gold only fill remaining empty slots after protocol ones.
    """
    families = discover_stem_families(artifacts)
    by_key = {f.stem_key: f for f in families}
    bindings: dict[int, str] = {}
    used: set[str] = set()

    for idx in SLOT_INDEXES:
        key = str((existing or {}).get(idx) or "").strip()
        if key and key in by_key and key not in used:
            bindings[idx] = key
            used.add(key)

    # Auto-migrate only protocol-valid families (cohort + gold). Legacy stems
    # without cohort stay in History as unslotted — Case 3–5 remain empty until
    # the user opens New case / Confirm.
    for fam in families:
        if not fam.has_protocol_cohort:
            continue
        if fam.stem_key in used:
            continue
        empty = next((i for i in SLOT_INDEXES if i not in bindings), None)
        if empty is None:
            break
        bindings[empty] = fam.stem_key
        used.add(fam.stem_key)
    return bindings


def resolve_slots(
    artifacts: Sequence[RunArtifact],
    bindings: Mapping[int, str],
) -> list[CaseSlot]:
    """Resolve slot views (stem + gold + runs) from bindings + artifacts."""
    by_key = {f.stem_key: f for f in discover_stem_families(artifacts)}
    slots: list[CaseSlot] = []
    for idx in SLOT_INDEXES:
        key = str(bindings.get(idx) or "").strip()
        fam = by_key.get(key) if key else None
        if not fam:
            slots.append(empty_slot(idx))
            continue
        slots.append(
            CaseSlot(
                index=idx,
                label=f"Case {idx}",
                stem_key=fam.stem_key,
                stem=fam.stem,
                gold_reference=fam.gold_reference,
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
    persist: bool = True,
) -> tuple[list[CaseSlot], dict[int, str]]:
    """Load/migrate sticky bindings, resolve Case 1–5, optionally persist hashes."""
    disk = load_bindings(workspace)
    # Session overrides win for in-flight New case bindings before first run.
    merged: dict[int, str] = dict(disk)
    for idx, key in (session_bindings or {}).items():
        try:
            i = int(idx)
        except (TypeError, ValueError):
            continue
        k = str(key or "").strip()
        if i in SLOT_INDEXES and k:
            merged[i] = k
    bindings = migrate_bindings(artifacts, merged)
    if persist:
        try:
            save_bindings(workspace, bindings)
        except OSError:
            pass
    return resolve_slots(artifacts, bindings), bindings


def next_empty_slot(slots: Sequence[CaseSlot]) -> int | None:
    for slot in slots:
        if not slot.filled:
            return slot.index
    return None


def bind_stem_to_slot(
    bindings: Mapping[int, str],
    *,
    slot_index: int,
    case_stem: str,
) -> dict[int, str]:
    """Bind (or re-bind) a slot to a clinical stem. Cap remains 1–5."""
    if slot_index not in SLOT_INDEXES:
        raise ValueError(f"slot_index must be 1–{MAX_CASE_SLOTS}")
    key = stem_key(case_stem)
    if not key:
        raise ValueError("case stem is empty")
    out = dict(bindings)
    # If this stem already owns another slot, keep that slot (no silent swap).
    for idx, existing in list(out.items()):
        if existing == key and idx != slot_index:
            return out
    out[slot_index] = key
    return out


def slot_for_stem_key(
    bindings: Mapping[int, str], stem_key_value: str
) -> int | None:
    key = str(stem_key_value or "").strip()
    if not key:
        return None
    for idx in SLOT_INDEXES:
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
