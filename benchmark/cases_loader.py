"""Load structured benchmark cases from JSON."""

from __future__ import annotations

from pathlib import Path
from typing import List

from benchmark.config import CASES_DIR
from benchmark.schema import Case


def list_case_ids() -> List[str]:
    # Custom Case (caseC) first, then Demo 1 / Demo 2 — ids unchanged for History
    found = {p.stem for p in CASES_DIR.glob("case*.json")}
    preferred = [c for c in ("caseC", "caseA", "caseB") if c in found]
    rest = sorted(found - set(preferred))
    return preferred + rest


def case_display_name(case_id: str) -> str:
    """Human label; internal ids stay caseA/B/C so artifacts keep matching."""
    return {
        "caseC": "Custom Case",
        "caseA": "Demo Case 1",
        "caseB": "Demo Case 2",
    }.get(case_id, case_id)


def load_case(case_id: str) -> Case:
    path = CASES_DIR / f"{case_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Case not found: {path}")
    return Case.model_validate_json(path.read_text(encoding="utf-8"))


def load_case_path(path: Path) -> Case:
    return Case.model_validate_json(path.read_text(encoding="utf-8"))
