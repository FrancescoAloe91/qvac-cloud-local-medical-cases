"""Load structured benchmark cases from JSON."""

from __future__ import annotations

from pathlib import Path
from typing import List

from benchmark.config import CASES_DIR
from benchmark.schema import Case


def list_case_ids() -> List[str]:
    """Active benchmark cases. Demo files, if present locally, stay archived."""
    return ["caseC"] if (CASES_DIR / "caseC.json").exists() else []


def case_display_name(case_id: str) -> str:
    """Human label, including read-only labels for archived artifacts."""
    return {
        "caseC": "Real Case · user reference",
        "caseA": "Legacy archived case A",
        "caseB": "Legacy archived case B",
    }.get(case_id, case_id)


def load_case(case_id: str) -> Case:
    path = CASES_DIR / f"{case_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Case not found: {path}")
    return Case.model_validate_json(path.read_text(encoding="utf-8"))


def load_case_path(path: Path) -> Case:
    return Case.model_validate_json(path.read_text(encoding="utf-8"))
