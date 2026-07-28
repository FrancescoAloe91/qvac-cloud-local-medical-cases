"""Offline judge calibration helper (no paid API calls).

Compares stored per-section score components against human expected ranges
in fixtures/calibration/*.json. The judge is "calibrated" only when fixtures
are reviewed and this check passes. The whole-run verifier is not calibration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = ROOT / "fixtures" / "calibration"


def load_fixture(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "expected" not in data:
        raise ValueError(f"Invalid calibration fixture: {path}")
    return data


def check_section(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> List[str]:
    """Return human-readable violations for one section."""
    failures: List[str] = []
    mapping = {
        "coverage": ("coverage_min", "coverage_max"),
        "quality": ("quality_min", "quality_max"),
        "discipline": ("discipline_min", "discipline_max"),
        "score": ("score_min", "score_max"),
    }
    for key, (lo_k, hi_k) in mapping.items():
        if key not in observed:
            continue
        if lo_k not in expected and hi_k not in expected:
            continue
        value = float(observed[key])
        lo = float(expected.get(lo_k, float("-inf")))
        hi = float(expected.get(hi_k, float("inf")))
        if value < lo or value > hi:
            failures.append(f"{key}={value} outside [{lo}, {hi}]")
    return failures


def compare_fixture(
    fixture: Mapping[str, Any],
    observed_sections: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Offline compare observed section metrics to fixture expected ranges."""
    expected = fixture.get("expected") or {}
    section_results: Dict[str, Any] = {}
    ok = True
    for section_id, ranges in expected.items():
        obs = observed_sections.get(section_id) or {}
        fails = check_section(obs, ranges if isinstance(ranges, Mapping) else {})
        section_results[section_id] = {
            "ok": not fails,
            "failures": fails,
            "observed": dict(obs),
        }
        if fails:
            ok = False
    return {
        "fixture_id": fixture.get("fixture_id"),
        "reviewed": bool(fixture.get("reviewed_by") and fixture.get("reviewed_at")),
        "ok": ok,
        "sections": section_results,
        "note": (
            "Calibrated only when reviewed_by/reviewed_at are set and ok=True. "
            "Verifier ≠ calibration."
        ),
    }


def iter_fixtures(directory: Optional[Path] = None) -> List[Path]:
    root = directory or DEFAULT_FIXTURES
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.json") if p.is_file())


def summarize_directory(
    observed_by_fixture: Mapping[str, Mapping[str, Mapping[str, Any]]],
    directory: Optional[Path] = None,
) -> Dict[str, Any]:
    rows = []
    for path in iter_fixtures(directory):
        fixture = load_fixture(path)
        fid = str(fixture.get("fixture_id") or path.stem)
        observed = observed_by_fixture.get(fid) or {}
        rows.append(compare_fixture(fixture, observed))
    return {
        "n": len(rows),
        "all_ok": all(r.get("ok") for r in rows) if rows else False,
        "all_reviewed": all(r.get("reviewed") for r in rows) if rows else False,
        "calibrated": bool(rows)
        and all(r.get("ok") and r.get("reviewed") for r in rows),
        "results": rows,
    }
