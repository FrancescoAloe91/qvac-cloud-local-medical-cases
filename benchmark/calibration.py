"""Offline judge calibration helper (no paid API calls).

Compares stored per-section score components against human expected ranges
in fixtures/calibration/*.json. The judge is "calibrated" only when fixtures
are reviewed and this check passes with nontrivial constraints.
The whole-run verifier is not calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from benchmark.schema import RunArtifact

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = ROOT / "fixtures" / "calibration"

# Full-domain ranges do not constrain anything — reject as tautological.
_TAUTOLOGY = {
    ("coverage", 0.0, 1.0),
    ("quality", 0.0, 1.0),
    ("discipline", 0.0, 1.0),
    ("score", 0.0, 100.0),
    ("score", 0, 100),
}


def load_fixture(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed calibration fixture JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Invalid calibration fixture (not an object): {path}")
    if "expected" not in data or not isinstance(data.get("expected"), dict):
        raise ValueError(f"Invalid calibration fixture (missing expected): {path}")
    if not data["expected"]:
        raise ValueError(f"Invalid calibration fixture (empty expected): {path}")
    return data


def _range_is_tautological(metric: str, lo: float, hi: float) -> bool:
    if metric in ("coverage", "quality", "discipline"):
        return lo <= 0.0 and hi >= 1.0
    if metric == "score":
        return lo <= 0.0 and hi >= 100.0
    return False


def check_section(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> List[str]:
    """Return human-readable violations for one section.

    Missing expected metrics fail. Full-domain ranges fail as non-meaningful.
    """
    failures: List[str] = []
    mapping = {
        "coverage": ("coverage_min", "coverage_max"),
        "quality": ("quality_min", "quality_max"),
        "discipline": ("discipline_min", "discipline_max"),
        "score": ("score_min", "score_max"),
    }
    expected_keys = set()
    for key, (lo_k, hi_k) in mapping.items():
        if lo_k in expected or hi_k in expected:
            expected_keys.add(key)
    if not expected_keys:
        failures.append("no metric ranges declared")
        return failures

    for key in sorted(expected_keys):
        lo_k, hi_k = mapping[key]
        if key not in observed:
            failures.append(f"missing observed metric: {key}")
            continue
        try:
            value = float(observed[key])
        except (TypeError, ValueError):
            failures.append(f"non-numeric observed {key}={observed.get(key)!r}")
            continue
        lo = float(expected.get(lo_k, float("-inf")))
        hi = float(expected.get(hi_k, float("inf")))
        if _range_is_tautological(key, lo, hi):
            failures.append(
                f"{key} range [{lo}, {hi}] is full-domain (not a meaningful constraint)"
            )
            continue
        if value < lo or value > hi:
            failures.append(f"{key}={value} outside [{lo}, {hi}]")
    return failures


def compare_fixture(
    fixture: Mapping[str, Any],
    observed_sections: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Offline compare observed section metrics to fixture expected ranges."""
    expected = fixture.get("expected") or {}
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Fixture expected map is empty or invalid")
    section_results: Dict[str, Any] = {}
    ok = True
    for section_id, ranges in expected.items():
        if not isinstance(ranges, Mapping):
            raise ValueError(f"Malformed expected section {section_id}")
        obs = observed_sections.get(section_id)
        if obs is None:
            fails = [f"missing observed section: {section_id}"]
        else:
            fails = check_section(obs, ranges)
        section_results[section_id] = {
            "ok": not fails,
            "failures": fails,
            "observed": dict(obs or {}),
        }
        if fails:
            ok = False
    reviewed = bool(fixture.get("reviewed_by") and fixture.get("reviewed_at"))
    return {
        "fixture_id": fixture.get("fixture_id"),
        "reviewed": reviewed,
        "ok": ok,
        "sections": section_results,
        "calibrated": bool(ok and reviewed),
        "note": (
            "Calibrated only when reviewed_by/reviewed_at are set, constraints are "
            "nontrivial, and ok=True. Verifier ≠ calibration."
        ),
    }


def observed_sections_from_artifact(
    artifact: RunArtifact, *, candidate_key: str
) -> Dict[str, Dict[str, float]]:
    """Extract per-section coverage/quality/discipline/score from a saved judgment."""
    judgment = next(
        (j for j in artifact.judgments if j.candidate_key == candidate_key),
        None,
    )
    if judgment is None:
        raise ValueError(f"No judgment for candidate_key={candidate_key!r}")
    out: Dict[str, Dict[str, float]] = {}
    for qs in judgment.question_scores or []:
        sid = str(qs.question_id)
        row: Dict[str, float] = {"score": float(qs.score)}
        if qs.recall is not None:
            row["coverage"] = float(qs.recall)
        if qs.quality is not None:
            row["quality"] = float(qs.quality)
        # discipline often derived; use precision as proxy when present
        if qs.precision is not None:
            row["discipline"] = float(qs.precision)
        out[sid] = row
    return out


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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Judge calibration helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser(
        "extract-observed",
        help="Extract observed section metrics from a saved run artifact JSON",
    )
    ex.add_argument("artifact", type=Path)
    ex.add_argument("--candidate-key", required=True)
    ex.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.cmd == "extract-observed":
        art = RunArtifact.model_validate_json(
            args.artifact.read_text(encoding="utf-8")
        )
        observed = observed_sections_from_artifact(
            art, candidate_key=args.candidate_key
        )
        text = json.dumps(observed, indent=2)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
