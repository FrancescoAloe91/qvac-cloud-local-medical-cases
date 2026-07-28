#!/usr/bin/env python3
"""Offline compare judge outputs to fixtures/calibration expected ranges.

Usage (no API):
  PYTHONPYCACHEPREFIX=.pycache .venv/bin/python scripts/compare_judge_calibration.py \\
    --observed path/to/observed.json

observed.json shape:
  {"example-partial-diagnosis": {"diagnosis": {"coverage": 0.5, "quality": 0.8, "discipline": 1.0, "score": 68}}}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.calibration import summarize_directory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=ROOT / "fixtures" / "calibration",
        help="Directory of calibration fixture JSON files",
    )
    parser.add_argument(
        "--observed",
        type=Path,
        required=True,
        help="JSON map fixture_id -> section_id -> {coverage,quality,discipline,score}",
    )
    args = parser.parse_args()
    observed = json.loads(args.observed.read_text(encoding="utf-8"))
    summary = summarize_directory(observed, directory=args.fixtures)
    print(json.dumps(summary, indent=2))
    if not summary.get("calibrated"):
        print(
            "\nNot calibrated: fixtures must be human-reviewed and within range. "
            "Verifier ≠ calibration.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
