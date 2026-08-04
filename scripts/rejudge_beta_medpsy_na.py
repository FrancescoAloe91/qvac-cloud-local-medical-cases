#!/usr/bin/env python3
"""Rejudge Beta MedPsy N/A rows with think-stripped answers (DeepSeek API).

Default is dry-run (list eligible targets; no History rewrite). Pass ``--write``
to persist rejudged artifacts in place under the owner workspace.

Usage:
  python scripts/rejudge_beta_medpsy_na.py
  python scripts/rejudge_beta_medpsy_na.py --dry-run
  python scripts/rejudge_beta_medpsy_na.py --write
  python scripts/rejudge_beta_medpsy_na.py --write --owner artifacts/owners/<id>

Requires OPENROUTER_API_KEY for ``--write`` (env or repo `.env`).
Does not interrupt live Streamlit / collect processes — only rewrites finished
beta-*.json artifacts when ``--write`` is set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.beta_rejudge import (  # noqa: E402
    rejudge_owner_beta_medpsy_na,
    resolve_openrouter_key,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--owner",
        type=Path,
        default=ROOT / "artifacts/owners/893e6a29cf690fbef4d6aee2",
        help="Owner artifact directory",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Persist rejudged artifacts in place (default: dry-run, no rewrite)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List targets only (default; use with or without --write conflict check)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path (default: owner/_beta_medpsy_rejudge_report.json)",
    )
    args = ap.parse_args()
    if args.write and args.dry_run:
        print("Use either --write or --dry-run, not both.", file=sys.stderr)
        return 2
    dry_run = not args.write
    owner = args.owner.resolve()
    key = resolve_openrouter_key()
    if not dry_run and not key.startswith("sk-or-"):
        print(
            "OPENROUTER_API_KEY missing. Export it or put it in .env, then re-run.\n"
            "  export OPENROUTER_API_KEY=sk-or-v1-...\n"
            "  python scripts/rejudge_beta_medpsy_na.py --write",
            file=sys.stderr,
        )
        return 2
    summary = rejudge_owner_beta_medpsy_na(
        owner,
        api_key=key,
        dry_run=dry_run,
        limit=args.limit,
    )
    report_path = args.report or (owner / "_beta_medpsy_rejudge_report.json")
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "reports"}, indent=2))
    print(f"report={report_path}")
    for r in summary.get("reports") or []:
        rid = r.get("run_id")
        if r.get("dry_run"):
            print(f"  dry {rid}: keys={r.get('keys')}")
        else:
            print(
                f"  {rid}: recovered={r.get('recovered')} still_na={r.get('still_na')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
