"""CLI: python -m benchmark run|dry-run|list-cases"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency)."""
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def cmd_list_cases(_: argparse.Namespace) -> int:
    from benchmark.cases_loader import list_case_ids, load_case

    for cid in list_case_ids():
        case = load_case(cid)
        print(f"{cid}: {case.title} ({len(case.questions)} questions)")
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    from benchmark.runner import dry_run_estimate

    case_ids = [args.case or "caseC"]
    gold = Path(args.gold_file).read_text(encoding="utf-8") if args.gold_file else ""
    stem = Path(args.stem_file).read_text(encoding="utf-8") if args.stem_file else ""
    est = dry_run_estimate(
        case_ids,
        args.n,
        models_path=Path(args.models) if args.models else None,
        skip_qvac=args.skip_qvac,
        gold_reference=gold,
        case_stem_override=stem,
    )
    print(json.dumps(est, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from benchmark.report import print_summary_table
    from benchmark.runner import run_n

    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "ERROR: set OPENROUTER_API_KEY (or add it to .env)",
            file=sys.stderr,
        )
        return 2

    from benchmark.workspace import scoped_artifacts_dir

    out = Path(args.out) if args.out else scoped_artifacts_dir()
    if not args.gold_file or not args.stem_file:
        print(
            "ERROR: gold-only runs require --stem-file and --gold-file "
            "(confirmed five-section JSON).",
            file=sys.stderr,
        )
        return 2
    gold = Path(args.gold_file).read_text(encoding="utf-8")
    stem = Path(args.stem_file).read_text(encoding="utf-8")
    arts, summary = run_n(
        args.case,
        n=args.n,
        models_path=Path(args.models) if args.models else None,
        skip_qvac=args.skip_qvac,
        out_dir=out,
        seed=args.seed,
        gold_reference=gold,
        case_stem_override=stem,
    )
    print(print_summary_table(summary))
    for a in arts:
        if a.notes:
            print(f"note ({a.run_id}): {a.notes}")
        print(f"artifact: {out / (a.run_id + '.json')}  cost=${a.total_cost_usd:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmark",
        description="Reproducible QVAC vs cloud health benchmark (OpenRouter + LLM judge).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-cases", help="List available cases")
    p_list.set_defaults(func=cmd_list_cases)

    p_dry = sub.add_parser("dry-run", help="Estimate OpenRouter cost without calling APIs")
    p_dry.add_argument("--case", default="caseC", help="Case id (default: caseC)")
    p_dry.add_argument("--n", type=int, default=1, help="Number of runs")
    p_dry.add_argument("--models", default=None, help="Path to models.yaml")
    p_dry.add_argument("--skip-qvac", action="store_true")
    p_dry.add_argument("--stem-file", default=None, help="Clinical case text file")
    p_dry.add_argument("--gold-file", default=None, help="Confirmed five-section gold JSON")
    p_dry.set_defaults(func=cmd_dry_run)

    p_run = sub.add_parser("run", help="Run benchmark and write artifacts/")
    p_run.add_argument("--case", default="caseC", help="Case id (gold-only: caseC)")
    p_run.add_argument("--n", type=int, default=1, help="Repeated runs for distribution")
    p_run.add_argument("--models", default=None, help="Path to models.yaml")
    p_run.add_argument("--out", default=None, help="Artifacts directory")
    p_run.add_argument("--skip-qvac", action="store_true", help="Cloud-only candidates")
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--stem-file", required=True, help="Clinical case text file")
    p_run.add_argument("--gold-file", required=True, help="Confirmed five-section gold JSON")
    p_run.set_defaults(func=cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
