"""Artifact I/O and multi-run statistics."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Dict, List

from benchmark.schema import MultiRunSummary, RunArtifact


def write_artifact(artifact: RunArtifact, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{artifact.run_id}.json"
    path.write_text(
        artifact.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return path


def write_summary(summary: MultiRunSummary, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{summary.case_id}-summary-n{summary.n}.json"
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return path


def summarize_runs(artifacts: List[RunArtifact]) -> MultiRunSummary:
    if not artifacts:
        return MultiRunSummary(case_id="", n=0)
    case_id = artifacts[0].case_id
    scores: Dict[str, List[float]] = {}
    for art in artifacts:
        for row in art.ranking:
            scores.setdefault(row["key"], []).append(float(row["accuracy"]))

    stats: Dict[str, Dict[str, float]] = {}
    outliers: List[str] = []
    for key, vals in scores.items():
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        med = statistics.median(vals)
        if len(vals) >= 4:
            q1, _, q3 = statistics.quantiles(vals, n=4)
            iqr = q3 - q1
        elif len(vals) >= 2:
            s = sorted(vals)
            iqr = s[-1] - s[0]
        else:
            iqr = 0.0
        stats[key] = {
            "mean": round(mean, 2),
            "median": round(med, 2),
            "std": round(std, 2),
            "iqr": round(iqr, 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "n": float(len(vals)),
        }
        # Flag high variance
        if len(vals) >= 3 and std > 15:
            outliers.append(f"{key}: high variance std={std:.1f}")
        # Flag bimodal-ish: large gap mid sorted
        if len(vals) >= 4:
            s = sorted(vals)
            mid_gap = s[len(s) // 2] - s[len(s) // 2 - 1]
            if mid_gap > 25:
                outliers.append(f"{key}: possible bimodal gap={mid_gap:.1f}")

    ranking_mean = [
        {
            "key": k,
            "accuracy_mean": v["mean"],
            "median": v["median"],
            "std": v["std"],
            "iqr": v["iqr"],
        }
        for k, v in stats.items()
    ]
    ranking_mean.sort(key=lambda r: r["accuracy_mean"], reverse=True)
    for i, row in enumerate(ranking_mean, 1):
        row["rank"] = i

    total_cost = sum(a.total_cost_usd for a in artifacts)
    return MultiRunSummary(
        case_id=case_id,
        n=len(artifacts),
        candidate_stats=stats,
        ranking_mean=ranking_mean,
        run_ids=[a.run_id for a in artifacts],
        total_cost_usd=round(total_cost, 6),
        outliers=outliers,
    )


def print_summary_table(summary: MultiRunSummary) -> str:
    lines = [
        f"Case {summary.case_id} · N={summary.n} · cost≈${summary.total_cost_usd:.4f}",
        f"{'Rank':<6}{'Model':<16}{'Mean%':>8}{'Med%':>8}{'Std':>8}{'IQR':>8}",
        "-" * 56,
    ]
    for row in summary.ranking_mean:
        lines.append(
            f"{row['rank']:<6}{row['key']:<16}"
            f"{row['accuracy_mean']:>8.1f}{row.get('median', 0):>8.1f}"
            f"{row['std']:>8.1f}{row.get('iqr', 0):>8.1f}"
        )
    if summary.outliers:
        lines.append("Outliers / warnings:")
        for o in summary.outliers:
            lines.append(f"  - {o}")
    return "\n".join(lines)


def list_run_artifacts(out_dir: Path) -> List[Path]:
    """Newest-first JSON run files (excludes multi-run summary files)."""
    if not out_dir.is_dir():
        return []
    paths = [
        p
        for p in out_dir.glob("*.json")
        if "-summary-" not in p.name and p.is_file()
    ]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


def load_artifact(path: Path) -> RunArtifact:
    return RunArtifact.model_validate_json(path.read_text(encoding="utf-8"))
