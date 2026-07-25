"""Artifact I/O and multi-run statistics."""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmark.cases_loader import load_case
from benchmark.prompts import use_gold_ground_truth
from benchmark.schema import MultiRunSummary, RunArtifact
from benchmark.scoring import (
    WEIGHTED_CAP,
    linear_item_score,
    semantic_item_score,
    soft_alignment_from_checklist,
)

# Multi-run mean reliability from CV% = 100 × std / mean
# High and Medium ceilings are 10 percentage points apart.
CV_HIGH_MAX = 20.0  # High if CV ≤ 20%
CV_MEDIUM_MAX = 30.0  # Medium if CV ≤ 30%; else Low


def reliability_from_cv(cv_pct: float) -> str:
    """Map coefficient of variation (%) → high / medium / low."""
    if cv_pct <= CV_HIGH_MAX:
        return "high"
    if cv_pct <= CV_MEDIUM_MAX:
        return "medium"
    return "low"


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
        # Coefficient of variation (%) — simple reliability signal for the mean
        cv_pct = round(100.0 * std / mean, 1) if mean > 1e-6 else 0.0
        reliability = reliability_from_cv(cv_pct)
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
            "cv_pct": cv_pct,
            "reliability": reliability,
            "iqr": round(iqr, 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "n": float(len(vals)),
        }
        # Flag high variance
        if len(vals) >= 3 and std > 15:
            outliers.append(f"{key}: high variance std={std:.1f} (CV {cv_pct}%)")
        if reliability == "low" and len(vals) >= 2:
            outliers.append(f"{key}: mean less reliable (CV {cv_pct}%)")
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
            "cv_pct": v["cv_pct"],
            "reliability": v["reliability"],
            "iqr": v["iqr"],
            "min": v["min"],
            "max": v["max"],
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
        f"{'Rank':<6}{'Model':<12}{'Mean%':>7}{'±Std':>7}{'CV%':>6}{'Rel':>8}{'Med%':>7}",
        "-" * 60,
    ]
    for row in summary.ranking_mean:
        lines.append(
            f"{row['rank']:<6}{row['key']:<12}"
            f"{row['accuracy_mean']:>7.1f}{row['std']:>7.1f}"
            f"{row.get('cv_pct', 0):>6.1f}{str(row.get('reliability', '—')):>8}"
            f"{row.get('median', 0):>7.1f}"
        )
    if summary.outliers:
        lines.append("Reliability notes:")
        for o in summary.outliers:
            lines.append(f"  - {o}")
    return "\n".join(lines)


def reliability_caption(summary: MultiRunSummary) -> str:
    """One-line plain-language guide for the multi-run mean."""
    if summary.n < 2:
        return "Single run — no variance yet."
    cvs = [float(r.get("cv_pct") or 0) for r in summary.ranking_mean]
    worst = max(cvs) if cvs else 0.0
    band_key = reliability_from_cv(worst)
    band = {
        "high": "High confidence",
        "medium": "Moderate confidence",
        "low": "Lower confidence — means jump between runs",
    }[band_key]
    return (
        f"{band} across N={summary.n} · "
        f"CV% = std/mean (High ≤{CV_HIGH_MAX:.0f}% · Med ≤{CV_MEDIUM_MAX:.0f}% · else Low) · "
        f"error bars = ±1 std"
    )


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


def _parse_rationale_metrics(rationale: str) -> Optional[Dict[str, float]]:
    """Extract align/m/a/quality/spec from stored judge rationale lines."""
    text = rationale or ""
    al = re.search(r"align=([0-9.]+)", text)
    q = re.search(r"quality=([0-9.]+)", text)
    s = re.search(r"spec=([0-9.]+)", text)
    m = re.search(r"m=(\d+)/(\d+)", text)
    a = re.search(r"a=(\d+)/(\d+)", text)
    if al and q and s:
        return {
            "alignment": float(al.group(1)),
            "quality": float(q.group(1)),
            "spec": float(s.group(1)),
        }
    if m and a and q and s:
        return {
            "m_hit": float(m.group(1)),
            "m_total": float(m.group(2)),
            "a_hit": float(a.group(1)),
            "a_total": float(a.group(2)),
            "quality": float(q.group(1)),
            "spec": float(s.group(1)),
        }
    return None


def rescore_artifact_current_formula(art: RunArtifact) -> Dict[str, Any]:
    """
    Recompute section scores + weighted accuracy with the *current* host formula.

    Uses metrics embedded in stored rationales (no API). Gold runs → semantic
    50/30/20; rubric runs → checklist+quality weights. Returns a lightweight
    ranking dict suitable for summarize_runs / charts.
    """
    cfg = art.models_config or {}
    gold_ref = str(cfg.get("gold_reference") or "")
    gold_mode = use_gold_ground_truth(gold_ref)
    try:
        case = load_case(art.case_id)
        section_w = {q.id: q.weight for q in case.questions}
    except Exception:
        section_w = {}

    ranking_rows: List[Dict[str, Any]] = []
    per_model_sections: Dict[str, Dict[str, float]] = {}

    for j in art.judgments:
        secs: Dict[str, float] = {}
        for qs in j.question_scores:
            parsed = _parse_rationale_metrics(qs.rationale or "")
            if not parsed:
                secs[qs.question_id] = float(qs.score)
                continue
            q = float(parsed["quality"])
            spec = float(parsed["spec"])
            if "alignment" in parsed:
                align = float(parsed["alignment"])
                secs[qs.question_id] = semantic_item_score(
                    alignment=align, quality=q, specificity=spec
                )
            elif gold_mode:
                align = soft_alignment_from_checklist(
                    m_hit=int(parsed["m_hit"]),
                    m_total=int(parsed["m_total"]),
                    a_hit=int(parsed["a_hit"]),
                    a_total=max(int(parsed["a_total"]), 1),
                    quality=q,
                )
                secs[qs.question_id] = semantic_item_score(
                    alignment=align, quality=q, specificity=spec
                )
            else:
                secs[qs.question_id] = linear_item_score(
                    m_hit=int(parsed["m_hit"]),
                    m_total=int(parsed["m_total"]),
                    a_hit=int(parsed["a_hit"]),
                    a_total=max(int(parsed["a_total"]), 1),
                    quality=q,
                    specificity=spec,
                )
        per_model_sections[j.candidate_key] = secs
        if section_w:
            keys = [k for k in section_w if k in secs]
            tw = sum(section_w[k] for k in keys) or 1.0
            acc = sum(secs[k] * section_w[k] for k in keys) / tw
        else:
            acc = (
                sum(secs.values()) / len(secs) if secs else float(j.weighted_accuracy)
            )
        ranking_rows.append(
            {
                "key": j.candidate_key,
                "accuracy": round(min(acc, WEIGHTED_CAP), 2),
                "label": j.candidate_key,
            }
        )

    ranking_rows.sort(key=lambda r: -float(r["accuracy"]))
    for i, row in enumerate(ranking_rows, 1):
        row["rank"] = i

    return {
        "run_id": art.run_id,
        "case_id": art.case_id,
        "n_index": art.n_index,
        "gold_mode": gold_mode,
        "ranking": ranking_rows,
        "sections": per_model_sections,
        "stored_ranking": list(art.ranking or []),
    }


def artifacts_for_case(
    out_dir: Path, case_id: str, *, limit: Optional[int] = None
) -> List[Tuple[Path, RunArtifact]]:
    """Newest-first artifacts for one case that have judgments/ranking."""
    out: List[Tuple[Path, RunArtifact]] = []
    for p in list_run_artifacts(out_dir):
        try:
            art = load_artifact(p)
        except Exception:
            continue
        if art.case_id != case_id:
            continue
        if not art.judgments and not art.ranking:
            continue
        out.append((p, art))
        if limit is not None and len(out) >= limit:
            break
    return out


def rebuild_multi_from_history(
    out_dir: Path,
    case_id: str,
    *,
    n: int = 5,
) -> Dict[str, Any]:
    """
    Offline Multi×N: take the N newest runs for case_id, rescore with the
    current formula, return summarize_runs-compatible summary + per-run rows.
    Zero API cost.
    """
    n = max(2, min(int(n), 20))
    pairs = artifacts_for_case(out_dir, case_id, limit=n)
    if len(pairs) < 2:
        return {
            "ok": False,
            "reason": f"Need at least 2 saved runs for {case_id} (found {len(pairs)}).",
            "available": len(pairs),
        }

    rescored_arts: List[RunArtifact] = []
    per_run: List[Dict[str, Any]] = []
    for path, art in pairs:
        scored = rescore_artifact_current_formula(art)
        # Minimal RunArtifact clone with new ranking accuracies
        clone = art.model_copy(deep=True)
        clone.ranking = scored["ranking"]
        # Keep judgments but update weighted_accuracy for consistency
        by_key = {r["key"]: r["accuracy"] for r in scored["ranking"]}
        for j in clone.judgments:
            if j.candidate_key in by_key:
                j.weighted_accuracy = float(by_key[j.candidate_key])
        rescored_arts.append(clone)
        per_run.append(
            {
                "path": str(path),
                "run_id": art.run_id,
                "finished_at": art.finished_at,
                "ranking": scored["ranking"],
                "gold_mode": scored["gold_mode"],
            }
        )

    summary = summarize_runs(rescored_arts)
    return {
        "ok": True,
        "available": len(pairs),
        "n_used": len(rescored_arts),
        "summary": summary,
        "per_run": per_run,
        "formula": "gold 50/30/20 semantic · or rubric quality-weighted (current code)",
        "api_cost_usd": 0.0,
    }
