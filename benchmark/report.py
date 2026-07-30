"""Artifact I/O and multi-run statistics."""

from __future__ import annotations

import re
import statistics
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmark.cases_loader import load_case
from benchmark.gold import case_family_key, load_confirmed_gold
from benchmark.prompts import use_gold_ground_truth
from benchmark.schema import MultiRunSummary, RunArtifact
from benchmark.scoring import (
    WEIGHTED_CAP,
    graded_clinical_score,
    linear_item_score,
    semantic_item_score,
    soft_alignment_from_checklist,
)
from lib.model_labels import CURRENT_ROSTER_KEYS, is_current_roster_key

# Multi-run mean reliability from CV% = 100 × std / mean
# Five bands (ceilings): Super High ≤5 · High ≤10 · Medium ≤15 · Low ≤20 · else Very Low
CV_SUPER_HIGH_MAX = 5.0
CV_HIGH_MAX = 10.0
CV_MEDIUM_MAX = 15.0
CV_LOW_MAX = 20.0


def reliability_from_cv(cv_pct: float) -> str:
    """Map coefficient of variation (%) → super_high / high / medium / low / very_low."""
    if cv_pct <= CV_SUPER_HIGH_MAX:
        return "super_high"
    if cv_pct <= CV_HIGH_MAX:
        return "high"
    if cv_pct <= CV_MEDIUM_MAX:
        return "medium"
    if cv_pct <= CV_LOW_MAX:
        return "low"
    return "very_low"


def write_artifact(artifact: RunArtifact, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{artifact.run_id}.json"
    _atomic_write(path, artifact.model_dump_json(indent=2))
    return path


def write_summary(summary: MultiRunSummary, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{summary.case_id}-summary-n{summary.n}.json"
    _atomic_write(path, summary.model_dump_json(indent=2))
    return path


def _atomic_write(path: Path, text: str) -> None:
    """Crash-safe replace: readers see either the old or complete new JSON."""
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except OSError:
            pass


def summarize_runs(
    artifacts: List[RunArtifact],
    *,
    allow_mixed_cohorts: bool = False,
    min_valid_for_ranking: int = 5,
) -> MultiRunSummary:
    if not artifacts:
        return MultiRunSummary(case_id="", n=0)
    min_rank_n = max(1, int(min_valid_for_ranking))
    # Pool only on cohort_id (requested recipe + scoring gold). execution_cohort_id
    # is audit metadata — best-effort routing / per-run N/A must not abort Multi means.
    cohort_ids = {(a.cohort_id or "") for a in artifacts}
    if "" in cohort_ids:
        raise ValueError("Cannot summarize runs with empty cohort_id")
    if len(cohort_ids) > 1 and not allow_mixed_cohorts:
        raise ValueError("Cannot summarize mixed cohorts")
    case_ids = {a.case_id for a in artifacts}
    if allow_mixed_cohorts and (len(cohort_ids) > 1 or len(case_ids) > 1):
        case_id = "portfolio"
    else:
        case_id = artifacts[0].case_id
    batch_ids = {(a.batch_id or "") for a in artifacts}
    non_empty_batches = {b for b in batch_ids if b}
    paired_batch_id = next(iter(non_empty_batches)) if len(non_empty_batches) == 1 else None
    scores: Dict[str, List[float]] = {}
    subscales: Dict[str, Dict[str, List[float]]] = {}
    requested: Dict[str, int] = {}
    failures: Dict[str, Dict[str, int]] = {}
    for art in artifacts:
        for row in art.ranking:
            key = str(row.get("key") or "")
            if not is_current_roster_key(key):
                continue  # drop legacy Band B / old cloud keys from means
            requested[key] = requested.get(key, 0) + 1
            status = str(row.get("status") or "ok")
            accuracy = row.get("accuracy")
            if status != "ok" or accuracy is None:
                reason = str(row.get("status_note") or status or "unknown")
                failures.setdefault(key, {})[reason] = (
                    failures.setdefault(key, {}).get(reason, 0) + 1
                )
                continue
            scores.setdefault(key, []).append(float(accuracy))
            for component in ("coverage", "quality", "discipline"):
                value = row.get(component)
                if value is not None:
                    subscales.setdefault(key, {}).setdefault(component, []).append(
                        float(value)
                    )

    stats: Dict[str, Dict[str, Any]] = {}
    outliers: List[str] = []
    for key in requested:
        vals = scores.get(key, [])
        if not vals:
            stats[key] = {
                "mean": None,
                "median": None,
                "std": None,
                "cv_pct": None,
                "reliability": "no_valid_observations",
                "iqr": None,
                "min": None,
                "max": None,
                "n": 0.0,
                "n_runs": 0.0,
                "n_requested": float(requested[key]),
                "n_valid": 0.0,
                "n_failed": float(requested[key]),
                "failure_rate": 1.0,
                "failure_reasons": failures.get(key, {}),
            }
            continue
        mean = statistics.fmean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else None
        med = statistics.median(vals)
        # Coefficient of variation (%) — simple reliability signal for the mean
        cv_pct = round(100.0 * std / mean, 1) if std is not None and mean > 1e-6 else None
        reliability = "exploratory" if len(vals) >= 5 else "insufficient_n"
        if len(vals) >= 4:
            q1, _, q3 = statistics.quantiles(vals, n=4)
            iqr = q3 - q1
        elif len(vals) >= 2:
            s = sorted(vals)
            iqr = s[-1] - s[0]
        else:
            iqr = 0.0
        n_runs = len(vals)
        stats[key] = {
            "mean": round(mean, 2),
            "median": round(med, 2),
            "std": round(std, 2) if std is not None else None,
            "cv_pct": cv_pct,
            "reliability": reliability,
            "iqr": round(iqr, 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "n": float(n_runs),
            "n_runs": float(n_runs),
            "n_requested": float(requested.get(key, n_runs)),
            "n_valid": float(n_runs),
            "n_failed": float(max(0, requested.get(key, n_runs) - n_runs)),
            "failure_rate": round(
                max(0, requested.get(key, n_runs) - n_runs)
                / max(requested.get(key, n_runs), 1),
                4,
            ),
            "failure_reasons": failures.get(key, {}),
            **{
                f"{component}_mean": (
                    round(statistics.fmean(values), 2) if values else None
                )
                for component, values in subscales.get(key, {}).items()
            },
        }
        # Flag high variance (prefer N≥5 for stable CV reads)
        if len(vals) >= 3 and std is not None and std > 15:
            outliers.append(f"{key}: high variance std={std:.1f} (CV {cv_pct}%)")
        # Flag bimodal-ish: large gap mid sorted
        if len(vals) >= 4:
            s = sorted(vals)
            mid_gap = s[len(s) // 2] - s[len(s) // 2 - 1]
            if mid_gap > 25:
                outliers.append(f"{key}: possible bimodal gap={mid_gap:.1f}")

    all_keys = set(requested)
    eligible_keys = {
        key for key in all_keys if len(scores.get(key, [])) >= min_rank_n
    }
    # Include every requested model so Failed % is visible even when a model
    # drops below min_valid_for_ranking (common at Multi N=5: one N/A ⇒ excluded
    # from competitive rank, which previously hid the only non-zero Failed cells).
    ranking_mean = []
    for k, v in stats.items():
        mean_raw = (
            statistics.fmean(scores[k]) if scores.get(k) else None
        )
        n_valid = int(v.get("n_valid") or v.get("n_runs") or v.get("n") or 0)
        is_eligible = k in eligible_keys and mean_raw is not None
        ranking_mean.append(
            {
                "key": k,
                "accuracy_mean": v["mean"],
                "accuracy_mean_raw": mean_raw,
                "median": v["median"],
                "std": v["std"],
                "cv_pct": v["cv_pct"],
                "reliability": v["reliability"],
                "iqr": v["iqr"],
                "min": v["min"],
                "max": v["max"],
                "n_runs": n_valid,
                "n_requested": int(v.get("n_requested") or 0),
                "n_failed": int(v.get("n_failed") or 0),
                "failure_rate": v.get("failure_rate"),
                "failure_reasons": dict(v.get("failure_reasons") or {}),
                "coverage_mean": v.get("coverage_mean"),
                "quality_mean": v.get("quality_mean"),
                "discipline_mean": v.get("discipline_mean"),
                "eligible": is_eligible,
                "exploratory": True,
            }
        )
    ranking_mean.sort(
        key=lambda r: (
            0 if r.get("eligible") else 1,
            -float(
                r["accuracy_mean_raw"]
                if r.get("accuracy_mean_raw") is not None
                else -1
            ),
            -float(r.get("failure_rate") or 0),
            str(r.get("key") or ""),
        ),
    )
    last_mean: Optional[float] = None
    last_rank = 0
    eligible_i = 0
    for row in ranking_mean:
        if not row.get("eligible") or row.get("accuracy_mean_raw") is None:
            row["rank"] = None
            continue
        eligible_i += 1
        mean_value = float(row["accuracy_mean_raw"])
        if last_mean is None or mean_value != last_mean:
            last_mean = mean_value
            last_rank = eligible_i
        row["rank"] = last_rank

    paired_values: Dict[str, List[float]] = {key: [] for key in all_keys}
    paired_components: Dict[str, Dict[str, List[float]]] = {
        key: {component: [] for component in ("coverage", "quality", "discipline")}
        for key in all_keys
    }
    paired_n = 0
    # Paired sensitivity requires the same non-empty batch_id on every artifact.
    # Mixed batches are rejected for paired analysis (means still use all runs).
    if paired_batch_id is not None and "" not in batch_ids:
        for art in artifacts:
            if art.batch_id != paired_batch_id:
                continue
            by_key = {
                str(row.get("key") or ""): row
                for row in art.ranking
                if is_current_roster_key(str(row.get("key") or ""))
            }
            if not all_keys or any(
                key not in by_key
                or str(by_key[key].get("status") or "ok") != "ok"
                or by_key[key].get("accuracy") is None
                for key in all_keys
            ):
                continue
            paired_n += 1
            for key in all_keys:
                paired_values[key].append(float(by_key[key]["accuracy"]))
                for component in ("coverage", "quality", "discipline"):
                    value = by_key[key].get(component)
                    if value is not None:
                        paired_components[key][component].append(float(value))

    paired_ranking: List[Dict[str, Any]] = []
    if paired_n >= 5:
        paired_ranking = []
        for key, values in paired_values.items():
            if len(values) != paired_n:
                continue
            mean_raw = statistics.fmean(values)
            paired_ranking.append(
                {
                    "key": key,
                    "accuracy_mean": round(mean_raw, 2),
                    "accuracy_mean_raw": mean_raw,
                    "n_runs": paired_n,
                    **{
                        f"{component}_mean": (
                            round(statistics.fmean(component_values), 2)
                            if len(component_values) == paired_n
                            else None
                        )
                        for component, component_values in paired_components[key].items()
                    },
                    "paired": True,
                    "exploratory": True,
                }
            )
        paired_ranking.sort(
            key=lambda row: float(row["accuracy_mean_raw"]),
            reverse=True,
        )
        last_mean = None
        last_rank = 0
        for index, row in enumerate(paired_ranking, 1):
            mean_value = float(row["accuracy_mean_raw"])
            if last_mean is None or mean_value != last_mean:
                last_mean = mean_value
                last_rank = index
            row["rank"] = last_rank

    from benchmark.costing import batch_total_cost_usd

    # Sum of per-run costs + extraction once (not ×N).
    total_cost = batch_total_cost_usd(artifacts)
    excluded = sorted(all_keys - eligible_keys)
    exec_ids = {
        (getattr(a, "execution_cohort_id", None) or "") for a in artifacts
    }
    exec_ids_present = {e for e in exec_ids if e}
    if len(exec_ids_present) > 1:
        if case_id == "portfolio":
            outliers.append(
                "execution_cohort_id varied across runs (audit only; mean still same "
                "cohort_id — e.g. per-run N/A or best-effort routing)."
            )
        else:
            outliers.append(
                "Same-case Multi mean pools on cohort_id (requested recipe); "
                "execution_cohort_id varied — routes/N/A/GGUF may differ "
                "(audit only; Portfolio cross-case means unchanged)."
            )
    return MultiRunSummary(
        case_id=case_id,
        n=min((len(scores.get(key, [])) for key in eligible_keys), default=0),
        candidate_stats=stats,
        ranking_mean=ranking_mean,
        paired_ranking=paired_ranking,
        paired_n=paired_n,
        run_ids=[a.run_id for a in artifacts],
        total_cost_usd=round(total_cost, 6),
        outliers=outliers
        + (
            []
            if not excluded
            else [
                "Unranked until "
                + str(min_rank_n)
                + " valid observation(s) (still listed with Failed %): "
                + ", ".join(
                    f"{key} ({len(scores.get(key, []))}/{min_rank_n})"
                    for key in excluded
                )
                + ". Technical failures are N/A; other models keep their valid data."
            ]
        ),
    )


def print_summary_table(summary: MultiRunSummary) -> str:
    lines = [
        f"Case {summary.case_id} · N={summary.n} · cost≈${summary.total_cost_usd:.4f}",
        f"{'Rank':<6}{'Model':<12}{'Mean%':>7}{'±Std':>7}{'CV%':>6}{'Rel':>11}"
        f"{'Med%':>7}{'Runs':>6}{'Fail%':>7}",
        "-" * 78,
    ]
    for row in summary.ranking_mean:
        std = row.get("std")
        cv = row.get("cv_pct")
        mean = row.get("accuracy_mean")
        med = row.get("median")
        fail_pct = 100.0 * float(row.get("failure_rate") or 0)
        rank = row.get("rank")
        lines.append(
            f"{('—' if rank is None else rank):<6}{row['key']:<12}"
            f"{('N/A' if mean is None else f'{float(mean):.1f}'):>7}"
            f"{(f'{std:.1f}' if std is not None else '—'):>7}"
            f"{(f'{cv:.1f}' if cv is not None else '—'):>6}"
            f"{str(row.get('reliability', '—')):>11}"
            f"{('—' if med is None else f'{float(med):.1f}'):>7}"
            f"{int(row.get('n_runs') or 0):>6}"
            f"{fail_pct:>6.0f}%"
        )
    if summary.outliers:
        lines.append("Reliability notes:")
        for o in summary.outliers:
            lines.append(f"  - {o}")
    return "\n".join(lines)


def reliability_caption(summary: MultiRunSummary) -> str:
    """One-line plain-language guide for the multi-run mean."""
    ranked = [
        r
        for r in (summary.ranking_mean or [])
        if r.get("rank") is not None and r.get("eligible", True)
    ]
    if not ranked:
        return (
            "No model has enough valid runs for exploratory ranking yet. "
            "Failed % = share of pooled runs that are technical N/A "
            "(collect/judge/timeout/partial/empty) — not a clinical zero; "
            "means use only scored runs."
        )
    eligible = len(ranked)
    return (
        f"Exploratory ranking for {eligible} model(s) with enough valid runs · "
        "each mean shows its own N; technical N/A never discard other models' data · "
        "Failed % = technical N/A rate across requested runs "
        "(collect/judge/timeout/partial/empty) · "
        "C/Q/D = coverage/quality/discipline (quality independent of coverage) · "
        "sample SD + median/IQR do not measure clinical generalization."
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


def _try_offline_recover_judgment(art: RunArtifact, judgment) -> Optional[Any]:
    """Re-validate stored judge JSON with current local salvage (no API)."""
    if judgment.status not in {
        "judge_schema_invalid",
        "judge_evidence_invalid",
        "judge_transport_failed",
    }:
        return None
    raw = (judgment.raw_judge_json or "").strip()
    if not raw:
        return None
    gold_ref = str((art.models_config or {}).get("gold_reference") or "")
    if not gold_ref:
        return None
    cand = next(
        (c for c in (art.candidates or []) if c.candidate_key == judgment.candidate_key),
        None,
    )
    if cand is None:
        return None
    try:
        from benchmark.judge import (
            _extract_json,
            _score_sections_from_payload,
            _weighted_accuracy,
            _weighted_subscale,
        )

        case = load_case(art.case_id)
        data = _extract_json(raw)
        accepted, errors = _score_sections_from_payload(
            case,
            cand,
            data if isinstance(data, dict) else {},
            gold_reference=gold_ref,
            target_ids={q.id for q in case.questions},
        )
        if errors or len(accepted) != len(case.questions):
            return None
        q_scores = [accepted[q.id] for q in case.questions]
        recovered = judgment.model_copy(deep=True)
        recovered.question_scores = q_scores
        recovered.weighted_accuracy = _weighted_accuracy(case, q_scores)
        recovered.coverage_score = _weighted_subscale(case, q_scores, "recall")
        recovered.quality_score = _weighted_subscale(case, q_scores, "quality")
        recovered.discipline_score = _weighted_subscale(case, q_scores, "precision")
        recovered.status = "valid"
        recovered.failure_reason = (
            (recovered.failure_reason or "").strip()
            + " | offline-recovered with current evidence/schema salvage"
        ).strip(" |")
        return recovered
    except Exception:
        return None


def rescore_artifact_current_formula(art: RunArtifact) -> Dict[str, Any]:
    """
    Recompute section scores + weighted accuracy with the *current* host formula.

    Uses structured claim decisions already stored in artifacts (no API).
    Legacy artifacts fall back to metrics embedded in their rationales.
    N/A judgments with stored judge JSON may be recovered offline when the
    failure was presentation/schema salvageable under current rules.

    graded-clinical-v3 stored quality may already be host-clamped; when the
    original unclamped quality cannot be recovered, preserve the stored ranking
    and never silently stamp the artifact as v4.
    """
    from benchmark.scoring import SCORING_VERSION

    stored_version = str(art.scoring_version or "").strip()
    can_apply_v4 = stored_version == SCORING_VERSION or stored_version.endswith("-v4")
    # Pre-v4 graded artifacts: quality may be clamped; preserve ranking.
    if stored_version.startswith("graded-clinical-v3") or (
        stored_version.startswith("graded-clinical") and not can_apply_v4
    ):
        return {
            "run_id": art.run_id,
            "case_id": art.case_id,
            "n_index": art.n_index,
            "gold_mode": use_gold_ground_truth(
                str((art.models_config or {}).get("gold_reference") or "")
            ),
            "ranking": list(art.ranking or []),
            "sections": {},
            "stored_ranking": list(art.ranking or []),
            "recovered_keys": [],
            "unrecovered_na": [],
            "effective_judgments": list(art.judgments or []),
            "formula": stored_version or "graded-clinical-v3",
            "preserved_stored_ranking": True,
            "scoring_version_stamp": stored_version or "graded-clinical-v3",
        }

    cfg = art.models_config or {}
    gold_ref = str(cfg.get("gold_reference") or "")
    gold_mode = use_gold_ground_truth(gold_ref)
    # Ranking-only fixtures / incomplete offline payloads: keep stored ranks
    # rather than wiping the mean (rebuild still needs ≥5 valid observations).
    if not art.judgments and art.ranking:
        return {
            "run_id": art.run_id,
            "case_id": art.case_id,
            "n_index": art.n_index,
            "gold_mode": gold_mode,
            "ranking": list(art.ranking or []),
            "sections": {},
            "stored_ranking": list(art.ranking or []),
            "recovered_keys": [],
            "unrecovered_na": [],
            "effective_judgments": [],
            "formula": stored_version or SCORING_VERSION,
            "preserved_stored_ranking": True,
            "scoring_version_stamp": stored_version or SCORING_VERSION,
        }
    try:
        case = load_case(art.case_id)
        section_w = {q.id: q.weight for q in case.questions}
    except Exception:
        case = None
        section_w = {}

    ranking_rows: List[Dict[str, Any]] = []
    per_model_sections: Dict[str, Dict[str, float]] = {}
    recovered_keys: List[str] = []
    unrecovered_na: List[Dict[str, str]] = []
    effective_judgments = []

    for j in art.judgments:
        working = j
        if j.status != "valid":
            recovered = _try_offline_recover_judgment(art, j)
            if recovered is not None:
                working = recovered
                recovered_keys.append(j.candidate_key)
            else:
                unrecovered_na.append(
                    {
                        "key": j.candidate_key,
                        "status": str(j.status or "n/a"),
                        "reason": (j.failure_reason or "")[:160],
                    }
                )
                ranking_rows.append(
                    {
                        "key": j.candidate_key,
                        "accuracy": None,
                        "accuracy_raw": None,
                        "label": j.candidate_key,
                        "status": "n/a",
                        "status_note": str(j.status or "n/a"),
                        "rank": None,
                    }
                )
                effective_judgments.append(j)
                continue
        effective_judgments.append(working)
        secs: Dict[str, float] = {}
        for qs in working.question_scores:
            if qs.claim_coverage and qs.recall is not None:
                secs[qs.question_id] = graded_clinical_score(
                    coverage=qs.recall,
                    quality=qs.quality if qs.quality is not None else 0.5,
                    discipline=qs.precision if qs.precision is not None else 1.0,
                )
                continue
            total_claims = len(qs.matched_claim_ids) + len(qs.missed_claim_ids)
            if total_claims:
                # Legacy binary artifacts cannot recover partial coverage. Use
                # their matched ratio as a conservative proxy, treat historically
                # over-broad "unsupported" labels as neutral, and retain explicit
                # contradiction penalties.
                coverage = len(qs.matched_claim_ids) / total_claims
                discipline = max(
                    0.0,
                    1.0 - (0.75 * len(qs.contradictions)) / total_claims,
                )
                score = graded_clinical_score(
                    coverage=coverage,
                    quality=qs.quality if qs.quality is not None else 0.5,
                    discipline=discipline,
                )
                secs[qs.question_id] = score
                continue
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
        per_model_sections[working.candidate_key] = secs
        if section_w:
            keys = [k for k in section_w if k in secs]
            tw = sum(section_w[k] for k in keys) or 1.0
            acc = sum(secs[k] * section_w[k] for k in keys) / tw
        else:
            acc = (
                sum(secs.values()) / len(secs)
                if secs
                else float(working.weighted_accuracy)
            )
        ranking_rows.append(
            {
                "key": working.candidate_key,
                "accuracy": round(min(acc, WEIGHTED_CAP), 2),
                "accuracy_raw": float(min(acc, WEIGHTED_CAP)),
                "label": working.candidate_key,
                "status": "ok",
                "coverage": working.coverage_score,
                "quality": working.quality_score,
                "discipline": working.discipline_score,
            }
        )

    ranking_rows.sort(
        key=lambda r: (
            0 if r.get("status") == "ok" else 1,
            -float(r["accuracy_raw"] if r.get("accuracy_raw") is not None else -1),
        )
    )
    last_score: Optional[float] = None
    last_rank = 0
    for i, row in enumerate(ranking_rows, 1):
        if row.get("status") != "ok":
            row["rank"] = None
            continue
        score = float(row["accuracy_raw"])
        if last_score is None or score != last_score:
            last_rank = i
            last_score = score
        row["rank"] = last_rank

    return {
        "run_id": art.run_id,
        "case_id": art.case_id,
        "n_index": art.n_index,
        "gold_mode": gold_mode,
        "ranking": ranking_rows,
        "sections": per_model_sections,
        "stored_ranking": list(art.ranking or []),
        "recovered_keys": recovered_keys,
        "unrecovered_na": unrecovered_na,
        "effective_judgments": effective_judgments,
        "formula": SCORING_VERSION,
        "preserved_stored_ranking": False,
        "scoring_version_stamp": SCORING_VERSION,
    }


def artifacts_for_case(
    out_dir: Path,
    case_id: str,
    *,
    limit: Optional[int] = None,
    preloaded: Optional[Sequence[RunArtifact]] = None,
) -> List[Tuple[Optional[Path], RunArtifact]]:
    """Newest-first artifacts for one case that have judgments/ranking."""
    out: List[Tuple[Optional[Path], RunArtifact]] = []
    if preloaded is not None:
        # Newest-first assumption: caller passes newest-first or we reverse append order
        for art in preloaded:
            if art.case_id != case_id:
                continue
            if not art.judgments and not art.ranking:
                continue
            out.append((None, art))
            if limit is not None and len(out) >= limit:
                break
        return out
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


def find_case_family_cohorts(
    out_dir: Path,
    *,
    case_stem: str,
    reference_raw: str,
    case_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Group local history by cohort under one case-family key.

    Family = normalized clinical stem + free-form reference raw text.
    Different confirmed gold contracts (claim splits) stay as separate cohorts.
    Never merges cohorts. Newest cohort first.
    """
    stem = (case_stem or "").strip()
    raw = (reference_raw or "").strip()
    if not stem or len(raw) < 40:
        return []
    target = case_family_key(case_stem=stem, reference_raw=raw)
    # cohort_id -> aggregate
    buckets: Dict[str, Dict[str, Any]] = {}
    for path in list_run_artifacts(out_dir):
        try:
            art = load_artifact(path)
        except Exception:
            continue
        if case_id and art.case_id != case_id:
            continue
        if not art.cohort_id:
            continue
        if not is_mean_poolable_run(art):
            continue
        cfg = art.models_config or {}
        gold_ref = str(cfg.get("gold_reference") or "").strip()
        art_stem = str(cfg.get("case_stem") or "")
        if not gold_ref:
            continue
        try:
            gold = load_confirmed_gold(gold_ref)
        except Exception:
            continue
        family = case_family_key(
            case_stem=art_stem, reference_raw=gold.raw_text
        )
        if family != target:
            continue
        bucket = buckets.get(art.cohort_id)
        finished = art.finished_at or art.started_at or ""
        if bucket is None:
            buckets[art.cohort_id] = {
                "cohort_id": art.cohort_id,
                "cohort_short": art.cohort_id[:12],
                "run_count": 1,
                "latest_finished_at": finished,
                "case_stem": art_stem,
                "gold_reference": gold_ref,
                "family_key": family,
                "case_id": art.case_id,
            }
        else:
            bucket["run_count"] = int(bucket["run_count"]) + 1
            if finished >= str(bucket.get("latest_finished_at") or ""):
                bucket["latest_finished_at"] = finished
                # Prefer newest artifact's exact gold JSON for restore.
                bucket["gold_reference"] = gold_ref
                bucket["case_stem"] = art_stem
    rows = list(buckets.values())
    rows.sort(
        key=lambda r: str(r.get("latest_finished_at") or ""),
        reverse=True,
    )
    return rows


def artifact_roster_keys(art: RunArtifact) -> frozenset:
    """Candidate keys from models_config (fallback: collected candidates)."""
    keys: List[str] = []
    for c in (art.models_config or {}).get("candidates") or []:
        if isinstance(c, dict) and c.get("key"):
            keys.append(str(c["key"]))
    if not keys:
        keys = [str(c.candidate_key) for c in (art.candidates or []) if c.candidate_key]
    return frozenset(keys)


def _has_valid_judged_scores(art: RunArtifact) -> bool:
    if any(getattr(j, "status", None) == "valid" for j in (art.judgments or [])):
        return True
    return any(
        str(r.get("status") or "ok") == "ok" and r.get("accuracy") is not None
        for r in (art.ranking or [])
    )


def is_mean_poolable_run(art: RunArtifact) -> bool:
    """True when a finished roster may enter Multi/Portfolio means.

    ``complete`` = every candidate scored valid.
    ``partial`` = at least one technical N/A (candidate_partial / collect /
    judge / timeout / empty) while others may still be valid — those N/A must
    count toward Failed %, so partial runs stay in the pool.

    ``cancelled`` / ``failed`` = abort or hard stop; excluded from official /
    rebuild means (STOP / crash stamps must not dilute Failed or means).
    """
    status = str(art.run_status or "complete").strip().lower()
    return status in {"complete", "partial"}


def list_portfolio_runs(
    out_dir: Path,
    *,
    n: int = 5,
    scoring_version: str = "graded-clinical-v4",
    track: str = "controlled",
    model_ids: Optional[Sequence[str]] = None,
    preloaded: Optional[Sequence[RunArtifact]] = None,
) -> List[Tuple[Optional[Path], RunArtifact]]:
    """Newest-first poolable runs across cases matching protocol filters.

    Filters: same scoring_version, track, complete|partial + ≥1 valid judgment.
    ``partial`` keeps per-model technical N/A in Failed %; cancelled/failed stay out.
    Roster shapes may differ: a run is kept when its keys intersect
    ``model_ids`` (or any current-roster key if omitted). Per-model means
    later use only observations that exist (different N is OK). Chronological
    by finished_at (then started_at), newest first. Cap at ``n`` (1–30).
    """
    n = max(1, min(int(n), 30))
    want_keys = frozenset(model_ids) if model_ids is not None else frozenset(
        CURRENT_ROSTER_KEYS
    )
    want_sv = str(scoring_version or "").strip()
    want_track = str(track or "").strip()
    matched: List[Tuple[Optional[Path], RunArtifact, str]] = []
    if preloaded is not None:
        source: List[Tuple[Optional[Path], RunArtifact]] = [
            (None, a) for a in preloaded
        ]
    else:
        source = []
        for path in list_run_artifacts(out_dir):
            try:
                source.append((path, load_artifact(path)))
            except Exception:
                continue
    for path, art in source:
        if not is_mean_poolable_run(art):
            continue
        if str(art.scoring_version or "").strip() != want_sv:
            continue
        if str(art.benchmark_track or "").strip() != want_track:
            continue
        if not art.cohort_id:
            continue
        art_keys = artifact_roster_keys(art)
        # Heterogeneous rosters OK — keep any run that shares ≥1 wanted key.
        if not (art_keys & want_keys):
            continue
        if not _has_valid_judged_scores(art):
            continue
        when = art.finished_at or art.started_at or ""
        matched.append((path, art, when))
    matched.sort(key=lambda t: t[2], reverse=True)
    return [(p, a) for p, a, _ in matched[:n]]


def _offline_rescore_pair(
    path: Optional[Path], art: RunArtifact
) -> Tuple[RunArtifact, Dict[str, Any]]:
    """Rescore one artifact offline; return clone + per-run row dict."""
    scored = rescore_artifact_current_formula(art)
    clone = art.model_copy(deep=True)
    clone.ranking = scored["ranking"]
    if scored.get("effective_judgments"):
        clone.judgments = list(scored["effective_judgments"])
    else:
        by_key = {
            r["key"]: r["accuracy"]
            for r in scored["ranking"]
            if r.get("accuracy") is not None
        }
        for j in clone.judgments:
            if j.candidate_key in by_key:
                j.weighted_accuracy = float(by_key[j.candidate_key])
    note = (clone.notes or "").strip()
    recovery_note = ""
    if scored.get("recovered_keys"):
        recovery_note = "offline-recovered: " + ", ".join(scored["recovered_keys"])
    if recovery_note and recovery_note not in note:
        clone.notes = (note + " | " + recovery_note).strip(" |")
    reproducibility = dict(clone.reproducibility or {})
    reproducibility["offline_rescore"] = {
        "formula": scored.get("formula") or "graded-clinical-v4",
        "preserved_stored_ranking": bool(scored.get("preserved_stored_ranking")),
        "recovered_keys": list(scored.get("recovered_keys") or []),
        "unrecovered_na": list(scored.get("unrecovered_na") or []),
        "stored_ranking": list(scored.get("stored_ranking") or []),
    }
    clone.reproducibility = reproducibility
    # Never silently rewrite an older scoring_version to v4.
    stamp = scored.get("scoring_version_stamp")
    if stamp and not scored.get("preserved_stored_ranking"):
        clone.scoring_version = str(stamp)
    per_run = {
        "path": str(path) if path else f"memory:{art.run_id}",
        "run_id": art.run_id,
        "case_id": art.case_id,
        "finished_at": art.finished_at,
        "ranking": scored["ranking"],
        "gold_mode": scored["gold_mode"],
        "recovered_keys": scored.get("recovered_keys") or [],
        "unrecovered_na": scored.get("unrecovered_na") or [],
    }
    return clone, per_run


def _mean_ranks_from_per_run(per_run: List[Dict[str, Any]]) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {}
    for pr in per_run:
        for row in pr.get("ranking") or []:
            key = str(row.get("key") or "")
            if not is_current_roster_key(key):
                continue
            if str(row.get("status") or "ok") != "ok" or row.get("rank") is None:
                continue
            buckets.setdefault(key, []).append(float(row["rank"]))
    return {
        key: round(statistics.fmean(vals), 2)
        for key, vals in buckets.items()
        if vals
    }


def rebuild_multi_from_history(
    out_dir: Path,
    case_id: str,
    *,
    n: int = 5,
    cohort_id: Optional[str] = None,
    preloaded: Optional[Sequence[RunArtifact]] = None,
) -> Dict[str, Any]:
    """
    Offline Multi×N: take the N newest runs for case_id, rescore with the
    current formula, return summarize_runs-compatible summary + per-run rows.
    Zero API cost.

    When ``cohort_id`` is set (e.g. after restoring a prior confirmed gold),
    rebuild that immutable cohort instead of the newest case cohort.
    """
    n = max(1, min(int(n), 30))
    all_pairs = artifacts_for_case(
        out_dir, case_id, limit=None, preloaded=preloaded
    )
    if not all_pairs:
        return {
            "ok": False,
            "reason": f"No saved runs for {case_id}.",
            "available": 0,
            "scope": "same_case",
        }
    if cohort_id:
        target_cohort = cohort_id
    else:
        target_cohort = all_pairs[0][1].cohort_id
    if not target_cohort:
        legacy_pairs = all_pairs[:n]
        return {
            "ok": False,
            "reason": (
                "Newest runs are legacy artifacts without a cohort manifest. "
                "They remain available as experimental history but cannot enter an "
                "official mean under the new protocol."
            ),
            "available": len(legacy_pairs),
            "legacy_auto_rescore": True,
            "scope": "same_case",
            "per_run": [
                rescore_artifact_current_formula(artifact)
                for _, artifact in legacy_pairs
            ],
        }
    # Cancelled / failed abort stamps stay out. Partial runs (per-model technical
    # N/A) stay in so Failed % is honest — same rule as portfolio.
    pairs = [
        pair
        for pair in all_pairs
        if pair[1].cohort_id == target_cohort
        and is_mean_poolable_run(pair[1])
    ][:n]
    if len(pairs) < 1:
        return {
            "ok": False,
            "reason": (
                f"Need at least 1 same-cohort run (complete or partial with scores; "
                f"found {len(pairs)}). Cancelled/failed aborts are excluded. "
                "Different stems, references, protocols or model configs cannot be mixed."
            ),
            "available": len(pairs),
            "cohort_id": target_cohort,
            "scope": "same_case",
        }

    rescored_arts: List[RunArtifact] = []
    per_run: List[Dict[str, Any]] = []
    for path, art in pairs:
        clone, row = _offline_rescore_pair(path, art)
        rescored_arts.append(clone)
        per_run.append(row)

    summary = summarize_runs(rescored_arts, min_valid_for_ranking=1)
    return {
        "ok": True,
        "available": len(pairs),
        "n_used": len(rescored_arts),
        "summary": summary,
        "per_run": per_run,
        "formula": "reference-relative Clinical Composite Score · same immutable cohort only",
        "api_cost_usd": 0.0,
        "cohort_id": target_cohort,
        "official": True,
        "scope": "same_case",
        "n_cases": 1,
        "mean_rank": _mean_ranks_from_per_run(per_run),
    }


def rebuild_portfolio_from_history(
    out_dir: Path,
    *,
    n: int = 5,
    scoring_version: str = "graded-clinical-v4",
    track: str = "controlled",
    model_ids: Optional[Sequence[str]] = None,
    preloaded: Optional[Sequence[RunArtifact]] = None,
) -> Dict[str, Any]:
    """Offline exploratory mean across last N poolable runs (all cases).

    Same track + scoring_version. Roster shapes may differ — each model keeps
    its own valid N / failures. Partial runs with technical N/A are included.
    Never merges incompatible scoring versions. Zero API cost. Not clinical
    validation.
    """
    n = max(1, min(int(n), 30))
    want_keys = list(model_ids) if model_ids is not None else list(CURRENT_ROSTER_KEYS)
    # Probe availability beyond n for UI messaging.
    all_eligible = list_portfolio_runs(
        out_dir,
        n=30,
        scoring_version=scoring_version,
        track=track,
        model_ids=want_keys,
        preloaded=preloaded,
    )
    pairs = all_eligible[:n]
    n_cases = len({a.case_id for _, a in pairs})
    if len(pairs) < 1:
        return {
            "ok": False,
            "reason": (
                f"Need at least 1 portfolio-eligible run "
                f"(found {len(pairs)}; {n_cases} distinct case(s)). "
                "Filters: same track + scoring_version, complete|partial + "
                "≥1 valid judgment; roster shapes may differ (per-model N). "
                "Cancelled/failed aborts stay out. "
                "Different scoring versions are never pooled."
            ),
            "available": len(all_eligible),
            "n_cases": n_cases,
            "scope": "portfolio",
            "scoring_version": scoring_version,
            "track": track,
        }

    rescored_arts: List[RunArtifact] = []
    per_run: List[Dict[str, Any]] = []
    for path, art in pairs:
        clone, row = _offline_rescore_pair(path, art)
        rescored_arts.append(clone)
        per_run.append(row)

    summary = summarize_runs(
        rescored_arts, allow_mixed_cohorts=True, min_valid_for_ranking=1
    )
    mean_rank = _mean_ranks_from_per_run(per_run)
    # Attach mean rank onto ranking_mean rows when present.
    if hasattr(summary, "ranking_mean"):
        for row in summary.ranking_mean:
            key = str(row.get("key") or "")
            if key in mean_rank:
                row["mean_rank"] = mean_rank[key]
    return {
        "ok": True,
        "available": len(all_eligible),
        "n_used": len(rescored_arts),
        "n_cases": n_cases,
        "summary": summary,
        "per_run": per_run,
        "formula": (
            "exploratory cross-case mean · mixed roster shapes OK · "
            "per-model valid N · reference-relative scores · not clinical validation"
        ),
        "api_cost_usd": 0.0,
        "official": False,
        "scope": "portfolio",
        "scoring_version": scoring_version,
        "track": track,
        "mean_rank": mean_rank,
        "case_ids": sorted({a.case_id for _, a in pairs}),
    }


def persist_rescored_artifacts(
    out_dir: Path,
    case_id: str,
    *,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Offline rescore recent artifacts in place; keep prior ranking in reproducibility.

    Includes legacy artifacts without cohort_id so History rebuild last 5/10 can
    read stamped ``offline_rescore`` metadata (official means still require a cohort).
    """
    pairs = artifacts_for_case(out_dir, case_id, limit=limit)
    written: List[str] = []
    comparisons: List[Dict[str, Any]] = []
    recovered_total = 0
    unrecovered_total = 0
    for path, art in pairs:
        scored = rescore_artifact_current_formula(art)
        clone = art.model_copy(deep=True)
        old_rank = [
            {
                "key": r.get("key"),
                "accuracy": r.get("accuracy"),
                "status": r.get("status", "ok"),
                "rank": r.get("rank"),
            }
            for r in (art.ranking or [])
        ]
        clone.ranking = scored["ranking"]
        if scored.get("effective_judgments"):
            clone.judgments = list(scored["effective_judgments"])
        reproducibility = dict(clone.reproducibility or {})
        reproducibility["offline_rescore"] = {
            "formula": scored.get("formula") or "graded-clinical-v4",
            "preserved_stored_ranking": bool(scored.get("preserved_stored_ranking")),
            "recovered_keys": list(scored.get("recovered_keys") or []),
            "unrecovered_na": list(scored.get("unrecovered_na") or []),
            "stored_ranking": old_rank,
        }
        clone.reproducibility = reproducibility
        stamp = scored.get("scoring_version_stamp")
        if stamp and not scored.get("preserved_stored_ranking"):
            clone.scoring_version = str(stamp)
        if scored.get("recovered_keys"):
            note = (clone.notes or "").strip()
            tag = "offline-recovered: " + ", ".join(scored["recovered_keys"])
            if tag not in note:
                clone.notes = (note + " | " + tag).strip(" |")
        write_artifact(clone, path.parent if path.parent != out_dir else out_dir)
        # write_artifact uses run_id filename; ensure we overwrite the same path
        if path.name != f"{clone.run_id}.json":
            _atomic_write(path, clone.model_dump_json(indent=2))
        written.append(str(path))
        recovered_total += len(scored.get("recovered_keys") or [])
        unrecovered_total += len(scored.get("unrecovered_na") or [])
        comparisons.append(
            {
                "path": str(path),
                "run_id": art.run_id,
                "cohort_id": art.cohort_id,
                "old": old_rank,
                "new": scored["ranking"],
                "recovered_keys": scored.get("recovered_keys") or [],
                "unrecovered_na": scored.get("unrecovered_na") or [],
            }
        )
    return {
        "ok": True,
        "written": written,
        "comparisons": comparisons,
        "recovered_total": recovered_total,
        "unrecovered_total": unrecovered_total,
        "api_cost_usd": 0.0,
    }
