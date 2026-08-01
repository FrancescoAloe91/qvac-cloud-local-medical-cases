"""Artifact I/O and multi-run statistics."""

from __future__ import annotations

import re
import statistics
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from benchmark.cases_loader import load_case
from benchmark.case_slots import (
    artifact_stem_key,
    count_distinct_stem_keys,
)
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
from lib.model_labels import (
    CURRENT_ROSTER_KEYS,
    DEFAULT_ACTIVE_ROSTER_KEYS,
    is_current_roster_key,
)

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
            try:
                acc_f = float(accuracy)
            except (TypeError, ValueError):
                failures.setdefault(key, {})["invalid_accuracy"] = (
                    failures.setdefault(key, {}).get("invalid_accuracy", 0) + 1
                )
                continue
            # Exact 0 composite is not a successful observation for means
            # (same as technical N/A). Low non-zero scores remain valid.
            if acc_f == 0.0:
                failures.setdefault(key, {})["zero_score"] = (
                    failures.setdefault(key, {}).get("zero_score", 0) + 1
                )
                continue
            scores.setdefault(key, []).append(acc_f)
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
    # Rank by mean whenever a model has ≥1 scored observation. Technical N/A
    # never drops a model from the table; incomplete coverage is marked partial
    # so Failed % stays honest without hiding the mean-based rank.
    eligible_keys = {
        key for key in all_keys if len(scores.get(key, [])) >= 1
    }
    ranking_mean = []
    for k, v in stats.items():
        mean_raw = (
            statistics.fmean(scores[k]) if scores.get(k) else None
        )
        n_valid = int(v.get("n_valid") or v.get("n_runs") or v.get("n") or 0)
        n_req = int(v.get("n_requested") or 0)
        n_failed = int(v.get("n_failed") or 0)
        is_eligible = mean_raw is not None
        is_partial = bool(
            is_eligible and (n_failed > 0 or (n_req > 0 and n_valid < n_req))
        )
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
                "n_requested": n_req,
                "n_failed": n_failed,
                "failure_rate": v.get("failure_rate"),
                "failure_reasons": dict(v.get("failure_reasons") or {}),
                "coverage_mean": v.get("coverage_mean"),
                "quality_mean": v.get("quality_mean"),
                "discipline_mean": v.get("discipline_mean"),
                "eligible": is_eligible,
                "partial": is_partial,
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
                key not in by_key or not _is_scored_ranking_row(by_key[key])
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
    no_score = sorted(all_keys - eligible_keys)
    partial_keys = sorted(
        str(r.get("key") or "")
        for r in ranking_mean
        if r.get("partial") and r.get("key")
    )
    below_exploratory = sorted(
        key
        for key in eligible_keys
        if len(scores.get(key, [])) < min_rank_n
    )
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
    extra_notes: List[str] = []
    if partial_keys:
        extra_notes.append(
            "Partial (ranked by mean of scored runs; Failed % > 0 or scored < "
            "requested): "
            + ", ".join(
                f"{key} ({len(scores.get(key, []))}/"
                f"{int(stats.get(key, {}).get('n_requested') or 0)})"
                for key in partial_keys
            )
            + ". Technical N/A never discard other models' valid data."
        )
    if no_score:
        extra_notes.append(
            "Listed with Failed % but unranked (no scored observations): "
            + ", ".join(no_score)
            + "."
        )
    if below_exploratory and min_rank_n > 1:
        extra_notes.append(
            "Below exploratory N="
            + str(min_rank_n)
            + " but still mean-ranked"
            + (" · partial" if partial_keys else "")
            + ": "
            + ", ".join(
                f"{key} ({len(scores.get(key, []))}/{min_rank_n})"
                for key in below_exploratory
            )
            + "."
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
        outliers=outliers + extra_notes,
    )


def planned_on_device_model_contract(
    slots: Sequence[Mapping[str, Any]],
) -> List[Dict[str, str]]:
    """Stable cohort recipe from planned on-device slots (not per-run sidecar labels).

    Sidecar ``meta.model`` can oscillate (``medpsy-1.7b-q4`` vs
    ``medpsy-1.7b-q4_k_m-imat``); hashing collected labels splits one Multi
    batch into mixed cohorts and aborts the mean. Planned roster ``model``
    ids stay constant across iterations.
    """
    rows: List[Dict[str, str]] = []
    for slot in slots:
        key = str(slot.get("key") or "").strip()
        if not key:
            continue
        rows.append(
            {
                "key": key,
                "model": str(slot.get("model") or key).strip(),
                "provider": str(slot.get("provider") or "qvac").strip() or "qvac",
            }
        )
    rows.sort(key=lambda r: r["key"])
    return rows


def summarize_multi_batch(
    artifacts: Sequence[RunArtifact],
    *,
    min_valid_for_ranking: int = 5,
) -> Tuple[Optional[MultiRunSummary], Optional[str]]:
    """UI-safe Multi wrap-up: never raise; majority-cohort fallback when mixed.

    Strict ``summarize_runs`` still rejects mixed/empty cohorts for history
    rebuild. A finished Multi batch must still show per-run tabs and, when
    possible, a mean from the largest same-cohort subset.
    """
    arts = list(artifacts or [])
    if len(arts) < 2:
        return None, None
    cohort_err: Optional[BaseException] = None
    try:
        return (
            summarize_runs(arts, min_valid_for_ranking=min_valid_for_ranking),
            None,
        )
    except ValueError as exc:
        cohort_err = exc
    except Exception as exc:  # noqa: BLE001 — UI must not crash Streamlit
        return (
            None,
            f"Mean ranking unavailable: {type(exc).__name__}: {exc}. "
            f"{len(arts)} run artifacts were saved — open per-run tabs below.",
        )

    usable = [a for a in arts if str(a.cohort_id or "").strip()]
    if len(usable) < 2:
        return (
            None,
            f"Mean ranking unavailable: {cohort_err}. "
            f"{len(arts)} run artifacts were saved — open per-run tabs below.",
        )

    counts = Counter(str(a.cohort_id) for a in usable)
    majority_cid, _ = counts.most_common(1)[0]
    same = [a for a in usable if str(a.cohort_id) == majority_cid]
    if len(same) < 2:
        return (
            None,
            f"Mean ranking unavailable: {cohort_err}. "
            f"{len(arts)} run artifacts were saved — open per-run tabs below.",
        )
    try:
        summary = summarize_runs(
            same, min_valid_for_ranking=min_valid_for_ranking
        )
    except Exception as exc2:  # noqa: BLE001
        return (
            None,
            f"Mean ranking unavailable: {exc2}. "
            f"{len(arts)} run artifacts were saved — open per-run tabs below.",
        )

    dropped = len(arts) - len(same)
    note = (
        f"Mean used {len(same)}/{len(arts)} runs with matching cohort_id "
        f"(dropped {dropped} mixed/empty-label run(s)): {cohort_err}"
    )
    summary.outliers = list(summary.outliers or []) + [note]
    return summary, note


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
        if rank is None:
            rank_s = "—"
        elif row.get("partial"):
            rank_s = f"{rank}·p"
        else:
            rank_s = str(rank)
        lines.append(
            f"{rank_s:<6}{row['key']:<12}"
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


def reliability_caption(
    summary: MultiRunSummary, *, successful_only: bool = False
) -> str:
    """One-line plain-language guide for the multi-run mean."""
    ranked = [
        r
        for r in (summary.ranking_mean or [])
        if r.get("rank") is not None and r.get("eligible", True)
    ]
    if not ranked:
        if successful_only:
            return (
                "No model has a successful scored observation for Rebuild mean yet. "
                "Means use only error-free non-zero scored runs "
                "(technical N/A and exact-zero composites skipped)."
            )
        return (
            "No model has a scored observation for mean ranking yet. "
            "Failed % = share of pooled runs that are technical N/A "
            "(collect/judge/timeout/partial/empty) or exact-zero composite — "
            "not a clinical low score; means use only non-zero scored runs."
        )
    eligible = len(ranked)
    if successful_only:
        return (
            f"Mean ranking for {eligible} model(s) with ≥1 successful scored run · "
            "each mean is over its last ≤N successful non-zero runs "
            "(technical N/A and exact-zero composites skipped; "
            "older successful History used) · models with only failures are omitted · "
            "C/Q/D = coverage/quality/discipline (quality independent of coverage) · "
            "sample SD + median/IQR do not measure clinical generalization."
        )
    n_partial = sum(1 for r in ranked if r.get("partial"))
    partial_bit = (
        f" · {n_partial} partial (ranked by mean of scored runs; badge shows incomplete coverage)"
        if n_partial
        else ""
    )
    return (
        f"Mean ranking for {eligible} model(s) with ≥1 scored run{partial_bit} · "
        "each mean shows its own N; technical N/A never discard other models' data · "
        "Failed % = technical N/A or exact-zero rate across requested runs "
        "(collect/judge/timeout/partial/empty/zero_score) · "
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
    # Beta (and any non-graded protocol): never silently re-stamp as graded-v4.
    if stored_version and not stored_version.startswith("graded-clinical"):
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
            "formula": stored_version,
            "preserved_stored_ranking": True,
            "scoring_version_stamp": stored_version,
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
    n: Optional[int] = 5,
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
    by finished_at (then started_at), newest first.

    ``n`` caps how many **run documents** are returned (1–30). Pass ``n=None``
    to return every eligible run (needed for Portfolio rebuild, where N is a
    per-model observation cap rather than a global run-document slice).
    """
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
    pairs = [(p, a) for p, a, _ in matched]
    if n is None:
        return pairs
    cap = max(1, min(int(n), 100))
    return pairs[:cap]


def _is_scored_ranking_row(row: Dict[str, Any]) -> bool:
    """True when a ranking row is a successful non-zero scored observation.

    Technical N/A (status ≠ ok / missing accuracy) and exact clinical composite
    ``0`` are not successful for mean pooling or fill-N. Low non-zero scores
    (e.g. 5–10) remain valid unless already marked technical failure.
    """
    status = str(row.get("status") or "ok")
    if status != "ok":
        return False
    accuracy = row.get("accuracy")
    if accuracy is None:
        return False
    try:
        return float(accuracy) != 0.0
    except (TypeError, ValueError):
        return False


def _classify_rebuild_observation(row: Dict[str, Any]) -> str:
    """Bucket one ranking row for ops reliability: scored | zero | technical_na."""
    if _is_scored_ranking_row(row):
        return "scored"
    status = str(row.get("status") or "ok")
    accuracy = row.get("accuracy")
    if status == "ok" and accuracy is not None:
        try:
            if float(accuracy) == 0.0:
                return "zero"
        except (TypeError, ValueError):
            return "technical_na"
    return "technical_na"


def collect_rebuild_ops_reliability(
    rescored_pairs: Sequence[Tuple[RunArtifact, Dict[str, Any]]],
    *,
    n: int,
    model_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Count zeros + technical N/A seen while filling scored-N (ops honesty).

    Same newest-first walk as Rebuild mean fill — does **not** change the
    scored-only mean pool. For each model, walk until N successful non-zero
    scores are reached (or History ends) and tally every observation seen in
    that window: scored, exact-zero composites, technical N/A.
    """
    n = max(1, min(int(n), 100))
    want = (
        [str(k) for k in model_ids]
        if model_ids is not None
        else list(CURRENT_ROSTER_KEYS)
    )
    want_set = frozenset(k for k in want if is_current_roster_key(k))
    tallies: Dict[str, Dict[str, int]] = {
        key: {"n_scored": 0, "n_zero": 0, "n_technical_na": 0, "n_seen": 0}
        for key in want_set
    }
    for clone, _row in rescored_pairs:
        for r in list(clone.ranking or []):
            key = str(r.get("key") or "")
            if key not in want_set:
                continue
            bucket = tallies[key]
            if bucket["n_scored"] >= n:
                continue
            kind = _classify_rebuild_observation(r)
            bucket["n_seen"] += 1
            if kind == "scored":
                bucket["n_scored"] += 1
            elif kind == "zero":
                bucket["n_zero"] += 1
            else:
                bucket["n_technical_na"] += 1

    rows: List[Dict[str, Any]] = []
    for key in want:
        if key not in tallies:
            continue
        t = tallies[key]
        seen = int(t["n_seen"])
        n_zero = int(t["n_zero"])
        n_na = int(t["n_technical_na"])
        n_scored = int(t["n_scored"])
        n_excluded = n_zero + n_na
        denom = max(seen, 1)

        def _pct(count: int) -> float:
            return round(100.0 * count / denom, 1) if seen else 0.0

        rows.append(
            {
                "key": key,
                "n_scored": n_scored,
                "n_zero": n_zero,
                "n_technical_na": n_na,
                "n_excluded": n_excluded,
                "n_seen": seen,
                "pct_scored": _pct(n_scored),
                "pct_zero": _pct(n_zero),
                "pct_technical_na": _pct(n_na),
                "pct_excluded": _pct(n_excluded),
            }
        )
    rows.sort(
        key=lambda r: (
            -int(r.get("n_excluded") or 0),
            -int(r.get("n_seen") or 0),
            str(r.get("key") or ""),
        )
    )
    return rows


def _trim_rescored_to_per_model_n(
    rescored_pairs: Sequence[Tuple[RunArtifact, Dict[str, Any]]],
    *,
    n: int,
    model_ids: Optional[Sequence[str]] = None,
    keep_failures: bool = False,
) -> Tuple[List[RunArtifact], List[Dict[str, Any]], Dict[str, int]]:
    """Keep ≤N newest *successful non-zero* scores per model; N/A/0 do not fill N.

    ``rescored_pairs`` must already be newest-first. Technical N/A / errors and
    exact-zero composites are skipped for the scored-N cap — the walk continues
    into older History until each model has N non-zero scored rows (or history
    ends).

    Rebuild mean defaults to ``keep_failures=False``: only successful non-zero
    scored rows enter the pool (clean comparison; no Failed%/partial theater).
    When ``keep_failures=True``, N/A and zero rows seen while filling the scored
    quota are retained so Failed % can reflect the scan window. Run documents
    that retain no rows are omitted. Returns (arts, per_run rows, per-model
    **scored** counts).
    """
    n = max(1, min(int(n), 100))
    want = (
        frozenset(str(k) for k in model_ids)
        if model_ids is not None
        else frozenset(CURRENT_ROSTER_KEYS)
    )
    scored_counts: Dict[str, int] = {}
    arts: List[RunArtifact] = []
    per_run: List[Dict[str, Any]] = []
    for clone, row in rescored_pairs:
        kept_ranking: List[Dict[str, Any]] = []
        for r in list(clone.ranking or []):
            key = str(r.get("key") or "")
            if key not in want or not is_current_roster_key(key):
                continue
            scored_n = scored_counts.get(key, 0)
            if scored_n >= n:
                # Already have N scored — stop collecting scored and N/A for key.
                continue
            if _is_scored_ranking_row(r):
                kept_ranking.append(r)
                scored_counts[key] = scored_n + 1
            elif keep_failures:
                # Technical N/A or exact-zero: keep for Failed %; do not
                # advance the successful scored cap.
                kept_ranking.append(r)
        if not kept_ranking:
            continue
        trimmed = clone.model_copy(deep=True)
        trimmed.ranking = kept_ranking
        arts.append(trimmed)
        trimmed_row = dict(row)
        trimmed_row["ranking"] = kept_ranking
        per_run.append(trimmed_row)
    return arts, per_run, scored_counts


def _finalize_clean_rebuild_summary(summary: MultiRunSummary) -> MultiRunSummary:
    """Rebuild mean view: only ≥1 successful score; never surface partial badges."""
    cleaned: List[Dict[str, Any]] = []
    for row in summary.ranking_mean or []:
        if not row.get("eligible") or row.get("accuracy_mean_raw") is None:
            continue
        if row.get("accuracy_mean") is None:
            continue
        item = dict(row)
        item["partial"] = False
        item["n_failed"] = 0
        item["failure_rate"] = 0.0
        item["failure_reasons"] = {}
        n_valid = int(item.get("n_runs") or item.get("n_valid") or 0)
        item["n_requested"] = n_valid
        cleaned.append(item)
    cleaned.sort(
        key=lambda r: (
            -float(r.get("accuracy_mean_raw") or -1),
            str(r.get("key") or ""),
        )
    )
    last_mean: Optional[float] = None
    last_rank = 0
    for index, row in enumerate(cleaned, 1):
        mean_value = float(row["accuracy_mean_raw"])
        if last_mean is None or mean_value != last_mean:
            last_mean = mean_value
            last_rank = index
        row["rank"] = last_rank
    summary.ranking_mean = cleaned
    kept_keys = {r.get("key") for r in cleaned}
    for key, stats in list((summary.candidate_stats or {}).items()):
        if key not in kept_keys:
            continue
        n_valid = int(stats.get("n_valid") or stats.get("n_runs") or stats.get("n") or 0)
        stats["n_failed"] = 0.0
        stats["failure_rate"] = 0.0
        stats["failure_reasons"] = {}
        stats["n_requested"] = float(n_valid)
    summary.outliers = [
        note
        for note in (summary.outliers or [])
        if "Partial (" not in note
        and "unranked (no scored observations)" not in note
        and " · partial" not in note
    ]
    return summary


def rebuild_model_ids(
    optional_legacy_keys: Optional[Sequence[str]] = None,
) -> List[str]:
    """Default 9-roster keys plus any opted-in optional/legacy slots."""
    keys = list(DEFAULT_ACTIVE_ROSTER_KEYS)
    for key in optional_legacy_keys or ():
        k = str(key or "")
        if k and k not in keys and is_current_roster_key(k):
            keys.append(k)
    return keys


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
    model_ids: Optional[Sequence[str]] = None,
    preloaded: Optional[Sequence[RunArtifact]] = None,
    scoring_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Offline Multi×N: same immutable cohort only; ``n`` = max **successful**
    scored observations per model (newest first). Technical N/A does not fill N
    and is omitted from the Rebuild mean pool — older successful rows are used.
    Rescore with the current formula; return summarize_runs-compatible summary +
    per-run rows. Zero API cost. No partial badge theater.

    When ``cohort_id`` is set (e.g. after restoring a prior confirmed gold),
    rebuild that immutable cohort instead of the newest case cohort.
    ``model_ids`` defaults to the active 9-roster (optional legacy only when
    passed in by the UI).
    When ``scoring_version`` is set, only artifacts with that stamp are pooled
    (keeps Beta ``beta-comprehension-v1`` out of graded same-case Rebuild).
    """
    n = max(1, min(int(n), 100))
    want_keys = list(model_ids) if model_ids is not None else rebuild_model_ids()
    want_sv = str(scoring_version or "").strip() or None
    all_pairs = artifacts_for_case(
        out_dir, case_id, limit=None, preloaded=preloaded
    )
    if want_sv:
        all_pairs = [
            pair
            for pair in all_pairs
            if str(pair[1].scoring_version or "").strip() == want_sv
        ]
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
    # Cancelled / failed abort stamps stay out. Runs with run_status=partial may
    # still contribute *successful* per-model scores; N/A rows themselves are
    # dropped by the clean trim. Load full cohort history; N caps successful
    # observations per model (not global docs).
    pairs = [
        pair
        for pair in all_pairs
        if pair[1].cohort_id == target_cohort
        and is_mean_poolable_run(pair[1])
        and (
            want_sv is None
            or str(pair[1].scoring_version or "").strip() == want_sv
        )
    ]
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

    rescored_pairs: List[Tuple[RunArtifact, Dict[str, Any]]] = []
    for path, art in pairs:
        clone, row = _offline_rescore_pair(path, art)
        rescored_pairs.append((clone, row))

    ops_reliability = collect_rebuild_ops_reliability(
        rescored_pairs, n=n, model_ids=want_keys
    )
    rescored_arts, per_run, _counts = _trim_rescored_to_per_model_n(
        rescored_pairs, n=n, model_ids=want_keys, keep_failures=False
    )
    if not rescored_arts:
        return {
            "ok": False,
            "reason": (
                "Need at least 1 same-cohort successful non-zero scored "
                "observation (found 0 after per-model trim). Cancelled/failed "
                "aborts, technical N/A, and exact-zero composites are excluded "
                "from Rebuild mean."
            ),
            "available": len(pairs),
            "cohort_id": target_cohort,
            "scope": "same_case",
            "ops_reliability": ops_reliability,
        }

    summary = _finalize_clean_rebuild_summary(
        summarize_runs(rescored_arts, min_valid_for_ranking=1)
    )
    return {
        "ok": True,
        "available": len(pairs),
        "n_used": len(rescored_arts),
        "n_per_model_cap": n,
        "summary": summary,
        "per_run": per_run,
        "formula": (
            "reference-relative Clinical Composite Score · same immutable cohort · "
            "≤N successful non-zero scored obs/model; N/A and exact-zero skipped, "
            "older successful used"
        ),
        "api_cost_usd": 0.0,
        "cohort_id": target_cohort,
        "official": True,
        "scope": "same_case",
        "n_cases": 1,
        "mean_rank": _mean_ranks_from_per_run(per_run),
        "successful_only": True,
        "model_ids": list(want_keys),
        "ops_reliability": ops_reliability,
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
    """Offline exploratory mean: ≤N **successful non-zero** scores per model.

    Loads every eligible run (same track + scoring_version, complete|partial),
    then for each roster key keeps that model's own newest ≤N *successful*
    non-zero ranking rows. Technical N/A and exact-zero composites do not fill
    N and are omitted from the Rebuild mean pool — the walk continues into
    older History. Cloud models with older history still appear when recent
    global runs were medical-only. Never invents scores; never merges
    incompatible scoring versions. Zero API cost. Not clinical validation.
    No partial badge theater.
    """
    n = max(1, min(int(n), 100))
    want_keys = list(model_ids) if model_ids is not None else rebuild_model_ids()
    # All eligible run documents — N is applied per model, not as a global slice.
    all_eligible = list_portfolio_runs(
        out_dir,
        n=None,
        scoring_version=scoring_version,
        track=track,
        model_ids=want_keys,
        preloaded=preloaded,
    )
    n_cases_all = count_distinct_stem_keys(a for _, a in all_eligible)
    if len(all_eligible) < 1:
        return {
            "ok": False,
            "reason": (
                f"Need at least 1 portfolio-eligible run "
                f"(found {len(all_eligible)}; {n_cases_all} distinct clinical "
                f"case stem(s)). "
                "Filters: same track + scoring_version, complete|partial + "
                "≥1 valid judgment; roster shapes may differ "
                "(≤N successful observations per model). "
                "Cancelled/failed aborts stay out. "
                "Different scoring versions are never pooled. "
                "Mixed-case portfolio means are exploratory — not clinical validation."
            ),
            "available": len(all_eligible),
            "n_cases": n_cases_all,
            "scope": "portfolio",
            "scoring_version": scoring_version,
            "track": track,
        }

    rescored_pairs: List[Tuple[RunArtifact, Dict[str, Any]]] = []
    for path, art in all_eligible:
        clone, row = _offline_rescore_pair(path, art)
        rescored_pairs.append((clone, row))

    ops_reliability = collect_rebuild_ops_reliability(
        rescored_pairs, n=n, model_ids=want_keys
    )
    rescored_arts, per_run, _counts = _trim_rescored_to_per_model_n(
        rescored_pairs, n=n, model_ids=want_keys, keep_failures=False
    )
    if not rescored_arts:
        return {
            "ok": False,
            "reason": (
                "Need at least 1 portfolio-eligible successful non-zero "
                "observation after per-model trim. Filters: same track + "
                "scoring_version, complete|partial; technical N/A and "
                "exact-zero composites omitted from Rebuild mean."
            ),
            "available": len(all_eligible),
            "n_cases": n_cases_all,
            "scope": "portfolio",
            "scoring_version": scoring_version,
            "track": track,
            "ops_reliability": ops_reliability,
        }

    summary = _finalize_clean_rebuild_summary(
        summarize_runs(
            rescored_arts, allow_mixed_cohorts=True, min_valid_for_ranking=1
        )
    )
    mean_rank = _mean_ranks_from_per_run(per_run)
    # Attach mean rank onto ranking_mean rows when present.
    if hasattr(summary, "ranking_mean"):
        for row in summary.ranking_mean:
            key = str(row.get("key") or "")
            if key in mean_rank:
                row["mean_rank"] = mean_rank[key]
    n_cases = count_distinct_stem_keys(rescored_arts)
    return {
        "ok": True,
        "available": len(all_eligible),
        "n_used": len(rescored_arts),
        "n_per_model_cap": n,
        "n_cases": n_cases,
        "summary": summary,
        "per_run": per_run,
        "formula": (
            "exploratory mixed-case portfolio mean · mixed roster shapes OK · "
            "≤N successful non-zero scored obs/model; N/A and exact-zero skipped, "
            "older successful used · reference-relative scores · not clinical "
            "validation · gold contracts are not merged"
        ),
        "api_cost_usd": 0.0,
        "official": False,
        "scope": "portfolio",
        "scoring_version": scoring_version,
        "track": track,
        "successful_only": True,
        "model_ids": list(want_keys),
        "mean_rank": mean_rank,
        "case_ids": sorted({a.case_id for a in rescored_arts}),
        "ops_reliability": ops_reliability,
        "mixed_case_exploratory": True,
    }


def _trim_rescored_balanced_round_robin(
    rescored_pairs: Sequence[Tuple[RunArtifact, Dict[str, Any]]],
    *,
    n: int,
    model_ids: Sequence[str],
    ordered_stem_keys: Sequence[str],
) -> Tuple[List[RunArtifact], List[Dict[str, Any]], Dict[str, int]]:
    """≤N successful non-zero scores/model via Case1→K→1… round-robin.

    For each model, walk ordered case stems in a cycle and take the newest
    unused successful observation on that stem before advancing. Empty stems
    are skipped; a full cycle with zero picks stops the fill for that model.
    Unlike portfolio trim (global newest-N), this balances case weight so a
    recent binge on one case cannot dominate the mean.
    """
    from collections import defaultdict, deque

    n = max(1, min(int(n), 100))
    want = frozenset(str(k) for k in model_ids)
    slots = [str(s).strip() for s in ordered_stem_keys if str(s).strip()]
    if not slots:
        return [], [], {}

    # Newest-first queues of (art, ranking_row, per_run_row) per (model, stem).
    queues: Dict[Tuple[str, str], deque] = defaultdict(deque)
    for clone, row in rescored_pairs:
        sk = artifact_stem_key(clone)
        if sk not in slots:
            continue
        for r in list(clone.ranking or []):
            key = str(r.get("key") or "")
            if key not in want or not is_current_roster_key(key):
                continue
            if not _is_scored_ranking_row(r):
                continue
            queues[(key, sk)].append((clone, r, row))

    # Per model: round-robin picks.
    picks_by_run: Dict[str, Dict[str, Any]] = {}
    # run_id -> {clone, per_run_row, ranking_rows[]}
    scored_counts: Dict[str, int] = {k: 0 for k in want}

    for model in want:
        picks = 0
        slot_i = 0
        no_progress = 0
        while picks < n and no_progress < len(slots):
            sk = slots[slot_i % len(slots)]
            slot_i += 1
            q = queues.get((model, sk))
            if not q:
                no_progress += 1
                continue
            clone, r, row = q.popleft()
            run_id = str(clone.run_id or "")
            bucket = picks_by_run.get(run_id)
            if bucket is None:
                bucket = {
                    "clone": clone,
                    "row": row,
                    "ranking": [],
                }
                picks_by_run[run_id] = bucket
            bucket["ranking"].append(dict(r))
            picks += 1
            scored_counts[model] = picks
            no_progress = 0

    # Preserve newest-first order of selected run docs.
    order_index = {
        str(clone.run_id or ""): i for i, (clone, _) in enumerate(rescored_pairs)
    }
    arts: List[RunArtifact] = []
    per_run: List[Dict[str, Any]] = []
    for run_id, bucket in sorted(
        picks_by_run.items(),
        key=lambda kv: order_index.get(kv[0], 10**9),
    ):
        clone = bucket["clone"].model_copy(deep=True)
        clone.ranking = list(bucket["ranking"])
        arts.append(clone)
        trimmed_row = dict(bucket["row"])
        trimmed_row["ranking"] = list(bucket["ranking"])
        per_run.append(trimmed_row)
    return arts, per_run, scored_counts


def rebuild_balanced_cases_from_history(
    out_dir: Path,
    *,
    n: int = 5,
    scoring_version: str = "graded-clinical-v4",
    track: str = "controlled",
    model_ids: Optional[Sequence[str]] = None,
    ordered_stem_keys: Optional[Sequence[str]] = None,
    preloaded: Optional[Sequence[RunArtifact]] = None,
) -> Dict[str, Any]:
    """Offline mean: ≤N obs/model round-robin across Case slots (1→K→1…).

    Same filters as portfolio (track + scoring_version, scored-only), but
    observations are filled by cycling case stems so each case is weighted
    roughly equally — unlike portfolio newest-N which can overweight the
    cases you ran most recently. Exploratory; not clinical validation.
    """
    n = max(1, min(int(n), 100))
    want_keys = list(model_ids) if model_ids is not None else rebuild_model_ids()
    all_eligible = list_portfolio_runs(
        out_dir,
        n=None,
        scoring_version=scoring_version,
        track=track,
        model_ids=want_keys,
        preloaded=preloaded,
    )
    # Ordered stems: caller Case1…K bindings, else discover by newest activity.
    slots = [str(s).strip() for s in (ordered_stem_keys or []) if str(s).strip()]
    if not slots:
        seen = []
        seen_set = set()
        for _, art in all_eligible:  # newest first
            sk = artifact_stem_key(art)
            if sk and sk not in seen_set:
                seen_set.add(sk)
                seen.append(sk)
        slots = list(reversed(seen))  # oldest-discovered first ≈ Case order proxy
    n_cases_all = len(slots) if slots else count_distinct_stem_keys(
        a for _, a in all_eligible
    )
    if len(all_eligible) < 1 or not slots:
        return {
            "ok": False,
            "reason": (
                "Need at least 1 eligible run and ≥1 Case stem for balanced "
                "round-robin rebuild (Case1→…→K→1…). Same track + scoring_version."
            ),
            "available": len(all_eligible),
            "n_cases": n_cases_all,
            "scope": "balanced_cases",
            "scoring_version": scoring_version,
            "track": track,
        }

    rescored_pairs: List[Tuple[RunArtifact, Dict[str, Any]]] = []
    for path, art in all_eligible:
        clone, row = _offline_rescore_pair(path, art)
        rescored_pairs.append((clone, row))

    # Ops window still uses global newest-N scan for honesty chart.
    ops_reliability = collect_rebuild_ops_reliability(
        rescored_pairs, n=n, model_ids=want_keys
    )
    rescored_arts, per_run, _counts = _trim_rescored_balanced_round_robin(
        rescored_pairs,
        n=n,
        model_ids=want_keys,
        ordered_stem_keys=slots,
    )
    if not rescored_arts:
        return {
            "ok": False,
            "reason": (
                "Need at least 1 successful non-zero observation after "
                "balanced Case round-robin trim."
            ),
            "available": len(all_eligible),
            "n_cases": n_cases_all,
            "scope": "balanced_cases",
            "scoring_version": scoring_version,
            "track": track,
            "ops_reliability": ops_reliability,
        }

    summary = _finalize_clean_rebuild_summary(
        summarize_runs(
            rescored_arts, allow_mixed_cohorts=True, min_valid_for_ranking=1
        )
    )
    mean_rank = _mean_ranks_from_per_run(per_run)
    if hasattr(summary, "ranking_mean"):
        for row in summary.ranking_mean:
            key = str(row.get("key") or "")
            if key in mean_rank:
                row["mean_rank"] = mean_rank[key]
    n_cases = count_distinct_stem_keys(rescored_arts)
    return {
        "ok": True,
        "available": len(all_eligible),
        "n_used": len(rescored_arts),
        "n_per_model_cap": n,
        "n_cases": n_cases,
        "summary": summary,
        "per_run": per_run,
        "formula": (
            "exploratory balanced-cases mean · Case1→K round-robin · "
            "≤N successful non-zero scored obs/model · each case weighted "
            "roughly equally · N/A and exact-zero skipped · not clinical validation"
        ),
        "api_cost_usd": 0.0,
        "official": False,
        "scope": "balanced_cases",
        "scoring_version": scoring_version,
        "track": track,
        "successful_only": True,
        "model_ids": list(want_keys),
        "mean_rank": mean_rank,
        "case_ids": sorted({a.case_id for a in rescored_arts}),
        "ordered_stem_keys": list(slots),
        "ops_reliability": ops_reliability,
        "mixed_case_exploratory": True,
        "balanced_round_robin": True,
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
