"""Blind LLM-as-judge over structured Q&A + rubric."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from benchmark import openrouter
from benchmark.prompts import judge_system, judge_user, use_gold_ground_truth
from benchmark.schema import Case, CandidateAnswer, JudgeResult, QuestionScore
from benchmark.scoring import (
    ITEM_SCORE_CAP,
    WEIGHTED_CAP,
    ensure_unique_accuracies,
    linear_item_score,
    scoring_legend,
    semantic_item_score,
    soft_alignment_from_checklist,
    stem_specificity,
)


def _normalize_judge_data(data: Any) -> Dict[str, Any]:
    """R1 sometimes returns a bare list of question_scores instead of an object."""
    if data is None:
        return {}
    if isinstance(data, list):
        if not data:
            return {"question_scores": []}
        if all(isinstance(x, dict) and "question_id" in x for x in data):
            return {"question_scores": data}
        if len(data) == 1 and isinstance(data[0], dict):
            return _normalize_judge_data(data[0])
        return {"question_scores": [x for x in data if isinstance(x, dict)]}
    if isinstance(data, dict):
        qs = data.get("question_scores")
        if isinstance(qs, dict):
            return {**data, "question_scores": [qs]}
        if qs is not None and not isinstance(qs, list):
            return {**data, "question_scores": []}
        return data
    return {}


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if parsed is None:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
    if parsed is None:
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        else:
            return {}
    return _normalize_judge_data(parsed)


def _weighted_accuracy(case: Case, scores: List[QuestionScore]) -> float:
    by_id = {s.question_id: s.score for s in scores}
    total_w = 0.0
    acc = 0.0
    for q in case.questions:
        if q.id not in by_id:
            continue
        acc += by_id[q.id] * q.weight
        total_w += q.weight
    if total_w <= 0:
        return 0.0
    return round(min(acc / total_w, WEIGHTED_CAP), 2)


def _as_pos_int(value: Any, default: int = 1) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default
    return max(1, min(n, 24))


def _score_from_judge_item(
    case: Case,
    item: Dict[str, Any],
    *,
    answer_text: str,
    gold_mode: bool = False,
) -> QuestionScore:
    qid = str(item.get("question_id", ""))
    qdef = next((q for q in case.questions if q.id == qid), None)

    err_raw = item.get("errors")
    if isinstance(err_raw, list):
        raw_errors = [str(e) for e in err_raw]
    elif err_raw is None or err_raw == "":
        raw_errors = []
    else:
        raw_errors = [str(err_raw)]
    rationale = str(item.get("rationale") or "")
    evidence = str(item.get("evidence") or "")
    spec = stem_specificity(case, answer_text)

    try:
        quality = float(item.get("quality"))
    except (TypeError, ValueError):
        quality = None

    # --- GOLD: semantic alignment (no checklist atomization) ---
    if gold_mode:
        alignment = None
        try:
            if item.get("alignment") is not None:
                alignment = float(item.get("alignment"))
        except (TypeError, ValueError):
            alignment = None

        # Fallback for older judge JSON that only had m/a counts
        if alignment is None and quality is not None:
            try:
                m_hit = int(item.get("m_hit"))
                a_hit = int(item.get("a_hit"))
            except (TypeError, ValueError):
                m_hit = a_hit = None
            if m_hit is not None and a_hit is not None:
                alignment = soft_alignment_from_checklist(
                    m_hit=m_hit,
                    m_total=_as_pos_int(item.get("m_total"), 1),
                    a_hit=a_hit,
                    a_total=_as_pos_int(item.get("a_total"), 1),
                    quality=quality,
                )

        if alignment is None or quality is None:
            try:
                fallback = float(item.get("score", 0))
            except (TypeError, ValueError):
                fallback = 0.0
            score = round(min(max(fallback, 0.0), min(92.0, ITEM_SCORE_CAP)), 2)
            raw_errors = list(raw_errors) + ["schema_fallback"]
            rationale = (
                f"schema_fallback≤{score} spec={spec:.2f} | {rationale}"
            ).strip(" |")
        else:
            quality = min(float(quality), 0.92)
            alignment = min(max(float(alignment), 0.0), 1.0)
            score = semantic_item_score(
                alignment=alignment,
                quality=quality,
                specificity=spec,
            )
            rationale = (
                f"align={alignment:.2f} quality={quality:.2f} "
                f"spec={spec:.2f} → {score}"
                + (f" | {rationale}" if rationale else "")
            )
    else:
        # --- RUBRIC: checklist + quality (quality-weighted) ---
        m_rub = len(qdef.rubric.must_include) if qdef else 0
        a_rub = len(qdef.rubric.acceptable) if qdef else 0
        m_total = m_rub if m_rub > 0 else _as_pos_int(item.get("m_total"), 1)
        a_total = a_rub if a_rub > 0 else _as_pos_int(item.get("a_total"), 1)
        m_total = max(m_total, 1)
        a_total = max(a_total, 1)

        try:
            m_hit = int(item.get("m_hit"))
        except (TypeError, ValueError):
            m_hit = None
        try:
            a_hit = int(item.get("a_hit"))
        except (TypeError, ValueError):
            a_hit = None

        if m_hit is None or a_hit is None or quality is None:
            try:
                fallback = float(item.get("score", 0))
            except (TypeError, ValueError):
                fallback = 0.0
            score = round(min(max(fallback, 0.0), min(92.0, ITEM_SCORE_CAP)), 2)
            raw_errors = list(raw_errors) + ["schema_fallback"]
            rationale = (
                f"schema_fallback≤{score} spec={spec:.2f} | {rationale}"
            ).strip(" |")
        else:
            quality = min(float(quality), 0.92)
            score = linear_item_score(
                m_hit=m_hit,
                m_total=m_total,
                a_hit=a_hit,
                a_total=a_total,
                quality=quality,
                specificity=spec,
            )
            rationale = (
                f"m={m_hit}/{m_total} a={a_hit}/{a_total} "
                f"quality={quality:.2f} spec={spec:.2f} → {score}"
                + (f" | {rationale}" if rationale else "")
            )

    trap_hints = ("must_not", "safety trap", "nitrate", "lithium without", "ignored")
    if any(any(h in e.lower() for h in trap_hints) for e in raw_errors):
        if qdef and qdef.kind in ("safety", "diagnosis") and score > 20:
            score = min(score, 20.0)
            rationale += " | must_not/safety cap≤20"

    return QuestionScore(
        question_id=qid,
        score=float(score),
        rationale=rationale,
        evidence=evidence,
        errors=raw_errors,
    )


def _zero_judgment(
    case: Case,
    candidate: CandidateAnswer,
    judge_model: str,
    rationale: str,
    meta,
    raw: str = "",
) -> JudgeResult:
    zeros = [
        QuestionScore(
            question_id=q.id,
            score=0.0,
            rationale=rationale,
            errors=["candidate_error" if "Candidate" in rationale else "judge_error"],
        )
        for q in case.questions
    ]
    return JudgeResult(
        blind_id=candidate.blind_id,
        candidate_key=candidate.candidate_key,
        question_scores=zeros,
        weighted_accuracy=0.0,
        judge_model=judge_model,
        judge_meta=meta,
        raw_judge_json=raw,
    )


def _candidate_has_answer(candidate: CandidateAnswer) -> bool:
    raw = (candidate.raw_response or "").strip()
    if raw:
        return True
    return any((v or "").strip() for v in (candidate.answers or {}).values())


def judgment_flags(j: JudgeResult) -> List[str]:
    flags: List[str] = []
    seen = set()
    for qs in j.question_scores or []:
        for e in qs.errors or []:
            if e and e not in seen:
                seen.add(e)
                flags.append(str(e))
    if j.judge_meta and j.judge_meta.error:
        err = f"transport:{j.judge_meta.error[:80]}"
        if err not in seen:
            flags.append(err)
    return flags


def is_failed_judgment(j: JudgeResult) -> bool:
    """True when accuracy 0 is not a fair clinical grade (transport / empty / crash)."""
    markers = {
        "judge_error",
        "candidate_error",
        "judge_retry_failed",
        "empty_answer",
        "judge_exception",
    }
    flags = set(judgment_flags(j))
    if flags & markers:
        return True
    if j.judge_meta and j.judge_meta.error:
        return True
    return False


def systemic_judge_failure(judgments: List[JudgeResult]) -> bool:
    """Stop Multi ×N when the judge path is systemically broken (save credits)."""
    if not judgments:
        return True
    failed = sum(1 for j in judgments if is_failed_judgment(j))
    n = len(judgments)
    if failed >= max(2, (n + 1) // 2):
        return True
    if failed >= 1 and all(j.weighted_accuracy == 0 for j in judgments):
        return True
    return False


def _is_rejudgable_failure(j: JudgeResult) -> bool:
    """Transport / empty-judge failures worth one serial retry (not empty candidate)."""
    if not is_failed_judgment(j):
        return False
    flags = set(judgment_flags(j))
    if flags & {"empty_answer", "candidate_error"}:
        return False
    err = ((j.judge_meta.error if j.judge_meta else None) or "").lower()
    return bool(
        flags
        & {"judge_error", "judge_retry_failed", "judge_exception"}
        or "empty body" in err
        or "incomplete" in err
        or "timeout" in err
        or "429" in err
        or "502" in err
        or "503" in err
        or "504" in err
    )


def judge_candidate(
    case: Case,
    candidate: CandidateAnswer,
    judge_model: str,
    temperature: float = 0.0,
    gold_reference: str = "",
) -> JudgeResult:
    gold_mode = use_gold_ground_truth(gold_reference)
    messages = [
        {"role": "system", "content": judge_system()},
        {
            "role": "user",
            "content": judge_user(
                case, candidate.blind_id, candidate.answers, gold_reference=gold_reference
            ),
        },
    ]
    raw, meta = "", None
    data: Dict[str, Any] = {}
    for attempt in range(2):
        raw, meta = openrouter.chat(
            judge_model,
            messages,
            temperature=temperature,
            max_tokens=8192,
            response_format={"type": "json_object"},
            max_attempts=3 if attempt == 0 else 2,
            timeout=240.0,
        )
        if meta.error:
            if attempt == 0 and openrouter.is_retryable_error(meta.error):
                continue
            return _zero_judgment(
                case,
                candidate,
                judge_model,
                f"Judge failed: {meta.error}",
                meta,
                raw=raw,
            )
        data = _extract_json(raw)
        if not isinstance(data, dict):
            data = {}
        qs = data.get("question_scores") or []
        if isinstance(qs, list) and qs:
            break
        if attempt == 0:
            # Empty / unusable JSON after a paid call — one extra attempt
            continue
        z = _zero_judgment(
            case,
            candidate,
            judge_model,
            "Judge retry failed: empty or unusable JSON",
            meta,
            raw=raw or "",
        )
        for qs_row in z.question_scores:
            qs_row.errors = ["judge_retry_failed"]
        return z

    assert meta is not None
    q_scores: List[QuestionScore] = []
    for item in data.get("question_scores") or []:
        if not isinstance(item, dict):
            continue
        try:
            qid = str(item.get("question_id", ""))
            ans = (candidate.answers or {}).get(qid) or candidate.raw_response or ""
            q_scores.append(
                _score_from_judge_item(
                    case, item, answer_text=ans, gold_mode=gold_mode
                )
            )
        except (TypeError, ValueError, AttributeError):
            continue

    have = {s.question_id for s in q_scores}
    for q in case.questions:
        if q.id not in have:
            q_scores.append(
                QuestionScore(
                    question_id=q.id,
                    score=0.0,
                    rationale="Missing from judge JSON",
                    errors=["missing_score"],
                )
            )

    return JudgeResult(
        blind_id=candidate.blind_id,
        candidate_key=candidate.candidate_key,
        question_scores=q_scores,
        weighted_accuracy=_weighted_accuracy(case, q_scores),
        judge_model=judge_model,
        judge_meta=meta,
        raw_judge_json=raw,
    )


def judge_candidates_parallel(
    case: Case,
    candidates: List[CandidateAnswer],
    judge_model: str,
    *,
    temperature: float = 0.0,
    gold_reference: str = "",
    max_workers: Optional[int] = None,
) -> List[JudgeResult]:
    """Score all candidates in parallel; enforce unique accuracies."""
    if not candidates:
        return []

    def _one_safe(cand: CandidateAnswer) -> JudgeResult:
        if cand.meta.error and not (cand.raw_response or "").strip():
            return _zero_judgment(
                case,
                cand,
                judge_model,
                f"Candidate error: {cand.meta.error}",
                cand.meta,
            )
        if not _candidate_has_answer(cand):
            z = _zero_judgment(
                case,
                cand,
                judge_model,
                "Empty answer — not judged (skipped DeepSeek to save credits)",
                cand.meta,
            )
            for qs in z.question_scores:
                qs.errors = ["empty_answer"]
            return z
        return judge_candidate(
            case,
            cand,
            judge_model,
            temperature=temperature,
            gold_reference=gold_reference,
        )

    workers = max(1, min(len(candidates), max_workers or len(candidates)))
    by_key: Dict[str, JudgeResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one_safe, c): c for c in candidates}
        for fut in as_completed(futs):
            cand = futs[fut]
            try:
                by_key[cand.candidate_key] = fut.result()
            except Exception as exc:
                z = _zero_judgment(
                    case,
                    cand,
                    judge_model,
                    f"Judge exception: {type(exc).__name__}: {exc}",
                    cand.meta,
                )
                for qs in z.question_scores:
                    qs.errors = ["judge_exception"]
                by_key[cand.candidate_key] = z
    ordered = [by_key[c.candidate_key] for c in candidates if c.candidate_key in by_key]

    # Serial re-judge for transport empties (e.g. Claude answer OK but R1 body empty).
    # Parallel storms sometimes drop one response; one quiet retry usually recovers.
    cand_by = {c.candidate_key: c for c in candidates}
    for j in list(ordered):
        if not _is_rejudgable_failure(j):
            continue
        cand = cand_by.get(j.candidate_key)
        if not cand or not _candidate_has_answer(cand):
            continue
        time.sleep(1.5)
        retry = judge_candidate(
            case,
            cand,
            judge_model,
            temperature=temperature,
            gold_reference=gold_reference,
        )
        by_key[j.candidate_key] = retry
    ordered = [by_key[c.candidate_key] for c in candidates if c.candidate_key in by_key]

    # Unique accuracies only among fair grades
    fair = [j for j in ordered if not is_failed_judgment(j)]
    failed = [j for j in ordered if is_failed_judgment(j)]
    if fair:
        unique_fair, _notes = ensure_unique_accuracies(case, fair)
        by_u = {j.candidate_key: j for j in unique_fair}
        for j in failed:
            by_u[j.candidate_key] = j
        ordered = [by_u[c.candidate_key] for c in candidates if c.candidate_key in by_u]
    return ordered


def build_ranking(judgments: List[JudgeResult]) -> List[Dict[str, Any]]:
    rows = []
    for j in judgments:
        failed = is_failed_judgment(j)
        flags = judgment_flags(j)
        rows.append(
            {
                "key": j.candidate_key,
                "blind_id": j.blind_id,
                "accuracy": j.weighted_accuracy,
                "status": "error" if failed else "ok",
                "status_note": ", ".join(flags[:4]) if flags else "",
            }
        )
    rows.sort(
        key=lambda r: (
            0 if r["status"] == "ok" else 1,
            -float(r["accuracy"] or 0),
        )
    )
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def explain_run_scores(case: Case, judgments: List[JudgeResult]) -> Dict[str, Any]:
    """Human-readable scoring explanation for the Results KPI panel."""
    legend = scoring_legend(case)
    per_model = []
    for j in judgments:
        by_q = {qs.question_id: qs for qs in j.question_scores}
        weakest = min(j.question_scores, key=lambda qs: qs.score) if j.question_scores else None
        strongest = max(j.question_scores, key=lambda qs: qs.score) if j.question_scores else None
        per_model.append(
            {
                "key": j.candidate_key,
                "accuracy": j.weighted_accuracy,
                "weakest": (
                    f"{weakest.question_id}={weakest.score}" if weakest else "—"
                ),
                "strongest": (
                    f"{strongest.question_id}={strongest.score}" if strongest else "—"
                ),
                "safety": by_q["safety"].score if "safety" in by_q else None,
                "diagnosis": by_q["diagnosis"].score if "diagnosis" in by_q else None,
            }
        )
    per_model.sort(key=lambda r: r["accuracy"], reverse=True)
    return {
        **legend,
        "per_model": per_model,
        "note": (
            f"100% is not used (item cap {ITEM_SCORE_CAP}, run cap {WEIGHTED_CAP}). "
            "Ties broken by safety → quality → stem specificity → diagnosis."
        ),
    }
