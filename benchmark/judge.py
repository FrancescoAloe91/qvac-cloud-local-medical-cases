"""Blind LLM-as-judge over structured Q&A + rubric."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from benchmark import openrouter
from benchmark.prompts import judge_system, judge_user
from benchmark.schema import Case, CandidateAnswer, JudgeResult, QuestionScore


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


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
    return round(acc / total_w, 2)


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


def judge_candidate(
    case: Case,
    candidate: CandidateAnswer,
    judge_model: str,
    temperature: float = 0.0,
    gold_reference: str = "",
) -> JudgeResult:
    messages = [
        {"role": "system", "content": judge_system()},
        {
            "role": "user",
            "content": judge_user(
                case, candidate.blind_id, candidate.answers, gold_reference=gold_reference
            ),
        },
    ]
    raw, meta = openrouter.chat(
        judge_model,
        messages,
        temperature=temperature,
        max_tokens=8192,
        response_format={"type": "json_object"},
    )
    if meta.error:
        return _zero_judgment(
            case,
            candidate,
            judge_model,
            f"Judge failed: {meta.error}",
            meta,
            raw=raw,
        )

    data = _extract_json(raw)
    q_scores: List[QuestionScore] = []
    for item in data.get("question_scores") or []:
        try:
            q_scores.append(
                QuestionScore(
                    question_id=str(item.get("question_id", "")),
                    score=float(item.get("score", 0)),
                    rationale=str(item.get("rationale") or ""),
                    evidence=str(item.get("evidence") or ""),
                    errors=[str(e) for e in (item.get("errors") or [])],
                )
            )
        except (TypeError, ValueError):
            continue

    # Ensure every question has a score
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
    """Score all candidates in parallel (OpenRouter calls overlap)."""
    if not candidates:
        return []

    def _one(cand: CandidateAnswer) -> JudgeResult:
        if cand.meta.error and not cand.raw_response:
            return _zero_judgment(
                case,
                cand,
                judge_model,
                f"Candidate error: {cand.meta.error}",
                cand.meta,
            )
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
        futs = {pool.submit(_one, c): c for c in candidates}
        for fut in as_completed(futs):
            cand = futs[fut]
            by_key[cand.candidate_key] = fut.result()
    return [by_key[c.candidate_key] for c in candidates if c.candidate_key in by_key]


def build_ranking(judgments: List[JudgeResult]) -> List[Dict[str, Any]]:
    rows = [
        {
            "key": j.candidate_key,
            "blind_id": j.blind_id,
            "accuracy": j.weighted_accuracy,
        }
        for j in judgments
    ]
    rows.sort(key=lambda r: r["accuracy"], reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows
