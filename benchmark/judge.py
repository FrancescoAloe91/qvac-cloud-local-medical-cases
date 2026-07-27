"""Blind LLM-as-judge over structured Q&A + rubric."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Dict, List, Optional

from benchmark import openrouter
from benchmark.gold import load_confirmed_gold
from benchmark.run_control import is_cancelled
from benchmark.prompts import judge_system, judge_user
from benchmark.schema import Case, CandidateAnswer, JudgeResult, QuestionScore
from benchmark.scoring import (
    WEIGHTED_CAP,
    claim_correctness_score,
    evidence_discipline_score,
    graded_clinical_score,
    scoring_legend,
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


def _evidence_normalized(text: str) -> str:
    """Normalize whitespace/case and presentation-only Markdown markers."""
    without_markdown = re.sub(r"[*_`#]+", "", text or "")
    return re.sub(r"\s+", " ", without_markdown.strip()).casefold()


def _evidence_quote_present(quote: str, answer_norm: str) -> bool:
    normalized = _evidence_normalized(quote)
    if normalized and normalized in answer_norm:
        return True
    # Judges occasionally combine two exact, non-contiguous bullet sentences
    # into one quote. Accept only when every substantial sentence is verbatim.
    chunks = [
        _evidence_normalized(chunk)
        for chunk in re.split(r"(?<=[.!?])\s+", quote or "")
        if len(_evidence_normalized(chunk)) >= 12
    ]
    return len(chunks) >= 2 and all(chunk in answer_norm for chunk in chunks)


def _score_from_judge_item(
    case: Case,
    item: Dict[str, Any],
    *,
    answer_text: str,
    gold_reference: str,
) -> QuestionScore:
    qid = str(item.get("question_id", ""))
    gold = load_confirmed_gold(gold_reference)
    if qid not in gold.sections:
        raise ValueError(f"Unknown question_id: {qid}")
    reference_claims = gold.sections[qid].claims
    reference_ids = {claim.id for claim in reference_claims}

    assessments_raw = item.get("claim_assessments")
    if isinstance(assessments_raw, list):
        answer_norm = _evidence_normalized(answer_text)
        coverage_by_id: Dict[str, float] = {}
        evidence_rows: List[Dict[str, Any]] = []
        for row in assessments_raw:
            if not isinstance(row, dict):
                raise ValueError("Invalid claim_assessments object")
            claim_id = str(row.get("reference_claim_id") or "")
            if claim_id not in reference_ids or claim_id in coverage_by_id:
                raise ValueError("Unknown or duplicate graded reference claim id")
            try:
                coverage = min(max(float(row.get("coverage")), 0.0), 1.0)
            except (TypeError, ValueError) as exc:
                raise ValueError("Claim coverage must be a 0-1 number") from exc
            quotes_raw = row.get("candidate_quotes")
            if not isinstance(quotes_raw, list):
                raise ValueError("candidate_quotes must be a list")
            quotes = [str(value).strip() for value in quotes_raw if str(value).strip()]
            if coverage > 0 and not quotes:
                raise ValueError("Nonzero claim coverage requires candidate evidence")
            if any(not _evidence_quote_present(quote, answer_norm) for quote in quotes):
                raise ValueError("Judge evidence quote is not present in candidate answer")
            coverage_by_id[claim_id] = coverage
            evidence_rows.append(
                {
                    "reference_claim_id": claim_id,
                    "coverage": coverage,
                    "candidate_quotes": quotes,
                    "rationale": str(row.get("rationale") or ""),
                }
            )
        if set(coverage_by_id) != reference_ids:
            raise ValueError(f"Judge did not grade every reference claim for {qid}")

        additions_raw = item.get("additional_claims")
        if not isinstance(additions_raw, list):
            raise ValueError("additional_claims must be a list")
        additions: List[Dict[str, Any]] = []
        allowed_classes = {
            "helpful",
            "neutral",
            "unsupported",
            "contradictory",
            "dangerous",
        }
        for row in additions_raw:
            if not isinstance(row, dict):
                raise ValueError("Invalid additional_claims object")
            quote = str(row.get("candidate_quote") or "").strip()
            classification = str(row.get("classification") or "").strip().lower()
            if not quote or not _evidence_quote_present(quote, answer_norm):
                raise ValueError(
                    "Added-content quote is not present in candidate answer"
                )
            if classification not in allowed_classes:
                raise ValueError("Invalid added-content classification")
            try:
                severity = min(max(float(row.get("severity")), 0.0), 1.0)
            except (TypeError, ValueError) as exc:
                raise ValueError("Added-content severity must be a 0-1 number") from exc
            additions.append(
                {
                    "candidate_quote": quote,
                    "classification": classification,
                    "severity": severity,
                    "rationale": str(row.get("rationale") or ""),
                }
            )

        try:
            quality = min(max(float(item.get("quality")), 0.0), 1.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("quality must be a 0-1 number") from exc
        weighted_total = sum(1.5 if claim.critical else 1.0 for claim in reference_claims)
        coverage = sum(
            coverage_by_id[claim.id] * (1.5 if claim.critical else 1.0)
            for claim in reference_claims
        ) / max(weighted_total, 1.0)
        discipline = evidence_discipline_score(
            additions,
            total_reference=len(reference_claims),
        )
        score = graded_clinical_score(
            coverage=coverage,
            quality=quality,
            discipline=discipline,
        )
        unsupported = [
            str(row.get("rationale") or row["candidate_quote"])
            for row in additions
            if row["classification"] == "unsupported"
        ]
        contradictions = [
            str(row.get("rationale") or row["candidate_quote"])
            for row in additions
            if row["classification"] in {"contradictory", "dangerous"}
        ]
        err_raw = item.get("errors")
        raw_errors = [str(value) for value in err_raw] if isinstance(err_raw, list) else []
        rationale = str(item.get("rationale") or "")
        matched = [
            claim_id for claim_id, value in coverage_by_id.items() if value >= 0.75
        ]
        missed = [
            claim_id for claim_id, value in coverage_by_id.items() if value <= 0.25
        ]
        return QuestionScore(
            question_id=qid,
            score=score,
            rationale=(
                f"graded={score:.2f} coverage={coverage:.4f} "
                f"quality={quality:.4f} discipline={discipline:.4f}"
                + (f" | {rationale}" if rationale else "")
            ),
            evidence=json.dumps(evidence_rows, ensure_ascii=False),
            errors=raw_errors,
            matched_claim_ids=matched,
            missed_claim_ids=missed,
            unsupported_claims=unsupported,
            contradictions=contradictions,
            claim_coverage=coverage_by_id,
            added_content=additions,
            precision=discipline,
            recall=round(coverage, 4),
            quality=quality,
        )

    def _strings(value: Any, *, field: str) -> List[str]:
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list")
        return [str(v) for v in value]

    matched = _strings(item.get("matched_claim_ids"), field="matched_claim_ids")
    missed = _strings(item.get("missed_claim_ids"), field="missed_claim_ids")
    if set(matched) - reference_ids or set(missed) - reference_ids:
        raise ValueError(f"Judge returned unknown claim ids for {qid}")
    if set(matched) & set(missed):
        raise ValueError(f"Claim both matched and missed for {qid}")
    if set(matched) | set(missed) != reference_ids:
        raise ValueError(f"Judge did not classify every reference claim for {qid}")

    evidence_raw = item.get("evidence")
    if not isinstance(evidence_raw, list):
        raise ValueError("evidence must be a list")
    answer_norm = _evidence_normalized(answer_text)
    evidence_by_id: Dict[str, str] = {}
    for ev in evidence_raw:
        if not isinstance(ev, dict):
            raise ValueError("Invalid evidence object")
        claim_id = str(ev.get("reference_claim_id") or "")
        quote = str(ev.get("candidate_quote") or "").strip()
        if claim_id not in reference_ids or not quote:
            raise ValueError("Invalid evidence claim or quote")
        if not _evidence_quote_present(quote, answer_norm):
            raise ValueError("Judge evidence quote is not present in candidate answer")
        evidence_by_id[claim_id] = quote
    if any(claim_id not in evidence_by_id for claim_id in matched):
        raise ValueError("Every matched claim requires candidate evidence")

    def _claim_objects(field: str) -> tuple[List[str], List[str]]:
        value = item.get(field)
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list")
        labels: List[str] = []
        quotes: List[str] = []
        for row in value:
            if not isinstance(row, dict):
                raise ValueError(f"Invalid {field} object")
            quote = str(row.get("candidate_quote") or "").strip()
            if not quote or not _evidence_quote_present(quote, answer_norm):
                raise ValueError(f"{field} quote is not present in candidate answer")
            quotes.append(quote)
            labels.append(str(row.get("reason") or quote))
        return labels, quotes

    unsupported, _ = _claim_objects("unsupported_claims")
    contradictions, _ = _claim_objects("contradictions")
    try:
        quality = min(max(float(item.get("quality")), 0.0), 1.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("quality must be a 0-1 number") from exc
    score, precision, recall = claim_correctness_score(
        matched=len(matched),
        total_reference=len(reference_ids),
        unsupported=len(unsupported),
        contradictions=len(contradictions),
        quality=quality,
    )
    err_raw = item.get("errors")
    raw_errors = [str(v) for v in err_raw] if isinstance(err_raw, list) else []
    rationale = str(item.get("rationale") or "")

    return QuestionScore(
        question_id=qid,
        score=float(score),
        rationale=(
            f"balanced={score:.2f} coverage={recall:.4f} "
            f"quality={quality:.4f} precision={precision:.4f}"
            + (f" | {rationale}" if rationale else "")
        ),
        evidence=json.dumps(evidence_raw, ensure_ascii=False),
        errors=raw_errors,
        matched_claim_ids=matched,
        missed_claim_ids=missed,
        unsupported_claims=unsupported,
        contradictions=contradictions,
        precision=precision,
        recall=recall,
        quality=quality,
    )


def _zero_judgment(
    case: Case,
    candidate: CandidateAnswer,
    judge_model: str,
    rationale: str,
    meta,
    raw: str = "",
) -> JudgeResult:
    """Create an invalid/N/A observation.

    Internal section values remain numeric for schema compatibility, but `status`
    prevents them from entering rankings or statistics.
    """
    lower = rationale.lower()
    if "candidate error" in lower:
        status = "collect_failed"
        marker = "candidate_error"
    elif "partial candidate" in lower:
        status = "candidate_partial"
        marker = "candidate_partial"
    elif "empty answer" in lower:
        status = "candidate_empty"
        marker = "empty_answer"
    elif "timeout" in lower:
        status = "timed_out"
        marker = "judge_error"
    elif "schema" in lower or "json" in lower:
        status = "judge_schema_invalid"
        marker = "judge_retry_failed"
    else:
        status = "judge_transport_failed"
        marker = "judge_error"
    zeros = [
        QuestionScore(
            question_id=q.id,
            score=0.0,
            rationale=rationale,
            errors=[marker],
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
        status=status,
        failure_reason=rationale,
    )


def _candidate_has_answer(candidate: CandidateAnswer) -> bool:
    raw = (candidate.raw_response or "").strip()
    if raw:
        return True
    return any((v or "").strip() for v in (candidate.answers or {}).values())


def _candidate_missing_sections(case: Case, candidate: CandidateAnswer) -> List[str]:
    return [
        q.id
        for q in case.questions
        if not ((candidate.answers or {}).get(q.id) or "").strip()
    ]


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
    """True when this observation is technical N/A, not a clinical grade."""
    if j.status != "valid":
        return True
    markers = {
        "judge_error",
        "candidate_error",
        "candidate_partial",
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
    api_key: Optional[str] = None,
    verifier_model: str = "",
    allow_verifier: bool = True,
) -> JudgeResult:
    load_confirmed_gold(gold_reference)  # validate before any paid request
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
    last_validation_error = ""
    for attempt in range(2):
        raw, meta = openrouter.chat(
            judge_model,
            messages,
            temperature=temperature,
            max_tokens=8192,
            response_format={"type": "json_object"},
            max_attempts=3 if attempt == 0 else 2,
            timeout=240.0,
            api_key=api_key,
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
        try:
            if not isinstance(qs, list) or not qs:
                raise ValueError("empty or unusable JSON")
            q_scores: List[QuestionScore] = []
            for item in qs:
                if not isinstance(item, dict):
                    raise ValueError("question_scores contains a non-object")
                if not isinstance(item.get("claim_assessments"), list):
                    raise ValueError("graded claim_assessments schema is required")
                qid = str(item.get("question_id", ""))
                answer = (
                    (candidate.answers or {}).get(qid)
                    or candidate.raw_response
                    or ""
                )
                q_scores.append(
                    _score_from_judge_item(
                        case,
                        item,
                        answer_text=answer,
                        gold_reference=gold_reference,
                    )
                )
            required = {q.id for q in case.questions}
            have = {score.question_id for score in q_scores}
            if have != required or len(q_scores) != len(required):
                raise ValueError(
                    f"section mismatch: expected {sorted(required)}, got {sorted(have)}"
                )
            assert meta is not None
            return JudgeResult(
                blind_id=candidate.blind_id,
                candidate_key=candidate.candidate_key,
                question_scores=q_scores,
                weighted_accuracy=_weighted_accuracy(case, q_scores),
                judge_model=judge_model,
                judge_meta=meta,
                raw_judge_json=raw,
                status="valid",
            )
        except (TypeError, ValueError, AttributeError) as exc:
            last_validation_error = str(exc)
            if attempt == 0:
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Deterministic validation rejected the previous output: "
                            f"{last_validation_error}. Return a complete corrected JSON "
                            "object and never invent evidence."
                        ),
                    }
                ]
                continue

    assert meta is not None
    if allow_verifier and verifier_model and verifier_model != judge_model:
        verified = judge_candidate(
            case,
            candidate,
            verifier_model,
            temperature=0.0,
            gold_reference=gold_reference,
            api_key=api_key,
            verifier_model="",
            allow_verifier=False,
        )
        if verified.status == "valid":
            verified.failure_reason = (
                f"Primary judge invalid ({judge_model}: {last_validation_error}); "
                f"accepted independent verifier {verifier_model}"
            )
            return verified
    return _zero_judgment(
        case,
        candidate,
        judge_model,
        f"Judge schema/evidence invalid after retry: {last_validation_error}",
        meta,
        raw=raw or "",
    )


class PipelinedJudge:
    """Judge candidates as soon as each finishes collect (overlap with later collects).

    Submit from the Streamlit/script thread; call :meth:`poll` periodically on that
    same thread so ``on_progress`` can update the UI. Finish with :meth:`finalize`.
    """

    def __init__(
        self,
        case: Case,
        judge_model: str,
        *,
        temperature: float = 0.0,
        gold_reference: str = "",
        max_workers: Optional[int] = None,
        expected_total: int = 0,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        api_key: Optional[str] = None,
        verifier_model: str = "",
        run_scope: str = "",
    ) -> None:
        self.case = case
        self.judge_model = judge_model
        self.temperature = float(temperature)
        self.gold_reference = gold_reference or ""
        self.expected_total = max(0, int(expected_total or 0))
        self.on_progress = on_progress
        self.api_key = api_key
        self.verifier_model = verifier_model
        self.run_scope = run_scope or f"pipe-{id(self)}"
        n_hint = max(self.expected_total, 1)
        self._workers = max(1, min(n_hint, max_workers or n_hint))
        self._pool = ThreadPoolExecutor(max_workers=self._workers)
        self._pending: Dict[Future, CandidateAnswer] = {}
        self._by_key: Dict[str, JudgeResult] = {}
        self._candidates: List[CandidateAnswer] = []
        self._finished = 0
        self._shutdown = False
        _register_active_pipe(self)

    @property
    def candidates(self) -> List[CandidateAnswer]:
        return list(self._candidates)

    @property
    def submitted(self) -> int:
        return len(self._candidates)

    @property
    def done_count(self) -> int:
        return self._finished

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def total(self) -> int:
        return max(self.expected_total, len(self._candidates))

    def set_expected_total(self, n: int) -> None:
        self.expected_total = max(0, int(n))

    def _emit(self, evt: Dict[str, Any]) -> None:
        if self.on_progress is None:
            return
        try:
            self.on_progress(evt)
        except Exception:
            pass

    def _one_safe(self, cand: CandidateAnswer) -> JudgeResult:
        # No on_progress from worker threads — UI updates stay on the script thread.
        # Never spend DeepSeek on errored / empty collects (even if partial text exists).
        if cand.meta.error:
            return _zero_judgment(
                self.case,
                cand,
                self.judge_model,
                f"Candidate error: {cand.meta.error}",
                cand.meta,
            )
        if not _candidate_has_answer(cand):
            z = _zero_judgment(
                self.case,
                cand,
                self.judge_model,
                "Empty answer — not judged (skipped DeepSeek to save credits)",
                cand.meta,
            )
            for qs in z.question_scores:
                qs.errors = ["empty_answer"]
            return z
        if (cand.meta.finish_reason or "").lower() in {
            "length",
            "max_tokens",
            "content_filter",
        }:
            return _zero_judgment(
                self.case,
                cand,
                self.judge_model,
                f"Partial candidate output — finish_reason={cand.meta.finish_reason}",
                cand.meta,
            )
        missing = _candidate_missing_sections(self.case, cand)
        if missing:
            return _zero_judgment(
                self.case,
                cand,
                self.judge_model,
                "Partial candidate output — missing required sections: "
                + ", ".join(missing),
                cand.meta,
            )
        return judge_candidate(
            self.case,
            cand,
            self.judge_model,
            temperature=self.temperature,
            gold_reference=self.gold_reference,
            api_key=self.api_key,
            verifier_model=self.verifier_model,
        )

    def submit(self, cand: CandidateAnswer) -> bool:
        """Queue one candidate for DeepSeek. Returns False if duplicate / closed."""
        if self._shutdown or is_cancelled(self.run_scope):
            return False
        if any(c.candidate_key == cand.candidate_key for c in self._candidates):
            return False
        self._candidates.append(cand)
        self._emit(
            {
                "phase": "queued",
                "key": cand.candidate_key,
                "label": cand.display_label or cand.label or cand.candidate_key,
                "done": self._finished,
                "total": self.total,
            }
        )
        fut = self._pool.submit(self._one_safe, cand)
        self._pending[fut] = cand
        return True

    def _harvest(self, fut: Future) -> None:
        cand = self._pending.pop(fut, None)
        if cand is None:
            return
        try:
            result = fut.result()
        except Exception as exc:
            result = _zero_judgment(
                self.case,
                cand,
                self.judge_model,
                f"Judge exception: {type(exc).__name__}: {exc}",
                cand.meta,
            )
            for qs in result.question_scores:
                qs.errors = ["judge_exception"]
        self._by_key[cand.candidate_key] = result
        self._finished += 1
        self._emit(
            {
                "phase": "done",
                "key": cand.candidate_key,
                "label": cand.display_label or cand.label or cand.candidate_key,
                "done": self._finished,
                "total": self.total,
                "accuracy": result.weighted_accuracy,
                "failed": is_failed_judgment(result),
                "note": (result.judge_meta.error if result.judge_meta else None) or "",
            }
        )

    def poll(self) -> int:
        """Non-blocking: harvest finished futures. Call from the UI/script thread."""
        n = 0
        for fut in list(self._pending):
            if fut.done():
                self._harvest(fut)
                n += 1
        return n

    def finalize(self) -> List[JudgeResult]:
        """Wait for judges, retry recoverable failures, or stop cooperatively."""
        while self._pending:
            if is_cancelled(self.run_scope):
                self.close(cancel_pending=True)
                break
            done, _ = wait(
                tuple(self._pending.keys()),
                return_when=FIRST_COMPLETED,
            )
            for fut in done:
                if fut in self._pending:
                    self._harvest(fut)

        ordered = [
            self._by_key[c.candidate_key]
            for c in self._candidates
            if c.candidate_key in self._by_key
        ]
        cand_by = {c.candidate_key: c for c in self._candidates}
        for j in list(ordered):
            if is_cancelled(self.run_scope):
                break
            if not _is_rejudgable_failure(j):
                continue
            cand = cand_by.get(j.candidate_key)
            if not cand or not _candidate_has_answer(cand):
                continue
            self._emit(
                {
                    "phase": "retry",
                    "key": cand.candidate_key,
                    "label": cand.display_label or cand.label or cand.candidate_key,
                    "done": self._finished,
                    "total": self.total,
                }
            )
            time.sleep(1.5)
            retry = judge_candidate(
                self.case,
                cand,
                self.judge_model,
                temperature=self.temperature,
                gold_reference=self.gold_reference,
                api_key=self.api_key,
                verifier_model=self.verifier_model,
            )
            self._by_key[j.candidate_key] = retry
            self._emit(
                {
                    "phase": "retry_done",
                    "key": cand.candidate_key,
                    "label": cand.display_label or cand.label or cand.candidate_key,
                    "done": self._finished,
                    "total": self.total,
                    "accuracy": retry.weighted_accuracy,
                    "failed": is_failed_judgment(retry),
                }
            )
        ordered = [
            self._by_key[c.candidate_key]
            for c in self._candidates
            if c.candidate_key in self._by_key
        ]

        self.close(cancel_pending=False)
        return ordered

    def close(self, *, cancel_pending: bool = False) -> None:
        """Idempotent pool shutdown. ``cancel_pending=True`` on STOP (best-effort)."""
        self._shutdown = True
        if cancel_pending:
            for fut in list(self._pending):
                fut.cancel()
            self._pending.clear()
        try:
            self._pool.shutdown(wait=False, cancel_futures=bool(cancel_pending))
        except TypeError:
            try:
                self._pool.shutdown(wait=False)
            except Exception:
                pass
        except Exception:
            pass
        _unregister_active_pipe(self)


# Registry is partitioned by an unguessable Streamlit session/run scope.
_ACTIVE_PIPES: Dict[str, List["PipelinedJudge"]] = {}


def _register_active_pipe(pipe: "PipelinedJudge") -> None:
    bucket = _ACTIVE_PIPES.setdefault(pipe.run_scope, [])
    if pipe not in bucket:
        bucket.append(pipe)


def _unregister_active_pipe(pipe: "PipelinedJudge") -> None:
    bucket = _ACTIVE_PIPES.get(pipe.run_scope, [])
    try:
        bucket.remove(pipe)
    except ValueError:
        pass
    if not bucket:
        _ACTIVE_PIPES.pop(pipe.run_scope, None)


def abandon_all_pipelines(run_scope: str) -> int:
    """Cancel only pipelines owned by one session/run scope."""
    n = 0
    for pipe in list(_ACTIVE_PIPES.get(run_scope, [])):
        try:
            pipe.close(cancel_pending=True)
            n += 1
        except Exception:
            pass
    _ACTIVE_PIPES.pop(run_scope, None)
    return n


def judge_candidates_parallel(
    case: Case,
    candidates: List[CandidateAnswer],
    judge_model: str,
    *,
    temperature: float = 0.0,
    gold_reference: str = "",
    max_workers: Optional[int] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    api_key: Optional[str] = None,
    verifier_model: str = "",
    run_scope: str = "",
) -> List[JudgeResult]:
    """Score all candidates in parallel while preserving legitimate ties.

    ``on_progress`` receives dicts:
      {"phase": "queued"|"done"|"retry"|"retry_done", "key", "label", "done", "total", ...}
    """
    if not candidates:
        return []
    pipe = PipelinedJudge(
        case,
        judge_model,
        temperature=temperature,
        gold_reference=gold_reference,
        max_workers=max_workers or len(candidates),
        expected_total=len(candidates),
        on_progress=on_progress,
        api_key=api_key,
        verifier_model=verifier_model,
        run_scope=run_scope,
    )
    for c in candidates:
        pipe.submit(c)
    return pipe.finalize()


def build_ranking(judgments: List[JudgeResult]) -> List[Dict[str, Any]]:
    rows = []
    for j in judgments:
        failed = is_failed_judgment(j)
        flags = judgment_flags(j)
        rows.append(
            {
                "key": j.candidate_key,
                "blind_id": j.blind_id,
                "accuracy": None if failed else j.weighted_accuracy,
                "status": "n/a" if failed else "ok",
                "status_note": ", ".join(flags[:4]) if flags else "",
            }
        )
    rows.sort(
        key=lambda r: (
            0 if r["status"] == "ok" else 1,
            -float(r["accuracy"] or 0),
        )
    )
    last_score: Optional[float] = None
    last_rank = 0
    for i, row in enumerate(rows, 1):
        if row["status"] != "ok":
            row["rank"] = None
            continue
        score = float(row["accuracy"])
        if last_score is None or score != last_score:
            last_rank = i
            last_score = score
        row["rank"] = last_rank
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
            "Correctness balances graded coverage, clinical quality, and discipline. "
            "Technical failures are N/A; exact ties keep the same rank."
        ),
    }
