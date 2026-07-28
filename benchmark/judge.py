"""Blind LLM-as-judge over structured Q&A + rubric."""

from __future__ import annotations

import json
import queue
import re
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Dict, List, Optional

from benchmark import openrouter
from benchmark.gold import load_confirmed_gold
from benchmark.run_control import is_cancelled
from benchmark.prompts import judge_system, judge_user
from benchmark.schema import (
    Case,
    CandidateAnswer,
    JudgeResult,
    ModelCallMeta,
    QuestionScore,
    utc_now_iso,
)
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
        qs = (
            data.get("question_scores")
            if "question_scores" in data
            else data.get("scores", data.get("sections", data.get("results")))
        )
        if qs is None:
            direct = [
                {"question_id": key, **value}
                for key, value in data.items()
                if key in {"diagnosis", "tests", "urgency", "safety", "plan"}
                and isinstance(value, dict)
            ]
            if direct:
                qs = direct
        if isinstance(qs, dict):
            if all(isinstance(value, dict) for value in qs.values()):
                qs = [
                    {"question_id": key, **value}
                    for key, value in qs.items()
                ]
            else:
                qs = [qs]
        if qs is not None and not isinstance(qs, list):
            return {**data, "question_scores": []}
        normalized = dict(data)
        if qs is not None:
            normalized["question_scores"] = [
                _normalize_judge_item(item)
                for item in qs
                if isinstance(item, dict)
            ]
        return normalized
    return {}


def _as_unit_float(
    value: Any,
    *,
    field: str,
    default: Optional[float] = None,
) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        if default is not None:
            return default
        raise ValueError(f"{field} must be numeric")
    text = str(value).strip()
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    try:
        number = float(text)
    except (TypeError, ValueError) as exc:
        if default is not None:
            return default
        raise ValueError(f"{field} must be numeric") from exc
    if is_percent or 1.0 < number <= 100.0:
        number /= 100.0
    # Clamp tiny float noise; reject clearly out-of-range values.
    if number < 0.0:
        number = 0.0
    if number > 1.0:
        if number <= 1.0001:
            number = 1.0
        else:
            raise ValueError(f"{field} must be between 0 and 1")
    return number


def _legacy_item_to_assessments(item: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Reshape binary matched/missed + evidence into graded claim_assessments."""
    matched = item.get("matched_claim_ids")
    missed = item.get("missed_claim_ids")
    if not isinstance(matched, list) or not isinstance(missed, list):
        return None
    evidence_by_id: Dict[str, List[str]] = {}
    evidence_raw = item.get("evidence")
    if isinstance(evidence_raw, list):
        for row in evidence_raw:
            if not isinstance(row, dict):
                continue
            claim_id = str(
                row.get("reference_claim_id")
                or row.get("claim_id")
                or row.get("id")
                or ""
            )
            quote = str(
                row.get("candidate_quote") or row.get("quote") or ""
            ).strip()
            if claim_id and quote:
                evidence_by_id.setdefault(claim_id, []).append(quote)
    assessments: List[Dict[str, Any]] = []
    for claim_id in matched:
        cid = str(claim_id)
        assessments.append(
            {
                "reference_claim_id": cid,
                "coverage": 1.0,
                "candidate_quotes": list(evidence_by_id.get(cid) or []),
            }
        )
    for claim_id in missed:
        cid = str(claim_id)
        assessments.append(
            {
                "reference_claim_id": cid,
                "coverage": 0.0,
                "candidate_quotes": list(evidence_by_id.get(cid) or []),
            }
        )
    return assessments


def _legacy_item_to_additions(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    additions: List[Dict[str, Any]] = []
    for field, classification, severity in (
        ("unsupported_claims", "unsupported", 1.0),
        ("contradictions", "contradictory", 1.0),
    ):
        rows = item.get(field)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, str):
                quote = row.strip()
                if quote:
                    additions.append(
                        {
                            "candidate_quote": quote,
                            "classification": classification,
                            "severity": severity,
                        }
                    )
                continue
            if not isinstance(row, dict):
                continue
            quote = str(
                row.get("candidate_quote") or row.get("quote") or ""
            ).strip()
            if not quote:
                continue
            additions.append(
                {
                    "candidate_quote": quote,
                    "classification": str(
                        row.get("classification") or classification
                    ).strip().lower(),
                    "severity": row.get("severity", severity),
                    "rationale": str(row.get("reason") or row.get("rationale") or ""),
                }
            )
    return additions


def _normalize_judge_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Repair unambiguous local schema variants without changing clinical text."""
    normalized = dict(item)
    normalized["question_id"] = str(
        item.get("question_id")
        or item.get("section_id")
        or item.get("section")
        or item.get("id")
        or ""
    )
    assessments = item.get(
        "claim_assessments",
        item.get("assessments", item.get("claim_grades")),
    )
    # "claims" is ambiguous (gold claims vs assessments); only use when graded-shaped.
    if assessments is None and isinstance(item.get("claims"), (list, dict)):
        claims_val = item["claims"]
        sample = None
        if isinstance(claims_val, list):
            sample = next((row for row in claims_val if isinstance(row, dict)), None)
        elif isinstance(claims_val, dict):
            sample = next(
                (row for row in claims_val.values() if isinstance(row, dict)),
                None,
            )
        if sample and (
            "coverage" in sample
            or "candidate_quotes" in sample
            or "candidate_quote" in sample
            or "reference_claim_id" in sample
            or "score" in sample
            or "grade" in sample
        ):
            assessments = claims_val
    if assessments is None:
        assessments = _legacy_item_to_assessments(item)
    if isinstance(assessments, dict):
        assessments = [
            {"reference_claim_id": claim_id, **value}
            if isinstance(value, dict)
            else {"reference_claim_id": claim_id, "coverage": value}
            for claim_id, value in assessments.items()
        ]
    if isinstance(assessments, list):
        repaired_assessments = []
        for row in assessments:
            if not isinstance(row, dict):
                repaired_assessments.append(row)
                continue
            repaired = dict(row)
            repaired["reference_claim_id"] = str(
                row.get("reference_claim_id")
                or row.get("claim_id")
                or row.get("id")
                or ""
            )
            if repaired.get("coverage") is None:
                repaired["coverage"] = row.get("score", row.get("grade", 0.0))
            if repaired.get("coverage") is None:
                repaired["coverage"] = 0.0
            quotes = row.get(
                "candidate_quotes",
                row.get("quotes", row.get("candidate_quote", row.get("evidence"))),
            )
            if quotes is None:
                quotes = []
            if isinstance(quotes, str):
                quotes = [quotes]
            if not isinstance(quotes, list):
                quotes = []
            repaired["candidate_quotes"] = [
                str(value).strip() for value in quotes if str(value).strip()
            ]
            repaired_assessments.append(repaired)
        normalized["claim_assessments"] = repaired_assessments
    additions = item.get("additional_claims", item.get("added_content"))
    if additions is None and (
        isinstance(item.get("unsupported_claims"), list)
        or isinstance(item.get("contradictions"), list)
    ):
        additions = _legacy_item_to_additions(item)
    if additions is None:
        additions = []
    if isinstance(additions, dict):
        additions = [additions]
    if isinstance(additions, list):
        repaired_additions = []
        for row in additions:
            if not isinstance(row, dict):
                repaired_additions.append(row)
                continue
            repaired = dict(row)
            repaired["candidate_quote"] = str(
                row.get("candidate_quote")
                or row.get("quote")
                or row.get("evidence")
                or ""
            ).strip()
            repaired["classification"] = str(
                row.get("classification")
                or row.get("label")
                or row.get("type")
                or row.get("class")
                or ""
            ).strip().lower()
            if repaired.get("severity") is None:
                repaired["severity"] = 1.0
            repaired_additions.append(repaired)
        normalized["additional_claims"] = repaired_additions
    if "quality" not in normalized or normalized.get("quality") is None:
        for key in ("clinical_quality", "quality_score", "clinical_score"):
            if item.get(key) is not None:
                normalized["quality"] = item.get(key)
                break
    if isinstance(normalized.get("errors"), str):
        normalized["errors"] = [normalized["errors"]]
    elif normalized.get("errors") is None:
        normalized["errors"] = []
    return normalized


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip().lstrip("\ufeff")
    if not text:
        return {}
    candidates: List[str] = [text]
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    parsed: Any = None
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            parsed = None
        for match in re.finditer(r"[\{\[]", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start() :])
                break
            except json.JSONDecodeError:
                continue
        if parsed is not None:
            break
    if parsed is None:
        return {}
    return _normalize_judge_data(parsed)


def _weighted_accuracy_raw(case: Case, scores: List[QuestionScore]) -> float:
    """Unrounded Clinical Composite Score for tie detection."""
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
    return min(acc / total_w, WEIGHTED_CAP)


def _weighted_accuracy(case: Case, scores: List[QuestionScore]) -> float:
    """Clinical Composite Score; ranking ties use this unrounded value."""
    return _weighted_accuracy_raw(case, scores)


def _weighted_subscale(case: Case, scores: List[QuestionScore], field: str) -> float:
    by_id = {score.question_id: score for score in scores}
    total_weight = 0.0
    value = 0.0
    for question in case.questions:
        score = by_id.get(question.id)
        component = getattr(score, field, None) if score else None
        if component is None:
            continue
        value += float(component) * question.weight
        total_weight += question.weight
    return round(100.0 * value / total_weight, 2) if total_weight > 0 else 0.0


def _as_pos_int(value: Any, default: int = 1) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default
    return max(1, min(n, 24))


# Clinically meaningful numeric punctuation kept through evidence normalization.
# Presentation punctuation (commas, colons, dashes used as styling, etc.) is dropped.
_EVIDENCE_TOKEN_RE = re.compile(
    r"(?:"
    r"\d+(?:[.,]\d+)+"  # decimals / thousand-grouped numerics: 1.0, 1,5, 1.000,5
    r"|\d+\s*[–—−\-]\s*\d+(?:[.,]\d+)*"  # ranges: 10-20, 10–20, 0.5-1.0
    r"|[±]\s*\d+(?:[.,]\d+)*"  # ±0.5
    r"|\d+\s*/\s*\d+(?:[.,]\d+)*"  # ratios / doses: 1/2, 10/20
    r"|\w+"
    r")",
    flags=re.UNICODE,
)


def _evidence_normalized(text: str) -> str:
    """Normalize presentation without erasing clinically meaningful numerics.

    Markdown/HTML, case, Unicode width, list markers, and styling punctuation
    stay flexible. Decimals, numeric ranges, ±, and slash ratios are preserved
    so ``10–20 mg`` does not match ``10 20 mg`` and ``1.0 mg`` does not match
    ``1 0 mg``.
    """
    canonical = unicodedata.normalize("NFKC", text or "")
    canonical = (
        canonical.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u00a0", " ")
        .replace("\u2026", " ")
    )
    without_html = re.sub(r"<[^>]+>", " ", canonical)
    without_markdown = re.sub(r"[*_`#~>]+", "", without_html)
    # Cloud answers often use Markdown bullets / numbered lists that judges omit.
    without_markdown = re.sub(
        r"(?m)^\s*(?:[-*•]|\d+[.)])\s+",
        "",
        without_markdown,
    )
    tokens = []
    for raw in _EVIDENCE_TOKEN_RE.findall(without_markdown.casefold()):
        token = re.sub(r"\s+", "", raw)
        if token:
            tokens.append(token)
    return " ".join(tokens)


def _evidence_token_sequence(text: str) -> List[str]:
    normalized = _evidence_normalized(text)
    return normalized.split() if normalized else []


def _token_sequence_present(needle: List[str], haystack: List[str]) -> bool:
    """True when needle tokens appear as a contiguous sequence in haystack."""
    if not needle:
        return False
    n = len(needle)
    if n > len(haystack):
        return False
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return True
    return False


def _evidence_quote_present(quote: str, answer_norm: str) -> bool:
    """Require a contiguous token sequence (word-boundary safe).

    Short tokens like ``renal`` must not match inside ``adrenal``.
    """
    answer_tokens = answer_norm.split() if answer_norm else []
    needle = _evidence_token_sequence(quote)
    if needle and _token_sequence_present(needle, answer_tokens):
        return True
    # Judges occasionally combine two exact, non-contiguous bullet sentences
    # into one quote. Accept only when every substantial sentence is verbatim.
    chunks = [
        _evidence_token_sequence(chunk)
        for chunk in re.split(r"(?<=[.!?])\s+", quote or "")
        if len(_evidence_normalized(chunk)) >= 12
    ]
    if len(chunks) >= 2 and all(
        _token_sequence_present(chunk, answer_tokens) for chunk in chunks
    ):
        return True
    # Partial contiguous spans (e.g. ≥50% of a long quote) are intentionally
    # rejected: half a clinical sentence must not count as full evidence.
    return False


_CLASSIFICATION_ALIASES = {
    "helpful": "helpful",
    "help": "helpful",
    "useful": "helpful",
    "neutral": "neutral",
    "ok": "neutral",
    "unsupported": "unsupported",
    "unsupport": "unsupported",
    "not_supported": "unsupported",
    "not-supported": "unsupported",
    "speculation": "unsupported",
    "contradictory": "contradictory",
    "contradiction": "contradictory",
    "conflicts": "contradictory",
    "dangerous": "dangerous",
    "harmful": "dangerous",
    "unsafe": "dangerous",
}


def _normalize_classification(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_")
    return _CLASSIFICATION_ALIASES.get(text, text)


def _filter_present_quotes(quotes: List[str], answer_norm: str) -> List[str]:
    return [quote for quote in quotes if _evidence_quote_present(quote, answer_norm)]


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
            coverage = _as_unit_float(
                row.get("coverage"),
                field="Claim coverage",
                default=0.0,
            )
            quotes_raw = row.get("candidate_quotes")
            if quotes_raw is None:
                quotes_raw = []
            if not isinstance(quotes_raw, list):
                raise ValueError("candidate_quotes must be a list")
            quotes = [str(value).strip() for value in quotes_raw if str(value).strip()]
            present_quotes = _filter_present_quotes(quotes, answer_norm)
            # Coverage>0 needs a presentable quote. Presentation salvage already
            # ran in _filter_present_quotes; if nothing remains, zero coverage
            # locally — never invent evidence and never force a paid retry.
            if coverage > 0 and not present_quotes:
                coverage = 0.0
                present_quotes = []
            if coverage <= 0:
                present_quotes = []
            coverage_by_id[claim_id] = coverage
            evidence_rows.append(
                {
                    "reference_claim_id": claim_id,
                    "coverage": coverage,
                    "candidate_quotes": present_quotes,
                    "rationale": str(row.get("rationale") or ""),
                }
            )
        if set(coverage_by_id) != reference_ids:
            raise ValueError(f"Judge did not grade every reference claim for {qid}")

        additions_raw = item.get("additional_claims")
        if additions_raw is None:
            additions_raw = []
        if not isinstance(additions_raw, list):
            raise ValueError("additional_claims must be a list")
        additions: List[Dict[str, Any]] = []
        unverified_harm_dropped: List[Dict[str, Any]] = []
        allowed_classes = {
            "helpful",
            "neutral",
            "unsupported",
            "contradictory",
            "dangerous",
        }
        harm_classes = {"contradictory", "dangerous"}
        for row in additions_raw:
            if not isinstance(row, dict):
                continue
            quote = str(row.get("candidate_quote") or "").strip()
            classification = _normalize_classification(row.get("classification"))
            if classification not in allowed_classes:
                continue
            quote_ok = bool(quote) and _evidence_quote_present(quote, answer_norm)
            # Every penalized addition needs a quote present in the candidate
            # answer. Unverifiable labels (including dangerous/contradictory) are
            # dropped with an audit marker — never invent evidence or presume guilt.
            if not quote_ok:
                if classification in harm_classes:
                    unverified_harm_dropped.append(
                        {
                            "classification": classification,
                            "candidate_quote": quote,
                            "rationale": str(row.get("rationale") or ""),
                            "audit": "judge_unverified_harm_dropped",
                        }
                    )
                continue
            severity = _as_unit_float(
                row.get("severity"),
                field="Added-content severity",
                default=1.0,
            )
            additions.append(
                {
                    "candidate_quote": quote,
                    "classification": classification,
                    "severity": severity,
                    "rationale": str(row.get("rationale") or ""),
                }
            )

        quality = _as_unit_float(item.get("quality"), field="quality")
        coverage = sum(coverage_by_id[claim.id] for claim in reference_claims) / max(
            len(reference_claims), 1
        )
        # Quality is independent of coverage (graded-clinical-v4). The judge
        # remains responsible for low quality on clinically bad answers.
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
        for dropped in unverified_harm_dropped:
            raw_errors.append(
                "judge_unverified_harm_dropped:"
                f"{dropped.get('classification') or 'unknown'}"
            )
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
            continue
        claim_id = str(ev.get("reference_claim_id") or "")
        quote = str(ev.get("candidate_quote") or "").strip()
        # Drop unverifiable / unknown evidence rows locally (no paid retry).
        if claim_id not in reference_ids or not quote:
            continue
        if not _evidence_quote_present(quote, answer_norm):
            continue
        evidence_by_id[claim_id] = quote
    # Matched claims without a presentable quote demote to missed locally.
    still_matched = [cid for cid in matched if cid in evidence_by_id]
    demoted = [cid for cid in matched if cid not in evidence_by_id]
    matched = still_matched
    if demoted:
        missed = list(dict.fromkeys([*missed, *demoted]))

    def _claim_objects(field: str) -> tuple[List[str], List[str]]:
        value = item.get(field)
        if value is None:
            value = []
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list")
        labels: List[str] = []
        quotes: List[str] = []
        for row in value:
            if isinstance(row, str):
                quote = row.strip()
                if quote and _evidence_quote_present(quote, answer_norm):
                    quotes.append(quote)
                    labels.append(quote)
                continue
            if not isinstance(row, dict):
                continue
            quote = str(row.get("candidate_quote") or row.get("quote") or "").strip()
            # Drop unverifiable rows locally instead of forcing a paid retry.
            if not quote or not _evidence_quote_present(quote, answer_norm):
                continue
            quotes.append(quote)
            labels.append(str(row.get("reason") or quote))
        return labels, quotes

    unsupported, _ = _claim_objects("unsupported_claims")
    contradictions, _ = _claim_objects("contradictions")
    quality = _as_unit_float(item.get("quality"), field="quality")
    recall_est = len(matched) / max(len(reference_ids), 1)
    quality = min(quality, recall_est)
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


def _empty_judge_meta(judge_model: str, *, provider: str = "openrouter") -> ModelCallMeta:
    """Judge-side meta with zero cost — never reuse candidate meta (fake duplication)."""
    return ModelCallMeta(
        model=judge_model or "judge",
        provider=provider,
        cost_usd=0.0,
        paid_attempts=[],
    )


def _append_paid_attempt(
    meta: ModelCallMeta,
    *,
    role: str,
    prior: Optional[ModelCallMeta] = None,
) -> ModelCallMeta:
    """Accumulate paid attempt cost/tokens into append-only metadata."""
    attempt = {
        "role": role,
        "model": meta.model,
        "cost_usd": float(meta.cost_usd or 0.0),
        "prompt_tokens": int(meta.prompt_tokens or 0),
        "completion_tokens": int(meta.completion_tokens or 0),
        "error": meta.error or "",
    }
    attempts = list((prior.paid_attempts if prior else None) or [])
    attempts.append(attempt)
    prior_cost = float(prior.cost_usd or 0.0) if prior else 0.0
    prior_prompt = int(prior.prompt_tokens or 0) if prior else 0
    prior_completion = int(prior.completion_tokens or 0) if prior else 0
    return meta.model_copy(
        update={
            "cost_usd": round(prior_cost + float(meta.cost_usd or 0.0), 8),
            "prompt_tokens": prior_prompt + int(meta.prompt_tokens or 0),
            "completion_tokens": prior_completion + int(meta.completion_tokens or 0),
            "paid_attempts": attempts,
            "retry_count": max(0, len(attempts) - 1),
        }
    )


def _zero_judgment(
    case: Case,
    candidate: CandidateAnswer,
    judge_model: str,
    rationale: str,
    meta,
    raw: str = "",
    *,
    primary_judge_model: str = "",
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
    elif "timeout" in lower or "deadline" in lower or "wall-clock" in lower:
        status = "timed_out"
        marker = "judge_timeout"
    elif "evidence" in lower:
        status = "judge_evidence_invalid"
        marker = "judge_evidence_invalid"
    elif "schema" in lower or "json" in lower:
        status = "judge_schema_invalid"
        marker = "judge_schema_invalid"
    elif "cancel" in lower:
        status = "cancelled"
        marker = "cancelled"
    else:
        status = "judge_transport_failed"
        marker = "judge_transport_failed"
    # Never attach candidate meta as judge_meta — that duplicates candidate cost.
    if meta is None or not isinstance(meta, ModelCallMeta):
        judge_meta = _empty_judge_meta(judge_model)
    elif candidate.meta is not None and meta is candidate.meta:
        judge_meta = _empty_judge_meta(judge_model)
    else:
        judge_meta = meta
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
        primary_judge_model=primary_judge_model or judge_model,
        judge_meta=judge_meta,
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
    """Stop only when the judge infrastructure produced no valid observations."""
    if not judgments:
        return True
    if any(not is_failed_judgment(j) for j in judgments):
        return False
    infrastructure_statuses = {
        "judge_transport_failed",
        "judge_schema_invalid",
        "judge_evidence_invalid",
        "timed_out",
    }
    return any(j.status in infrastructure_statuses for j in judgments)


def _is_rejudgable_failure(j: JudgeResult) -> bool:
    """Transport failures worth one pipeline-level retry (not empty candidate).

    Schema/evidence invalidation already spends the in-judge corrective attempt
    inside ``judge_candidate``. Re-queueing those here flipped finished N/A rows
    back to 75% "corrective retry" with a frozen clock — skip them.
    """
    if not is_failed_judgment(j):
        return False
    if int(getattr(j, "retry_count", 0) or 0) >= 1:
        return False
    flags = set(judgment_flags(j))
    if flags & {
        "empty_answer",
        "candidate_error",
        "candidate_partial",
        "judge_evidence_invalid",
        "judge_schema_invalid",
        "judge_retry_failed",
    }:
        return False
    if j.status in {
        "judge_schema_invalid",
        "judge_evidence_invalid",
        "candidate_partial",
        "candidate_empty",
        "collect_failed",
        "cancelled",
    }:
        return False
    if j.status in {"judge_transport_failed", "timed_out"}:
        return True
    err = ((j.judge_meta.error if j.judge_meta else None) or "").lower()
    return bool(
        flags & {"judge_error", "judge_exception"}
        or "empty body" in err
        or "incomplete" in err
        or "timeout" in err
        or "429" in err
        or "502" in err
        or "503" in err
        or "504" in err
    )


def _score_sections_from_payload(
    case: Case,
    candidate: CandidateAnswer,
    payload: Dict[str, Any],
    *,
    gold_reference: str,
    target_ids: set[str],
) -> tuple[Dict[str, QuestionScore], Dict[str, str]]:
    """Local normalize + score; never spends tokens."""
    qs = payload.get("question_scores") or []
    accepted: Dict[str, QuestionScore] = {}
    section_errors: Dict[str, str] = {}
    if not isinstance(qs, list) or not qs:
        return accepted, {qid: "empty or unusable JSON" for qid in sorted(target_ids)}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for raw_item in qs:
        if not isinstance(raw_item, dict):
            continue
        item = _normalize_judge_item(raw_item)
        qid = str(item.get("question_id") or "")
        if qid in target_ids:
            grouped.setdefault(qid, []).append(item)
    for qid in target_ids:
        items = grouped.get(qid) or []
        if len(items) != 1:
            section_errors[qid] = (
                "missing section" if not items else "duplicate section"
            )
            continue
        item = items[0]
        has_graded = isinstance(item.get("claim_assessments"), list)
        has_legacy = isinstance(item.get("matched_claim_ids"), list)
        if not has_graded and not has_legacy:
            section_errors[qid] = "graded claim_assessments schema is required"
            continue
        answer = (
            (candidate.answers or {}).get(qid) or candidate.raw_response or ""
        )
        try:
            accepted[qid] = _score_from_judge_item(
                case,
                item,
                answer_text=answer,
                gold_reference=gold_reference,
            )
        except (TypeError, ValueError, AttributeError) as exc:
            section_errors[qid] = str(exc)
    return accepted, section_errors


# Primary judge often hits finish_reason=length at 8k on long Claude/OpenAI
# answers (see caseC-a0d375f208). Give headroom; repairs go section-by-section.
_JUDGE_PRIMARY_MAX_TOKENS = 16384
_JUDGE_SECTION_MAX_TOKENS = 4096


def _judge_output_truncated(meta: Optional[ModelCallMeta], *, cap: int) -> bool:
    if meta is None:
        return False
    reason = (meta.finish_reason or "").strip().lower()
    if reason in {"length", "max_tokens", "max_length"}:
        return True
    # Some providers omit finish_reason but still stop exactly at the cap.
    return int(meta.completion_tokens or 0) >= max(1, int(cap) - 4)


def _repair_prompt_for_sections(
    retry_ids: set[str],
    section_errors: Dict[str, str],
) -> str:
    details = "; ".join(
        f"{qid}: {section_errors.get(qid, 'invalid')}" for qid in sorted(retry_ids)
    )
    return (
        "Local validation rejected ONLY these sections after deterministic repair: "
        f"{sorted(retry_ids)}. Errors: {details}. "
        "Return ONE compact JSON object with question_scores containing exactly "
        "those rejected sections (claim_assessments + additional_claims + quality). "
        "Use short verbatim candidate quotes (≤40 words; Markdown/whitespace OK). "
        "Omit long rationales. Never invent evidence. Finish the JSON completely."
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
    progress_callback: Optional[Callable[[str, int], None]] = None,
    allowed_providers: Optional[List[str]] = None,
    require_parameters: bool = False,
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
    accumulated_meta: Optional[ModelCallMeta] = None
    last_validation_error = ""
    required = {q.id for q in case.questions}
    accepted: Dict[str, QuestionScore] = {}

    if progress_callback:
        progress_callback("judge request", 25)
    raw, meta = openrouter.chat(
        judge_model,
        messages,
        temperature=temperature,
        max_tokens=_JUDGE_PRIMARY_MAX_TOKENS,
        response_format={"type": "json_object"},
        max_attempts=3,
        timeout=300.0,
        api_key=api_key,
        allowed_providers=allowed_providers,
        require_parameters=require_parameters,
    )
    accumulated_meta = _append_paid_attempt(meta, role="primary", prior=None)
    meta = accumulated_meta
    primary_truncated = _judge_output_truncated(
        meta, cap=_JUDGE_PRIMARY_MAX_TOKENS
    )
    if progress_callback:
        progress_callback("validating response", 70)
    if meta.error:
        return _zero_judgment(
            case,
            candidate,
            judge_model,
            f"Judge failed: {meta.error}",
            meta,
            raw=raw,
            primary_judge_model=judge_model,
        )

    data = _extract_json(raw)
    scored, section_errors = _score_sections_from_payload(
        case,
        candidate,
        data if isinstance(data, dict) else {},
        gold_reference=gold_reference,
        target_ids=required,
    )
    accepted.update(scored)
    retry_ids = required - set(accepted)

    # Paid repair only for sections local salvage could not accept. Presentation /
    # invented-quote mismatches already zero coverage locally and never land here.
    # When the primary hit the length cap (common on long Claude/OpenAI answers),
    # repair ONE section at a time — a single multi-section corrective also truncates.
    if retry_ids:
        details = "; ".join(
            f"{qid}: {section_errors.get(qid, 'invalid')}"
            for qid in sorted(retry_ids)
        )
        last_validation_error = details
        per_section = primary_truncated or len(retry_ids) >= 2
        # UI shows 75% only when a paid repair HTTP call is about to start.
        if progress_callback:
            progress_callback("corrective retry", 75)

        section_batches: List[set[str]] = (
            [{qid} for qid in sorted(retry_ids)]
            if per_section
            else [set(retry_ids)]
        )
        for batch in section_batches:
            if not batch:
                continue
            repair_messages = messages + [
                {
                    "role": "user",
                    "content": _repair_prompt_for_sections(batch, section_errors),
                }
            ]
            raw_repair, meta_repair = openrouter.chat(
                judge_model,
                repair_messages,
                temperature=temperature,
                max_tokens=_JUDGE_SECTION_MAX_TOKENS,
                response_format={"type": "json_object"},
                max_attempts=2,
                timeout=180.0,
                api_key=api_key,
                allowed_providers=allowed_providers,
                require_parameters=require_parameters,
            )
            accumulated_meta = _append_paid_attempt(
                meta_repair, role="corrective_retry", prior=accumulated_meta
            )
            meta = accumulated_meta
            if raw_repair:
                raw = raw_repair
            if meta_repair.error:
                last_validation_error = (
                    f"{details}; corrective retry failed: {meta_repair.error}"
                )
                continue
            repair_data = _extract_json(raw_repair)
            repaired, repair_errors = _score_sections_from_payload(
                case,
                candidate,
                repair_data if isinstance(repair_data, dict) else {},
                gold_reference=gold_reference,
                target_ids=batch,
            )
            accepted.update(repaired)
            section_errors.update(repair_errors)

        if progress_callback:
            progress_callback("validating response", 88)
        retry_ids = required - set(accepted)
        if retry_ids:
            last_validation_error = "; ".join(
                f"{qid}: {section_errors.get(qid, 'invalid')}"
                for qid in sorted(retry_ids)
            )

    if retry_ids:
        assert meta is not None
        failed = _zero_judgment(
            case,
            candidate,
            judge_model,
            f"Judge schema/evidence invalid after retry: {last_validation_error}",
            meta,
            raw=raw or "",
            primary_judge_model=judge_model,
        )
        failed.retry_count = 1
        return failed

    q_scores = [accepted[q.id] for q in case.questions]
    assert meta is not None
    return JudgeResult(
        blind_id=candidate.blind_id,
        candidate_key=candidate.candidate_key,
        question_scores=q_scores,
        weighted_accuracy=_weighted_accuracy(case, q_scores),
        coverage_score=_weighted_subscale(case, q_scores, "recall"),
        quality_score=_weighted_subscale(case, q_scores, "quality"),
        discipline_score=_weighted_subscale(case, q_scores, "precision"),
        judge_model=judge_model,
        primary_judge_model=judge_model,
        judge_meta=meta,
        raw_judge_json=raw,
        status="valid",
        retry_count=1 if accumulated_meta and len(accumulated_meta.paid_attempts) > 1 else 0,
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
        max_wall_s: float = 900.0,
        benchmark_track: str = "controlled",
        max_retries: Optional[int] = None,
        judge_allowed_providers: Optional[List[str]] = None,
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
        self.benchmark_track = benchmark_track
        self.judge_allowed_providers = list(judge_allowed_providers or [])
        self.max_wall_s = max(30.0, float(max_wall_s))
        # Wall-clock starts at the first submit(), not at pipeline construction.
        self._pipeline_started: Optional[float] = None
        self.started_at = utc_now_iso()
        n_hint = max(self.expected_total, 1)
        self._workers = max(1, min(n_hint, max_workers or n_hint))
        self.max_retries = max(
            0, int(n_hint if max_retries is None else max_retries)
        )
        self._pool = ThreadPoolExecutor(max_workers=self._workers)
        self._pending: Dict[Future, CandidateAnswer] = {}
        self._by_key: Dict[str, JudgeResult] = {}
        self._candidates: List[CandidateAnswer] = []
        self._worker_progress: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._started_at: Dict[str, float] = {}
        self._progress_state: Dict[str, Dict[str, Any]] = {}
        self._finished = 0
        self._shutdown = False
        _register_active_pipe(self)

    def _elapsed_wall_s(self) -> float:
        if self._pipeline_started is None:
            return 0.0
        return max(0.0, time.monotonic() - self._pipeline_started)

    def _budget_remaining_s(self) -> float:
        if self._pipeline_started is None:
            return self.max_wall_s
        return self.max_wall_s - self._elapsed_wall_s()

    def _deadline_reached(self) -> bool:
        return (
            self._pipeline_started is not None
            and self._elapsed_wall_s() >= self.max_wall_s
        )

    def _finish_pending_as(self, *, status: str, reason: str, marker: str) -> None:
        for fut, cand in list(self._pending.items()):
            fut.cancel()
            result = _zero_judgment(
                self.case,
                cand,
                self.judge_model,
                reason,
                None,
                primary_judge_model=self.judge_model,
            )
            result.status = status
            result.failure_reason = reason
            for score in result.question_scores:
                score.errors = [marker]
            self._by_key[cand.candidate_key] = result
            self._finished += 1
            self._emit(
                {
                    "phase": "done",
                    "key": cand.candidate_key,
                    "label": cand.display_label or cand.label or cand.candidate_key,
                    "done": self._finished,
                    "total": self.total,
                    "accuracy": 0.0,
                    "failed": True,
                    "note": reason,
                    "stage": status,
                    "percent": 100,
                    "elapsed_s": max(
                        0.0,
                        time.monotonic()
                        - self._started_at.get(cand.candidate_key, time.monotonic()),
                    ),
                }
            )
        self._pending.clear()

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
                None,
                primary_judge_model=self.judge_model,
            )
        if not _candidate_has_answer(cand):
            z = _zero_judgment(
                self.case,
                cand,
                self.judge_model,
                "Empty answer — not judged (skipped DeepSeek to save credits)",
                None,
                primary_judge_model=self.judge_model,
            )
            for qs in z.question_scores:
                qs.errors = ["empty_answer"]
            return z
        if (cand.meta.finish_reason or "").lower() == "content_filter":
            return _zero_judgment(
                self.case,
                cand,
                self.judge_model,
                f"Partial candidate output — finish_reason={cand.meta.finish_reason}",
                None,
                primary_judge_model=self.judge_model,
            )
        # A hit length cap is not itself a failure: judge the answer whenever every
        # required section carries content, and fail only on real absence.
        missing = _candidate_missing_sections(self.case, cand)
        if missing:
            return _zero_judgment(
                self.case,
                cand,
                self.judge_model,
                "Partial candidate output — missing required sections: "
                + ", ".join(missing),
                None,
                primary_judge_model=self.judge_model,
            )
        return judge_candidate(
            self.case,
            cand,
            self.judge_model,
            temperature=self.temperature,
            gold_reference=self.gold_reference,
            api_key=self.api_key,
            verifier_model="",
            allowed_providers=(
                self.judge_allowed_providers
                if self.benchmark_track == "controlled"
                else None
            ),
            require_parameters=False,
            progress_callback=lambda stage, percent: self._worker_progress.put(
                {
                    "key": cand.candidate_key,
                    "label": cand.display_label or cand.label or cand.candidate_key,
                    "stage": stage,
                    "percent": percent,
                }
            ),
        )

    def _verifier_is_independent(self) -> bool:
        verifier = (self.verifier_model or "").strip()
        if not verifier or verifier == self.judge_model:
            return False
        candidate_models = {
            str(c.meta.model or "").strip()
            for c in self._candidates
        } | {
            str(c.meta.requested_model or "").strip()
            for c in self._candidates
        }
        return verifier not in candidate_models

    def _needs_whole_run_verifier(self) -> bool:
        recoverable_statuses = {
            "judge_transport_failed",
            "judge_schema_invalid",
            "judge_evidence_invalid",
            "timed_out",
        }
        technical_failures = sum(
            result.status in recoverable_statuses
            for result in self._by_key.values()
        )
        fixed_cohort = max(self.expected_total, len(self._candidates), 1)
        threshold = max(2, (3 * fixed_cohort + 9) // 10)
        return (
            self._verifier_is_independent()
            and technical_failures >= threshold
        )

    def _wait_with_heartbeats(
        self,
        futures: Dict[Future, CandidateAnswer],
        *,
        timeout: float,
        stage: str,
        percent: int,
    ) -> tuple[set[Future], set[Future]]:
        """Wait while keeping elapsed_s moving for active rows (no frozen UI clock)."""
        deadline = time.monotonic() + max(0.0, timeout)
        pending: set[Future] = set(futures)
        done: set[Future] = set()
        while pending and time.monotonic() < deadline:
            if is_cancelled(self.run_scope) or self._deadline_reached():
                break
            slice_timeout = min(1.0, max(0.05, deadline - time.monotonic()))
            finished, pending = wait(
                tuple(pending),
                timeout=slice_timeout,
                return_when=FIRST_COMPLETED,
            )
            done.update(finished)
            self._drain_worker_progress()
            now = time.monotonic()
            for fut, cand in futures.items():
                if fut not in pending:
                    continue
                key = cand.candidate_key
                self._emit(
                    {
                        "phase": "progress",
                        "key": key,
                        "label": cand.display_label or cand.label or key,
                        "stage": stage,
                        "percent": percent,
                        "elapsed_s": max(
                            0.0, now - self._started_at.get(key, now)
                        ),
                        "done": self._finished,
                        "total": self.total,
                        "active_attempt": True,
                    }
                )
        return done, pending

    def _verify_whole_run(self) -> None:
        remaining = self._budget_remaining_s()
        if remaining <= 0:
            # Do not launch a whole-run verifier with an exhausted wall-clock budget.
            return
        # Methodological contract: once the verifier activates, it re-judges the
        # entire fixed eligible set and becomes the sole effective judge. Keeping
        # primary DeepSeek scores alongside Qwen would mix judge cohorts.
        eligible = [
            cand
            for cand in self._candidates
            if not cand.meta.error
            and _candidate_has_answer(cand)
            and not _candidate_missing_sections(self.case, cand)
            and (cand.meta.finish_reason or "").lower() != "content_filter"
        ]
        if not eligible:
            return
        workers = min(self._workers, len(eligible))
        executor = ThreadPoolExecutor(max_workers=max(1, workers))
        futures: Dict[Future, CandidateAnswer] = {}
        for cand in eligible:
            self._started_at[cand.candidate_key] = time.monotonic()
            self._progress_state[cand.candidate_key] = {
                "stage": "independent verifier",
                "percent": 92,
            }
            self._emit(
                {
                    "phase": "retry",
                    "key": cand.candidate_key,
                    "label": cand.display_label or cand.label or cand.candidate_key,
                    "stage": "independent verifier",
                    "percent": 92,
                    "done": self._finished,
                    "total": self.total,
                    "elapsed_s": 0.0,
                    "active_attempt": True,
                }
            )
            futures[
                executor.submit(
                    judge_candidate,
                    self.case,
                    cand,
                    self.verifier_model,
                    0.0,
                    self.gold_reference,
                    self.api_key,
                    "",
                    False,
                )
            ] = cand
        done, pending = self._wait_with_heartbeats(
            futures,
            timeout=max(0.1, remaining),
            stage="independent verifier",
            percent=92,
        )
        for fut in done:
            cand = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = _zero_judgment(
                    self.case,
                    cand,
                    self.verifier_model,
                    f"Verifier exception: {type(exc).__name__}: {exc}",
                    None,
                    primary_judge_model=self.judge_model,
                )
            result.primary_judge_model = self.judge_model
            result.judge_model = self.verifier_model
            if result.judge_meta and isinstance(result.judge_meta, ModelCallMeta):
                # Merge prior primary/repair paid_attempts with verifier attempts
                # (same append-only pattern as corrective retry).
                prior = self._by_key.get(cand.candidate_key)
                prior_meta = (
                    prior.judge_meta
                    if prior is not None and isinstance(prior.judge_meta, ModelCallMeta)
                    else None
                )
                verifier_attempts = list(result.judge_meta.paid_attempts or [])
                for attempt in verifier_attempts:
                    if attempt.get("role") == "primary":
                        attempt["role"] = "verifier"
                prior_attempts = list((prior_meta.paid_attempts if prior_meta else None) or [])
                merged_attempts = prior_attempts + verifier_attempts
                prior_cost = float(prior_meta.cost_usd or 0.0) if prior_meta else 0.0
                result.judge_meta = result.judge_meta.model_copy(
                    update={
                        "cost_usd": round(
                            prior_cost + float(result.judge_meta.cost_usd or 0.0),
                            8,
                        ),
                        "prompt_tokens": int(
                            (prior_meta.prompt_tokens if prior_meta else 0) or 0
                        )
                        + int(result.judge_meta.prompt_tokens or 0),
                        "completion_tokens": int(
                            (prior_meta.completion_tokens if prior_meta else 0) or 0
                        )
                        + int(result.judge_meta.completion_tokens or 0),
                        "paid_attempts": merged_attempts,
                    }
                )
            result.failure_reason = (
                f"Whole-run verifier activated after primary judge failure; "
                f"effective judge={self.verifier_model}. "
                + (result.failure_reason or "")
            ).strip()
            self._by_key[cand.candidate_key] = result
            self._emit(
                {
                    "phase": "retry_done",
                    "key": cand.candidate_key,
                    "label": cand.display_label or cand.label or cand.candidate_key,
                    "done": self._finished,
                    "total": self.total,
                    "accuracy": result.weighted_accuracy,
                    "coverage": result.coverage_score,
                    "quality": result.quality_score,
                    "discipline": result.discipline_score,
                    "failed": is_failed_judgment(result),
                    "stage": "complete",
                    "percent": 100,
                    "elapsed_s": max(
                        0.0,
                        time.monotonic()
                        - self._started_at.get(cand.candidate_key, time.monotonic()),
                    ),
                }
            )
        for fut in pending:
            cand = futures[fut]
            fut.cancel()
            timed_out = _zero_judgment(
                self.case,
                cand,
                self.verifier_model,
                "Whole-run verifier exceeded the judge wall-clock budget",
                None,
                primary_judge_model=self.judge_model,
            )
            timed_out.status = "timed_out"
            timed_out.failure_reason = "Whole-run verifier timed out"
            self._by_key[cand.candidate_key] = timed_out
            self._emit(
                {
                    "phase": "retry_done",
                    "key": cand.candidate_key,
                    "label": cand.display_label or cand.label or cand.candidate_key,
                    "done": self._finished,
                    "total": self.total,
                    "accuracy": 0.0,
                    "failed": True,
                    "stage": "complete",
                    "percent": 100,
                    "elapsed_s": max(
                        0.0,
                        time.monotonic()
                        - self._started_at.get(cand.candidate_key, time.monotonic()),
                    ),
                }
            )
        executor.shutdown(wait=False, cancel_futures=True)

    def _ensure_terminal_rows(self) -> None:
        """Guarantee exactly one terminal result for every submitted candidate."""
        for cand in self._candidates:
            if cand.candidate_key in self._by_key:
                continue
            result = _zero_judgment(
                self.case,
                cand,
                self.judge_model,
                "Judge pipeline ended without a terminal result",
                None,
                primary_judge_model=self.judge_model,
            )
            result.status = "judge_transport_failed"
            result.failure_reason = "Missing terminal judge result"
            for score in result.question_scores:
                score.errors = ["missing_terminal_result"]
            self._by_key[cand.candidate_key] = result
            self._finished += 1

    def _drain_worker_progress(self) -> None:
        while True:
            try:
                evt = self._worker_progress.get_nowait()
            except queue.Empty:
                break
            key = str(evt.get("key") or "")
            self._progress_state[key] = dict(evt)
            self._emit(
                {
                    "phase": "progress",
                    **evt,
                    "elapsed_s": max(
                        0.0, time.monotonic() - self._started_at.get(key, time.monotonic())
                    ),
                    "done": self._finished,
                    "total": self.total,
                }
            )

    def _emit_heartbeats(self) -> None:
        now = time.monotonic()
        for cand in self._pending.values():
            key = cand.candidate_key
            state = self._progress_state.get(key) or {
                "stage": "queued",
                "percent": 10,
            }
            self._emit(
                {
                    "phase": "progress",
                    "key": key,
                    "label": cand.display_label or cand.label or key,
                    "stage": state.get("stage") or "judging",
                    "percent": int(state.get("percent") or 10),
                    "elapsed_s": max(0.0, now - self._started_at.get(key, now)),
                    "done": self._finished,
                    "total": self.total,
                }
            )

    def submit(self, cand: CandidateAnswer) -> bool:
        """Queue one candidate for DeepSeek. Returns False if duplicate / closed."""
        if self._shutdown or is_cancelled(self.run_scope):
            return False
        if any(c.candidate_key == cand.candidate_key for c in self._candidates):
            return False
        if self._pipeline_started is None:
            self._pipeline_started = time.monotonic()
            self.started_at = utc_now_iso()
        self._candidates.append(cand)
        self._started_at[cand.candidate_key] = time.monotonic()
        self._progress_state[cand.candidate_key] = {
            "stage": "queued",
            "percent": 10,
        }
        self._emit(
            {
                "phase": "queued",
                "key": cand.candidate_key,
                "label": cand.display_label or cand.label or cand.candidate_key,
                "done": self._finished,
                "total": self.total,
                "stage": "queued",
                "percent": 10,
                "elapsed_s": 0.0,
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
                None,
                primary_judge_model=self.judge_model,
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
                "coverage": result.coverage_score,
                "quality": result.quality_score,
                "discipline": result.discipline_score,
                "failed": is_failed_judgment(result),
                "note": (result.failure_reason or "")
                or ((result.judge_meta.error if result.judge_meta else None) or ""),
                "failure_reason": result.failure_reason or "",
                "status": result.status,
                "stage": "complete",
                "percent": 100,
                "elapsed_s": max(
                    0.0,
                    time.monotonic()
                    - self._started_at.get(cand.candidate_key, time.monotonic()),
                ),
            }
        )

    def poll(self) -> int:
        """Non-blocking: harvest finished futures. Call from the UI/script thread."""
        self._drain_worker_progress()
        if self._deadline_reached() and self._pending:
            self._finish_pending_as(
                status="timed_out",
                reason=(
                    f"Judge pipeline exceeded {self.max_wall_s:.0f}s wall-clock budget"
                ),
                marker="judge_timeout",
            )
            self.close(cancel_pending=True)
            self._drain_worker_progress()
            return 0
        n = 0
        for fut in list(self._pending):
            if fut.done():
                self._harvest(fut)
                n += 1
        self._drain_worker_progress()
        return n

    def finalize(self) -> List[JudgeResult]:
        """Wait for judges, retry recoverable failures, or stop cooperatively."""
        while self._pending:
            if is_cancelled(self.run_scope):
                self._finish_pending_as(
                    status="cancelled",
                    reason="Judging cancelled by user",
                    marker="cancelled",
                )
                self.close(cancel_pending=True)
                break
            if self._deadline_reached():
                self._finish_pending_as(
                    status="timed_out",
                    reason=f"Judge pipeline exceeded {self.max_wall_s:.0f}s wall-clock budget",
                    marker="judge_timeout",
                )
                self.close(cancel_pending=True)
                break
            done, _ = wait(
                tuple(self._pending.keys()),
                timeout=1.0,
                return_when=FIRST_COMPLETED,
            )
            self._drain_worker_progress()
            if not done:
                self._emit_heartbeats()
            for fut in done:
                if fut in self._pending:
                    self._harvest(fut)

        ordered = [
            self._by_key[c.candidate_key]
            for c in self._candidates
            if c.candidate_key in self._by_key
        ]
        cand_by = {c.candidate_key: c for c in self._candidates}
        retry_attempts = 0
        for j in list(ordered):
            if is_cancelled(self.run_scope):
                break
            if retry_attempts >= self.max_retries:
                break
            remaining = self._budget_remaining_s()
            if remaining <= 0:
                break
            if not _is_rejudgable_failure(j):
                continue
            cand = cand_by.get(j.candidate_key)
            if not cand or not _candidate_has_answer(cand):
                continue
            # Real new attempt: reset the live clock so the row never freezes at
            # a stale elapsed_s from the previous terminal N/A paint.
            self._started_at[cand.candidate_key] = time.monotonic()
            self._progress_state[cand.candidate_key] = {
                "stage": "corrective retry",
                "percent": 75,
            }
            self._emit(
                {
                    "phase": "retry",
                    "key": cand.candidate_key,
                    "label": cand.display_label or cand.label or cand.candidate_key,
                    "done": self._finished,
                    "total": self.total,
                    "stage": "corrective retry",
                    "percent": 75,
                    "elapsed_s": 0.0,
                    "active_attempt": True,
                }
            )
            time.sleep(0.35)
            retry_attempts += 1
            prior_meta = j.judge_meta if isinstance(j.judge_meta, ModelCallMeta) else None
            retry_future = self._pool.submit(
                judge_candidate,
                self.case,
                cand,
                self.judge_model,
                temperature=self.temperature,
                gold_reference=self.gold_reference,
                api_key=self.api_key,
                verifier_model="",
                allowed_providers=(
                    self.judge_allowed_providers
                    if self.benchmark_track == "controlled"
                    else None
                ),
                require_parameters=False,
                progress_callback=lambda stage, percent: self._worker_progress.put(
                    {
                        "key": cand.candidate_key,
                        "label": cand.display_label
                        or cand.label
                        or cand.candidate_key,
                        "stage": stage,
                        "percent": percent,
                    }
                ),
            )
            remaining = max(0.0, self._budget_remaining_s())
            retry_done, retry_pending = self._wait_with_heartbeats(
                {retry_future: cand},
                timeout=remaining,
                stage="corrective retry",
                percent=75,
            )
            if retry_future in retry_done:
                retry = retry_future.result()
                if prior_meta and retry.judge_meta:
                    # Keep primary-attempt cost when the pipeline-level retry replaces
                    # an earlier failed observation.
                    merged_attempts = list(prior_meta.paid_attempts or []) + list(
                        retry.judge_meta.paid_attempts or []
                    )
                    retry.judge_meta = retry.judge_meta.model_copy(
                        update={
                            "cost_usd": round(
                                float(prior_meta.cost_usd or 0.0)
                                + float(retry.judge_meta.cost_usd or 0.0),
                                8,
                            ),
                            "prompt_tokens": int(prior_meta.prompt_tokens or 0)
                            + int(retry.judge_meta.prompt_tokens or 0),
                            "completion_tokens": int(prior_meta.completion_tokens or 0)
                            + int(retry.judge_meta.completion_tokens or 0),
                            "paid_attempts": merged_attempts,
                        }
                    )
            else:
                for fut in retry_pending:
                    fut.cancel()
                retry = _zero_judgment(
                    self.case,
                    cand,
                    self.judge_model,
                    "Corrective retry exceeded the judge wall-clock budget",
                    prior_meta,
                    primary_judge_model=self.judge_model,
                )
                retry.status = "timed_out"
                retry.failure_reason = "Corrective retry timed out"
            retry.retry_count += 1
            retry.primary_judge_model = self.judge_model
            self._drain_worker_progress()
            self._by_key[j.candidate_key] = retry
            self._emit(
                {
                    "phase": "retry_done",
                    "key": cand.candidate_key,
                    "label": cand.display_label or cand.label or cand.candidate_key,
                    "done": self._finished,
                    "total": self.total,
                    "accuracy": retry.weighted_accuracy,
                    "coverage": retry.coverage_score,
                    "quality": retry.quality_score,
                    "discipline": retry.discipline_score,
                    "failed": is_failed_judgment(retry),
                    "stage": "complete",
                    "percent": 100,
                    "elapsed_s": max(
                        0.0,
                        time.monotonic()
                        - self._started_at.get(cand.candidate_key, time.monotonic()),
                    ),
                }
            )
        if (
            self._needs_whole_run_verifier()
            and not is_cancelled(self.run_scope)
            and self._budget_remaining_s() > 0
        ):
            self._verify_whole_run()
        self._ensure_terminal_rows()
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

    def cancel_and_snapshot(self) -> Dict[str, Any]:
        """Cancel pending work and retain every submitted candidate/result."""
        if self._pending:
            self._finish_pending_as(
                status="cancelled",
                reason="Judging cancelled by user",
                marker="cancelled",
            )
        snapshot = {
            "case": self.case,
            "candidates": list(self._candidates),
            "judgments": [
                self._by_key[candidate.candidate_key]
                for candidate in self._candidates
                if candidate.candidate_key in self._by_key
            ],
            "judge_model": self.judge_model,
            "gold_reference": self.gold_reference,
            "started_at": self.started_at,
            "benchmark_track": self.benchmark_track,
        }
        self.close(cancel_pending=True)
        return snapshot


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


def abandon_all_pipelines(run_scope: str) -> List[Dict[str, Any]]:
    """Cancel pipelines in one scope and return persistence-ready snapshots."""
    snapshots: List[Dict[str, Any]] = []
    for pipe in list(_ACTIVE_PIPES.get(run_scope, [])):
        try:
            snapshots.append(pipe.cancel_and_snapshot())
        except Exception:
            pass
    _ACTIVE_PIPES.pop(run_scope, None)
    return snapshots


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
    benchmark_track: str = "controlled",
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
        benchmark_track=benchmark_track,
    )
    for c in candidates:
        pipe.submit(c)
    return pipe.finalize()


def build_ranking(judgments: List[JudgeResult]) -> List[Dict[str, Any]]:
    rows = []
    for j in judgments:
        failed = is_failed_judgment(j)
        flags = judgment_flags(j)
        # Ties use the stored unrounded composite when available; display is rounded.
        raw_accuracy = None if failed else float(j.weighted_accuracy)
        rows.append(
            {
                "key": j.candidate_key,
                "blind_id": j.blind_id,
                "accuracy": None if failed else round(raw_accuracy, 2),
                "accuracy_raw": raw_accuracy,
                "coverage": None if failed else j.coverage_score,
                "quality": None if failed else j.quality_score,
                "discipline": None if failed else j.discipline_score,
                "status": "n/a" if failed else "ok",
                "status_note": ", ".join(flags[:4]) if flags else "",
                "primary_judge": j.primary_judge_model or j.judge_model,
                "effective_judge": j.judge_model,
            }
        )
    rows.sort(
        key=lambda r: (
            0 if r["status"] == "ok" else 1,
            -float(r["accuracy_raw"] if r["accuracy_raw"] is not None else -1),
        )
    )
    last_score: Optional[float] = None
    last_rank = 0
    for i, row in enumerate(rows, 1):
        if row["status"] != "ok":
            row["rank"] = None
            continue
        score = float(row["accuracy_raw"])
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
            "Technical failures are N/A; exact ties on unrounded scores keep the same rank."
        ),
    }
