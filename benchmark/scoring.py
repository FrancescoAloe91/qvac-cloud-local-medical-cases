"""Deterministic linear scoring helpers (scientifically transparent)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from benchmark.schema import Case, CandidateAnswer, JudgeResult, QuestionScore

# Hard ceilings: a true 100% is not used in this benchmark.
ITEM_SCORE_CAP = 96.5
WEIGHTED_CAP = 97.0

# Continuous mix: checklist + clinical quality + stem specificity
W_MUST = 0.45
W_ACCEPT = 0.20
W_QUALITY = 0.25
W_SPEC = 0.10


def linear_item_score(
    *,
    m_hit: int,
    m_total: int,
    a_hit: int,
    a_total: int,
    quality: float,
    specificity: float,
) -> float:
    mt = max(int(m_total), 1)
    at = max(int(a_total), 1)
    m = min(max(int(m_hit), 0), mt) / mt
    a = min(max(int(a_hit), 0), at) / at
    q = min(max(float(quality), 0.0), 1.0)
    s = min(max(float(specificity), 0.0), 1.0)
    raw = 100.0 * (W_MUST * m + W_ACCEPT * a + W_QUALITY * q + W_SPEC * s)
    return round(min(raw, ITEM_SCORE_CAP), 2)


def stem_anchor_terms(case: Case) -> List[str]:
    """Case-specific anchors a strong answer should touch (not generic filler)."""
    blob = f"{case.stem} {case.gold_summary}".lower()
    anchors: List[str] = []
    for n in re.findall(r"\d+(?:[.,]\d+)?", blob):
        if len(n) >= 2:
            anchors.append(n)
    # Salient clinical tokens from stem/gold (length filter drops stopwords-ish noise)
    stop = {
        "with",
        "from",
        "that",
        "this",
        "have",
        "been",
        "were",
        "their",
        "about",
        "after",
        "before",
        "patient",
        "history",
        "years",
        "year",
        "days",
        "hour",
        "hours",
        "brought",
        "family",
        "denies",
        "reports",
        "treated",
        "followed",
    }
    for tok in re.findall(r"[a-z][a-z\-]{3,}", blob):
        if tok in stop:
            continue
        anchors.append(tok)
    # Always include rubric must_include lexicon
    for q in case.questions:
        for m in q.rubric.must_include:
            for piece in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", m.lower()):
                anchors.append(piece)
    # de-dupe preserving order
    seen = set()
    out: List[str] = []
    for a in anchors:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out[:80]


def stem_specificity(case: Case, answer: str) -> float:
    """Fraction of stem/gold anchors present in the answer (0–1)."""
    text = (answer or "").lower()
    if not text.strip():
        return 0.0
    anchors = stem_anchor_terms(case)
    if not anchors:
        return 0.0
    hits = sum(1 for a in anchors if a in text)
    # Saturate: hitting ~40% of anchors is already very specific
    return round(min(1.0, hits / max(12, int(0.35 * len(anchors)))), 3)


def mean_question_metric(scores: Sequence[QuestionScore], attr: str) -> float:
    vals = []
    for s in scores:
        # metrics embedded in rationale as "quality=0.75" / stored loosely
        if attr == "score":
            vals.append(float(s.score))
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _parse_metric(rationale: str, key: str) -> float:
    m = re.search(rf"{key}=([0-9.]+)", rationale or "")
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except ValueError:
        return 0.0


def safety_score(case: Case, result: JudgeResult) -> float:
    for qs in result.question_scores:
        q = next((x for x in case.questions if x.id == qs.question_id), None)
        if q and q.kind == "safety":
            return float(qs.score)
    return 0.0


def ensure_unique_accuracies(
    case: Case,
    judgments: List[JudgeResult],
) -> Tuple[List[JudgeResult], List[str]]:
    """
    Guarantee strictly different weighted accuracies (transparent tie-break).

    Order key (clinical, not random):
      1) raw weighted accuracy
      2) safety section score (discriminator)
      3) mean quality
      4) mean stem specificity
      5) diagnosis section score
      6) candidate_key (stable)
    Then assign unique display accuracies preserving that order, capped at WEIGHTED_CAP.
    """
    notes: List[str] = []
    if len(judgments) <= 1:
        for j in judgments:
            j.weighted_accuracy = min(float(j.weighted_accuracy), WEIGHTED_CAP)
        return judgments, notes

    def diag_score(j: JudgeResult) -> float:
        for qs in j.question_scores:
            if qs.question_id == "diagnosis":
                return float(qs.score)
        return 0.0

    def mean_q(j: JudgeResult) -> float:
        vals = [_parse_metric(qs.rationale, "quality") for qs in j.question_scores]
        vals = [v for v in vals if v > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def mean_spec(j: JudgeResult) -> float:
        vals = [_parse_metric(qs.rationale, "spec") for qs in j.question_scores]
        vals = [v for v in vals if v > 0]
        return sum(vals) / len(vals) if vals else 0.0

    decorated = []
    for j in judgments:
        raw = min(float(j.weighted_accuracy), WEIGHTED_CAP)
        decorated.append(
            (
                raw,
                safety_score(case, j),
                mean_q(j),
                mean_spec(j),
                diag_score(j),
                j.candidate_key,
                j,
            )
        )
    decorated.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4], t[5]), reverse=True)

    # Spread: top gets min(raw, CAP), each next at least 0.05 below previous (unique)
    assigned: List[float] = []
    for i, row in enumerate(decorated):
        raw = row[0]
        if i == 0:
            acc = round(min(raw, WEIGHTED_CAP), 2)
        else:
            # Keep clinical magnitude when possible, but force uniqueness downward
            acc = round(min(raw, assigned[-1] - 0.05), 2)
            if acc == assigned[-1]:
                acc = round(assigned[-1] - 0.05, 2)
        if i > 0 and raw >= assigned[-1] - 0.001:
            notes.append(
                f"{row[5]}: tie-break vs higher rank "
                f"(safety/quality/spec/diagnosis) → {acc} (raw {raw})"
            )
        assigned.append(acc)
        row[6].weighted_accuracy = acc

    # Return in original candidate order
    by_key = {j.candidate_key: j for j in judgments}
    return [by_key[j.candidate_key] for j in judgments], notes


def scoring_legend(case: Case) -> Dict[str, Any]:
    weights = {q.id: q.weight for q in case.questions}
    top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "formula": (
            f"item = 100×({W_MUST}·must + {W_ACCEPT}·acceptable + "
            f"{W_QUALITY}·quality + {W_SPEC}·stem_specificity), "
            f"capped at {ITEM_SCORE_CAP}; weighted mean capped at {WEIGHTED_CAP} "
            "(100% not used)."
        ),
        "section_weights": weights,
        "heaviest_sections": [f"{k} ({v:.0%})" for k, v in top[:3]],
        "main_discriminators": [
            q.id for q in case.questions if q.kind == "safety" or q.weight >= 0.25
        ],
    }
