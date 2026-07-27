"""Deterministic claim-based scoring helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from benchmark.schema import Case, QuestionScore

# Hard ceilings: a true 100% is not used in this benchmark.
ITEM_SCORE_CAP = 96.5
WEIGHTED_CAP = 100.0
SCORING_VERSION = "graded-clinical-v3"

# Rubric mode (no gold): checklist still exists, but quality dominates.
W_MUST = 0.30
W_ACCEPT = 0.20
W_QUALITY = 0.40
W_SPEC = 0.10

# Gold mode: semantic closeness to the gold thesis — no checklist atomization.
W_ALIGN = 0.50
W_GOLD_QUALITY = 0.30
W_GOLD_SPEC = 0.20


def claim_correctness_score(
    *,
    matched: int,
    total_reference: int,
    unsupported: int = 0,
    contradictions: int = 0,
    quality: float = 1.0,
) -> Tuple[float, float, float]:
    """Legacy v2 rescoring helper retained for archived binary artifacts.

    Coverage remains primary (55%), clinical quality contributes 25%, and
    precision contributes 20%. Contradictions receive a separate bounded penalty
    so one error matters without collapsing an otherwise useful answer.
    """
    total = max(int(total_reference), 1)
    tp = min(max(int(matched), 0), total)
    fn = total - tp
    fp = max(int(unsupported), 0) + max(int(contradictions), 0)
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    clinical_quality = min(max(float(quality), 0.0), 1.0)
    base = 100.0 * (
        0.55 * recall
        + 0.25 * clinical_quality
        + 0.20 * precision
    )
    contradiction_penalty = min(20.0, 7.5 * max(int(contradictions), 0))
    score = max(0.0, base - contradiction_penalty)
    return round(score, 2), round(precision, 4), round(recall, 4)


def graded_clinical_score(
    *,
    coverage: float,
    quality: float,
    discipline: float,
) -> float:
    """Continuous section score: 50% coverage, 35% quality, 15% discipline."""
    cov = min(max(float(coverage), 0.0), 1.0)
    qual = min(max(float(quality), 0.0), 1.0)
    disc = min(max(float(discipline), 0.0), 1.0)
    return round(100.0 * (0.50 * cov + 0.35 * qual + 0.15 * disc), 2)


def evidence_discipline_score(
    additions: Sequence[Mapping[str, Any]],
    *,
    total_reference: int,
) -> float:
    """Proportional penalty for genuinely problematic added content.

    Helpful and neutral additions are never penalized. Unsupported speculation is
    mild; contradictions and dangerous advice matter progressively more.
    """
    factors = {
        "helpful": 0.0,
        "neutral": 0.0,
        "unsupported": 0.25,
        "contradictory": 0.75,
        "dangerous": 1.0,
    }
    penalty = 0.0
    for item in additions:
        classification = str(item.get("classification") or "").strip().lower()
        if classification not in factors:
            raise ValueError(f"Unknown added-content classification: {classification}")
        try:
            severity = min(max(float(item.get("severity", 0.0)), 0.0), 1.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("Added-content severity must be a 0-1 number") from exc
        penalty += factors[classification] * severity
    scale = max(int(total_reference), 1)
    return round(max(0.0, 1.0 - penalty / scale), 4)


def linear_item_score(
    *,
    m_hit: int,
    m_total: int,
    a_hit: int,
    a_total: int,
    quality: float,
    specificity: float,
) -> float:
    """Rubric-mode score: soft checklist + clinical quality + stem specificity."""
    mt = max(int(m_total), 1)
    at = max(int(a_total), 1)
    m = min(max(int(m_hit), 0), mt) / mt
    a = min(max(int(a_hit), 0), at) / at
    q = min(max(float(quality), 0.0), 1.0)
    s = min(max(float(specificity), 0.0), 1.0)
    raw = 100.0 * (W_MUST * m + W_ACCEPT * a + W_QUALITY * q + W_SPEC * s)
    return round(min(raw, ITEM_SCORE_CAP), 2)


def semantic_item_score(
    *,
    alignment: float,
    quality: float,
    specificity: float,
) -> float:
    """
    Gold-mode score: how close the answer is *in clinical meaning* to the gold.

    alignment = holistic semantic match for that section (diagnosis thesis,
    workup intent, safety advice, next steps) — synonyms and near-equivalent
    formulations count; exact wording / acronyms do not.
    """
    al = min(max(float(alignment), 0.0), 1.0)
    q = min(max(float(quality), 0.0), 1.0)
    s = min(max(float(specificity), 0.0), 1.0)
    raw = 100.0 * (W_ALIGN * al + W_GOLD_QUALITY * q + W_GOLD_SPEC * s)
    return round(min(raw, ITEM_SCORE_CAP), 2)


def soft_alignment_from_checklist(
    *,
    m_hit: int,
    m_total: int,
    a_hit: int,
    a_total: int,
    quality: float,
) -> float:
    """
    Bridge when an older judge payload has m/a but no alignment field.
    Blends checklist coverage with quality so near-miss meaning is not zeroed.
    """
    mt = max(int(m_total), 1)
    at = max(int(a_total), 1)
    m = min(max(int(m_hit), 0), mt) / mt
    a = min(max(int(a_hit), 0), at) / at
    q = min(max(float(quality), 0.0), 1.0)
    # Quality carries semantic credit; checklist is a soft hint, not a veto.
    return round(min(1.0, 0.40 * m + 0.15 * a + 0.45 * q), 3)


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


def scoring_guide_markdown() -> str:
    """Load canonical scoring explainer for UI / docs."""
    path = Path(__file__).resolve().parent / "SCORING.md"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return (
            "Section score: 50% graded reference coverage + 35% clinical quality + "
            "15% evidence discipline. Helpful/neutral additions are not penalized. "
            "Technical failures remain N/A and ties remain ties."
        )


def scoring_legend(case: Case) -> Dict[str, Any]:
    weights = {q.id: q.weight for q in case.questions}
    top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "formula": (
            "section = 50% graded coverage + 35% clinical quality + "
            "15% evidence discipline; helpful/neutral additions are unpenalized; "
            "weighted mean uses "
            "predeclared section weights; ties remain ties"
        ),
        "section_weights": weights,
        "heaviest_sections": [f"{k} ({v:.0%})" for k, v in top[:3]],
        "main_discriminators": [
            q.id for q in case.questions if q.kind == "safety" or q.weight >= 0.25
        ],
    }
