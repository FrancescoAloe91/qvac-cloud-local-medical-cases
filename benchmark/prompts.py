"""Shared candidate + judge prompts (identical for every model)."""

from __future__ import annotations

import json
from typing import Dict

from benchmark.schema import Case


def candidate_system() -> str:
    return (
        "You are an expert physician taking a structured clinical benchmark. "
        "Answer each numbered question clearly and specifically. "
        "Do not invent patient identifiers. Be concise but complete."
    )


def candidate_user(case: Case) -> str:
    lines = [
        "CLINICAL CASE:",
        case.stem.strip(),
        "",
        "Answer ALL of the following questions. Use this exact format:",
        "",
    ]
    for i, q in enumerate(case.questions, 1):
        lines.append(f"Q{i} [{q.id}]: {q.text}")
        lines.append(f"A{i}:")
        lines.append("")
    lines.append(
        "Fill every A# answer. Do not skip questions. "
        "For urgency use one of: critical, high, moderate, low."
    )
    return "\n".join(lines)


def parse_candidate_answers(case: Case, raw: str) -> Dict[str, str]:
    """Best-effort parse of A1/A2… or [id]: blocks. Falls back to whole text."""
    answers: Dict[str, str] = {}
    text = raw or ""
    for i, q in enumerate(case.questions, 1):
        marker = f"A{i}:"
        start = text.find(marker)
        if start < 0:
            alt = f"[{q.id}]"
            start = text.lower().find(alt.lower())
            if start >= 0:
                start = text.find(":", start)
                if start < 0:
                    continue
                start += 1
            else:
                continue
        else:
            start += len(marker)
        end = len(text)
        for j in range(i + 1, len(case.questions) + 1):
            nxt = text.find(f"A{j}:", start)
            if nxt >= 0:
                end = nxt
                break
        answers[q.id] = text[start:end].strip()

    n_q = len(case.questions)
    n_filled = sum(1 for q in case.questions if (answers.get(q.id) or "").strip())
    if text.strip() and n_filled == 0 and len(text.strip()) > 80:
        answers[case.questions[0].id] = (
            "[UNSTRUCTURED FULL RESPONSE — score only content relevant to "
            f"{case.questions[0].id}; apply linear formula on relevant parts only]\n"
            + text.strip()
        )
    return answers


def judge_system() -> str:
    return (
        "Act as a senior clinical evaluation engineer grading a competitive benchmark. "
        "Score anonymized candidates against a fixed rubric on a CONTINUOUS linear 0–100 scale. "
        "Frontier models often cover the checklist; you MUST still differentiate them with the "
        "quality term — identical 100s across candidates should be rare. "
        "Synonyms count. Do not guess vendors. Return ONLY valid JSON."
    )


def judge_user(
    case: Case,
    blind_id: str,
    answers: Dict[str, str],
    gold_reference: str = "",
) -> str:
    rubric_payload = []
    for q in case.questions:
        rubric_payload.append(
            {
                "question_id": q.id,
                "kind": q.kind,
                "weight": q.weight,
                "question": q.text,
                "rubric": q.rubric.model_dump(),
                "candidate_answer": answers.get(q.id, ""),
            }
        )
    schema = {
        "blind_id": blind_id,
        "question_scores": [
            {
                "question_id": "string",
                "score": "0-100 continuous (decimals OK, e.g. 81.4)",
                "rationale": "m_hit/m_tot, a_hit/a_tot, quality=0-1, formula result",
                "evidence": "short quote",
                "errors": ["missing checklist items and quality gaps"],
                "m_hit": "int",
                "m_total": "int",
                "a_hit": "int",
                "a_total": "int",
                "quality": "0-1 float",
            }
        ],
    }
    gold_block = ""
    if gold_reference.strip():
        gold_block = (
            "CONFIRMED REFERENCE / TARGET:\n"
            f"{gold_reference.strip()}\n\n"
            "Prefer this reference over conflicting rubric notes for diagnosis/safety.\n"
            "If acceptable/must_include are empty, score vs this reference + sound practice.\n\n"
        )
    return (
        f"CASE STEM:\n{case.stem}\n\n"
        f"{gold_block}"
        f"Evaluate {blind_id} only.\n\n"
        f"ITEMS:\n{json.dumps(rubric_payload, ensure_ascii=False, indent=2)}\n\n"
        "LINEAR FORMULA (mandatory — no score bands):\n"
        "Treat each must_include[] and acceptable[] entry as ONE atomic checklist point "
        "(count them separately; do not merge blobs).\n"
        "m = m_hit / max(m_total, 1)   # clinical-meaning matches, synonyms OK\n"
        "a = a_hit / max(a_total, 1)\n"
        "quality ∈ [0,1] continuous, based ONLY on this question's answer:\n"
        "  + case-specific anchors from the stem (numbers, drugs, eGFR, timelines, named risks)\n"
        "  + prioritization / ranking when asked\n"
        "  + actionable specificity (not vague 'stabilize / get labs / consult')\n"
        "  − fluff, hedging, generic textbook padding without stem anchors\n"
        "  − missing nuances called out in rubric.notes\n"
        "  Typical strong cloud answer: quality 0.55–0.85. quality ≥ 0.95 is rare.\n"
        "score = round(100 × (0.55×m + 0.25×a + 0.20×quality), 1)\n"
        "Examples:\n"
        "  full checklist m=1,a=1 but quality=0.7 → 100×(0.55+0.25+0.14)=94.0\n"
        "  full checklist m=1,a=1 quality=1.0 → 100.0 (exceptional only)\n"
        "  m=2/3,a=1/2,quality=0.6 → 100×(0.55×0.667+0.25×0.5+0.20×0.6)=64.2\n"
        "HARD RULES:\n"
        "- must_not / ignored required safety trap → score = min(score, 20) "
        "(main safety trap often ≤ 10).\n"
        "- Empty → 0.\n"
        "- Completing the checklist alone is NOT enough for 100; without near-perfect "
        "quality, even complete answers land ~88–96.\n"
        "- Prefer differentiation: if two answers both cover the checklist but one is more "
        "stem-specific, their scores MUST differ.\n"
        "- Return m_hit,m_total,a_hit,a_total,quality in each question_scores item.\n"
        "- Synonyms OK (cath≈PCI; tox≈UDS; CKD≈renal/eGFR; admission≈inpatient).\n\n"
        f"Return JSON only, shape:\n{json.dumps(schema, indent=2)}"
    )
