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
        "LINEAR FORMULA (mandatory — continuous; 100% is NOT a normal outcome):\n"
        "Treat each must_include[] and acceptable[] entry as ONE atomic checklist point.\n"
        "m = m_hit / max(m_total, 1)   # clinical meaning, synonyms OK\n"
        "a = a_hit / max(a_total, 1)\n"
        "quality ∈ [0,1] = clinical QUALITY of diagnosis/reasoning for THIS question:\n"
        "  correctness of primary call, differential coherence, prioritization,\n"
        "  case-specific meaning (not generic textbook paste), absence of dangerous advice.\n"
        "  Typical strong cloud answer: quality 0.55–0.82. Do NOT give quality > 0.90 "
        "unless truly exceptional.\n"
        "The host adds stem_specificity in code and RECOMPUTES the numeric score.\n"
        "Host: score = 100×(0.45·m + 0.20·a + 0.25·quality + 0.10·spec), CAP 96.5/item.\n"
        "You must return m_hit,m_total,a_hit,a_total,quality (host ignores your score field).\n"
        "HARD RULES:\n"
        "- must_not / ignored safety trap → list in errors[] (host caps ≤20).\n"
        "- Empty → m_hit=0,a_hit=0,quality=0.\n"
        "- Two checklist-complete answers with different diagnostic nuance MUST get "
        "different quality values.\n"
        "- Synonyms OK (cath≈PCI; tox≈UDS; CKD≈renal/eGFR; admission≈inpatient).\n\n"
        f"Return JSON only, shape:\n{json.dumps(schema, indent=2)}"
    )
