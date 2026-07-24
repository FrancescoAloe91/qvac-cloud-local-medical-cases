"""Shared candidate + judge prompts (identical for every model)."""

from __future__ import annotations

import json
from typing import Dict, List

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
    # Prefer markers A1:, A2: …
    for i, q in enumerate(case.questions, 1):
        marker = f"A{i}:"
        start = text.find(marker)
        if start < 0:
            # try [question_id]
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
        # end at next A{n+1}: or end
        end = len(text)
        for j in range(i + 1, len(case.questions) + 1):
            nxt = text.find(f"A{j}:", start)
            if nxt >= 0:
                end = nxt
                break
        answers[q.id] = text[start:end].strip()

    n_q = len(case.questions)
    n_filled = sum(1 for q in case.questions if (answers.get(q.id) or "").strip())
    # Sparse / unstructured: do NOT copy the full essay into every question
    # (that inflated all dimensions toward 100). Leave gaps empty → score 0
    # for that item, unless almost nothing parsed — then one shared blob flagged.
    if text.strip() and n_filled == 0 and len(text.strip()) > 80:
        # Completely unstructured: attach once under first question only;
        # others stay empty so the judge cannot rubber-stamp every axis.
        answers[case.questions[0].id] = (
            "[UNSTRUCTURED FULL RESPONSE — score only content relevant to "
            f"{case.questions[0].id}; cap at 70 unless clearly focused]\n"
            + text.strip()
        )
    elif text.strip() and 0 < n_filled < n_q:
        # Leave missing slots empty (harder / fairer discrimination).
        pass
    return answers


def judge_system() -> str:
    return (
        "Act as a senior clinical evaluation engineer grading a competitive benchmark. "
        "Score anonymized candidates (Candidate 1, Candidate 2, …) against a fixed rubric. "
        "Use a CONTINUOUS linear 0–100 scale (not score bands/steps). "
        "Synonyms count; do not inflate. Do not guess vendors. "
        "Return ONLY valid JSON matching the schema."
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
                "score": "0-100 number (continuous; decimals OK, e.g. 73.5)",
                "rationale": "short — state coverage fraction used",
                "evidence": "quote from candidate answer",
                "errors": ["missing/wrong items"],
            }
        ],
    }
    gold_block = ""
    if gold_reference.strip():
        gold_block = (
            "CONFIRMED REFERENCE DIAGNOSIS / TARGET (ground truth for this run):\n"
            f"{gold_reference.strip()}\n\n"
            "When the reference conflicts with the static rubric notes, prefer the reference "
            "for diagnosis/safety meaning.\n"
            "If rubric acceptable/must_include lists are empty, score primarily against this reference "
            "and sound clinical practice for the stem.\n\n"
        )
    return (
        f"CASE STEM:\n{case.stem}\n\n"
        f"{gold_block}"
        f"Evaluate {blind_id} only. Do not guess which vendor it is.\n\n"
        f"ITEMS:\n{json.dumps(rubric_payload, ensure_ascii=False, indent=2)}\n\n"
        "LINEAR SCORING (mandatory — continuous 0–100, NO step bands):\n"
        "For each question compute a coverage fraction, then score = 100 × that fraction.\n"
        "1) must_include: count how many required concepts are present by CLINICAL MEANING "
        "(synonyms/paraphrases OK). Let m_hit / m_total be that ratio (if m_total=0, treat as 1).\n"
        "2) acceptable[]: count how many distinct acceptable points are covered. "
        "Let a_hit / a_total (if a_total=0, treat as 1).\n"
        "3) Combine linearly: coverage = 0.70 × (m_hit/m_total) + 0.30 × (a_hit/a_total)\n"
        "4) score = round(100 × coverage, 1)  // e.g. 2/3 must + 1/2 acceptable → "
        "0.7×0.667 + 0.3×0.5 = 0.617 → 61.7\n"
        "5) must_not / safety trap violated or ignored when required: "
        "score = min(score, 20) (or ≤10 for the main safety trap on that item).\n"
        "6) Empty answer → 0. Unstructured blob that barely addresses this question_id → "
        "apply the same formula on what is relevant only (usually low).\n"
        "7) Do NOT snap to bands like 40/60/70/80/90. Do NOT default to 100. "
        "Identical round numbers across candidates should be uncommon if coverage differs.\n"
        "8) Do NOT reward verbosity/grammar — only clinical content coverage.\n"
        "9) Synonym examples: cath≈PCI lab; STEMI≈ST-elevation MI; tox≈UDS; "
        "lithium caution≈avoid Li in CKD; admission≈inpatient/hospitalize.\n\n"
        f"Return JSON only, shape:\n{json.dumps(schema, indent=2)}"
    )
