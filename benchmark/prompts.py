"""Shared candidate + judge prompts (identical for every model)."""

from __future__ import annotations

import json
from typing import Dict

from benchmark.schema import Case

# Soft budget in the prompt + hard API/sidecar cap (same for every candidate).
CANDIDATE_MAX_OUTPUT_TOKENS = 3000


def use_gold_ground_truth(gold_reference: str = "") -> bool:
    """If the user pasted a confirmed diagnosis/gold, score 0–100 against that."""
    return bool((gold_reference or "").strip())


def candidate_system() -> str:
    return (
        "You are an expert physician taking a structured clinical benchmark. "
        "Answer each numbered question clearly and specifically. "
        "Do not invent patient identifiers. Be concise but complete. "
        f"Hard length budget: at most {CANDIDATE_MAX_OUTPUT_TOKENS} tokens for the entire "
        "reply (all A# answers together). Stopping earlier is fine; do not pad. "
        "Cover every question within that budget — put the most important clinical "
        "points first so nothing critical is left for the end."
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
        "For urgency use one of: critical, high, moderate, low. "
        f"Stay within {CANDIDATE_MAX_OUTPUT_TOKENS} tokens total for the whole answer."
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

    n_filled = sum(1 for q in case.questions if (answers.get(q.id) or "").strip())
    # Fewer than 2 sections parsed → treat as unstructured prose; give judge the
    # full text for every question so plan/safety are not silently empty.
    if text.strip() and n_filled < 2 and len(text.strip()) > 80:
        body = text.strip()
        for q in case.questions:
            answers[q.id] = (
                "[UNSTRUCTURED FULL RESPONSE — score only content relevant to "
                f"{q.id}; apply linear formula on relevant parts only]\n"
                + body
            )
    return answers


def judge_system() -> str:
    return (
        "Act as a senior clinician grading a competitive benchmark. "
        "Judge CLINICAL MEANING, not wording: synonyms, paraphrases, and near-equivalent "
        "formulations (same diagnosis / same next-step intent) count as matches. "
        "Do NOT atomize the gold into keyword checklists or demand exact acronyms. "
        "Do not guess vendors. Return ONLY valid JSON."
    )


def judge_user(
    case: Case,
    blind_id: str,
    answers: Dict[str, str],
    gold_reference: str = "",
) -> str:
    gold_mode = use_gold_ground_truth(gold_reference)
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

    if gold_mode:
        schema = {
            "blind_id": blind_id,
            "question_scores": [
                {
                    "question_id": "string",
                    "score": "0-100 continuous (host recomputes; optional)",
                    "rationale": "why this alignment/quality (1-2 sentences)",
                    "evidence": "short quote showing the clinical idea",
                    "errors": ["true clinical misses vs gold — not missing keywords"],
                    "alignment": "0-1 float: semantic closeness to gold for THIS section",
                    "quality": "0-1 float: clinical judgment quality for THIS section",
                }
            ],
        }
        mode_block = (
            "SCORING MODE = GOLD_WINS (confirmed diagnosis / reference is present).\n"
            "The GOLD REFERENCE is the SOLE ground truth. Ignore empty teaching rubrics.\n"
            "Do NOT split the gold into bullet checklists or count acronyms/phrases.\n"
            "For EACH section, judge holistically:\n"
            "  1) Diagnosis / framing — same clinical thesis? (near-equivalent OK)\n"
            "  2) Workup / tests — same investigative INTENT (bone+soft tissue, etc.)?\n"
            "  3) Urgency — same acuity band and red-flag thinking?\n"
            "  4) Safety / advice — same traps avoided and cautions given?\n"
            "  5) Plan / next steps — same stepwise strategy and priorities?\n"
            "Return alignment ∈ [0,1] = how close the candidate is in MEANING to the gold\n"
            "for that section (0.75–0.92 = strongly aligned; 1.0 almost never).\n"
            "Partial credit when the idea is right but incomplete; do not zero a section\n"
            "just because a synonym or imaging modality synonym differs.\n"
            "quality ∈ [0,1] = soundness of clinical reasoning on that section\n"
            "(prioritization, coherence, case-specificity, absence of dangerous advice).\n"
            "Host formula: score = 100×(0.50·alignment + 0.30·quality + 0.20·spec), CAP 96.5.\n"
            "You must return alignment and quality (host ignores your score field).\n\n"
            "GOLD REFERENCE:\n"
            f"{gold_reference.strip()}\n\n"
        )
        hard = (
            "HARD RULES:\n"
            "- Dangerous contradiction of gold MUST NOT / safety trap → errors[] "
            "(host may cap safety/diagnosis ≤20).\n"
            "- Empty answer → alignment=0, quality=0.\n"
            "- Reward smart near-matches (facet arthropathy≈facet hypertrophy; "
            "CT+MRI intent≈bone+soft-tissue imaging) even if wording differs.\n"
            "- Penalize wrong primary frame (e.g. fibromyalgia-as-primary, hip-only story).\n"
        )
    else:
        schema = {
            "blind_id": blind_id,
            "question_scores": [
                {
                    "question_id": "string",
                    "score": "0-100 continuous (host recomputes; optional)",
                    "rationale": "m_hit/m_tot, a_hit/a_tot, quality — by clinical meaning",
                    "evidence": "short quote",
                    "errors": ["true clinical misses — not missing exact words"],
                    "m_hit": "int",
                    "m_total": "int",
                    "a_hit": "int",
                    "a_total": "int",
                    "quality": "0-1 float",
                }
            ],
        }
        mode_block = (
            "SCORING MODE = RUBRIC_WINS (no confirmed diagnosis pasted).\n"
            "Ground truth = teaching rubric must_include[] / acceptable[] / must_not[].\n"
            "m_total = len(must_include), a_total = len(acceptable) for that question.\n"
            "Score coverage by CLINICAL MEANING (synonyms and near-equivalents OK).\n"
            "Do not require exact phrasing.\n\n"
        )
        hard = (
            "LINEAR FORMULA (host recomputes; 100% is NOT normal):\n"
            "m = m_hit / max(m_total, 1); a = a_hit / max(a_total, 1)\n"
            "quality ∈ [0,1] = clinical judgment (not writing style).\n"
            "Strong answer: quality 0.55–0.85; >0.90 only if exceptional.\n"
            "Host: score = 100×(0.30·m + 0.20·a + 0.40·quality + 0.10·spec), CAP 96.5.\n"
            "You must return m_hit,m_total,a_hit,a_total,quality.\n"
            "HARD RULES:\n"
            "- must_not / ignored safety trap → errors[] (host caps ≤20).\n"
            "- Empty answer → m_hit=0,a_hit=0,quality=0.\n"
            "- Synonyms OK (cath≈PCI; facet arthropathy≈facet syndrome).\n"
        )

    return (
        f"CASE STEM:\n{case.stem}\n\n"
        f"{mode_block}"
        f"Evaluate {blind_id} only.\n\n"
        f"ITEMS:\n{json.dumps(rubric_payload, ensure_ascii=False, indent=2)}\n\n"
        f"{hard}\n"
        f"Return JSON only, shape:\n{json.dumps(schema, indent=2)}"
    )
