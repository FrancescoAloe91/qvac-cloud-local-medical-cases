"""Shared candidate + judge prompts (identical for every model)."""

from __future__ import annotations

import json
from typing import Dict

from benchmark.gold import load_confirmed_gold
from benchmark.schema import Case

# Soft budget in the prompt + hard API/sidecar cap (same for every candidate).
CANDIDATE_MAX_OUTPUT_TOKENS = 3000


def use_gold_ground_truth(gold_reference: str = "") -> bool:
    """Compatibility predicate; all active runs require confirmed gold."""
    try:
        load_confirmed_gold(gold_reference)
        return True
    except ValueError:
        return False


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
        "Act as a senior clinician performing a blind, evidence-linked evaluation. "
        "Use the supplied case and frozen reference as the scoring anchor, but do not "
        "treat the reference as an exhaustive whitelist. Score reference coverage "
        "continuously by clinical meaning: 0 absent, 0.25 weak/implicit, 0.5 partial, "
        "0.75 substantial, 1 complete. Synonyms and faithful paraphrases count. "
        "Classify additional content as helpful, neutral, unsupported, contradictory, "
        "or dangerous. Clinically appropriate case-relevant additions are helpful or "
        "neutral, never unsupported merely because they are absent from the reference. "
        "Every nonzero coverage decision and every added-content classification must "
        "cite exact candidate text. "
        "Do not guess the model/vendor. Return ONLY valid JSON."
    )


def judge_user(
    case: Case,
    blind_id: str,
    answers: Dict[str, str],
    gold_reference: str = "",
) -> str:
    gold = load_confirmed_gold(gold_reference)
    items = []
    for q in case.questions:
        section = gold.sections[q.id]
        items.append(
            {
                "question_id": q.id,
                "kind": q.kind,
                "weight": q.weight,
                "question": q.text,
                "reference_claims": [
                    {"id": claim.id, "source_quote": claim.source_quote}
                    for claim in section.claims
                ],
                "candidate_answer": answers.get(q.id, ""),
            }
        )
    schema = {
        "blind_id": blind_id,
        "question_scores": [
            {
                "question_id": "one required section id",
                "claim_assessments": [
                    {
                        "reference_claim_id": "one reference claim id",
                        "coverage": "number from 0 to 1",
                        "candidate_quotes": [
                            "exact candidate quote; empty only when coverage is 0"
                        ],
                        "rationale": "optional one short sentence",
                    }
                ],
                "additional_claims": [
                    {
                        "candidate_quote": "exact candidate quote",
                        "classification": (
                            "helpful | neutral | unsupported | contradictory | dangerous"
                        ),
                        "severity": "number from 0 to 1",
                        "rationale": "optional one short sentence",
                    }
                ],
                "quality": "0-1 clinical coherence, prioritization, and usefulness",
                "rationale": "optional one short sentence",
                "errors": ["structured clinical concerns, never transport/schema errors"],
            }
        ],
    }
    return (
        f"CASE STEM:\n{case.stem}\n\n"
        "SCORING CONTRACT = FROZEN REFERENCE ANCHOR + GRADED CLINICAL REVIEW.\n"
        "Assess every reference claim exactly once with continuous coverage. "
        "Do not collapse partial coverage to binary matched/missed. "
        "Do not penalize useful or reasonable additions merely for being absent from "
        "the reference. Unsupported means speculative and potentially misleading; "
        "contradictory means incompatible with the case/reference; dangerous means "
        "likely harmful advice. Optional detail can be neutral. "
        "Do not score empty answers: the host treats them as technical N/A. "
        "Every candidate_quote must be verbatim text present in that section's answer. "
        "Quality is independent but evidence-grounded; use the full 0-1 continuum: "
        "0 only unusable/dangerous, 0.25 major deficiencies, 0.5 mixed but useful, "
        "0.75 strong with limited omissions, 1 exceptional. Do not reduce quality "
        "merely because a reasonable addition is absent from the reference.\n\n"
        f"Evaluate {blind_id} only.\n\n"
        f"ITEMS:\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        f"Return JSON only, shape:\n{json.dumps(schema, indent=2)}"
    )
