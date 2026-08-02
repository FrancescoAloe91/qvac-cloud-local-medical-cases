"""Candidate prompts for Comprehension track (no mandatory A1–A5 markers)."""

from __future__ import annotations

from typing import Dict, List

from benchmark.prompts import clinical_answer_text, parse_candidate_answers
from benchmark.schema import Case


def beta_candidate_system() -> str:
    return (
        "You are a careful clinical assistant. Given a patient presentation, "
        "write a single continuous clinical note covering: most likely diagnosis "
        "and key differentials; next tests; urgency and red flags; critical safety "
        "traps; and the initial management plan. Use clear prose. Do not use "
        "mandatory A1/A2/A3/A4/A5 labels. Do not invent labs or findings not "
        "supported by the presentation. If unsure, say what is uncertain."
    )


def beta_candidate_user(*, stem: str) -> str:
    return (
        "Patient presentation:\n\n"
        f"{stem.strip()}\n\n"
        "Write one undivided clinical response covering diagnosis, tests, "
        "urgency, safety traps, and initial plan."
    )


def beta_candidate_messages(*, stem: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": beta_candidate_system()},
        {"role": "user", "content": beta_candidate_user(stem=stem)},
    ]


def parse_beta_candidate_answers(case: Case, raw: str) -> Dict[str, str]:
    """Parse free-form Beta replies for the five graded dimensions.

    Prefer soft section markers when present; otherwise attribute the full
    undivided *clinical* note to every required section so the judge can score
    coverage against each gold chapter without forcing A1–A5 format repair.

    Reasoning / ``<think>`` scratchpads are stripped before photocopy so the
    judge never receives five think-contaminated dumps of the same raw stream.
    """
    clinical = clinical_answer_text(raw)
    if not clinical:
        return {}
    marked = parse_candidate_answers(case, clinical)
    order = [q.id for q in case.questions]
    if marked and all((marked.get(qid) or "").strip() for qid in order):
        return marked
    # Undivided prose: same cleaned note visible to each dimension scorer.
    return {qid: clinical for qid in order}
