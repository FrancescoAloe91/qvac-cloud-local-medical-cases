"""Shared candidate + judge prompts (identical for every model)."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

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


# --------------------------------------------------------------------------
# Tolerant candidate parsing
#
# Small on-device GGUF models reproduce the requested "A#:" layout only some of
# the time. They emit reasoning blocks, restate the question before answering,
# use Markdown headings, drop the colon, or answer out of order. None of that
# changes whether the clinical content is present, so the parser recovers the
# content by meaning of the heading instead of demanding one exact shape. It
# still never invents, copies, or reorders clinical text.
# --------------------------------------------------------------------------

# Reasoning/scratchpad wrappers emitted by reasoning-tuned local checkpoints.
_REASONING_TAGS = (
    "think",
    "thinking",
    "thought",
    "reasoning",
    "reflection",
    "scratchpad",
    "analysis",
)
_REASONING_BLOCK_RE = re.compile(
    r"<\s*(" + "|".join(_REASONING_TAGS) + r")\s*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_OPEN_RE = re.compile(
    r"<\s*(?:" + "|".join(_REASONING_TAGS) + r")\s*>",
    re.IGNORECASE,
)
_REASONING_CLOSE_RE = re.compile(
    r"<\s*/\s*(?:" + "|".join(_REASONING_TAGS) + r")\s*>",
    re.IGNORECASE,
)

# Section labels accepted as equivalent to the five canonical question ids.
# Only wording differs; the clinical meaning of each bucket is unchanged.
_SECTION_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "diagnosis": (
        "diagnosis",
        "diagnoses",
        "primary diagnosis",
        "most likely diagnosis",
        "likely diagnosis",
        "working diagnosis",
        "differential",
        "differential diagnosis",
        "diagnosis and differential",
        "dx",
        "impression",
        "assessment",
        "diagnosi",
        "diagnostico",
        "diagnostic",
    ),
    "tests": (
        "tests",
        "test",
        "testing",
        "next tests",
        "further tests",
        "diagnostic tests",
        "investigations",
        "investigation",
        "workup",
        "work-up",
        "work up",
        "labs",
        "laboratory",
        "esami",
        "indagini",
        "pruebas",
        "examens",
    ),
    "urgency": (
        "urgency",
        "urgency level",
        "acuity",
        "triage",
        "priority",
        "red flags",
        "urgency and red flags",
        "urgenza",
        "urgencia",
        "urgence",
    ),
    "safety": (
        "safety",
        "safety concerns",
        "safety issues",
        "safety traps",
        "pitfalls",
        "contraindications",
        "cautions",
        "precautions",
        "warnings",
        "sicurezza",
        "seguridad",
        "securite",
        "sécurité",
    ),
    "plan": (
        "plan",
        "management",
        "management plan",
        "treatment",
        "treatment plan",
        "initial management",
        "immediate management",
        "next steps",
        "therapy",
        "disposition",
        "piano",
        "gestione",
        "manejo",
        "prise en charge",
    ),
}

# Leading Markdown / list noise allowed before a heading on its own line.
_LINE_NOISE = r"[ \t]*(?:[>#*\-–—•▪●·+]+[ \t]*)*(?:\*{1,3}|_{1,3}|`{1,3})?[ \t]*"
# Optional "1." / "2)" ordinal in front of a label heading.
_ORDINAL = r"(?:\d{1,2}[.)][ \t]*)?"
# Words that may introduce a numbered answer/question marker.
_NUM_WORD = (
    r"(?:answers?|questions?|risposte?|domande?|respuestas?|preguntas?"
    r"|r[ée]ponses?|ans|a|q|r|d)"
)
# ":" / "." / ")" / "]" / free-standing dash. A dash counts only when it is
# detached from the surrounding words, so ordinary prose such as "A 5-day course
# of antibiotics" is never mistaken for an answer heading.
_SEP = r"(?::+|[.)\]](?=[ \t\n]|$)|(?<=[ \t])[-–—](?=[ \t*_`\n]|$))"
# Emphasis or quoting that may close a heading before the content starts.
_TRAILING = r"(?:[*_`]+)?[ \t]*"
# A heading may also end the line instead of carrying a separator ("## A1 [TESTS]").
_EOL = r"(?:\*{1,3}|_{1,3}|`{1,3})?[ \t]*(?=\n|$)"

_ALL_LABELS = sorted(
    {label for labels in _SECTION_SYNONYMS.values() for label in labels},
    key=len,
    reverse=True,
)
_LABEL_TO_ID = {
    label: section_id
    for section_id, labels in _SECTION_SYNONYMS.items()
    for label in labels
}
_LABEL_ALTERNATION = "|".join(re.escape(label) for label in _ALL_LABELS)

# "A1:", "### Q2)", "**Answer 3 —**", "## A1 [DIAGNOSIS]".
_NUMBERED_RE = re.compile(
    rf"(?im)^{_LINE_NOISE}{_NUM_WORD}[ \t]*\.?[ \t]*(?P<num>\d{{1,2}})"
    rf"(?:[ \t]*[\[(][ \t]*(?P<sid>[A-Za-z][A-Za-z /_-]{{1,30}}?)[ \t]*[\])])?"
    rf"[ \t]*(?:{_SEP}{_TRAILING}|{_EOL})"
)
# "... volume depletion. A1: hyperkalemia" — marker restated mid-line.
_INLINE_NUMBERED_RE = re.compile(
    rf"(?im)(?<=[.!?;:)\]])[ \t]+(?:\*{{1,3}}|_{{1,3}})?"
    rf"{_NUM_WORD}[ \t]*(?P<num>\d{{1,2}})"
    rf"(?:[ \t]*[\[(][ \t]*(?P<sid>[A-Za-z][A-Za-z /_-]{{1,30}}?)[ \t]*[\])])?"
    rf"[ \t]*:+{_TRAILING}"
)
# "## Plan", "**Safety:**", "[urgency] -", "3. Management:"
_LABEL_RE = re.compile(
    rf"(?im)^{_LINE_NOISE}{_ORDINAL}\[?[ \t]*"
    rf"(?P<label>{_LABEL_ALTERNATION})"
    rf"[ \t]*\]?[ \t]*(?:{_SEP}{_TRAILING}|{_EOL})"
)


class _Marker(NamedTuple):
    start: int
    end: int
    question_id: str
    priority: int


def strip_reasoning_blocks(text: str) -> str:
    """Drop <think>-style scratchpads so a private monologue is never scored.

    A block left unclosed by truncation is dropped only when the model also
    produced text outside it; otherwise the scratchpad is all there is and the
    caller decides what to do with it.
    """
    cleaned = _REASONING_BLOCK_RE.sub("\n", text)
    open_match = _REASONING_OPEN_RE.search(cleaned)
    if open_match and not _REASONING_CLOSE_RE.search(cleaned, open_match.end()):
        outside = cleaned[: open_match.start()]
        if outside.strip():
            cleaned = outside
    # Orphan closing tag (opening lost to a context window): keep what follows.
    orphan_close = _REASONING_CLOSE_RE.search(cleaned)
    if orphan_close and not _REASONING_OPEN_RE.search(cleaned[: orphan_close.start()]):
        cleaned = cleaned[orphan_close.end() :]
    return _REASONING_OPEN_RE.sub(" ", _REASONING_CLOSE_RE.sub(" ", cleaned))


def _resolve_label(raw_label: Optional[str]) -> Optional[str]:
    if not raw_label:
        return None
    key = re.sub(r"[\s_/-]+", " ", raw_label.strip().lower()).strip(" .:-")
    return _LABEL_TO_ID.get(key)


def _iter_markers(text: str, order: List[str]) -> Iterator[_Marker]:
    """Yield every recognizable section heading, highest-confidence first."""
    by_number = {index: qid for index, qid in enumerate(order, 1)}
    known_ids = set(order)

    def numbered(match: re.Match, priority: int) -> Optional[_Marker]:
        question_id = by_number.get(int(match.group("num")))
        if question_id is None:
            # Out-of-range numbering (e.g. a targeted retry that kept the
            # original A5 label) is usable only when it names its section.
            question_id = _resolve_label(match.group("sid"))
            if question_id is None or question_id not in known_ids:
                return None
        return _Marker(match.start(), match.end(), question_id, priority)

    for match in _NUMBERED_RE.finditer(text):
        marker = numbered(match, 0)
        if marker is not None:
            yield marker
    for match in _INLINE_NUMBERED_RE.finditer(text):
        marker = numbered(match, 1)
        if marker is not None:
            yield marker
    for match in _LABEL_RE.finditer(text):
        question_id = _resolve_label(match.group("label"))
        if question_id in known_ids:
            yield _Marker(match.start(), match.end(), question_id, 2)


def _accepted_markers(text: str, order: List[str]) -> List[_Marker]:
    markers = sorted(_iter_markers(text, order), key=lambda m: (m.start, m.priority))
    accepted: List[_Marker] = []
    for marker in markers:
        if accepted and marker.start < accepted[-1].end:
            continue  # nested inside a stronger heading, e.g. "Q1 [diagnosis]:"
        accepted.append(marker)
    return accepted


def parse_candidate_answers(case: Case, raw: str) -> Dict[str, str]:
    """Recover each clinical section from realistic formatting variation.

    Tolerant about presentation (heading wording, Markdown, ordering, casing,
    punctuation, reasoning blocks); strict about substance. A section stays
    absent when the model genuinely produced no attributable content for it.
    """
    normalized = unicodedata.normalize("NFKC", raw or "")
    text = strip_reasoning_blocks(normalized)
    if not text.strip():
        # Everything was inside an unterminated scratchpad; the monologue is
        # the only output there is, so keep it rather than dropping the answer.
        text = _REASONING_OPEN_RE.sub(" ", _REASONING_CLOSE_RE.sub(" ", normalized))

    order = [q.id for q in case.questions]
    markers = _accepted_markers(text, order)

    # A question restated then answered ("Q1 [diagnosis]: … A1: …") produces two
    # markers for the same section; both fragments belong to that section.
    chunks: Dict[str, List[str]] = {}
    for index, marker in enumerate(markers):
        end = markers[index + 1].start if index + 1 < len(markers) else len(text)
        body = text[marker.end : end].strip()
        if body:
            chunks.setdefault(marker.question_id, []).append(body)

    answers = {qid: "\n\n".join(parts) for qid, parts in chunks.items() if parts}

    # Fewer than two sections recovered → unstructured prose. Hand the judge the
    # whole body per question, flagged, so real content is graded on relevance
    # instead of being discarded as a formatting failure.
    body = text.strip()
    if body and len(answers) < 2 and len(body) > 80:
        for qid in order:
            answers[qid] = (
                "[UNSTRUCTURED FULL RESPONSE — score only content relevant to "
                f"{qid}; apply linear formula on relevant parts only]\n" + body
            )
    return answers


def missing_section_ids(case: Case, answers: Dict[str, str]) -> List[str]:
    """Question ids with no recovered candidate content."""
    return [q.id for q in case.questions if not ((answers or {}).get(q.id) or "").strip()]


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
        "Keep JSON compact: short verbatim quotes (≤40 words), omit long rationales, "
        "no chain-of-thought outside JSON. Prefer finishing all sections over verbose "
        "prose. Do not guess the model/vendor. Return ONLY valid JSON."
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
        "Keep quotes short (≤40 words). Omit optional rationales unless essential. "
        "Quality is independent but evidence-grounded; use the full 0-1 continuum: "
        "0 only unusable/dangerous, 0.25 major deficiencies, 0.5 mixed but useful, "
        "0.75 strong with limited omissions, 1 exceptional. Do not reduce quality "
        "merely because a reasonable addition is absent from the reference.\n\n"
        f"Evaluate {blind_id} only.\n\n"
        f"ITEMS:\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        f"Return JSON only, shape:\n{json.dumps(schema, indent=2)}"
    )
