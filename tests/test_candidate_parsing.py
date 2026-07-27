"""Tolerant recovery of clinical sections from realistic candidate formatting.

Small on-device GGUF models rarely reproduce the requested "A#:" layout exactly.
These cases pin down that presentation variation is recovered while genuinely
absent clinical content still terminates as N/A.
"""

from __future__ import annotations

from benchmark.cases_loader import load_case
from benchmark.gold import confirmed_gold, gold_json
from benchmark.judge import PipelinedJudge
from benchmark.prompts import (
    missing_section_ids,
    parse_candidate_answers,
    strip_reasoning_blocks,
)
from benchmark.runner import _collect_candidate_once, is_retryable_local_error
from benchmark.schema import CandidateAnswer, GoldClaim, GoldSection, ModelCallMeta

SECTIONS = ("diagnosis", "tests", "urgency", "safety", "plan")


def _contract() -> str:
    sections = {
        section: GoldSection(
            summary=f"{section} reference",
            claims=[
                GoldClaim(
                    id=f"{section}-1",
                    text=f"{section} reference",
                    source_quote=f"{section} reference",
                )
            ],
        )
        for section in SECTIONS
    }
    return gold_json(
        confirmed_gold(
            raw_text=" ".join(s.summary for s in sections.values()),
            sections=sections,
            extraction_model="test",
        )
    )


def _parse(raw: str):
    return parse_candidate_answers(load_case("caseC"), raw)


def test_markdown_headings_with_bracketed_ids_are_recovered():
    raw = """# Structured Clinical Analysis

## A1 [DIAGNOSIS]

Severe acute kidney injury with critical hyperkalemia.

## A2 [TESTS]

Renal ultrasound and serum CK.

## A3 [URGENCY]

Critical.

## A4 [SAFETY]

Avoid nephrotoxins and hold the ACE inhibitor.

## A5 [PLAN]

IV calcium gluconate, then insulin-glucose, then dialysis.
"""
    parsed = _parse(raw)

    assert missing_section_ids(load_case("caseC"), parsed) == []
    assert parsed["diagnosis"].startswith("Severe acute kidney injury")
    assert parsed["urgency"] == "Critical."
    assert "dialysis" in parsed["plan"]


def test_reasoning_block_is_not_scored_as_an_answer():
    raw = """<think>
Urgency: critical because hyperkalemia can cause cardiac arrest.
Safety: avoid NSAIDs, stop the ACE inhibitor.
</think>

Q1 [diagnosis]: Hyperkalemia with acute kidney injury. A1: AKI with hyperkalemia.

Q2 [tests]: ABG, repeat electrolytes, renal ultrasound.

Q3 [urgency]: Critical, K+ above 6.5 with ECG changes.

Q4 [safety]: Avoid NSAIDs and potassium-sparing agents.

Q5 [plan]: Calcium gluconate, insulin-glucose, dialysis if refractory.
"""
    parsed = _parse(raw)

    assert missing_section_ids(load_case("caseC"), parsed) == []
    # The private monologue must not leak into the graded text.
    assert "cardiac arrest" not in parsed["urgency"]
    assert "Critical" in parsed["urgency"]
    # A restated question plus its short answer both belong to that section.
    assert "Hyperkalemia with acute kidney injury" in parsed["diagnosis"]
    assert "AKI with hyperkalemia" in parsed["diagnosis"]
    # A section must not absorb the next question's content.
    assert "ABG" not in parsed["diagnosis"]


def test_inline_answer_marker_after_prose_is_recognized():
    raw = (
        "Q1 [diagnosis]: Acute tubular necrosis. A1: Ischemic ATN.\n"
        "Q2 [tests]: Renal ultrasound. A2: Ultrasound plus CK.\n"
        "Q3 [urgency]: Critical. A3: Critical.\n"
        "Q4 [safety]: No NSAIDs. A4: Avoid nephrotoxins.\n"
        "Q5 [plan]: Calcium then dialysis. A5: Calcium, insulin, dialysis.\n"
    )
    parsed = _parse(raw)

    assert missing_section_ids(load_case("caseC"), parsed) == []
    assert "Ischemic ATN" in parsed["diagnosis"]
    assert "Ultrasound plus CK" in parsed["tests"]


def test_synonym_labels_out_of_order_and_lowercase_are_recovered():
    raw = """**management**
Fluids, calcium, dialysis.

impression: acute kidney injury, KDIGO stage 3

3. investigations - renal ultrasound, CK, ABG

contraindications: avoid NSAIDs

TRIAGE: critical
"""
    parsed = _parse(raw)

    assert missing_section_ids(load_case("caseC"), parsed) == []
    assert parsed["plan"] == "Fluids, calcium, dialysis."
    assert "KDIGO stage 3" in parsed["diagnosis"]
    assert "renal ultrasound" in parsed["tests"]
    assert parsed["urgency"] == "critical"


def test_truncated_output_keeps_every_section_it_did_produce():
    raw = (
        "A1: Acute kidney injury.\n"
        "A2: Ultrasound and CK.\n"
        "A3: Critical.\n"
        "A4: Avoid nephrotoxins.\n"
        "A5: Calcium gluconate, insulin-glucose, then urgent dialy"
    )
    parsed = _parse(raw)

    assert missing_section_ids(load_case("caseC"), parsed) == []
    assert parsed["plan"].endswith("urgent dialy")


def test_truncated_but_complete_answer_is_judged_despite_length_stop(monkeypatch):
    case = load_case("caseC")
    candidate = CandidateAnswer(
        candidate_key="qvac",
        label="MedPsy",
        blind_id="Candidate 1",
        answers={q.id: f"{q.id} answer" for q in case.questions},
        raw_response="complete but cut off",
        meta=ModelCallMeta(
            model="medpsy-4b", provider="qvac", finish_reason="length"
        ),
    )
    pipe = PipelinedJudge(
        case,
        "judge",
        gold_reference=_contract(),
        expected_total=1,
        run_scope="test-truncation-tolerant",
    )
    judged = []

    def fake_judge(case_arg, cand, *args, **kwargs):
        judged.append(cand.candidate_key)
        raise RuntimeError("stop here — reaching the judge is what matters")

    monkeypatch.setattr("benchmark.judge.judge_candidate", fake_judge)
    try:
        pipe._one_safe(candidate)
    except RuntimeError:
        pass
    finally:
        pipe.close(cancel_pending=True)

    assert judged == ["qvac"]


def test_genuinely_missing_section_stays_na():
    case = load_case("caseC")
    candidate = CandidateAnswer(
        candidate_key="qvac",
        label="MedPsy",
        blind_id="Candidate 1",
        answers=_parse("A1: Acute kidney injury.\nA2: Ultrasound and CK.\n"),
        raw_response="A1: Acute kidney injury.\nA2: Ultrasound and CK.\n",
        meta=ModelCallMeta(model="medpsy-4b", provider="qvac"),
    )
    pipe = PipelinedJudge(
        case,
        "judge",
        gold_reference=_contract(),
        expected_total=1,
        run_scope="test-missing-stays-na",
    )
    result = pipe._one_safe(candidate)
    pipe.close(cancel_pending=True)

    assert result.status == "candidate_partial"
    assert "urgency" in result.failure_reason
    assert "plan" in result.failure_reason


def test_degenerate_repetition_is_not_credited_with_absent_sections():
    block = (
        "Q1 [diagnosis]: Lumbar spinal stenosis.\n\n"
        "Q2 [tests]: MRI lumbar spine.\n\n"
    )
    parsed = _parse(block * 8)

    assert missing_section_ids(load_case("caseC"), parsed) == [
        "urgency",
        "safety",
        "plan",
    ]


def test_prose_dash_is_not_mistaken_for_an_answer_heading():
    raw = "A1: Cellulitis.\nA 5-day course of antibiotics is reasonable here.\n"
    parsed = _parse(raw)

    assert "5-day course" in parsed["diagnosis"]
    assert missing_section_ids(load_case("caseC"), parsed) == [
        "tests",
        "urgency",
        "safety",
        "plan",
    ]


def test_unterminated_reasoning_block_is_kept_as_the_only_output():
    raw = "<think>\nThe likely diagnosis is acute kidney injury with hyperkalemia."
    assert "acute kidney injury" in strip_reasoning_blocks(raw)

    closed = "<think>\nscratch notes\n</think>\nA1: acute kidney injury"
    assert "scratch notes" not in strip_reasoning_blocks(closed)


def test_local_load_failure_terminalizes_as_a_collect_failed_row(monkeypatch):
    case = load_case("caseC")
    monkeypatch.setattr(
        "benchmark.qvac_bridge.load_model",
        lambda *args, **kwargs: {"ok": False, "error": "Failed to load medpsy.gguf"},
    )
    candidate = _collect_candidate_once(
        case,
        {
            "key": "qvac",
            "provider": "qvac",
            "label": "MedPsy",
            "gguf_path": "/models/medpsy.gguf",
        },
        "Candidate 1",
    )

    assert candidate.candidate_key == "qvac"
    assert candidate.meta.error == "Failed to load medpsy.gguf"
    assert is_retryable_local_error(candidate.meta.error)

    pipe = PipelinedJudge(
        case,
        "judge",
        gold_reference=_contract(),
        expected_total=1,
        run_scope="test-load-failure",
    )
    result = pipe._one_safe(candidate)
    pipe.close(cancel_pending=True)

    assert result.status == "collect_failed"
    assert result.candidate_key == "qvac"
