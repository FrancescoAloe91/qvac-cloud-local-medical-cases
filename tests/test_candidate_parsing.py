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


def test_long_unstructured_prose_does_not_fill_all_sections():
    raw = (
        "The patient likely has acute kidney injury with hyperkalemia. "
        "I would order renal ultrasound and repeat electrolytes. "
        "This is critical. Avoid NSAIDs and hold ACE inhibitors. "
        "Give calcium gluconate then insulin-glucose and consider dialysis. "
    ) * 3
    parsed = _parse(raw)
    assert missing_section_ids(load_case("caseC"), parsed) == [
        "diagnosis",
        "tests",
        "urgency",
        "safety",
        "plan",
    ]


def test_format_repair_messages_ask_for_a_markers():
    from benchmark.prompts import format_repair_messages

    case = load_case("caseC")
    msgs = format_repair_messages(case, "Loose prose about AKI and dialysis.")
    assert len(msgs) == 2
    assert "A1" in msgs[1]["content"]
    assert "Loose prose about AKI" in msgs[1]["content"]
    assert "do not invent" in msgs[1]["content"].casefold()

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


def test_long_diagnosis_only_prose_does_not_fill_other_sections():
    """No unstructured photocopy: one attributed section must not invent the rest."""
    prose = (
        "A1: Acute kidney injury with critical hyperkalemia secondary to ACE "
        "inhibitor use in the setting of volume depletion. The presentation is "
        "consistent with ischemic ATN versus acute interstitial nephritis; "
        "however the ECG changes and potassium above 6.5 dominate triage. "
        "This remains a single diagnosis block without separate markers for "
        "tests, urgency, safety, or plan despite discussing related ideas."
    )
    parsed = _parse(prose)
    assert "diagnosis" in parsed
    assert set(parsed) == {"diagnosis"}
    assert missing_section_ids(load_case("caseC"), parsed) == [
        "tests",
        "urgency",
        "safety",
        "plan",
    ]


def test_unstructured_wall_of_text_does_not_become_five_sections():
    wall = (
        "The patient most likely has acute coronary syndrome. Order troponin "
        "and ECG. This is urgent. Avoid NSAIDs. Give aspirin and heparin. "
        "Additional discussion of differentials and disposition without any "
        "section markers or numbered answer headings at all."
    )
    parsed = _parse(wall)
    assert parsed == {}
    assert missing_section_ids(load_case("caseC"), parsed) == list(SECTIONS)


def test_missing_after_parse_triggers_missing_section_ids():
    parsed = _parse("A1: Migraine.\nA3: Moderate.\n")
    assert set(parsed) == {"diagnosis", "urgency"}
    assert missing_section_ids(load_case("caseC"), parsed) == [
        "tests",
        "safety",
        "plan",
    ]


def test_prompt_footer_echo_is_not_a_plan_answer():
    from benchmark.prompts import is_unsubstantive_section

    case = load_case("caseC")
    assert is_unsubstantive_section(
        case,
        "plan",
        "Fill every A# answer. Do not skip questions. Stay within 3000 tokens.",
    )
    assert is_unsubstantive_section(
        case,
        "diagnosis",
        "What is the most likely primary diagnosis? Rank the top differential.\n\n"
        "Q2 [tests]: Which tests should be ordered next?",
    )


def test_prompt_template_echo_detected():
    from benchmark.prompts import is_prompt_template_echo

    case = load_case("caseC")
    raw = (
        "CLINICAL CASE:\n"
        + case.stem[:80]
        + "\n\nAnswer ALL of the following questions. Use this exact format:\n"
        "Q1 [diagnosis]: …\nA1:\n"
    )
    assert is_prompt_template_echo(raw, case) is True
    assert is_prompt_template_echo("A1: AKI\nA2: labs\n", case) is False


def test_one_line_q_a_markers_split_without_sentence_punctuation():
    raw = (
        "Q1 [diagnosis]: What is the most likely primary diagnosis? Rank the top "
        "differential. A1: Acute tubular necrosis (ATN) "
        "Q2 [tests]: Which tests should be ordered next? A2: CBC, BMP, ECG "
        "Q3 [urgency]: Urgency level (critical / high / moderate / low) and red flags. "
        "A3: Critical "
        "Q4 [safety]: Critical contraindications or safety traps. A4: None "
        "Q5 [plan]: Outline the initial management plan. "
        "A5: Calcium gluconate, then insulin-glucose, then dialysis."
    )
    parsed = _parse(raw)
    assert missing_section_ids(load_case("caseC"), parsed) == []
    assert "ATN" in parsed["diagnosis"]
    assert "CBC" in parsed["tests"]
    assert "Critical" in parsed["urgency"]
    assert "None" in parsed["safety"]
    assert "dialysis" in parsed["plan"]


def test_llama3_chat_format_pre_renders_assistant_header():
    from benchmark.prompts import local_chat_messages, render_llama3_instruct

    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "A1: answer"},
    ]
    packed = local_chat_messages(messages, {"chat_format": "llama3"})
    assert len(packed) == 1
    assert packed[0]["role"] == "user"
    body = packed[0]["content"]
    assert body.startswith("<|begin_of_text|>")
    assert "<|start_header_id|>system<|end_header_id|>" in body
    assert body.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    # Active medical peers keep role messages untouched (GGUF/SDK template).
    plain = local_chat_messages(messages, {"key": "local_med42"})
    assert plain == messages
    assert "assistant" in render_llama3_instruct(messages)


def test_medical_peers_rely_on_embedded_chat_template():
    from benchmark.qvac_variants import MEDICAL_PEER_SPECS

    by_key = {s["key"]: s for s in MEDICAL_PEER_SPECS}
    assert set(by_key) == {
        "local_medgemma",
        "local_med42",
        "local_ultramedical",
    }
    for key, spec in by_key.items():
        assert "chat_format" not in spec
    assert by_key["local_medgemma"]["gguf"] == "medgemma-1.5-4b-it-Q4_K_M.gguf"
    assert by_key["local_ultramedical"]["gguf"].startswith("Llama-3-8B-UltraMedical")


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
