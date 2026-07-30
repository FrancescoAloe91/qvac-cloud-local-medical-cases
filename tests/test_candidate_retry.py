from benchmark.cases_loader import load_case
from benchmark.prompts import (
    fold_system_into_user,
    is_unsubstantive_section,
    missing_section_ids,
    parse_candidate_answers,
    prefers_user_only_chat,
)
from benchmark.runner import _collect_candidate
from benchmark.schema import CandidateAnswer, ModelCallMeta


def _candidate(
    *, error=None, finish_reason="", answers=None, provider="openrouter", raw=None
):
    ans = answers if answers is not None else {}
    if raw is None:
        raw_response = "response" if ans else ""
    else:
        raw_response = raw
    return CandidateAnswer(
        candidate_key="candidate",
        label="Candidate",
        blind_id="Candidate 1",
        answers=ans,
        raw_response=raw_response,
        meta=ModelCallMeta(
            model="model",
            provider=provider,
            error=error,
            finish_reason=finish_reason,
            cost_usd=0.01,
        ),
    )


def test_candidate_transport_failure_retries_once(monkeypatch):
    case = load_case("caseC")
    responses = [
        _candidate(error="HTTP 503"),
        _candidate(answers={q.id: "answer" for q in case.questions}),
    ]
    calls = []

    def fake_once(*args, **kwargs):
        calls.append(1)
        return responses.pop(0)

    monkeypatch.setattr("benchmark.runner._collect_candidate_once", fake_once)
    result = _collect_candidate(
        case,
        {"key": "candidate", "provider": "openrouter"},
        "Candidate 1",
    )

    assert len(calls) == 2
    assert result.meta.retry_count == 1
    assert result.meta.cost_usd == 0.02


def test_candidate_missing_sections_trigger_one_targeted_regeneration(monkeypatch):
    case = load_case("caseC")
    responses = [
        _candidate(answers={"diagnosis": "only one section"}),
        _candidate(answers={q.id: "recovered" for q in case.questions[1:]}),
    ]
    requested_sections = []

    def fake_once(case_arg, *args, **kwargs):
        requested_sections.append([q.id for q in case_arg.questions])
        return responses.pop(0)

    monkeypatch.setattr("benchmark.runner._collect_candidate_once", fake_once)
    result = _collect_candidate(
        case,
        {"key": "candidate", "provider": "openrouter"},
        "Candidate 1",
    )

    assert requested_sections[0] == [q.id for q in case.questions]
    assert requested_sections[1] == [q.id for q in case.questions[1:]]
    assert result.answers["diagnosis"] == "only one section"
    assert result.meta.retry_count == 1


def test_local_missing_sections_one_multi_gap_targeted_call(monkeypatch):
    """Local recovery is ≤1 generate: one multi-gap call, never N sequential."""
    case = load_case("caseC")
    responses = [
        _candidate(
            answers={"diagnosis": "only one section"},
            provider="qvac",
        ),
        _candidate(
            answers={
                "tests": "labs",
                "urgency": "high",
                "safety": "hold ACE",
                "plan": "fluids",
            },
            provider="qvac",
        ),
    ]
    requested_sections = []

    def fake_once(case_arg, *args, **kwargs):
        requested_sections.append([q.id for q in case_arg.questions])
        return responses.pop(0)

    monkeypatch.setattr("benchmark.runner._collect_candidate_once", fake_once)
    result = _collect_candidate(
        case,
        {"key": "local_biomistral", "provider": "qvac", "model": "biomistral-7b-q4"},
        "Candidate 1",
    )

    assert requested_sections[0] == [q.id for q in case.questions]
    assert requested_sections[1] == ["tests", "urgency", "safety", "plan"]
    assert len(requested_sections) == 2
    assert missing_section_ids(case, result.answers) == []
    assert result.answers["plan"] == "fluids"
    assert result.meta.retry_count == 1


def test_local_format_repair_does_not_stack_targeted(monkeypatch):
    """Local honesty: format-repair XOR targeted — never both."""
    case = load_case("caseC")
    # Long unlabeled prose → format-repair path; repair still leaves gaps.
    prose = "Clinical impression of AKI with hyperkalemia and volume depletion. " * 3
    first = _candidate(answers={}, provider="qvac", raw=prose)
    repaired = _candidate(answers={"diagnosis": "AKI"}, provider="qvac")
    responses = [first, repaired]
    calls = []

    def fake_once(case_arg, *args, **kwargs):
        calls.append(
            {
                "ids": [q.id for q in case_arg.questions],
                "has_messages": bool(kwargs.get("messages")),
            }
        )
        return responses.pop(0)

    monkeypatch.setattr("benchmark.runner._collect_candidate_once", fake_once)
    # Bypass timed wrapper so the fake is invoked directly under local policy.
    def fake_recover(
        case_arg,
        cand_cfg,
        blind_id,
        on_event=None,
        benchmark_track="controlled",
        api_key=None,
        *,
        messages=None,
        timeout=None,
        template=None,
    ):
        return fake_once(case_arg, messages=messages)

    monkeypatch.setattr("benchmark.runner._recover_collect_once", fake_recover)
    result = _collect_candidate(
        case,
        {"key": "local_biomistral", "provider": "qvac", "model": "biomistral-7b-q4"},
        "Candidate 1",
    )

    assert len(calls) == 2  # first collect + one format-repair only
    assert calls[1]["has_messages"] is True
    assert result.answers.get("diagnosis") == "AKI"
    assert missing_section_ids(case, result.answers) == [
        q.id for q in case.questions[1:]
    ]


def test_local_recovery_timeout_leaves_gaps_as_na(monkeypatch):
    case = load_case("caseC")
    first = _candidate(
        answers={"diagnosis": "only one section"},
        provider="qvac",
    )
    calls = {"n": 0}

    def slow_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return first
        import time

        time.sleep(5)
        return _candidate(answers={"tests": "late"}, provider="qvac")

    monkeypatch.setattr("benchmark.runner._collect_candidate_once", slow_once)
    monkeypatch.setattr("benchmark.runner.LOCAL_RECOVERY_TIMEOUT_S", 0.05)
    result = _collect_candidate(
        case,
        {"key": "local_biomistral", "provider": "qvac", "model": "biomistral-7b-q4"},
        "Candidate 1",
    )

    assert result.answers.get("diagnosis") == "only one section"
    assert "tests" not in (result.answers or {})
    assert missing_section_ids(case, result.answers) == [
        q.id for q in case.questions[1:]
    ]


def test_candidate_retry_budget_stops_after_one_attempt(monkeypatch):
    case = load_case("caseC")
    calls = []

    def fake_once(*args, **kwargs):
        calls.append(1)
        return _candidate(answers={"diagnosis": "only one section"})

    monkeypatch.setattr("benchmark.runner._collect_candidate_once", fake_once)
    result = _collect_candidate(
        case,
        {"key": "candidate", "provider": "openrouter"},
        "Candidate 1",
    )

    assert len(calls) == 2
    assert result.meta.retry_count == 1
    assert missing_section_ids(case, result.answers) == [
        q.id for q in case.questions[1:]
    ]


def test_local_sidecar_failure_is_retried_once(monkeypatch):
    case = load_case("caseC")
    responses = [
        _candidate(error="Failed to load /models/medpsy-4b.gguf", provider="qvac"),
        _candidate(
            answers={q.id: "answer" for q in case.questions},
            provider="qvac",
        ),
    ]
    calls = []

    def fake_once(*args, **kwargs):
        calls.append(1)
        return responses.pop(0)

    monkeypatch.setattr("benchmark.runner._collect_candidate_once", fake_once)
    result = _collect_candidate(
        case,
        {"key": "qvac", "provider": "qvac"},
        "Candidate 1",
    )

    assert len(calls) == 2
    assert result.meta.retry_count == 1
    assert result.meta.error is None


def test_unknown_provider_error_is_not_retried(monkeypatch):
    case = load_case("caseC")
    calls = []

    def fake_once(*args, **kwargs):
        calls.append(1)
        return _candidate(error="Unknown provider: bogus")

    monkeypatch.setattr("benchmark.runner._collect_candidate_once", fake_once)
    result = _collect_candidate(
        case,
        {"key": "candidate", "provider": "bogus"},
        "Candidate 1",
    )

    assert len(calls) == 1
    assert result.meta.retry_count == 0


def test_candidate_truncation_recovers_only_missing_section(monkeypatch):
    case = load_case("caseC")
    first_answers = {q.id: "answer" for q in case.questions[:-1]}
    responses = [
        _candidate(finish_reason="length", answers=first_answers),
        _candidate(answers={"plan": "recovered plan"}),
    ]
    requested_sections = []

    def fake_once(case_arg, *args, **kwargs):
        requested_sections.append([q.id for q in case_arg.questions])
        return responses.pop(0)

    monkeypatch.setattr("benchmark.runner._collect_candidate_once", fake_once)
    result = _collect_candidate(
        case,
        {"key": "candidate", "provider": "openrouter"},
        "Candidate 1",
    )

    assert requested_sections[0] == [q.id for q in case.questions]
    assert requested_sections[1] == ["plan"]
    assert result.answers["diagnosis"] == "answer"
    assert result.answers["plan"] == "recovered plan"
    assert result.meta.retry_count == 1


def test_candidate_parser_normalizes_presentation_without_inventing_sections():
    case = load_case("caseC")
    raw = """
    **a 1 —** migraine
    ### A2) CBC and CMP
    A 3: moderate
    """
    parsed = parse_candidate_answers(case, raw)

    assert parsed["diagnosis"] == "migraine"
    assert parsed["tests"] == "CBC and CMP"
    assert parsed["urgency"] == "moderate"
    assert "safety" not in parsed
    assert "plan" not in parsed


def test_question_echo_is_not_a_substantive_section():
    case = load_case("caseC")
    q = case.questions[0]
    echo = f"Q1 [diagnosis]: {q.text}"
    assert is_unsubstantive_section(case, "diagnosis", echo)
    parsed = parse_candidate_answers(case, echo)
    assert "diagnosis" not in parsed
    assert missing_section_ids(case, parsed) == [x.id for x in case.questions]


def test_single_section_unlabeled_prose_is_attributed():
    case = load_case("caseC")
    one = case.model_copy(update={"questions": [case.questions[2]]})
    raw = "Critical acuity with ECG changes and potassium above 6.5."
    parsed = parse_candidate_answers(one, raw)
    assert parsed == {"urgency": raw}


def test_biomistral_folds_system_into_user():
    assert prefers_user_only_chat(
        {"key": "local_biomistral", "model": "biomistral-7b-q4"}
    )
    assert not prefers_user_only_chat({"key": "local_medgemma"})
    msgs = fold_system_into_user(
        [
            {"role": "system", "content": "Be a physician."},
            {"role": "user", "content": "Answer A1–A5."},
        ]
    )
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"].startswith("Be a physician.")
    assert "Answer A1–A5." in msgs[0]["content"]
