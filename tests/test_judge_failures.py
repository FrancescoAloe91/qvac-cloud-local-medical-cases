from __future__ import annotations

from concurrent.futures import Future

import pytest

from benchmark.cases_loader import load_case
from benchmark.gold import confirmed_gold, gold_json
from benchmark.judge import (
    PipelinedJudge,
    _evidence_normalized,
    _evidence_quote_present,
    _extract_json,
    _normalize_judge_data,
    _score_from_judge_item,
    _score_sections_from_payload,
    build_ranking,
    judge_candidate,
    systemic_judge_failure,
)
from benchmark.schema import (
    CandidateAnswer,
    GoldClaim,
    GoldSection,
    JudgeResult,
    ModelCallMeta,
    QuestionScore,
)


def _contract() -> str:
    sections = {}
    for section in ("diagnosis", "tests", "urgency", "safety", "plan"):
        quote = f"{section} reference"
        sections[section] = GoldSection(
            summary=quote,
            claims=[GoldClaim(id=f"{section}-1", text=quote, source_quote=quote)],
        )
    return gold_json(
        confirmed_gold(
            raw_text=" ".join(s.summary for s in sections.values()),
            sections=sections,
            extraction_model="test",
        )
    )


def test_valid_evidence_linked_item_scores_balanced_formula():
    case = load_case("caseC")
    answer = "The diagnosis is migraine."
    item = {
        "question_id": "diagnosis",
        "matched_claim_ids": ["diagnosis-1"],
        "missed_claim_ids": [],
        "unsupported_claims": [],
        "contradictions": [],
        "evidence": [
            {
                "candidate_quote": "diagnosis is migraine",
                "reference_claim_id": "diagnosis-1",
            }
        ],
        "quality": 0.8,
        "rationale": "Supported.",
        "errors": [],
    }
    score = _score_from_judge_item(
        case, item, answer_text=answer, gold_reference=_contract()
    )
    assert score.score == 95.0
    assert score.precision == 1.0
    assert score.recall == 1.0


def test_invented_evidence_is_rejected_not_scored_zero():
    case = load_case("caseC")
    item = {
        "question_id": "diagnosis",
        "matched_claim_ids": ["diagnosis-1"],
        "missed_claim_ids": [],
        "unsupported_claims": [],
        "contradictions": [],
        "evidence": [
            {
                "candidate_quote": "invented quote",
                "reference_claim_id": "diagnosis-1",
            }
        ],
        "quality": 0.8,
        "rationale": "",
        "errors": [],
    }
    with pytest.raises(ValueError, match="not present"):
        _score_from_judge_item(
            case,
            item,
            answer_text="The diagnosis is migraine.",
            gold_reference=_contract(),
        )


def test_markdown_only_quote_differences_are_accepted():
    case = load_case("caseC")
    item = {
        "question_id": "diagnosis",
        "matched_claim_ids": ["diagnosis-1"],
        "missed_claim_ids": [],
        "unsupported_claims": [],
        "contradictions": [],
        "evidence": [
            {
                "candidate_quote": "Severe acute kidney injury",
                "reference_claim_id": "diagnosis-1",
            }
        ],
        "quality": 0.8,
        "rationale": "",
        "errors": [],
    }
    score = _score_from_judge_item(
        case,
        item,
        answer_text="**Severe acute kidney injury**",
        gold_reference=_contract(),
    )
    assert score.score == 95.0


def test_punctuation_and_case_only_quote_differences_are_accepted():
    answer = _evidence_normalized(
        "URGENT—give calcium gluconate; then monitor the ECG."
    )
    assert _evidence_quote_present(
        "Urgent: give calcium gluconate, then monitor the ECG!",
        answer,
    )
    assert not _evidence_quote_present(
        "Urgent: give potassium, then monitor the ECG!",
        answer,
    )


def test_combined_noncontiguous_verbatim_sentences_are_accepted():
    answer = _evidence_normalized(
        "Give insulin now. An unrelated bullet. Monitor glucose hourly."
    )
    assert _evidence_quote_present(
        "Give insulin now. Monitor glucose hourly.",
        answer,
    )
    assert not _evidence_quote_present(
        "Give insulin now. Invented treatment.",
        answer,
    )


def test_graded_judge_item_preserves_partial_coverage_and_neutral_additions():
    case = load_case("caseC")
    item = {
        "question_id": "diagnosis",
        "claim_assessments": [
            {
                "reference_claim_id": "diagnosis-1",
                "coverage": 0.5,
                "candidate_quotes": ["probable migraine"],
                "rationale": "Correct direction but incomplete.",
            }
        ],
        "additional_claims": [
            {
                "candidate_quote": "Consider tension headache",
                "classification": "neutral",
                "severity": 0.5,
                "rationale": "Reasonable optional differential.",
            }
        ],
        "quality": 0.8,
        "rationale": "Useful but incomplete.",
        "errors": [],
    }
    score = _score_from_judge_item(
        case,
        item,
        answer_text="Probable migraine. Consider tension headache.",
        gold_reference=_contract(),
    )
    assert score.score == 68.0
    assert score.recall == 0.5
    assert score.precision == 1.0
    assert score.claim_coverage == {"diagnosis-1": 0.5}


def test_local_repair_accepts_unambiguous_schema_and_numeric_variants():
    repaired = _normalize_judge_data(
        {
            "sections": {
                "diagnosis": {
                    "claims": {
                        "diagnosis-1": {
                            "score": "50%",
                            "candidate_quote": "probable migraine",
                        }
                    },
                    "added_content": {
                        "quote": "Consider tension headache",
                        "classification": "neutral",
                        "severity": "50",
                    },
                    "clinical_quality": "80",
                    "ignored_extra_field": "harmless",
                }
            }
        }
    )
    item = repaired["question_scores"][0]
    score = _score_from_judge_item(
        load_case("caseC"),
        item,
        answer_text="Probable migraine. Consider tension headache.",
        gold_reference=_contract(),
    )

    assert score.recall == 0.5
    assert score.quality == 0.8
    assert score.precision == 1.0


def test_corrective_retry_requests_only_invalid_sections(monkeypatch):
    case = load_case("caseC")
    answers = {q.id: f"{q.id} answer" for q in case.questions}
    candidate = CandidateAnswer(
        candidate_key="candidate",
        label="Candidate",
        blind_id="Candidate 1",
        answers=answers,
        raw_response=" ".join(answers.values()),
        meta=ModelCallMeta(model="candidate", provider="test"),
    )

    def item(qid, quote):
        return {
            "question_id": qid,
            "claim_assessments": [
                {
                    "reference_claim_id": f"{qid}-1",
                    "coverage": 1,
                    "candidate_quotes": [quote],
                }
            ],
            "additional_claims": [],
            "quality": 1,
        }

    first = {
        "question_scores": [
            item(q.id, "invented evidence" if q.id == "diagnosis" else answers[q.id])
            for q in case.questions
        ]
    }
    second = {"question_scores": [item("diagnosis", answers["diagnosis"])]}
    payloads = [first, second]
    sent_messages = []

    def fake_chat(model, messages, **kwargs):
        sent_messages.append(messages)
        payload = payloads.pop(0)
        return (
            __import__("json").dumps(payload),
            ModelCallMeta(model=model, provider="openrouter"),
        )

    monkeypatch.setattr("benchmark.judge.openrouter.chat", fake_chat)
    result = judge_candidate(
        case,
        candidate,
        "judge",
        gold_reference=_contract(),
    )

    assert result.status == "valid"
    assert result.retry_count == 1
    assert len(result.question_scores) == 5
    assert "['diagnosis']" in sent_messages[1][-1]["content"]
    assert "rejected ONLY these sections" in sent_messages[1][-1]["content"]
    # Repair call must be cheaper/shorter than a full re-judge.
    assert sent_messages  # primary + targeted repair only
    assert len(sent_messages) == 2


def _judgment(key: str, score: float, status: str = "valid") -> JudgeResult:
    meta = ModelCallMeta(model="judge", provider="openrouter")
    return JudgeResult(
        blind_id=key,
        candidate_key=key,
        question_scores=[QuestionScore(question_id="diagnosis", score=score)],
        weighted_accuracy=score,
        judge_model="judge",
        judge_meta=meta,
        status=status,
        failure_reason="timeout" if status != "valid" else "",
    )


def test_ties_remain_ties_and_failures_are_na():
    ranking = build_ranking(
        [
            _judgment("a", 80.0),
            _judgment("b", 80.0),
            _judgment("c", 0.0, "timed_out"),
        ]
    )
    assert ranking[0]["accuracy"] == ranking[1]["accuracy"] == 80.0
    assert ranking[0]["rank"] == ranking[1]["rank"] == 1
    assert ranking[2]["accuracy"] is None
    assert ranking[2]["rank"] is None
    assert ranking[2]["status"] == "n/a"


def test_systemic_failure_requires_no_valid_judge_observation():
    valid = _judgment("valid", 75.0)
    transport = _judgment("transport", 0.0, "judge_transport_failed")
    partial = _judgment("partial", 0.0, "candidate_partial")

    assert not systemic_judge_failure([valid, transport, transport])
    assert systemic_judge_failure([transport, transport])
    assert not systemic_judge_failure([partial, partial])


def test_partial_candidate_is_na_without_spending_on_judge():
    case = load_case("caseC")
    candidate = CandidateAnswer(
        candidate_key="a",
        label="A",
        blind_id="Candidate 1",
        answers={"diagnosis": "migraine"},
        raw_response="A1: migraine",
        meta=ModelCallMeta(model="candidate", provider="openrouter"),
    )
    pipe = PipelinedJudge(
        case,
        "judge",
        gold_reference=_contract(),
        expected_total=1,
        run_scope="test-partial",
    )
    result = pipe._one_safe(candidate)
    pipe.close(cancel_pending=True)
    assert result.status == "candidate_partial"
    assert result.weighted_accuracy == 0.0  # internal compatibility only


def test_verifier_rejudges_only_residual_failures(monkeypatch):
    case = load_case("caseC")
    answers = {q.id: f"{q.id} answer" for q in case.questions}
    candidates = [
        CandidateAnswer(
            candidate_key=key,
            label=key,
            blind_id=f"Candidate {index}",
            answers=answers,
            raw_response="complete response",
            meta=ModelCallMeta(
                model=f"candidate-model-{index}",
                provider="openrouter",
            ),
        )
        for index, key in enumerate(("a", "b"), 1)
    ]
    pipe = PipelinedJudge(
        case,
        "primary-judge",
        gold_reference=_contract(),
        verifier_model="independent-verifier",
        expected_total=2,
        run_scope="test-whole-verifier",
    )
    pipe._candidates = candidates
    pipe._by_key = {
        "a": _judgment("a", 80.0),
        "b": _judgment("b", 0.0, "judge_schema_invalid"),
    }
    seen = []

    def fake_judge(case_arg, candidate, judge_model, *args, **kwargs):
        seen.append(candidate.candidate_key)
        result = _judgment(candidate.candidate_key, 70.0)
        result.judge_model = judge_model
        return result

    monkeypatch.setattr("benchmark.judge.judge_candidate", fake_judge)
    pipe._verify_whole_run()
    pipe.close(cancel_pending=True)

    # Already-valid scores must not be pulled back into verifier theater.
    assert seen == ["b"]
    assert pipe._by_key["a"].judge_model == "judge"
    assert pipe._by_key["b"].judge_model == "independent-verifier"


def test_verifier_requires_systemic_residual_failures():
    case = load_case("caseC")
    pipe = PipelinedJudge(
        case,
        "primary",
        expected_total=3,
        verifier_model="independent",
        run_scope="test-verifier-threshold",
    )
    pipe._candidates = [
        CandidateAnswer(
            candidate_key=key,
            label=key,
            blind_id=f"Candidate {index}",
            answers={q.id: "answer" for q in case.questions},
            raw_response="complete",
            meta=ModelCallMeta(model=f"model-{key}", provider="test"),
        )
        for index, key in enumerate(("a", "b", "c"), 1)
    ]
    pipe._by_key = {
        "a": _judgment("a", 80),
        "b": _judgment("b", 0, "judge_schema_invalid"),
        "c": _judgment("c", 82),
    }
    assert not pipe._needs_whole_run_verifier()

    pipe._by_key["c"] = _judgment("c", 0, "judge_transport_failed")
    assert pipe._needs_whole_run_verifier()
    pipe.close(cancel_pending=True)


def test_pipeline_deadline_synthesizes_explicit_na_row():
    case = load_case("caseC")
    candidate = CandidateAnswer(
        candidate_key="late",
        label="Late",
        blind_id="Candidate 1",
        answers={q.id: "answer" for q in case.questions},
        raw_response="complete",
        meta=ModelCallMeta(model="late-model", provider="test"),
    )
    pipe = PipelinedJudge(
        case,
        "primary-judge",
        expected_total=1,
        run_scope="test-deadline",
    )
    future = Future()
    pipe._candidates = [candidate]
    pipe._pending = {future: candidate}

    pipe._finish_pending_as(
        status="timed_out",
        reason="deadline",
        marker="judge_timeout",
    )
    pipe.close(cancel_pending=True)

    result = pipe._by_key["late"]
    assert result.status == "timed_out"
    assert result.failure_reason == "deadline"
    assert result.question_scores[0].errors == ["judge_timeout"]


def test_pipeline_cancellation_snapshot_retains_submitted_candidate():
    case = load_case("caseC")
    candidate = CandidateAnswer(
        candidate_key="submitted",
        label="Submitted",
        blind_id="Candidate 1",
        answers={q.id: "answer" for q in case.questions},
        raw_response="complete",
        meta=ModelCallMeta(
            model="submitted-model",
            provider="test",
            latency_s=2.5,
            tps=12.0,
        ),
    )
    pipe = PipelinedJudge(
        case,
        "primary-judge",
        expected_total=1,
        run_scope="test-cancel-snapshot",
    )
    future = Future()
    pipe._candidates = [candidate]
    pipe._pending = {future: candidate}

    snapshot = pipe.cancel_and_snapshot()

    assert snapshot["candidates"][0].meta.tps == 12.0
    assert snapshot["judgments"][0].status == "cancelled"
    assert snapshot["judgments"][0].failure_reason == "Judging cancelled by user"


def test_corrective_retry_emits_stage_progress(monkeypatch):
    case = load_case("caseC")
    candidate = CandidateAnswer(
        candidate_key="retry",
        label="Retry",
        blind_id="Candidate 1",
        answers={q.id: "answer" for q in case.questions},
        raw_response="complete",
        meta=ModelCallMeta(model="model", provider="test"),
    )
    events = []
    pipe = PipelinedJudge(
        case,
        "primary",
        expected_total=1,
        on_progress=events.append,
        run_scope="test-retry-progress",
    )
    pipe._candidates = [candidate]
    pipe._pipeline_started = __import__("time").monotonic()
    pipe._started_at[candidate.candidate_key] = __import__("time").monotonic()
    # Pipeline-level retry is reserved for transport/timeout — not schema/evidence.
    pipe._by_key = {
        candidate.candidate_key: _judgment(
            candidate.candidate_key, 0.0, "judge_transport_failed"
        )
    }

    def fake_judge(*args, progress_callback=None, **kwargs):
        if progress_callback:
            progress_callback("validating response", 88)
        return _judgment(candidate.candidate_key, 77.0)

    monkeypatch.setattr("benchmark.judge.judge_candidate", fake_judge)
    monkeypatch.setattr("benchmark.judge.time.sleep", lambda _: None)
    pipe.finalize()

    phases = [event.get("phase") for event in events]
    assert "retry" in phases
    assert "progress" in phases
    assert "retry_done" in phases
    retry_evt = next(evt for evt in events if evt.get("phase") == "retry")
    assert retry_evt.get("active_attempt") is True
    assert float(retry_evt.get("elapsed_s") or 0) >= 0.0
    assert any(evt.get("phase") == "retry_done" for evt in events)


@pytest.mark.parametrize("na_count", [1, 2])
def test_candidate_specific_na_terminalizes_iteration_and_allows_next(na_count):
    case = load_case("caseC")

    def completed_iteration(scope):
        pipe = PipelinedJudge(
            case,
            "primary",
            expected_total=4,
            run_scope=scope,
        )
        pipe._candidates = [
            CandidateAnswer(
                candidate_key=f"candidate-{index}",
                label=f"Candidate {index}",
                blind_id=f"Candidate {index}",
                answers={q.id: "answer" for q in case.questions},
                raw_response="complete",
                meta=ModelCallMeta(model=f"model-{index}", provider="test"),
            )
            for index in range(4)
        ]
        pipe._by_key = {
            candidate.candidate_key: _judgment(
                candidate.candidate_key,
                0.0 if index < na_count else 80.0 + index,
                "candidate_partial" if index < na_count else "valid",
            )
            for index, candidate in enumerate(pipe._candidates)
        }
        return pipe.finalize()

    first = completed_iteration(f"iteration-one-{na_count}")
    assert len(first) == 4
    assert len({row.candidate_key for row in first}) == 4
    assert sum(row.status != "valid" for row in first) == na_count
    assert not systemic_judge_failure(first)

    second = completed_iteration(f"iteration-two-{na_count}")
    assert len(second) == 4


def test_schema_invalid_does_not_reenter_pipeline_corrective_retry(monkeypatch):
    """Finished N/A must not flip back to 75% via finalize serial retry."""
    case = load_case("caseC")
    candidate = CandidateAnswer(
        candidate_key="claude",
        label="Claude",
        blind_id="Candidate 1",
        answers={q.id: "answer" for q in case.questions},
        raw_response="complete",
        meta=ModelCallMeta(model="anthropic/claude-sonnet-5", provider="openrouter"),
    )
    events = []
    pipe = PipelinedJudge(
        case,
        "primary",
        expected_total=1,
        on_progress=events.append,
        run_scope="test-no-regress-retry",
        max_retries=3,
    )
    failed = _judgment("claude", 0.0, "judge_schema_invalid")
    failed.retry_count = 1
    pipe._candidates = [candidate]
    pipe._by_key = {"claude": failed}
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("pipeline must not re-call judge for schema/evidence N/A")

    monkeypatch.setattr("benchmark.judge.judge_candidate", boom)
    monkeypatch.setattr("benchmark.judge.time.sleep", lambda _: None)
    results = pipe.finalize()
    assert calls["n"] == 0
    assert results[0].status == "judge_schema_invalid"
    assert not any(evt.get("phase") == "retry" for evt in events)


def test_finalize_completes_mixed_valid_and_na(monkeypatch):
    case = load_case("caseC")
    candidates = [
        CandidateAnswer(
            candidate_key=key,
            label=key,
            blind_id=f"Candidate {i}",
            answers={q.id: "answer" for q in case.questions},
            raw_response="complete",
            meta=ModelCallMeta(model=f"m-{key}", provider="openrouter"),
        )
        for i, key in enumerate(("chatgpt", "claude", "gemini"), 1)
    ]
    pipe = PipelinedJudge(
        case,
        "primary",
        expected_total=3,
        run_scope="test-mixed-terminal",
        verifier_model="",
    )
    pipe._candidates = candidates
    pipe._by_key = {
        "chatgpt": _judgment("chatgpt", 90.0),
        "claude": _judgment("claude", 0.0, "judge_schema_invalid"),
        "gemini": _judgment("gemini", 85.0),
    }
    monkeypatch.setattr(
        "benchmark.judge.judge_candidate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no retry")),
    )
    results = pipe.finalize()
    assert len(results) == 3
    by = {r.candidate_key: r for r in results}
    assert by["chatgpt"].status == "valid"
    assert by["claude"].status == "judge_schema_invalid"
    assert by["gemini"].status == "valid"


def test_numeric_punctuation_is_preserved_in_evidence_matching():
    answer = _evidence_normalized("Start calcium gluconate 10–20 mL and insulin 1.0 U.")
    assert _evidence_quote_present("calcium gluconate 10–20 mL", answer)
    assert _evidence_quote_present("insulin 1.0 U", answer)
    assert not _evidence_quote_present("calcium gluconate 10 20 mL", answer)
    assert not _evidence_quote_present("insulin 1 0 U", answer)


def test_short_token_does_not_match_inside_longer_word():
    answer = _evidence_normalized("Consider adrenal insufficiency in this context.")
    assert not _evidence_quote_present("renal", answer)
    assert _evidence_quote_present("adrenal insufficiency", answer)


def test_markdown_wrapped_string_numbers_pass_without_paid_retry(monkeypatch):
    case = load_case("caseC")
    answers = {
        q.id: (
            f"**{q.id.title()}** — give 10–20 mg now. "
            f"Primary plan for {q.id}."
        )
        for q in case.questions
    }
    candidate = CandidateAnswer(
        candidate_key="claude",
        label="Claude",
        blind_id="Candidate 1",
        answers=answers,
        raw_response="\n".join(answers.values()),
        meta=ModelCallMeta(model="anthropic/claude-sonnet-5", provider="openrouter"),
    )
    payload = {
        "question_scores": [
            {
                "question_id": q.id,
                "claim_assessments": [
                    {
                        "reference_claim_id": f"{q.id}-1",
                        "coverage": "100%",
                        "candidate_quotes": [
                            f"**{q.id.title()}** — give 10–20 mg now."
                        ],
                    }
                ],
                "additional_claims": [],
                "clinical_quality": "80",
            }
            for q in case.questions
        ]
    }
    calls = {"n": 0}

    def fake_chat(model, messages, **kwargs):
        calls["n"] += 1
        return (
            "```json\n" + __import__("json").dumps(payload) + "\n```",
            ModelCallMeta(model=model, provider="openrouter", cost_usd=0.01),
        )

    monkeypatch.setattr("benchmark.judge.openrouter.chat", fake_chat)
    stages: list[tuple[str, int]] = []
    result = judge_candidate(
        case,
        candidate,
        "judge",
        gold_reference=_contract(),
        progress_callback=lambda stage, pct: stages.append((stage, pct)),
    )
    assert result.status == "valid"
    assert result.retry_count == 0
    assert calls["n"] == 1
    assert ("corrective retry", 75) not in stages
    assert any(stage == "validating response" for stage, _ in stages)


def test_cloud_artifact_judge_payload_salvages_without_retry():
    """Gemini N/A in caseC-a91baa0a2c was evidence-presentation; local salvage scores it."""
    import json
    from pathlib import Path

    path = Path(
        "artifacts/owners/893e6a29cf690fbef4d6aee2/caseC-a91baa0a2c.json"
    )
    if not path.exists():
        pytest.skip("artifact not present")
    art = json.loads(path.read_text())
    gold = art["models_config"]["gold_reference"]
    case = load_case(art["case_id"])
    for key in ("gemini", "chatgpt", "claude"):
        j = next(jj for jj in art["judgments"] if jj["candidate_key"] == key)
        c = next(cc for cc in art["candidates"] if cc["candidate_key"] == key)
        cand = CandidateAnswer(
            candidate_key=key,
            label=key,
            blind_id=c.get("blind_id") or key,
            answers=c.get("answers") or {},
            raw_response=c.get("raw_response") or "",
            meta=ModelCallMeta(model="x", provider="test"),
        )
        data = _extract_json(j.get("raw_judge_json") or "")
        accepted, errors = _score_sections_from_payload(
            case,
            cand,
            data,
            gold_reference=gold,
            target_ids={q.id for q in case.questions},
        )
        assert not errors, (key, errors)
        assert set(accepted) == {q.id for q in case.questions}


def test_pipeline_deadline_starts_at_first_submit_not_construction():
    case = load_case("caseC")
    pipe = PipelinedJudge(
        case,
        "primary",
        expected_total=1,
        max_wall_s=30.0,
        run_scope="test-deadline-start",
    )
    assert pipe._pipeline_started is None
    assert pipe._budget_remaining_s() == 30.0
    candidate = CandidateAnswer(
        candidate_key="c",
        label="C",
        blind_id="Candidate 1",
        answers={q.id: "answer" for q in case.questions},
        raw_response="complete",
        meta=ModelCallMeta(
            model="x", provider="test", error="Failed to load medpsy.gguf"
        ),
    )
    pipe.submit(candidate)
    assert pipe._pipeline_started is not None
    pipe.poll()
    pipe.close(cancel_pending=True)
    assert pipe._by_key["c"].status == "collect_failed"
    assert pipe._by_key["c"].judge_meta.cost_usd == 0.0
    assert pipe._by_key["c"].judge_meta.model != candidate.meta.model or (
        pipe._by_key["c"].judge_meta is not candidate.meta
    )


def test_na_judgment_does_not_reuse_candidate_meta_cost():
    case = load_case("caseC")
    candidate = CandidateAnswer(
        candidate_key="chatgpt",
        label="ChatGPT",
        blind_id="Candidate 1",
        answers={"diagnosis": "migraine"},
        raw_response="A1: migraine",
        meta=ModelCallMeta(
            model="openai/gpt",
            provider="openrouter",
            cost_usd=0.42,
        ),
    )
    pipe = PipelinedJudge(
        case,
        "judge",
        gold_reference=_contract(),
        expected_total=1,
        run_scope="test-no-cost-dup",
    )
    result = pipe._one_safe(candidate)
    pipe.close(cancel_pending=True)
    assert result.status == "candidate_partial"
    assert float(result.judge_meta.cost_usd or 0.0) == 0.0
    assert result.judge_meta.paid_attempts == []

