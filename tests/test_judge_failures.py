from __future__ import annotations

import pytest

from benchmark.cases_loader import load_case
from benchmark.gold import confirmed_gold, gold_json
from benchmark.judge import (
    PipelinedJudge,
    _evidence_normalized,
    _evidence_quote_present,
    _score_from_judge_item,
    build_ranking,
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

