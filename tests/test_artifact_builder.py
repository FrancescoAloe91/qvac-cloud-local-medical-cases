import pytest

from benchmark.runner import _validate_judge_separation, build_run_artifact
from benchmark.schema import CandidateAnswer, JudgeResult, ModelCallMeta


def test_shared_artifact_builder_records_reproducibility_and_judge_cohort():
    candidate = CandidateAnswer(
        candidate_key="candidate",
        label="Candidate",
        blind_id="Candidate 1",
        answers={"diagnosis": "answer"},
        meta=ModelCallMeta(
            model="requested/model",
            requested_model="requested/model",
            routed_model="routed/model",
            routed_provider="provider",
            provider="openrouter",
        ),
    )
    judgment = JudgeResult(
        blind_id="Candidate 1",
        candidate_key="candidate",
        question_scores=[],
        weighted_accuracy=80,
        judge_model="independent/verifier",
        judge_meta=ModelCallMeta(model="independent/verifier", provider="openrouter"),
    )
    artifact = build_run_artifact(
        config_snapshot={
            "judge": {
                "model": "primary/judge",
                "verifier_model": "independent/verifier",
            }
        },
        run_id="run-1",
        case_id="caseC",
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:01:00Z",
        batch_id="batch-1",
        models_config={"judge": {"model": "primary/judge"}},
        candidates=[candidate],
        judgments=[judgment],
        ranking=[],
        benchmark_track="native_defaults",
    )

    manifest = artifact.reproducibility
    assert manifest["primary_judge"] == "primary/judge"
    assert manifest["effective_judges"] == ["independent/verifier"]
    assert manifest["verifier_activated"] is True
    assert manifest["candidate_temperature"] is None
    assert manifest["candidate_calls"][0]["requested_model"] == "requested/model"
    assert manifest["candidate_calls"][0]["routed_model"] == "routed/model"
    assert manifest["prompts_sha256"]
    assert manifest["scoring_sha256"]


def test_verifier_must_be_independent_of_candidates_and_extractor(monkeypatch):
    monkeypatch.setenv("BENCHMARK_GOLD_EXTRACTOR_MODEL", "google/extractor")
    with pytest.raises(ValueError, match="candidate roster"):
        _validate_judge_separation(
            {"judge": {"model": "primary", "verifier_model": "vendor/candidate"}},
            [{"model": "vendor/candidate"}],
        )
    with pytest.raises(ValueError, match="extractor model family"):
        _validate_judge_separation(
            {"judge": {"model": "primary", "verifier_model": "google/verifier"}},
            [{"model": "vendor/candidate"}],
        )
