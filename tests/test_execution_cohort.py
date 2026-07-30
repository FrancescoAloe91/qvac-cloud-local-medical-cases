"""execution_cohort_id and strict vs best-effort track behavior."""

from __future__ import annotations

from benchmark.gold import (
    execution_cohort_id,
    is_strict_track,
    uses_controlled_sampling,
)
from benchmark.openrouter import _provider_prefs
from benchmark.schema import (
    CandidateAnswer,
    GoldClaim,
    GoldSection,
    JudgeResult,
    ModelCallMeta,
)
from benchmark.gold import confirmed_gold


def _gold():
    raw = (
        "Diagnosis Alpha. Tests Beta. Urgency Gamma. Safety Delta. Plan Epsilon."
    )
    sections = {
        sid: GoldSection(
            summary=f"{sid} s",
            claims=[
                GoldClaim(
                    id=f"{sid}-1",
                    text=q,
                    source_quote=q,
                )
            ],
        )
        for sid, q in [
            ("diagnosis", "Diagnosis Alpha."),
            ("tests", "Tests Beta."),
            ("urgency", "Urgency Gamma."),
            ("safety", "Safety Delta."),
            ("plan", "Plan Epsilon."),
        ]
    }
    return confirmed_gold(raw_text=raw, sections=sections, extraction_model="m")


def test_track_helpers():
    assert uses_controlled_sampling("controlled")
    assert uses_controlled_sampling("strict_controlled")
    assert not uses_controlled_sampling("native_defaults")
    assert is_strict_track("strict_controlled")
    assert not is_strict_track("controlled")


def test_provider_prefs_strict_vs_best_effort():
    soft = _provider_prefs(
        allowed_providers=["OpenAI"], require_parameters=False, allow_fallbacks=True
    )
    assert soft["allow_fallbacks"] is True
    assert "require_parameters" not in soft
    hard = _provider_prefs(
        allowed_providers=["OpenAI"], require_parameters=True, allow_fallbacks=False
    )
    assert hard["allow_fallbacks"] is False
    assert hard["require_parameters"] is True


def test_execution_cohort_changes_with_routed_provider_and_gguf():
    gold = _gold()
    base_meta = ModelCallMeta(
        model="openai/gpt-5.5",
        provider="openrouter",
        requested_model="openai/gpt-5.5",
        routed_model="openai/gpt-5.5",
        routed_provider="OpenAI",
    )
    c1 = CandidateAnswer(
        candidate_key="chatgpt",
        label="c",
        blind_id="b1",
        answers={},
        meta=base_meta,
    )
    j1 = JudgeResult(
        candidate_key="chatgpt",
        blind_id="b1",
        question_scores=[],
        weighted_accuracy=50,
        judge_model="deepseek/deepseek-r1",
        judge_meta=ModelCallMeta(model="deepseek/deepseek-r1", provider="openrouter"),
    )
    a = execution_cohort_id(
        case_stem="case",
        gold=gold,
        prompt_version="gold-only-v1",
        benchmark_track="controlled",
        candidates=[c1],
        judgments=[j1],
    )
    c2 = CandidateAnswer(
        candidate_key="chatgpt",
        label="c",
        blind_id="b1",
        answers={},
        meta=base_meta.model_copy(update={"routed_provider": "Azure"}),
    )
    b = execution_cohort_id(
        case_stem="case",
        gold=gold,
        prompt_version="gold-only-v1",
        benchmark_track="controlled",
        candidates=[c2],
        judgments=[j1],
    )
    assert a != b
    c3 = CandidateAnswer(
        candidate_key="qvac",
        label="q",
        blind_id="b2",
        answers={},
        meta=ModelCallMeta(
            model="medpsy",
            provider="qvac",
            gguf_sha256="aaa",
            device="metal",
        ),
    )
    d1 = execution_cohort_id(
        case_stem="case",
        gold=gold,
        prompt_version="gold-only-v1",
        benchmark_track="controlled",
        candidates=[c3],
        judgments=[],
    )
    c4 = c3.model_copy(
        update={"meta": c3.meta.model_copy(update={"gguf_sha256": "bbb"})}
    )
    d2 = execution_cohort_id(
        case_stem="case",
        gold=gold,
        prompt_version="gold-only-v1",
        benchmark_track="controlled",
        candidates=[c4],
        judgments=[],
    )
    assert d1 != d2
