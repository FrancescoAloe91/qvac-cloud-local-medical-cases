"""Collect retry metadata merge keeps first-attempt provenance."""

from __future__ import annotations

from benchmark.runner import _merge_collect_meta
from benchmark.schema import CandidateAnswer, ModelCallMeta


def test_merge_collect_meta_keeps_first_attempt_tokens_and_route():
    first = CandidateAnswer(
        candidate_key="chatgpt",
        label="c",
        blind_id="b1",
        answers={},
        meta=ModelCallMeta(
            model="openai/gpt-first",
            provider="OpenAI",
            requested_model="openai/gpt-req",
            routed_model="openai/gpt-routed",
            routed_provider="OpenAI",
            prompt_tokens=10,
            completion_tokens=5,
            reasoning_tokens=3,
            cost_usd=0.01,
            latency_s=1.0,
            finish_reason="error",
            error="timeout",
            requested_providers=["OpenAI"],
            paid_attempts=[{"role": "primary", "cost_usd": 0.01}],
        ),
    )
    second = CandidateAnswer(
        candidate_key="chatgpt",
        label="c",
        blind_id="b1",
        answers={"A1": "ok"},
        meta=ModelCallMeta(
            model="openai/gpt-second",
            provider="OpenAI",
            prompt_tokens=8,
            completion_tokens=4,
            reasoning_tokens=1,
            cost_usd=0.02,
            latency_s=0.5,
        ),
    )
    _merge_collect_meta(first, second)
    assert second.meta.prompt_tokens == 18
    assert second.meta.completion_tokens == 9
    assert second.meta.reasoning_tokens == 4
    assert abs(float(second.meta.cost_usd or 0) - 0.03) < 1e-9
    assert second.meta.retry_count >= 1
    prior = second.meta.prior_attempts
    assert prior and prior[-1]["routed_model"] == "openai/gpt-routed"
    assert prior[-1]["reasoning_tokens"] == 3
    assert prior[-1]["cost_usd"] == 0.01
    assert second.meta.paid_attempts and second.meta.paid_attempts[0]["role"] == "primary"
