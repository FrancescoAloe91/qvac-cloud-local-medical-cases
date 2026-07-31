"""Cost estimate honesty: typical tokens (not max_tokens), History calibration."""

from __future__ import annotations

from benchmark import openrouter
from benchmark.cases_loader import load_case
from benchmark.config import load_models_config
from benchmark.costing import cost_estimate_priors_from_artifacts
from benchmark.runner import estimate_cost_breakdown
from benchmark.schema import (
    CandidateAnswer,
    JudgeResult,
    ModelCallMeta,
    RunArtifact,
)


def test_fallback_prices_include_verifier_and_extractor_models():
    assert "qwen/qwen3.5-397b-a17b" in openrouter._FALLBACK_PRICES
    assert "google/gemini-3.5-flash" in openrouter._FALLBACK_PRICES
    assert "openai/gpt-4o-mini" in openrouter._FALLBACK_PRICES
    assert "deepseek/deepseek-r1" in openrouter._FALLBACK_PRICES
    pin, pout = openrouter.model_prices_per_mtok("qwen/qwen3.5-397b-a17b")
    assert pin > 0 and pout > 0
    # Must not silently fall back to the generic (2.0, 10.0) default.
    assert (pin, pout) != (2.0, 10.0)


def test_estimate_uses_typical_judge_tokens_not_max_cap():
    cfg = load_models_config()
    case = load_case("caseC")
    case = case.model_copy(
        update={
            "stem": "Anonymized case. " * 40,
        }
    )
    gold = (
        "Diagnosis: NSTEMI. Tests: troponin, ECG. Urgency: urgent. "
        "Safety: no nitrates with sildenafil. Plan: dual antiplatelet. "
    ) * 8

    bd = estimate_cost_breakdown(
        cfg,
        case,
        include_qvac=True,
        gold_reference=gold,
        n=1,
        triple_qvac=False,
    )

    assert "extractor" in bd
    assert bd["extractor"]["model"] == "openai/gpt-4o-mini"
    assert "section_repair" in bd
    assert "verifier" in bd
    assert bd["extractor"]["estimated_usd"] > 0
    # Baseline must use typical outs — never bill the 16k API cap as expected.
    assert bd["judge"]["completion_tokens_per_call"] == 5500
    assert bd["judge"]["completion_tokens_cap"] == 16384
    assert bd["judge"]["completion_tokens_per_call"] < bd["judge"]["completion_tokens_cap"]
    assert bd["section_repair"]["completion_tokens_per_call"] == 4096
    assert bd["section_repair"].get("optional") is True
    assert bd["verifier"].get("optional") is True
    assert bd["verifier"]["model"] == "qwen/qwen3.5-397b-a17b"
    assert bd["reliability"] == "rough_estimate_often_over"

    baseline = float(bd["total_usd"])
    upper = float(bd["total_usd_upper"])
    repair = float(bd["section_repair"]["estimated_usd"])
    verifier = float(bd["verifier"]["estimated_usd"])
    extract = float(bd["extractor"]["estimated_usd"])
    judge = float(bd["judge"]["estimated_usd"])

    assert baseline > extract + judge * 0.5  # candidates + extract + judge
    assert upper >= baseline
    # Upper folds uncommon repair/verifier — not the old full-cohort ceiling.
    assert repair < float(bd["section_repair"]["estimated_usd_ceiling"])
    assert verifier < float(bd["verifier"]["estimated_usd_full_cohort"])


def test_estimate_multi_run_extractor_once():
    cfg = load_models_config()
    case = load_case("caseC")
    case = case.model_copy(update={"stem": "Case text for cost. " * 20})
    gold = "Reference diagnosis and plan. " * 30

    one = estimate_cost_breakdown(
        cfg, case, include_qvac=False, gold_reference=gold, n=1
    )
    five = estimate_cost_breakdown(
        cfg, case, include_qvac=False, gold_reference=gold, n=5
    )

    extract = float(one["extractor"]["estimated_usd"])
    # Extractor billed once per batch, not ×N.
    expected = extract + (float(one["total_usd"]) - extract) * 5
    assert abs(float(five["total_usd_for_n"]) - expected) < 1e-4
    assert float(five["total_usd_upper_for_n"]) >= float(five["total_usd_for_n"])


def test_estimate_skips_extractor_when_already_paid():
    cfg = load_models_config()
    case = load_case("caseC")
    case = case.model_copy(update={"stem": "Case text for cost. " * 20})
    gold = "Reference diagnosis and plan. " * 30
    with_ex = estimate_cost_breakdown(
        cfg, case, include_qvac=False, gold_reference=gold, n=1
    )
    without = estimate_cost_breakdown(
        cfg,
        case,
        include_qvac=False,
        gold_reference=gold,
        n=1,
        include_extractor=False,
        extraction_cost_usd=0.0,
    )
    assert float(with_ex["extractor"]["estimated_usd"]) > 0
    assert float(without["extractor"]["estimated_usd"]) == 0.0
    assert float(without["total_usd"]) < float(with_ex["total_usd"])


def test_confirmed_gold_extraction_cost_persists():
    from benchmark.gold import confirmed_gold, gold_json, load_confirmed_gold
    from benchmark.schema import GoldClaim, GoldSection

    sections = {
        sid: GoldSection(
            summary=f"{sid} ref",
            claims=[
                GoldClaim(
                    id=f"{sid}-1",
                    text=f"{sid} ref",
                    source_quote=f"{sid} ref",
                )
            ],
        )
        for sid in (
            "diagnosis",
            "tests",
            "urgency",
            "safety",
            "plan",
        )
    }
    raw = " ".join(s.summary for s in sections.values())
    gold = confirmed_gold(
        raw_text=raw,
        sections=sections,
        extraction_model="openai/gpt-4o-mini",
        extraction_cost_usd=0.0123,
    )
    loaded = load_confirmed_gold(gold_json(gold))
    assert loaded.extraction_cost_usd == 0.0123


def test_verifier_cost_merge_keeps_prior_attempts():
    """Whole-run verifier must append paid_attempts, not overwrite primary spend."""
    prior = ModelCallMeta(
        model="deepseek/deepseek-r1",
        provider="openrouter",
        cost_usd=0.05,
        prompt_tokens=100,
        completion_tokens=200,
        paid_attempts=[{"role": "primary", "cost_usd": 0.05}],
    )
    verifier = ModelCallMeta(
        model="qwen/qwen3.5-397b-a17b",
        provider="openrouter",
        cost_usd=0.02,
        prompt_tokens=50,
        completion_tokens=80,
        paid_attempts=[{"role": "primary", "cost_usd": 0.02}],
    )
    attempts = list(verifier.paid_attempts or [])
    for attempt in attempts:
        if attempt.get("role") == "primary":
            attempt["role"] = "verifier"
    merged = list(prior.paid_attempts or []) + attempts
    total = round(float(prior.cost_usd or 0) + float(verifier.cost_usd or 0), 8)
    assert total == 0.07
    assert [a["role"] for a in merged] == ["primary", "verifier"]
    assert len(merged) == 2


def _hist_art(run_cost: float, *, n_or: int = 3, n_local: int = 9) -> RunArtifact:
    cands = []
    judges = []
    for i in range(n_or):
        key = f"cloud{i}"
        cands.append(
            CandidateAnswer(
                candidate_key=key,
                label=key,
                blind_id=f"b-{key}",
                answers={},
                meta=ModelCallMeta(
                    model="openai/gpt-5.5",
                    provider="openrouter",
                    cost_usd=0.03,
                    prompt_tokens=700,
                    completion_tokens=1900,
                ),
            )
        )
        judges.append(
            JudgeResult(
                candidate_key=key,
                blind_id=f"b-{key}",
                status="valid",
                weighted_accuracy=50.0,
                question_scores=[],
                judge_model="deepseek/deepseek-r1",
                judge_meta=ModelCallMeta(
                    model="deepseek/deepseek-r1",
                    provider="openrouter",
                    cost_usd=0.016,
                    prompt_tokens=2200,
                    completion_tokens=5200,
                    paid_attempts=[{"role": "primary", "cost_usd": 0.016}],
                ),
            )
        )
    for i in range(n_local):
        key = f"local{i}"
        cands.append(
            CandidateAnswer(
                candidate_key=key,
                label=key,
                blind_id=f"b-{key}",
                answers={},
                meta=ModelCallMeta(
                    model="medpsy",
                    provider="qvac",
                    cost_usd=0.0,
                    completion_tokens=800,
                ),
            )
        )
        judges.append(
            JudgeResult(
                candidate_key=key,
                blind_id=f"b-{key}",
                status="valid",
                weighted_accuracy=50.0,
                question_scores=[],
                judge_model="deepseek/deepseek-r1",
                judge_meta=ModelCallMeta(
                    model="deepseek/deepseek-r1",
                    provider="openrouter",
                    cost_usd=0.015,
                    prompt_tokens=2000,
                    completion_tokens=5100,
                    paid_attempts=[{"role": "primary", "cost_usd": 0.015}],
                ),
            )
        )
    return RunArtifact(
        run_id=f"r-{run_cost}",
        case_id="caseC",
        started_at="t0",
        finished_at="t1",
        total_cost_usd=run_cost,
        candidates=cands,
        judgments=judges,
    )


def test_cost_estimate_priors_from_history():
    arts = [_hist_art(0.25 + i * 0.01) for i in range(5)]
    priors = cost_estimate_priors_from_artifacts(
        arts, scored_keys=12, openrouter_keys=3, min_samples=3
    )
    assert priors is not None
    assert priors["n_samples"] == 5
    assert 0.24 <= priors["run_cost_usd_typical"] <= 0.30
    # 3×5200 + 9×5100 → median 5100
    assert priors["judge_completion_tokens_typical"] == 5100
    assert priors["candidate_completion_tokens_typical"] == 1900


def test_estimate_calibrates_from_history_artifacts():
    cfg = load_models_config()
    case = load_case("caseC")
    case = case.model_copy(update={"stem": "Chest pain vignette. " * 30})
    gold = "Diagnosis ACS. Plan cath. " * 20
    history = [_hist_art(0.27) for _ in range(4)]
    bd = estimate_cost_breakdown(
        cfg,
        case,
        include_qvac=True,
        gold_reference=gold,
        n=10,
        triple_qvac=True,
        include_local_peers=True,
        include_medical_peers=True,
        history_artifacts=history,
        include_extractor=False,
        extraction_cost_usd=0.0,
    )
    assert bd["calibrated"] is True
    assert bd["estimate_source"] == "history_calibrated"
    # 10 × ~$0.27 ≈ $2.7 — not the old ~$5–18 max_tokens fantasy.
    assert 2.0 <= float(bd["total_usd_for_n"]) <= 4.0
    assert float(bd["total_usd_upper_for_n"]) >= float(bd["total_usd_for_n"])
    assert bd["judge"]["completion_tokens_per_call"] < 16384
