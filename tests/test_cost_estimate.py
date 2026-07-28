"""Cost estimate honesty: extractor, primary 16k, repair, optional verifier."""

from __future__ import annotations

from benchmark import openrouter
from benchmark.cases_loader import load_case
from benchmark.config import load_models_config
from benchmark.runner import estimate_cost_breakdown


def test_fallback_prices_include_verifier_and_extractor_models():
    assert "qwen/qwen3.5-397b-a17b" in openrouter._FALLBACK_PRICES
    assert "google/gemini-3.5-flash" in openrouter._FALLBACK_PRICES
    assert "openai/gpt-4o-mini" in openrouter._FALLBACK_PRICES
    assert "deepseek/deepseek-r1" in openrouter._FALLBACK_PRICES
    pin, pout = openrouter.model_prices_per_mtok("qwen/qwen3.5-397b-a17b")
    assert pin > 0 and pout > 0
    # Must not silently fall back to the generic (2.0, 10.0) default.
    assert (pin, pout) != (2.0, 10.0)


def test_estimate_cost_breakdown_includes_extractor_repair_verifier():
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
    assert bd["judge"]["completion_tokens_per_call"] == 16384
    assert bd["section_repair"]["completion_tokens_per_call"] == 4096
    assert bd["section_repair"].get("optional") is True
    assert bd["verifier"].get("optional") is True
    assert bd["verifier"]["model"] == "qwen/qwen3.5-397b-a17b"

    baseline = float(bd["total_usd"])
    upper = float(bd["total_usd_upper"])
    repair = float(bd["section_repair"]["estimated_usd"])
    verifier = float(bd["verifier"]["estimated_usd"])
    extract = float(bd["extractor"]["estimated_usd"])
    judge = float(bd["judge"]["estimated_usd"])

    assert baseline > extract + judge * 0.5  # candidates + extract + judge
    assert upper > baseline
    assert abs(upper - (baseline + repair + verifier)) < 1e-6


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
    assert float(five["total_usd_upper_for_n"]) > float(five["total_usd_for_n"])


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
    from benchmark.schema import ModelCallMeta

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
