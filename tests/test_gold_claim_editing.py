"""Gold claim editing + scoring-contract cohort identity."""

from __future__ import annotations

from benchmark.gold import (
    assign_deterministic_claim_ids,
    cohort_id,
    confirmed_gold,
    scoring_contract_dump,
)
from benchmark.schema import GoldClaim, GoldSection


def _sections(raw: str, *, extra_diag: str | None = None, summary_suffix: str = ""):
    bits = {
        "diagnosis": ["Diagnosis Alpha."],
        "tests": ["Tests Beta."],
        "urgency": ["Urgency Gamma."],
        "safety": ["Safety Delta."],
        "plan": ["Plan Epsilon."],
    }
    if extra_diag:
        bits["diagnosis"].append(extra_diag)
    out = {}
    for sid, quotes in bits.items():
        out[sid] = GoldSection(
            summary=f"{sid} summary{summary_suffix}",
            claims=[
                GoldClaim(id=f"{sid}-{i}", text=q, source_quote=q)
                for i, q in enumerate(quotes, 1)
            ],
        )
    assert all(q in raw for qs in bits.values() for q in qs)
    return out


def test_summary_change_does_not_change_cohort_id():
    raw = (
        "Diagnosis Alpha. Extra claim here. Tests Beta. Urgency Gamma. "
        "Safety Delta. Plan Epsilon."
    )
    g1 = confirmed_gold(
        raw_text=raw,
        sections=_sections(raw, summary_suffix=" A"),
        extraction_model="openai/gpt-4o-mini",
        extraction_cost_usd=0.01,
    )
    g2 = confirmed_gold(
        raw_text=raw,
        sections=_sections(raw, summary_suffix=" B"),
        extraction_model="openai/gpt-4o-mini",
        extraction_cost_usd=0.99,
    )
    cid = dict(
        case_stem="case",
        prompt_version="gold-only-v1",
        model_config={"candidates": [{"key": "a"}]},
        benchmark_track="controlled",
    )
    assert cohort_id(gold=g1, **cid) == cohort_id(gold=g2, **cid)
    assert "summary" not in scoring_contract_dump(g1)["sections"]["diagnosis"]


def test_add_claim_changes_cohort_and_ids_deterministic():
    raw = (
        "Diagnosis Alpha. Extra claim here. Tests Beta. Urgency Gamma. "
        "Safety Delta. Plan Epsilon."
    )
    base = _sections(raw)
    with_extra = _sections(raw, extra_diag="Extra claim here.")
    g1 = confirmed_gold(
        raw_text=raw, sections=base, extraction_model="m", extraction_cost_usd=0
    )
    g2 = confirmed_gold(
        raw_text=raw, sections=with_extra, extraction_model="m", extraction_cost_usd=0
    )
    cid = dict(
        case_stem="case",
        prompt_version="gold-only-v1",
        model_config={"x": 1},
        benchmark_track="controlled",
    )
    assert cohort_id(gold=g1, **cid) != cohort_id(gold=g2, **cid)
    assert [c.id for c in g2.sections["diagnosis"].claims] == [
        "diagnosis-1",
        "diagnosis-2",
    ]


def test_empty_section_rejected():
    raw = (
        "Diagnosis Alpha. Tests Beta. Urgency Gamma. Safety Delta. Plan Epsilon."
    )
    secs = _sections(raw)
    secs["diagnosis"] = GoldSection(summary="diagnosis summary", claims=[])
    try:
        confirmed_gold(raw_text=raw, sections=secs, extraction_model="m")
        assert False, "expected empty diagnosis to fail"
    except ValueError:
        pass


def test_assign_deterministic_claim_ids_renumbers():
    section = GoldSection(
        summary="tests summary",
        claims=[
            GoldClaim(id="x", text="Tests Beta.", source_quote="Tests Beta."),
            GoldClaim(id="y", text="Urgency Gamma.", source_quote="Urgency Gamma."),
        ],
    )
    out = assign_deterministic_claim_ids(
        {
            "diagnosis": GoldSection(
                summary="d",
                claims=[
                    GoldClaim(
                        id="a", text="Diagnosis Alpha.", source_quote="Diagnosis Alpha."
                    )
                ],
            ),
            "tests": section,
            "urgency": GoldSection(
                summary="u",
                claims=[
                    GoldClaim(
                        id="b", text="Safety Delta.", source_quote="Safety Delta."
                    )
                ],
            ),
            "safety": GoldSection(
                summary="s",
                claims=[
                    GoldClaim(
                        id="c", text="Plan Epsilon.", source_quote="Plan Epsilon."
                    )
                ],
            ),
            "plan": GoldSection(
                summary="p",
                claims=[
                    GoldClaim(
                        id="d", text="Diagnosis Alpha.", source_quote="Diagnosis Alpha."
                    )
                ],
            ),
        }
    )
    assert [c.id for c in out["tests"].claims] == ["tests-1", "tests-2"]
