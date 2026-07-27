from __future__ import annotations

import json

import pytest

from benchmark.gold import (
    SECTION_IDS,
    cohort_id,
    confirmed_gold,
    extract_json_object,
    load_confirmed_gold,
    parse_extraction,
)
from benchmark.schema import GoldSection
from benchmark.scoring import (
    claim_correctness_score,
    evidence_discipline_score,
    graded_clinical_score,
)


RAW = (
    "Diagnosis is migraine. Order brain MRI. Urgency is moderate. "
    "Avoid triptans in this patient. Start preventive therapy."
)


def _payload():
    source = {
        "diagnosis": "Diagnosis is migraine.",
        "tests": "Order brain MRI.",
        "urgency": "Urgency is moderate.",
        "safety": "Avoid triptans in this patient.",
        "plan": "Start preventive therapy.",
    }
    return {
        "sections": {
            section: {
                "summary": quote,
                "claims": [
                    {
                        "id": f"{section}-1",
                        "text": quote,
                        "source_quote": quote,
                        "critical": section == "safety",
                    }
                ],
            }
            for section, quote in source.items()
        }
    }


def test_extractor_requires_verbatim_source_quotes():
    sections = parse_extraction(RAW, _payload())
    assert set(sections) == set(SECTION_IDS)
    assert sections["safety"].claims[0].critical is False
    assert (
        sections["diagnosis"].claims[0].text
        == sections["diagnosis"].claims[0].source_quote
    )

    bad = _payload()
    bad["sections"]["diagnosis"]["claims"][0]["source_quote"] = "Invented diagnosis"
    try:
        parse_extraction(RAW, bad)
    except ValueError as exc:
        assert "verbatim source quote" in str(exc)
    else:
        raise AssertionError("invented source quote was accepted")


def test_extractor_rejects_duplicate_source_quotes_across_sections():
    payload = _payload()
    payload["sections"]["tests"]["claims"][0]["source_quote"] = (
        payload["sections"]["diagnosis"]["claims"][0]["source_quote"]
    )

    with pytest.raises(ValueError, match="Duplicate source quote"):
        parse_extraction(RAW, payload)


def test_loading_existing_gold_relocks_text_to_source_quote():
    gold = confirmed_gold(
        raw_text=RAW,
        sections=parse_extraction(RAW, _payload()),
        extraction_model="test",
    )
    gold.sections["diagnosis"].claims[0].text = "rewritten diagnosis"
    gold.sections["diagnosis"].claims[0].critical = True

    loaded = load_confirmed_gold(gold)

    claim = loaded.sections["diagnosis"].claims[0]
    assert claim.text == claim.source_quote
    assert claim.critical is False


def test_extractor_accepts_markdown_or_commentary_around_json():
    encoded = json.dumps(_payload())
    assert extract_json_object(f"```json\n{encoded}\n```") == _payload()
    assert extract_json_object(f"Prepared reference:\n{encoded}\nDone.") == _payload()


def test_extractor_accepts_sections_without_outer_wrapper():
    sections = _payload()["sections"]
    parsed = parse_extraction(RAW, sections)
    assert set(parsed) == set(SECTION_IDS)


def test_confirmed_gold_requires_all_five_nonempty_sections():
    sections = parse_extraction(RAW, _payload())
    sections["plan"] = GoldSection(summary="", claims=[])
    try:
        confirmed_gold(raw_text=RAW, sections=sections, extraction_model="test")
    except ValueError as exc:
        assert "plan" in str(exc)
    else:
        raise AssertionError("incomplete gold was accepted")


def test_claim_score_penalizes_unsupported_and_contradictions():
    perfect, p0, r0 = claim_correctness_score(matched=4, total_reference=4)
    unsupported, p1, r1 = claim_correctness_score(
        matched=4, total_reference=4, unsupported=1
    )
    contradicted, p2, r2 = claim_correctness_score(
        matched=4, total_reference=4, contradictions=1
    )
    assert perfect == 100.0
    assert unsupported < perfect
    assert contradicted < unsupported
    assert r0 == r1 == r2 == 1.0
    assert p0 > p1 == p2


def test_balanced_claim_score_uses_coverage_quality_and_precision():
    score, precision, recall = claim_correctness_score(
        matched=8,
        total_reference=10,
        unsupported=2,
        quality=0.9,
    )
    assert precision == 0.8
    assert recall == 0.8
    assert score == 82.5


def test_graded_score_treats_helpful_and_neutral_additions_proportionally():
    additions = [
        {"classification": "helpful", "severity": 1.0},
        {"classification": "neutral", "severity": 1.0},
        {"classification": "unsupported", "severity": 0.4},
    ]
    discipline = evidence_discipline_score(additions, total_reference=4)
    assert discipline == 0.9
    assert graded_clinical_score(
        coverage=0.8,
        quality=0.9,
        discipline=discipline,
    ) == 85.0


@pytest.mark.parametrize("classification", ["unsupported", "contradictory", "dangerous"])
def test_discipline_penalty_is_invariant_to_reference_claim_count(classification):
    additions = [{"classification": classification, "severity": 0.6}]
    assert evidence_discipline_score(
        additions, total_reference=1
    ) == evidence_discipline_score(additions, total_reference=20)


def test_cohort_changes_with_protocol_or_reference():
    gold = confirmed_gold(
        raw_text=RAW,
        sections=parse_extraction(RAW, _payload()),
        extraction_model="test",
    )
    common = dict(
        case_stem="A case",
        gold=gold,
        prompt_version="v1",
        model_config={"models": ["a"]},
    )
    controlled = cohort_id(**common, benchmark_track="controlled")
    native = cohort_id(**common, benchmark_track="native_defaults")
    assert controlled != native

