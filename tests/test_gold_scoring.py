from __future__ import annotations

import json

import pytest

from benchmark.gold import (
    SECTION_IDS,
    cohort_id,
    confirmed_gold,
    extract_json_object,
    gold_json,
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
    bad["sections"]["diagnosis"]["claims"][0]["text"] = "Invented diagnosis"
    try:
        parse_extraction(RAW, bad)
    except ValueError as exc:
        assert "verbatim source quote" in str(exc)
        assert "paraphrase not allowed" in str(exc)
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


def test_confirmed_and_loaded_gold_require_verbatim_source_quotes():
    sections = parse_extraction(RAW, _payload())
    gold = confirmed_gold(
        raw_text=RAW, sections=sections, extraction_model="test"
    )
    assert load_confirmed_gold(gold_json(gold)).raw_text == RAW.strip()

    bad = parse_extraction(RAW, _payload())
    bad["diagnosis"].claims[0].source_quote = "Invented migraine wording"
    with pytest.raises(ValueError, match="verbatim"):
        confirmed_gold(raw_text=RAW, sections=bad, extraction_model="test")

    loaded_bad = confirmed_gold(
        raw_text=RAW,
        sections=parse_extraction(RAW, _payload()),
        extraction_model="test",
    )
    loaded_bad.sections["diagnosis"].claims[0].source_quote = "Not in the raw text"
    with pytest.raises(ValueError, match="verbatim|source quote"):
        load_confirmed_gold(loaded_bad)


def test_overlapping_substantial_source_quotes_are_rejected():
    payload = _payload()
    payload["sections"]["tests"]["claims"][0]["source_quote"] = (
        "Diagnosis is migraine. Order brain MRI."
    )
    with pytest.raises(ValueError, match="Overlapping|verbatim|Duplicate"):
        parse_extraction(RAW, payload)


SHORT_MULTI = (
    "Dx: inferior STEMI. Order ECG. Check troponin. Urgency: immediate cath. "
    "Safety: no nitrates after sildenafil. Plan: activate cath lab."
)


def _short_multi_payload(*, bad_tests_2: bool = False):
    quotes = {
        "diagnosis": "Dx: inferior STEMI.",
        "urgency": "Urgency: immediate cath.",
        "safety": "Safety: no nitrates after sildenafil.",
        "plan": "Plan: activate cath lab.",
    }
    tests_claims = [
        {
            "id": "tests-1",
            "text": "Order ECG.",
            "source_quote": "Order ECG.",
            "critical": False,
        },
        {
            "id": "tests-2",
            "text": "Check troponin." if not bad_tests_2 else "serial troponins q3h",
            "source_quote": (
                "Check troponin." if not bad_tests_2 else "serial troponins q3h"
            ),
            "critical": False,
        },
    ]
    sections = {
        "tests": {"summary": "tests", "claims": tests_claims},
    }
    for section, quote in quotes.items():
        sections[section] = {
            "summary": quote,
            "claims": [
                {
                    "id": f"{section}-1",
                    "text": quote,
                    "source_quote": quote,
                    "critical": False,
                }
            ],
        }
    return {"sections": sections}


def test_short_multi_claim_reference_parses_when_quotes_verbatim():
    sections = parse_extraction(SHORT_MULTI, _short_multi_payload(bad_tests_2=False))
    assert len(sections["tests"].claims) == 2
    assert sections["tests"].claims[1].id == "tests-2"


def test_tests_2_paraphrase_is_rejected_with_clear_message():
    with pytest.raises(ValueError, match="tests-2.*paraphrase not allowed"):
        parse_extraction(SHORT_MULTI, _short_multi_payload(bad_tests_2=True))


def test_drop_invalid_claims_keeps_section_when_other_claims_ok():
    sections = parse_extraction(
        SHORT_MULTI,
        _short_multi_payload(bad_tests_2=True),
        drop_invalid_claims=True,
    )
    assert [c.id for c in sections["tests"].claims] == ["tests-1"]


def test_nfc_whitespace_normalize_accepts_matching_quote():
    raw = "Diagnosis is migraine.\n\nOrder brain MRI. Urgency is moderate. Avoid triptans in this patient. Start preventive therapy."
    payload = _payload()
    # Extra internal spaces + composed/compatible forms still match after NFC/ws fold.
    payload["sections"]["diagnosis"]["claims"][0]["source_quote"] = (
        "Diagnosis  is   migraine."
    )
    sections = parse_extraction(raw, payload)
    assert "migraine" in sections["diagnosis"].claims[0].source_quote.casefold()


def test_extract_with_chat_repairs_bad_tests_2_quote():
    from types import SimpleNamespace

    from benchmark.gold import extract_with_chat

    bad = _short_multi_payload(bad_tests_2=True)
    fixed = _short_multi_payload(bad_tests_2=False)
    calls = {"n": 0}

    def chat(model, messages, **kwargs):
        calls["n"] += 1
        meta = SimpleNamespace(
            error=None,
            cost_usd=0.01,
            prompt_tokens=10,
            completion_tokens=20,
        )
        if calls["n"] == 1:
            return json.dumps(bad), meta
        # Repair pass must see failure context and return fixed quotes.
        joined = " ".join(m["content"] for m in messages)
        assert "VALIDATION FAILURES" in joined or "quote" in joined.casefold()
        return json.dumps(fixed), meta

    sections, meta = extract_with_chat(
        SHORT_MULTI, model="test/model", chat=chat
    )
    assert calls["n"] == 2
    assert len(sections["tests"].claims) == 2
    assert meta.cost_usd in (0.01, 0.02) or getattr(meta, "cost_usd", None) == 0.01


def test_extract_with_chat_drops_unrepairable_claim():
    from types import SimpleNamespace

    from benchmark.gold import extract_with_chat

    bad = _short_multi_payload(bad_tests_2=True)
    # Repair still returns the paraphrase for tests-2.
    calls = {"n": 0}

    def chat(model, messages, **kwargs):
        calls["n"] += 1
        meta = SimpleNamespace(
            error=None,
            cost_usd=0.0,
            prompt_tokens=1,
            completion_tokens=1,
        )
        return json.dumps(bad), meta

    sections, _meta = extract_with_chat(
        SHORT_MULTI, model="test/model", chat=chat
    )
    assert calls["n"] == 2
    assert [c.id for c in sections["tests"].claims] == ["tests-1"]


def test_salvage_uses_claim_text_when_source_quote_paraphrased():
    payload = _payload()
    payload["sections"]["tests"]["claims"][0]["source_quote"] = "Get an MRI of the brain"
    payload["sections"]["tests"]["claims"][0]["text"] = "Order brain MRI."
    sections = parse_extraction(RAW, payload)
    assert sections["tests"].claims[0].source_quote == "Order brain MRI."

