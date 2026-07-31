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


def test_case6_default_pack_gold_prepares_via_local_qna():
    """Seeded Case 6 Q1–A5 gold must Prepare without OpenRouter (public UX P0)."""
    from benchmark.case_slots import load_default_pack
    from benchmark.gold import (
        LOCAL_QNA_EXTRACTOR_MODEL,
        confirmed_gold,
        extract_with_chat,
        looks_like_qna_reference,
        try_extract_qna_sections,
    )

    pack = load_default_pack()
    gold = pack[6]["gold_raw"]
    assert "Q1 [diagnosis]:" in gold and "A5:" in gold
    assert looks_like_qna_reference(gold)

    sections = try_extract_qna_sections(gold)
    assert sections is not None
    for sid in SECTION_IDS:
        assert sections[sid].claims, sid
        assert sections[sid].claims[0].source_quote in gold

    def boom(*_a, **_k):
        raise AssertionError("OpenRouter chat must not run for Q1–A5 Case 6 gold")

    out, meta = extract_with_chat(gold, model="openai/gpt-4o-mini", chat=boom)
    assert meta.model == LOCAL_QNA_EXTRACTOR_MODEL
    assert float(getattr(meta, "cost_usd", 0.0) or 0.0) == 0.0
    confirmed_gold(
        raw_text=gold, sections=out, extraction_model=LOCAL_QNA_EXTRACTOR_MODEL
    )


def test_case7_and_pack_qna_golds_prepare_locally():
    from benchmark.case_slots import load_default_pack
    from benchmark.gold import extract_with_chat, try_extract_qna_sections

    pack = load_default_pack()

    def boom(*_a, **_k):
        raise AssertionError("chat must not run for pack Q1–A5 gold")

    for slot in (2, 3, 4, 5, 6, 7):
        gold = pack[slot]["gold_raw"]
        assert try_extract_qna_sections(gold) is not None, slot
        sections, meta = extract_with_chat(gold, model="x", chat=boom)
        assert all(sections[sid].claims for sid in SECTION_IDS), slot
        assert str(meta.model).startswith("local/qna")


def test_drop_invalid_claims_removes_nested_overlapping_quotes():
    raw = (
        "Diagnosis is migraine with aura. Order brain MRI. Urgency is moderate. "
        "Avoid triptans in this patient. Start preventive therapy."
    )
    payload = {
        "sections": {
            "diagnosis": {
                "summary": "dx",
                "claims": [
                    {
                        "id": "diagnosis-1",
                        "text": "Diagnosis is migraine with aura.",
                        "source_quote": "Diagnosis is migraine with aura.",
                        "critical": False,
                    },
                    {
                        "id": "diagnosis-2",
                        "text": "migraine with aura",
                        "source_quote": "migraine with aura.",
                        "critical": False,
                    },
                ],
            },
            "tests": {
                "summary": "t",
                "claims": [
                    {
                        "id": "tests-1",
                        "text": "Order brain MRI.",
                        "source_quote": "Order brain MRI.",
                        "critical": False,
                    }
                ],
            },
            "urgency": {
                "summary": "u",
                "claims": [
                    {
                        "id": "urgency-1",
                        "text": "Urgency is moderate.",
                        "source_quote": "Urgency is moderate.",
                        "critical": False,
                    }
                ],
            },
            "safety": {
                "summary": "s",
                "claims": [
                    {
                        "id": "safety-1",
                        "text": "Avoid triptans in this patient.",
                        "source_quote": "Avoid triptans in this patient.",
                        "critical": False,
                    }
                ],
            },
            "plan": {
                "summary": "p",
                "claims": [
                    {
                        "id": "plan-1",
                        "text": "Start preventive therapy.",
                        "source_quote": "Start preventive therapy.",
                        "critical": False,
                    }
                ],
            },
        }
    }
    with pytest.raises(ValueError, match="Overlapping"):
        parse_extraction(raw, payload)
    sections = parse_extraction(raw, payload, drop_invalid_claims=True)
    assert [c.id for c in sections["diagnosis"].claims] == ["diagnosis-1"]


def test_extract_with_chat_recovers_from_nested_overlapping_quotes():
    from types import SimpleNamespace

    from benchmark.gold import extract_with_chat

    raw = (
        "Diagnosis is migraine with aura. Order brain MRI. Urgency is moderate. "
        "Avoid triptans in this patient. Start preventive therapy."
    )
    payload = {
        "sections": {
            sid: {
                "summary": quote,
                "claims": [
                    {
                        "id": f"{sid}-1",
                        "text": quote,
                        "source_quote": quote,
                        "critical": False,
                    }
                ],
            }
            for sid, quote in {
                "diagnosis": "Diagnosis is migraine with aura.",
                "tests": "Order brain MRI.",
                "urgency": "Urgency is moderate.",
                "safety": "Avoid triptans in this patient.",
                "plan": "Start preventive therapy.",
            }.items()
        }
    }
    # Nested overlap that previously left Prepare stuck red after quote-repair.
    payload["sections"]["diagnosis"]["claims"].append(
        {
            "id": "diagnosis-2",
            "text": "migraine with aura",
            "source_quote": "migraine with aura.",
            "critical": False,
        }
    )
    calls = {"n": 0}

    def chat(model, messages, **kwargs):
        calls["n"] += 1
        meta = SimpleNamespace(
            error=None, cost_usd=0.0, prompt_tokens=1, completion_tokens=1
        )
        return json.dumps(payload), meta

    sections, _meta = extract_with_chat(raw, model="test/model", chat=chat)
    assert calls["n"] == 2  # primary + repair, then drop nested claim
    assert [c.id for c in sections["diagnosis"].claims] == ["diagnosis-1"]


def test_format_prepare_error_is_actionable():
    from benchmark.gold import format_prepare_error

    msg = format_prepare_error(ValueError("Overlapping source quote across scoring claims: x"))
    assert "Prepare reference" in msg or "retry" in msg.casefold()
    key_msg = format_prepare_error(ValueError("An OpenRouter key is required"))
    assert "sk-or-v1" in key_msg or "OpenRouter" in key_msg

