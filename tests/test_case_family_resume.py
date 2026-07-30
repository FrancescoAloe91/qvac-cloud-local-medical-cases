"""Case-family history resume: restore exact confirmed gold, never merge splits."""

from __future__ import annotations

from pathlib import Path

from benchmark.gold import (
    case_family_key,
    cohort_id,
    confirmed_gold,
    gold_json,
    load_confirmed_gold,
    parse_extraction,
)
from benchmark.report import (
    find_case_family_cohorts,
    rebuild_multi_from_history,
    write_artifact,
)
from benchmark.schema import GoldClaim, RunArtifact


STEM = "45yo with thunderclap headache after exertion"
RAW = (
    "Diagnosis is subarachnoid hemorrhage until proven otherwise. "
    "Order non-contrast CT head then LP if CT negative. "
    "Urgency is emergent. Avoid lumbar puncture before imaging rules out mass. "
    "Start BP control and neurosurgery consult."
)


def _sections_a():
    source = {
        "diagnosis": "Diagnosis is subarachnoid hemorrhage until proven otherwise.",
        "tests": "Order non-contrast CT head then LP if CT negative.",
        "urgency": "Urgency is emergent.",
        "safety": "Avoid lumbar puncture before imaging rules out mass.",
        "plan": "Start BP control and neurosurgery consult.",
    }
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
            for sid, quote in source.items()
        }
    }
    return parse_extraction(RAW, payload)


def _sections_b_split():
    """Same raw reference, different claim split on diagnosis (two quotes)."""
    sections = _sections_a()
    # Split diagnosis into two contiguous substrings of RAW.
    q1 = "Diagnosis is subarachnoid hemorrhage"
    q2 = "until proven otherwise."
    assert q1 in RAW and q2 in RAW
    sections["diagnosis"].summary = "SAH until proven otherwise"
    sections["diagnosis"].claims = [
        GoldClaim(id="diagnosis-1", text=q1, source_quote=q1, critical=False),
        GoldClaim(id="diagnosis-2", text=q2, source_quote=q2, critical=False),
    ]
    return sections


def _gold(sections, *, cost: float = 0.01):
    return confirmed_gold(
        raw_text=RAW,
        sections=sections,
        extraction_model="test/extractor",
        extraction_cost_usd=cost,
    )


def _model_cfg(gold):
    return {
        "candidates": [{"key": "chatgpt", "model": "openai/gpt-test"}],
        "judge": {"model": "deepseek/deepseek-r1"},
    }


def _write_run(
    out_dir: Path,
    *,
    run_id: str,
    gold,
    finished_at: str,
    ranking_acc: float = 80.0,
) -> str:
    cid = cohort_id(
        case_stem=STEM,
        gold=gold,
        prompt_version="gold-only-v1",
        model_config=_model_cfg(gold),
        benchmark_track="controlled",
    )
    art = RunArtifact(
        run_id=run_id,
        case_id="caseC",
        started_at=finished_at,
        finished_at=finished_at,
        n_index=1,
        batch_id="batch-fam",
        models_config={
            "candidates": _model_cfg(gold)["candidates"],
            "judge": _model_cfg(gold)["judge"],
            "gold_reference": gold_json(gold),
            "case_stem": STEM,
        },
        ranking=[
            {
                "key": "chatgpt",
                "accuracy": ranking_acc,
                "status": "ok",
                "rank": 1,
                "coverage": 70.0,
                "quality": 80.0,
                "discipline": 90.0,
            }
        ],
        judgments=[],
        cohort_id=cid,
        scoring_version="graded-clinical-v4",
        prompt_version="gold-only-v1",
        benchmark_track="controlled",
    )
    write_artifact(art, out_dir)
    return cid


def test_case_family_key_ignores_whitespace_and_case():
    a = case_family_key(case_stem=STEM, reference_raw=RAW)
    b = case_family_key(
        case_stem="  " + STEM.upper() + "\n",
        reference_raw=RAW.replace("  ", " "),
    )
    # RAW has no double spaces; use expanded whitespace variant
    c = case_family_key(
        case_stem=STEM,
        reference_raw="  " + RAW.replace(" ", "  ") + "\n",
    )
    assert a == b == c


def test_same_family_different_claim_splits_list_separate_cohorts(tmp_path: Path):
    gold_a = _gold(_sections_a(), cost=0.01)
    gold_b = _gold(_sections_b_split(), cost=0.02)
    cid_a = _write_run(
        tmp_path, run_id="run-a1", gold=gold_a, finished_at="2026-01-01T10:00:00Z"
    )
    _write_run(
        tmp_path, run_id="run-a2", gold=gold_a, finished_at="2026-01-01T11:00:00Z"
    )
    cid_b = _write_run(
        tmp_path, run_id="run-b1", gold=gold_b, finished_at="2026-01-02T10:00:00Z"
    )
    assert cid_a != cid_b

    family = case_family_key(case_stem=STEM, reference_raw=RAW)
    rows = find_case_family_cohorts(
        tmp_path, case_stem=STEM, reference_raw=RAW, case_id="caseC"
    )
    assert len(rows) == 2
    assert {r["cohort_id"] for r in rows} == {cid_a, cid_b}
    assert all(r["family_key"] == family for r in rows)
    by_id = {r["cohort_id"]: r for r in rows}
    assert by_id[cid_a]["run_count"] == 2
    assert by_id[cid_b]["run_count"] == 1
    # Newest cohort first
    assert rows[0]["cohort_id"] == cid_b


def test_restore_exact_gold_recovers_same_cohort_id(tmp_path: Path):
    gold_a = _gold(_sections_a(), cost=0.01)
    cid = _write_run(
        tmp_path, run_id="run-r1", gold=gold_a, finished_at="2026-01-03T10:00:00Z"
    )
    for i in range(2, 6):
        _write_run(
            tmp_path,
            run_id=f"run-r{i}",
            gold=gold_a,
            finished_at=f"2026-01-03T1{i}:00:00Z",
        )

    rows = find_case_family_cohorts(
        tmp_path, case_stem=STEM, reference_raw=RAW, case_id="caseC"
    )
    assert len(rows) == 1
    restored = load_confirmed_gold(rows[0]["gold_reference"])
    # New confirm timestamp must not change cohort identity.
    restored = restored.model_copy(update={"confirmed_at": "2099-01-01T00:00:00Z"})
    again = cohort_id(
        case_stem=STEM,
        gold=restored,
        prompt_version="gold-only-v1",
        model_config=_model_cfg(restored),
        benchmark_track="controlled",
    )
    assert again == cid

    built = rebuild_multi_from_history(
        tmp_path, "caseC", n=5, cohort_id=cid
    )
    assert built["ok"] is True
    assert built["n_used"] == 5
    assert built["cohort_id"] == cid


def test_rebuild_does_not_mix_family_cohorts(tmp_path: Path):
    gold_a = _gold(_sections_a(), cost=0.01)
    gold_b = _gold(_sections_b_split(), cost=0.02)
    cid_a = _write_run(
        tmp_path, run_id="mix-a1", gold=gold_a, finished_at="2026-01-01T10:00:00Z"
    )
    for i in range(2, 6):
        _write_run(
            tmp_path,
            run_id=f"mix-a{i}",
            gold=gold_a,
            finished_at=f"2026-01-01T1{i}:00:00Z",
        )
    cid_b = _write_run(
        tmp_path, run_id="mix-b1", gold=gold_b, finished_at="2026-01-05T10:00:00Z"
    )
    # Newest overall is B (1 run) — without cohort filter, rebuild uses that cohort.
    newest = rebuild_multi_from_history(tmp_path, "caseC", n=5)
    assert newest["ok"] is True
    assert newest["cohort_id"] == cid_b
    assert newest["available"] == 1
    assert newest["n_used"] == 1

    # Explicit restore of A pools the five A runs only.
    restored = rebuild_multi_from_history(
        tmp_path, "caseC", n=5, cohort_id=cid_a
    )
    assert restored["ok"] is True
    assert restored["n_used"] == 5
    assert restored["cohort_id"] == cid_a
