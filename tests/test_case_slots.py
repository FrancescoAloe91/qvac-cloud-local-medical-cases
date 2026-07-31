"""Case slot mapping (1–5) + mean-scope helpers."""

from __future__ import annotations

from pathlib import Path

from benchmark.case_slots import (
    MAX_CASE_SLOTS,
    bind_stem_to_slot,
    count_distinct_stem_keys,
    discover_stem_families,
    ensure_owner_slots,
    filter_artifacts_for_slot,
    load_bindings,
    migrate_bindings,
    next_empty_slot,
    resolve_slots,
    save_bindings,
    slot_for_stem_key,
    slot_label_for_artifact,
    stem_key,
)
from benchmark.schema import RunArtifact


def _art(
    *,
    run_id: str,
    stem: str,
    finished_at: str,
    cohort_id: str = "",
    gold: str = "",
) -> RunArtifact:
    return RunArtifact(
        run_id=run_id,
        case_id="caseC",
        started_at=finished_at,
        finished_at=finished_at,
        cohort_id=cohort_id,
        models_config={
            "case_stem": stem,
            "gold_reference": gold,
            "candidates": [{"key": "chatgpt", "model": "openai/gpt-test"}],
            "judge": {"model": "deepseek/deepseek-r1"},
        },
        ranking=[
            {
                "key": "chatgpt",
                "accuracy": 80.0,
                "status": "ok",
                "rank": 1,
                "coverage": 70.0,
                "quality": 80.0,
                "discipline": 90.0,
            }
        ],
        judgments=[],
        scoring_version="graded-clinical-v4",
        prompt_version="gold-only-v1",
        benchmark_track="controlled",
        run_status="complete",
    )


STEM_A = "Patient: 58-year-old male. History: hypertension and chest pain."
STEM_B = "Patient: 34-year-old female. History: progressive neurological symptoms."
STEM_C = "Patient: 35-year-old male. Post-op follow-up with fever."
GOLD_A = '{"raw_text":"Dx A","sections":{},"extraction_model":"m","confirmed_at":"t"}'
# Minimal valid-looking placeholder is not loadable as ConfirmedGold; tests that
# need restore validation use discover/migrate which only require non-empty gold
# string for has_protocol_cohort. Keep a non-empty marker.
GOLD_MARK = '{"marker":true}'


def test_stem_key_normalizes_whitespace_and_case():
    a = stem_key(STEM_A)
    b = stem_key("  " + STEM_A.upper().replace("  ", " ") + "\n")
    assert a == b
    assert a != stem_key(STEM_B)


def test_migrate_assigns_two_protocol_families_to_case_1_and_2():
    arts = [
        _art(
            run_id="a1",
            stem=STEM_A,
            finished_at="2026-07-31T10:00:00Z",
            cohort_id="cohort-a",
            gold=GOLD_MARK,
        ),
        _art(
            run_id="a2",
            stem=STEM_A,
            finished_at="2026-07-31T11:00:00Z",
            cohort_id="cohort-a",
            gold=GOLD_MARK,
        ),
        _art(
            run_id="b1",
            stem=STEM_B,
            finished_at="2026-07-30T09:00:00Z",
            cohort_id="cohort-b",
            gold=GOLD_MARK,
        ),
        # Older stem without cohort — must not steal Case 1/2
        _art(
            run_id="c1",
            stem=STEM_C,
            finished_at="2026-07-20T09:00:00Z",
            gold=GOLD_MARK,
        ),
    ]
    bindings = migrate_bindings(arts)
    assert bindings[1] == stem_key(STEM_A)
    assert bindings[2] == stem_key(STEM_B)
    assert 3 not in bindings  # legacy non-protocol must not steal Case 3–5
    slots = resolve_slots(arts, bindings)
    assert slots[0].filled and slots[0].run_count == 2
    assert slots[1].filled and slots[1].stem == STEM_B
    assert not slots[2].filled and not slots[3].filled and not slots[4].filled


def test_migrate_preserves_existing_bindings_order():
    arts = [
        _art(
            run_id="a1",
            stem=STEM_A,
            finished_at="2026-07-31T12:00:00Z",
            cohort_id="ca",
            gold=GOLD_MARK,
        ),
        _art(
            run_id="b1",
            stem=STEM_B,
            finished_at="2026-07-31T13:00:00Z",
            cohort_id="cb",
            gold=GOLD_MARK,
        ),
    ]
    # Force B into Case 1 even though A is older — stickiness wins.
    existing = {1: stem_key(STEM_B), 2: stem_key(STEM_A)}
    bindings = migrate_bindings(arts, existing)
    assert bindings[1] == stem_key(STEM_B)
    assert bindings[2] == stem_key(STEM_A)


def test_next_empty_slot_and_bind_cap_at_five(tmp_path: Path):
    arts = [
        _art(
            run_id=f"r{i}",
            stem=f"Stem number {i} with enough text",
            finished_at=f"2026-07-0{i}T10:00:00Z",
            cohort_id=f"c{i}",
            gold=GOLD_MARK,
        )
        for i in range(1, 6)
    ]
    slots, bindings = ensure_owner_slots(tmp_path, arts)
    assert next_empty_slot(slots) is None
    assert len(bindings) == MAX_CASE_SLOTS
    # Binding a 6th distinct stem onto an occupied slot keeps prior stem if
    # the new stem is already mapped elsewhere — and does not create slot 6.
    out = bind_stem_to_slot(bindings, slot_index=1, case_stem="Brand new sixth case")
    assert set(out.keys()) <= set(range(1, 6))
    assert 6 not in out


def test_bind_stem_to_empty_slot_for_new_case(tmp_path: Path):
    arts = [
        _art(
            run_id="a1",
            stem=STEM_A,
            finished_at="2026-07-31T10:00:00Z",
            cohort_id="ca",
            gold=GOLD_MARK,
        ),
        _art(
            run_id="b1",
            stem=STEM_B,
            finished_at="2026-07-30T10:00:00Z",
            cohort_id="cb",
            gold=GOLD_MARK,
        ),
    ]
    slots, bindings = ensure_owner_slots(tmp_path, arts)
    assert next_empty_slot(slots) == 3
    bindings = bind_stem_to_slot(bindings, slot_index=3, case_stem=STEM_C)
    save_bindings(tmp_path, bindings)
    loaded = load_bindings(tmp_path)
    assert loaded[3] == stem_key(STEM_C)
    assert slot_for_stem_key(loaded, stem_key(STEM_A)) == 1


def test_filter_artifacts_for_selected_case_mean_scope():
    arts = [
        _art(
            run_id="a1",
            stem=STEM_A,
            finished_at="2026-07-31T10:00:00Z",
            cohort_id="ca",
            gold=GOLD_MARK,
        ),
        _art(
            run_id="b1",
            stem=STEM_B,
            finished_at="2026-07-30T10:00:00Z",
            cohort_id="cb",
            gold=GOLD_MARK,
        ),
    ]
    slots = resolve_slots(arts, migrate_bindings(arts))
    only_a = filter_artifacts_for_slot(arts, slots[0])
    assert [a.run_id for a in only_a] == ["a1"]
    assert count_distinct_stem_keys(arts) == 2
    assert slot_label_for_artifact(arts[1], migrate_bindings(arts)) == "Case 2"


def test_discover_prefers_newest_gold_and_cohort():
    arts = [
        _art(
            run_id="old",
            stem=STEM_A,
            finished_at="2026-07-01T10:00:00Z",
            cohort_id="old-cohort",
            gold="old-gold",
        ),
        _art(
            run_id="new",
            stem=STEM_A,
            finished_at="2026-07-31T10:00:00Z",
            cohort_id="new-cohort",
            gold="new-gold",
        ),
    ]
    fams = discover_stem_families(arts)
    assert len(fams) == 1
    assert fams[0].cohort_id == "new-cohort"
    assert fams[0].gold_reference == "new-gold"
    assert fams[0].run_count == 2
