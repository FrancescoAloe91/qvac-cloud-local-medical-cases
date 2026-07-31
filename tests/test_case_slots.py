"""Case slot mapping (base 1–5 + grow 6+) + mean-scope helpers."""

from __future__ import annotations

from pathlib import Path

from benchmark.case_slots import (
    BASE_CASE_SLOTS,
    SOFT_MAX_CASE_SLOTS,
    apply_default_pack_to_empty_slots,
    bind_stem_to_slot,
    count_distinct_stem_keys,
    discover_stem_families,
    ensure_owner_slots,
    filter_artifacts_for_slot,
    load_bindings,
    load_default_pack,
    load_drafts,
    load_slot_state,
    migrate_bindings,
    next_empty_slot,
    open_new_case_slot,
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


def test_base_full_new_case_grows_to_six(tmp_path: Path):
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
    slots, bindings, slot_count, _drafts = ensure_owner_slots(
        tmp_path, arts, apply_defaults=False
    )
    assert next_empty_slot(slots) is None
    assert slot_count == BASE_CASE_SLOTS
    assert len(bindings) == BASE_CASE_SLOTS
    new_idx, new_count = open_new_case_slot(slots, slot_count=slot_count)
    assert new_idx == 6
    assert new_count == 6
    save_bindings(tmp_path, bindings, slot_count=new_count)
    slots6, bindings6, count6, _ = ensure_owner_slots(
        tmp_path, arts, session_slot_count=new_count, apply_defaults=False
    )
    assert count6 == 6
    assert len(slots6) == 6
    assert not slots6[5].filled
    assert slots6[5].index == 6
    # Binding a 6th distinct stem onto Case 6 is allowed.
    out = bind_stem_to_slot(
        bindings6, slot_index=6, case_stem="Brand new sixth case stem text"
    )
    assert out[6] == stem_key("Brand new sixth case stem text")
    assert 6 in out


def test_new_case_prefers_empty_base_before_growing(tmp_path: Path):
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
    slots, _bindings, slot_count, _ = ensure_owner_slots(
        tmp_path, arts, apply_defaults=False
    )
    assert next_empty_slot(slots) == 3
    idx, count = open_new_case_slot(slots, slot_count=slot_count)
    assert idx == 3
    assert count == BASE_CASE_SLOTS


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
    slots, bindings, _count, _ = ensure_owner_slots(
        tmp_path, arts, apply_defaults=False
    )
    assert next_empty_slot(slots) == 3
    bindings = bind_stem_to_slot(bindings, slot_index=3, case_stem=STEM_C)
    save_bindings(tmp_path, bindings, slot_count=BASE_CASE_SLOTS)
    loaded = load_bindings(tmp_path)
    assert loaded[3] == stem_key(STEM_C)
    assert slot_for_stem_key(loaded, stem_key(STEM_A)) == 1


def test_slot_state_persists_grown_empty_case_six(tmp_path: Path):
    save_bindings(tmp_path, {1: stem_key(STEM_A)}, slot_count=6)
    bindings, count = load_slot_state(tmp_path)
    assert count == 6
    assert bindings[1] == stem_key(STEM_A)
    arts = [
        _art(
            run_id="a1",
            stem=STEM_A,
            finished_at="2026-07-31T10:00:00Z",
            cohort_id="ca",
            gold=GOLD_MARK,
        )
    ]
    slots, _, count2, _ = ensure_owner_slots(
        tmp_path, arts, apply_defaults=False
    )
    assert count2 == 6
    assert [s.index for s in slots] == list(range(1, 7))
    assert slots[5].index == 6 and not slots[5].filled


def test_open_new_case_respects_soft_max():
    slots = resolve_slots([], {}, slot_count=SOFT_MAX_CASE_SLOTS)
    # Fill every slot with a fake binding so none are empty.
    bindings = {i: f"k{i:02d}{'x' * 20}" for i in range(1, SOFT_MAX_CASE_SLOTS + 1)}
    slots = resolve_slots([], bindings, slot_count=SOFT_MAX_CASE_SLOTS)
    assert next_empty_slot(slots) is None
    try:
        open_new_case_slot(slots, slot_count=SOFT_MAX_CASE_SLOTS)
        raised = False
    except ValueError:
        raised = True
    assert raised


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


def test_migrate_does_not_auto_fill_slot_six():
    arts = [
        _art(
            run_id=f"r{i}",
            stem=f"Protocol stem number {i} long enough",
            finished_at=f"2026-07-{10 + i:02d}T10:00:00Z",
            cohort_id=f"c{i}",
            gold=GOLD_MARK,
        )
        for i in range(1, 8)
    ]
    # Even with slot_count=7, auto-migrate only fills base 1–5.
    bindings = migrate_bindings(arts, slot_count=7)
    assert set(bindings.keys()) == {1, 2, 3, 4, 5}
    assert 6 not in bindings and 7 not in bindings


def test_default_pack_loads_stemi_dka_stroke_qna():
    pack = load_default_pack()
    assert set(pack.keys()) == {3, 4, 5}
    assert "STEMI" in pack[3]["stem"] or "ST-segment elevation" in pack[3]["stem"]
    assert "Q1 [diagnosis]:" in pack[3]["gold_raw"]
    assert "A5:" in pack[3]["gold_raw"]
    assert "ketoacidosis" in pack[4]["gold_raw"].lower() or "DKA" in pack[4]["stem"]
    assert "stroke" in pack[5]["gold_raw"].lower()
    assert "Q5 [plan]:" in pack[5]["gold_raw"]


def test_default_pack_fills_empty_slots_only_preserves_case_1_2(tmp_path: Path):
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
    slots, bindings, _count, drafts = ensure_owner_slots(tmp_path, arts)
    assert bindings[1] == stem_key(STEM_A)
    assert bindings[2] == stem_key(STEM_B)
    assert 3 in bindings and 4 in bindings and 5 in bindings
    assert slots[2].filled and "ST-segment elevation" in slots[2].stem
    assert "Q1 [diagnosis]:" in slots[2].gold_raw
    assert "Q1 [diagnosis]:" in slots[3].gold_raw
    assert "Q1 [diagnosis]:" in slots[4].gold_raw
    assert slots[0].stem == STEM_A
    assert slots[1].stem == STEM_B
    # Drafts persisted for owner workspace.
    loaded = load_drafts(tmp_path)
    assert set(loaded.keys()) >= {3, 4, 5}
    assert "A1:" in loaded[3]["gold_raw"]


def test_apply_default_pack_does_not_overwrite_existing_binding():
    pack = load_default_pack()
    # Pretend Case 3 already owns an unrelated stem.
    existing_key = stem_key("Already bound unrelated stem for case three")
    bindings = {3: existing_key}
    drafts: dict[int, dict[str, str]] = {}
    out_b, out_d = apply_default_pack_to_empty_slots(bindings, drafts, pack=pack)
    assert out_b[3] == existing_key
    assert 3 not in out_d  # pack must not replace occupied slot
    assert 4 in out_b and 5 in out_b
    assert "Q1 [diagnosis]:" in out_d[4]["gold_raw"]


def test_resolve_slots_surfaces_draft_gold_without_artifact():
    pack = load_default_pack()
    bindings = {3: pack[3]["stem_key"]}
    drafts = {3: pack[3]}
    slots = resolve_slots([], bindings, drafts=drafts)
    assert slots[2].stem_key == pack[3]["stem_key"]
    assert slots[2].stem.startswith("Patient:")
    assert slots[2].gold_raw.startswith("Q1 [diagnosis]:")
    assert not slots[2].gold_reference
