"""Comprehension pack + protocol isolation from graded Rebuild."""

from __future__ import annotations

from pathlib import Path

from benchmark.beta_pack import (
    auto_freeze_beta_slot,
    beta_case_slot_of,
    count_beta_runs_by_slot,
    is_beta_artifact,
    list_beta_slots,
    load_beta_pack,
    merge_beta_slots,
    open_new_beta_case_slot,
    resolve_beta_gold_raw,
    synthetic_gold_raw_from_prose,
)
from benchmark.beta_prompts import (
    beta_candidate_system,
    beta_candidate_user,
    parse_beta_candidate_answers,
)
from benchmark.beta_protocol import CASE_ID, PROTOCOL_ID, SCORING_VERSION
from benchmark.cases_loader import load_case
from benchmark.gold import cohort_id, looks_like_qna_reference, try_extract_qna_sections
from benchmark.schema import ConfirmedGold, GoldClaim, GoldSection
from lib.charts import fig_judge_mean_accuracy_bars


def test_beta_pack_stems_and_prose():
    pack = load_beta_pack()
    assert pack.get("protocol_id") == PROTOCOL_ID
    assert int(pack.get("revision") or 0) >= 3
    slots = list_beta_slots(pack)
    assert len(slots) >= 10
    slot_ids = [int(s["slot"]) for s in slots]
    assert slot_ids[0] == 1
    assert slot_ids == list(range(1, len(slots) + 1))
    assert "AKI" in slots[0]["title"] or "hyperkalemia" in slots[0]["title"].lower()
    by_slot = {int(s["slot"]): s for s in slots}
    assert "septic" in by_slot[8]["title"].lower()
    assert "pe" in by_slot[9]["title"].lower() or "embolism" in by_slot[9]["title"].lower()
    assert "gi" in by_slot[10]["title"].lower() or "bleed" in by_slot[10]["title"].lower()
    for s in slots:
        assert s["stem"].strip()
        assert s["reference_prose"].strip()
        assert "Q1 [" not in s["reference_prose"]
        assert "A1:" not in s["reference_prose"]
        assert s["gold_raw"].strip()
        assert looks_like_qna_reference(s["gold_raw"])
        assert try_extract_qna_sections(s["gold_raw"]) is not None


def test_beta_prompts_are_free_form():
    sys_p = beta_candidate_system().lower()
    user_p = beta_candidate_user(stem="Patient: test").lower()
    assert "a1" not in sys_p or "mandatory" in sys_p
    assert "do not use" in sys_p and "a1" in sys_p
    assert "patient presentation" in user_p


def test_beta_scoring_version_differs_from_graded():
    assert SCORING_VERSION == "comprehension-v1"
    assert SCORING_VERSION != "graded-clinical-v4"


def test_mean_chart_includes_median_marker_and_dual_std():
    rows = [
        {
            "key": "chatgpt",
            "label": "ChatGPT",
            "model": "gpt",
            "eligible": True,
            "rank": 1,
            "accuracy_mean": 70.0,
            "std": 4.0,
            "cv_pct": 5.7,
            "n_runs": 10,
            "median": 71.0,
        }
    ]
    fig = fig_judge_mean_accuracy_bars(rows, hide_partial_labels=True)
    # bar + white outline whisker + black core whisker + median scatter
    assert len(fig.data) >= 4
    assert fig.data[0].type == "bar"
    assert getattr(fig.data[0].error_x, "array", None) in (None, ())
    # Dual whiskers on top of bars (white outline, black core).
    assert fig.data[1].type == "scatter"
    assert fig.data[2].type == "scatter"
    assert "255,255,255" in str(fig.data[1].error_x.color)
    assert "15,23,42" in str(fig.data[2].error_x.color)
    assert fig.data[-1].mode == "markers"
    title = (fig.layout.xaxis.title.text or "") if fig.layout.xaxis.title else ""
    assert "median" in title.lower() or "◆" in title


def test_mean_chart_rank_by_median_reorders_and_compacts():
    rows = [
        {
            "key": "chatgpt",
            "label": "ChatGPT",
            "model": "gpt",
            "eligible": True,
            "rank": 1,
            "accuracy_mean": 90.0,
            "std": 2.0,
            "cv_pct": 2.0,
            "n_runs": 30,
            "median": 70.0,
        },
        {
            "key": "claude",
            "label": "Claude",
            "model": "claude",
            "eligible": True,
            "rank": 2,
            "accuracy_mean": 80.0,
            "std": 3.0,
            "cv_pct": 4.0,
            "n_runs": 30,
            "median": 85.0,
        },
    ]
    by_mean = fig_judge_mean_accuracy_bars(
        rows, hide_partial_labels=True, height=280
    )
    by_med = fig_judge_mean_accuracy_bars(
        rows,
        hide_partial_labels=True,
        rank_by="median",
        compact=True,
        height=160,
    )
    # Plotly y is reversed (top = last); top bar label should follow rank order.
    mean_top = by_mean.data[1].y[-1]
    med_top = by_med.data[1].y[-1]
    assert "ChatGPT" in str(mean_top) or "OpenAI" in str(mean_top)
    assert "Claude" in str(med_top) or "Anthropic" in str(med_top)
    assert int(by_med.layout.height or 0) <= 160
    assert int(by_med.layout.height or 0) < int(by_mean.layout.height or 0)
    axis = (by_med.layout.xaxis.title.text or "") if by_med.layout.xaxis.title else ""
    assert "ranked by median" in axis.lower()


def test_comprehension_home_and_structured_page_exist():
    root = Path(__file__).resolve().parents[1]
    home = (root / "app.py").read_text(encoding="utf-8")
    assert "Comprehension" in home
    assert "structured_graded" in home
    assert "Beta comprehension" not in home
    assert (root / "pages" / "structured_graded.py").is_file()
    # Legacy URL = thin redirect (kept for mid-flight Multi sessions).
    legacy = (root / "pages" / "comprehension_redirect.py").read_text(encoding="utf-8")
    assert "switch_page" in legacy
    assert "app.py" in legacy


def test_beta_rebuild_mean_wires_ops_reliability_panels():
    """Comprehension Rebuild mean must show graded-style zeros/N/A table + chart."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "app.py").read_text(encoding="utf-8")
    assert "paint_rebuild_ops_reliability_panels" in src
    assert "ops_reliability" in src
    assert "_paint_beta_rebuild_mean_body" in src
    assert "ops_chart" in src


def test_beta_rebuild_mean_uses_shared_reliability_table_html():
    """Comprehension mean ranking must reuse reliability_table_html (not st.dataframe)."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "app.py").read_text(encoding="utf-8")
    assert "reliability_table_html" in src
    assert "successful_only=" in src
    assert "st.dataframe(" not in src


def test_beta_multi_finish_arms_mean_popup_like_graded():
    """After Multi×all / Multi×N, Comprehension must auto-rebuild and open mean KPI dialog."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "app.py").read_text(encoding="utf-8")
    assert "beta_mean_rebuild_dialog" in src
    assert "_arm_beta_mean_popup" in src
    assert "show_beta_mean_popup" in src
    assert "rebuild_balanced_cases_from_history" in src
    assert "mean KPI popup" in src or "Opening mean KPI popup" in src
    # Cost OK gate before streams (parity with Structured).
    assert "beta_pending_run" in src
    assert "render_spend_confirm_card" in src
    assert "Yes · start run" in (root / "lib" / "spend_confirm.py").read_text(
        encoding="utf-8"
    )
    assert "open_new_beta_case_slot" in src
    assert "run_boot_dialogs" in src
    # Honest Freeze copy + pack revision + balanced default.
    assert "gold_raw" in src and "narrative twin" in src
    assert "pack_revision" in src or "pack_rev" in src
    assert 'beta_rebuild_scope"] = "balanced_cases"' in src or (
        '["beta_rebuild_scope"] = "balanced_cases"' in src
    )


def test_ux_parity_shared_shell_on_both_tracks():
    """Boot, spend, tracks sidebar shared; Structured optional; wire ids renamed."""
    root = Path(__file__).resolve().parents[1]
    home = (root / "app.py").read_text(encoding="utf-8")
    structured = (root / "pages" / "structured_graded.py").read_text(encoding="utf-8")
    assert "from lib.track_sidebar import" in home
    assert "from lib.track_sidebar import" in structured
    assert "render_spend_confirm_card" in home
    assert "render_spend_confirm_card" in structured
    assert "run_boot_dialogs" in home
    assert "run_boot_dialogs" in structured
    assert "optional" in structured.lower() or "secondary" in structured.lower()
    pack = (root / "benchmark" / "default_cases" / "comprehension.json").read_text(
        encoding="utf-8"
    )
    assert '"title": "Comprehension' in pack
    assert '"id": "comprehension"' in pack
    assert '"protocol_id": "comprehension-v1"' in pack
    assert "beta-comprehension" not in pack.lower()
    from benchmark.beta_protocol import PROTOCOL_ID, SCORING_VERSION, CASE_ID

    assert PROTOCOL_ID == "comprehension-v1"
    assert SCORING_VERSION == "comprehension-v1"
    assert CASE_ID == "comprehension"
    from benchmark.report import case_ids_equivalent, scoring_versions_equivalent

    assert scoring_versions_equivalent("beta-comprehension-v1", "comprehension-v1")
    assert case_ids_equivalent("beta_comprehension", "comprehension")
    assert not scoring_versions_equivalent("comprehension-v1", "graded-clinical-v4")


def test_new_beta_case_slot_does_not_mutate_pack():
    pack_slots = list_beta_slots()
    n_pack = len(pack_slots)
    new_idx, drafts = open_new_beta_case_slot(pack_slots, custom_drafts={})
    assert new_idx == n_pack + 1
    assert new_idx in drafts
    merged = merge_beta_slots(pack_slots, drafts)
    assert len(merged) == n_pack + 1
    # Pack slots unchanged / not marked custom.
    for s in merged:
        if int(s["slot"]) <= n_pack:
            assert not s.get("custom")
        else:
            assert s.get("custom")
    # Never overwrite pack slot 1.
    drafts[1] = {
        "title": "evil",
        "stem": "x",
        "reference_prose": "y",
        "gold_raw": "",
    }
    merged2 = merge_beta_slots(pack_slots, drafts)
    assert merged2[0]["stem"] == pack_slots[0]["stem"]
    assert not merged2[0].get("custom")


def test_synthetic_gold_raw_from_prose_parses():
    prose = "Severe anaphylaxis with airway compromise; give IM epinephrine first."
    raw = synthetic_gold_raw_from_prose(prose)
    assert "Q1 [" in raw and "A5:" in raw
    resolved = resolve_beta_gold_raw(
        {"gold_raw": "", "reference_prose": prose, "custom": True}
    )
    from benchmark.gold import try_extract_qna_sections

    assert try_extract_qna_sections(resolved) is not None
    frozen = auto_freeze_beta_slot(
        {
            "slot": 99,
            "title": "Custom",
            "stem": "Patient: test",
            "reference_prose": prose,
            "gold_raw": "",
            "custom": True,
        }
    )
    assert frozen.get("custom_case") is True
    assert frozen.get("beta_reference_prose") == prose


def test_auto_freeze_all_pack_slots_for_multi_case():
    slots = list_beta_slots()
    assert len(slots) >= 10
    assert int(slots[0]["slot"]) == 1
    assert {int(s["slot"]) for s in slots} >= {8, 9, 10}
    for s in slots:
        frozen = auto_freeze_beta_slot(s)
        assert frozen.get("auto_confirmed") is True
        assert frozen.get("scoring_version") == SCORING_VERSION
        assert frozen.get("beta_stem")
        assert frozen.get("beta_reference_prose")
        assert "Q1 [" not in frozen["beta_reference_prose"]


def test_count_beta_runs_by_slot_includes_multi_all_rounds():
    """Multi×all rounds that hit a slot count toward that case's N."""

    class _Art:
        def __init__(self, *, case_id, scoring_version, slot=None):
            self.case_id = case_id
            self.scoring_version = scoring_version
            self.models_config = (
                {"beta_case_slot": slot} if slot is not None else {}
            )

    arts = [
        _Art(case_id=CASE_ID, scoring_version=SCORING_VERSION, slot=1),
        _Art(case_id=CASE_ID, scoring_version=SCORING_VERSION, slot=1),
        _Art(case_id=CASE_ID, scoring_version=SCORING_VERSION, slot=3),
        # graded / unrelated — must not count
        _Art(case_id="caseC", scoring_version="graded-clinical-v4", slot=None),
        # beta without slot metadata — skipped
        _Art(case_id=CASE_ID, scoring_version=SCORING_VERSION, slot=None),
    ]
    counts = count_beta_runs_by_slot(arts)
    assert counts == {1: 2, 3: 1}
    assert is_beta_artifact(arts[0])
    assert not is_beta_artifact(arts[3])
    assert beta_case_slot_of(arts[2]) == 3



def test_parse_beta_free_form_fills_all_sections():
    case = load_case("caseC")
    prose = (
        "Likely anaphylaxis with airway threat. Give IM epinephrine now. "
        "Critical urgency. Avoid delaying for antihistamines. ICU observation."
    )
    answers = parse_beta_candidate_answers(case, prose)
    assert set(answers) == {q.id for q in case.questions}
    assert all(answers[q.id] == prose for q in case.questions)


def test_parse_beta_strips_think_blocks_before_photocopy():
    """Judge payload must not receive <think> dumps in all five sections."""
    from benchmark.prompts import clinical_answer_text, sanitize_candidate_answers
    from benchmark.schema import CandidateAnswer, ModelCallMeta

    case = load_case("caseC")
    clinical = (
        "Anterior STEMI — activate primary PCI now. Continuous monitoring; "
        "do not delay reperfusion for troponin. Critical urgency with shock "
        "red flags. Hold metformin around contrast. Aspirin and cath-lab transfer."
    )
    raw = f"<think>\nPrivate scratch: ignore this monologue about QRS width.\n</think>\n{clinical}"
    answers = parse_beta_candidate_answers(case, raw)
    assert set(answers) == {q.id for q in case.questions}
    for qid, text in answers.items():
        assert "<think>" not in text.lower()
        assert "private scratch" not in text.lower()
        assert "anterior stemi" in text.lower()
        assert text == clinical_answer_text(raw)
    # Collect path stores parser output on CandidateAnswer.answers (not raw).
    cand = CandidateAnswer(
        candidate_key="qvac",
        label="MedPsy",
        blind_id="Candidate 1",
        answers=answers,
        raw_response=raw,
        meta=ModelCallMeta(model="medpsy", provider="qvac"),
    )
    assert all("<think>" not in (v or "").lower() for v in cand.answers.values())
    sanitized = sanitize_candidate_answers(
        {q.id: raw for q in case.questions}
    )
    assert all("<think>" not in (v or "").lower() for v in sanitized.values())
    assert all("anterior stemi" in v.lower() for v in sanitized.values())


def test_cohort_id_differs_for_beta_scoring_version():
    section = GoldSection(
        summary="x",
        claims=[GoldClaim(id="diagnosis-1", text="dx", source_quote="dx")],
    )
    gold = ConfirmedGold(
        raw_text="dx",
        sections={
            "diagnosis": section,
            "tests": section,
            "urgency": section,
            "safety": section,
            "plan": section,
        },
        confirmed_at="2026-01-01T00:00:00Z",
    )
    models = {"candidates": [{"key": "chatgpt"}]}
    a = cohort_id(
        case_stem="stem",
        gold=gold,
        prompt_version="gold-only-v1",
        model_config=models,
        benchmark_track="controlled",
        scoring_version="graded-clinical-v4",
    )
    b = cohort_id(
        case_stem="stem",
        gold=gold,
        prompt_version="comprehension-v1",
        model_config=models,
        benchmark_track="controlled",
        scoring_version=SCORING_VERSION,
    )
    assert a != b
    assert CASE_ID == "comprehension"
    assert PROTOCOL_ID == SCORING_VERSION


def test_comprehension_and_graded_pack_json_stay_in_sync():
    root = Path(__file__).resolve().parents[1]
    primary = root / "benchmark" / "default_cases" / "comprehension.json"
    legacy = root / "benchmark" / "default_cases" / "beta_comprehension.json"
    assert primary.is_file() and legacy.is_file()
    assert primary.read_text(encoding="utf-8") == legacy.read_text(encoding="utf-8")


def test_rebuild_cross_track_isolation(tmp_path: Path):
    from tests.test_portfolio_rebuild import ROSTER, _write
    from benchmark.report import (
        list_portfolio_runs,
        rebuild_balanced_cases_from_history,
        rebuild_portfolio_from_history,
        scoring_versions_equivalent,
    )
    from benchmark.beta_protocol import CASE_ID, SCORING_VERSION as COMP_SV

    for i in range(3):
        _write(
            tmp_path,
            run_id=f"g{i}",
            case_id="caseC",
            finished_at=f"2026-08-01T1{i}:00:00Z",
            cohort_id="cohort-g",
            acc=90.0 + i,
            scoring_version="graded-clinical-v4",
        )
    for i in range(3):
        _write(
            tmp_path,
            run_id=f"c{i}",
            case_id=CASE_ID,
            finished_at=f"2026-08-02T1{i}:00:00Z",
            cohort_id="cohort-c",
            acc=70.0 + i,
            scoring_version=COMP_SV,
        )
    # Legacy wire stamp must still dual-read into Comprehension pools only.
    _write(
        tmp_path,
        run_id="legacy-beta",
        case_id="beta_comprehension",
        finished_at="2026-08-02T20:00:00Z",
        cohort_id="cohort-legacy-beta",
        acc=66.0,
        scoring_version="beta-comprehension-v1",
    )
    graded = list_portfolio_runs(
        tmp_path, n=10, scoring_version="graded-clinical-v4", track="controlled"
    )
    comp = list_portfolio_runs(
        tmp_path, n=10, scoring_version=COMP_SV, track="controlled"
    )
    assert len(graded) == 3
    assert len(comp) == 4
    assert all(a.scoring_version == "graded-clinical-v4" for _, a in graded)
    assert all(
        scoring_versions_equivalent(a.scoring_version, COMP_SV) for _, a in comp
    )
    assert all(a.case_id == "caseC" for _, a in graded)
    built = rebuild_portfolio_from_history(
        tmp_path, n=5, scoring_version=COMP_SV, track="controlled", model_ids=ROSTER
    )
    assert built.get("ok")
    assert "caseC" not in (built.get("case_ids") or [])
    assert built.get("pack_revision_label") is not None
    graded_built = rebuild_portfolio_from_history(
        tmp_path,
        n=5,
        scoring_version="graded-clinical-v4",
        track="controlled",
        model_ids=ROSTER,
    )
    assert graded_built.get("ok")
    assert graded_built.get("case_ids") == ["caseC"]
    assert CASE_ID not in (graded_built.get("case_ids") or [])
    assert "beta_comprehension" not in (graded_built.get("case_ids") or [])
    balanced = rebuild_balanced_cases_from_history(
        tmp_path, n=5, scoring_version=COMP_SV, track="controlled", model_ids=ROSTER
    )
    assert balanced.get("ok")
    assert "caseC" not in (balanced.get("case_ids") or [])


def test_pack_revision_missing_matches_current_on_rebuild(tmp_path: Path):
    from tests.test_portfolio_rebuild import _write
    from benchmark.report import (
        list_portfolio_runs,
        rebuild_portfolio_from_history,
        write_artifact,
    )

    _write(
        tmp_path,
        run_id="legacy",
        case_id="caseC",
        finished_at="2026-08-01T10:00:00Z",
        cohort_id="cohort-legacy",
        acc=85.0,
    )
    _write(
        tmp_path,
        run_id="old-rev",
        case_id="caseC",
        finished_at="2026-08-01T11:00:00Z",
        cohort_id="cohort-old",
        acc=80.0,
    )
    old_art = list_portfolio_runs(
        tmp_path, n=5, scoring_version="graded-clinical-v4", track="controlled"
    )[0][1]
    old_art.models_config["pack_revision"] = 2
    write_artifact(old_art, tmp_path)

    all_runs = list_portfolio_runs(
        tmp_path, n=10, scoring_version="graded-clinical-v4", track="controlled"
    )
    assert len(all_runs) == 2
    filtered = list_portfolio_runs(
        tmp_path,
        n=10,
        scoring_version="graded-clinical-v4",
        track="controlled",
        pack_revision=3,
        current_pack_revision=3,
    )
    assert len(filtered) == 1
    assert filtered[0][1].run_id == "legacy"
    built = rebuild_portfolio_from_history(
        tmp_path,
        n=5,
        scoring_version="graded-clinical-v4",
        track="controlled",
        pack_revision=3,
        current_pack_revision=3,
    )
    assert built.get("ok")
    assert built.get("pack_revision_label") == "3"


def test_history_dual_read_includes_legacy_beta_stamp(tmp_path: Path):
    from benchmark.report import artifacts_for_case, write_artifact
    from benchmark.schema import RunArtifact

    for sv in ("beta-comprehension-v1", "comprehension-v1"):
        write_artifact(
            RunArtifact(
                run_id=f"hist-{sv}",
                case_id="comprehension",
                started_at="2026-08-01T00:00:00Z",
                finished_at="2026-08-01T00:01:00Z",
                scoring_version=sv,
                cohort_id=f"cohort-{sv}",
                ranking=[{"key": "chatgpt", "accuracy": 80.0, "status": "ok", "rank": 1}],
            ),
            tmp_path,
        )
    from benchmark.report import scoring_versions_equivalent
    from benchmark.beta_protocol import SCORING_VERSION

    hist = [
        a
        for _, a in artifacts_for_case(tmp_path, "comprehension", limit=10)
        if scoring_versions_equivalent(str(a.scoring_version or ""), SCORING_VERSION)
    ]
    assert len(hist) == 2


def test_cohort_id_omits_current_pack_rev_preserves_means():
    """pack_revision ≤3 must not change cohort_id vs pre-stamp History."""
    from benchmark.gold import COHORT_HASH_PACK_REVISION_FROM, cohort_id

    section = GoldSection(
        summary="x",
        claims=[GoldClaim(id="diagnosis-1", text="dx", source_quote="dx")],
    )
    gold = ConfirmedGold(
        raw_text="dx",
        sections={
            "diagnosis": section,
            "tests": section,
            "urgency": section,
            "safety": section,
            "plan": section,
        },
        confirmed_at="2026-01-01T00:00:00Z",
    )
    kwargs = dict(
        case_stem="stem",
        gold=gold,
        prompt_version="comprehension-v1",
        model_config={"candidates": [{"key": "chatgpt"}]},
        benchmark_track="controlled",
        scoring_version=SCORING_VERSION,
    )
    bare = cohort_id(**kwargs)
    stamped3 = cohort_id(**kwargs, pack_revision=3)
    stamped0 = cohort_id(**kwargs, pack_revision=0)
    assert bare == stamped3 == stamped0
    assert COHORT_HASH_PACK_REVISION_FROM == 4
    stamped4 = cohort_id(**kwargs, pack_revision=4)
    assert stamped4 != bare
