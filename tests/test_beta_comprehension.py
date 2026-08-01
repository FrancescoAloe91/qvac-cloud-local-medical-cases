"""Beta comprehension pack + protocol isolation from graded Rebuild."""

from __future__ import annotations

from pathlib import Path

from benchmark.beta_pack import (
    auto_freeze_beta_slot,
    beta_case_slot_of,
    count_beta_runs_by_slot,
    is_beta_artifact,
    list_beta_slots,
    load_beta_pack,
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
    slots = list_beta_slots(pack)
    assert len(slots) >= 7
    slot_ids = [int(s["slot"]) for s in slots]
    assert slot_ids[0] == 1
    assert slot_ids == list(range(1, len(slots) + 1))
    assert "AKI" in slots[0]["title"] or "hyperkalemia" in slots[0]["title"].lower()
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
    assert SCORING_VERSION == "beta-comprehension-v1"
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
    assert (root / "pages" / "structured_graded.py").is_file()
    # Legacy URL kept for in-flight Multi sessions
    assert (root / "pages" / "beta_comprehension.py").is_file()


def test_beta_rebuild_mean_wires_ops_reliability_panels():
    """Beta Rebuild mean must show graded-style zeros/N/A table + chart."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "pages" / "beta_comprehension.py").read_text(encoding="utf-8")
    assert "paint_rebuild_ops_reliability_panels" in src
    assert "ops_reliability" in src
    assert "_paint_beta_rebuild_mean_body" in src
    assert "ops_chart" in src


def test_beta_rebuild_mean_uses_shared_reliability_table_html():
    """Beta mean ranking must reuse graded reliability_table_html (not st.dataframe)."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "pages" / "beta_comprehension.py").read_text(encoding="utf-8")
    assert "reliability_table_html" in src
    assert "successful_only=" in src
    # Plain Streamlit dataframe was the divergent simpler table
    assert "st.dataframe(" not in src


def test_beta_multi_finish_arms_mean_popup_like_graded():
    """After Multi×all / Multi×N, Beta must auto-rebuild and open mean KPI dialog."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "pages" / "beta_comprehension.py").read_text(encoding="utf-8")
    assert "beta_mean_rebuild_dialog" in src
    assert "_arm_beta_mean_popup" in src
    assert "show_beta_mean_popup" in src
    assert "rebuild_balanced_cases_from_history" in src
    assert "mean KPI popup" in src or "Opening mean KPI popup" in src


def test_auto_freeze_all_pack_slots_for_multi_case():
    slots = list_beta_slots()
    assert len(slots) >= 7
    assert int(slots[0]["slot"]) == 1
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
        prompt_version="beta-comprehension-v1",
        model_config=models,
        benchmark_track="controlled",
        scoring_version=SCORING_VERSION,
    )
    assert a != b
    assert CASE_ID == "beta_comprehension"
    assert PROTOCOL_ID == SCORING_VERSION
