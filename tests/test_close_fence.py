"""CLOSE+FENCE package locks: cloud strip, rebuild copy, aliases, banners."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_h2_comprehension_strips_openrouter_on_cloud(monkeypatch):
    """Comprehension home must strip process-wide OPENROUTER like Structured."""
    from lib.deployment import capture_and_strip_openrouter_env

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + ("a" * 40))
    monkeypatch.setenv("STREAMLIT_CLOUD", "1")
    assert capture_and_strip_openrouter_env() == ""
    assert not (os.environ.get("OPENROUTER_API_KEY") or "").strip()

    home = (ROOT / "app.py").read_text(encoding="utf-8")
    structured = (ROOT / "pages" / "structured_graded.py").read_text(encoding="utf-8")
    assert "capture_and_strip_openrouter_env" in home
    assert "capture_and_strip_openrouter_env" in structured


def test_h2_openrouter_refuses_env_fallback_on_cloud(monkeypatch):
    from benchmark.openrouter import _resolve_api_key

    monkeypatch.setenv("STREAMLIT_CLOUD", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="BYOK|session|not used"):
        _resolve_api_key("")
    with pytest.raises(RuntimeError, match="BYOK|session|not used"):
        _resolve_api_key(None)
    assert _resolve_api_key("sk-or-v1-" + ("b" * 40)).startswith("sk-or-v1-")


def test_h2_render_and_public_bind_strip_host_key(monkeypatch):
    """Render / 0.0.0.0 hosts must enforce BYOK like Streamlit Cloud."""
    from lib.deployment import (
        capture_and_strip_openrouter_env,
        is_hosted_byok_required,
        is_local_install,
    )
    from benchmark.openrouter import _resolve_api_key

    monkeypatch.delenv("STREAMLIT_CLOUD", raising=False)
    monkeypatch.delenv("STREAMLIT_RUNTIME_ENVIRONMENT", raising=False)
    monkeypatch.delenv("STREAMLIT_SHARING_MODE", raising=False)
    monkeypatch.setenv("HOSTED_BYOK", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + ("d" * 40))
    assert is_hosted_byok_required() is True
    assert is_local_install() is False
    assert capture_and_strip_openrouter_env() == ""
    assert not (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    with pytest.raises(RuntimeError, match="BYOK|hosted|not used"):
        _resolve_api_key("")

    monkeypatch.delenv("HOSTED_BYOK", raising=False)
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + ("e" * 40))
    assert is_hosted_byok_required() is True
    assert capture_and_strip_openrouter_env() == ""

    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + ("f" * 40))
    assert is_hosted_byok_required() is True
    assert capture_and_strip_openrouter_env() == ""


def test_h2_local_env_fallback_still_works(monkeypatch):
    from benchmark import openrouter as or_mod
    from lib import deployment as dep

    monkeypatch.setattr(dep, "is_hosted_byok_required", lambda: False)
    monkeypatch.setattr(dep, "is_streamlit_cloud", lambda: False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + ("c" * 40))
    key = or_mod._resolve_api_key("")
    assert key.startswith("sk-or-v1-")


def test_h1_multi_all_shows_auto_freeze_banner_path():
    """Multi×all must surface auto-freeze banner + auto_confirmed freeze path."""
    home = (ROOT / "app.py").read_text(encoding="utf-8")
    # Banner on the Multi×all execution path (not only disclosure helpers).
    assert "disclosure.banner_auto_freeze" in home
    assert "multi_case" in home
    assert "auto_freeze_beta_slot" in home
    # Banner call sits in the multi_case branch before prepare_run.
    idx_banner = home.index('t("disclosure.banner_auto_freeze"')
    idx_multi = home.index("if _multi_case:")
    idx_prep = home.index("prepare_run(", idx_multi)
    assert idx_multi < idx_banner < idx_prep

    from benchmark.beta_pack import auto_freeze_beta_slot, list_beta_slots, load_beta_pack

    pack = load_beta_pack()
    slots = list_beta_slots(pack)
    assert slots
    frozen = auto_freeze_beta_slot(slots[0])
    assert frozen.get("auto_confirmed") is True


def test_h4_reliability_badge_stable_mean_not_super_high():
    from lib.benchmark_multi_ui import RELIABILITY_BAND_COLORS, reliability_badge
    from benchmark.report import (
        CV_HIGH_MAX,
        CV_LOW_MAX,
        CV_MEDIUM_MAX,
        CV_SUPER_HIGH_MAX,
        reliability_from_cv,
    )

    assert RELIABILITY_BAND_COLORS["super_high"][2] == "Stable mean"
    assert "Super High" not in RELIABILITY_BAND_COLORS["super_high"][2]
    html = reliability_badge("super_high")
    assert "Stable mean" in html
    assert "Super High" not in html
    # Numeric cutoffs unchanged (5/10/15/20).
    assert CV_SUPER_HIGH_MAX == 5.0
    assert CV_HIGH_MAX == 10.0
    assert CV_MEDIUM_MAX == 15.0
    assert CV_LOW_MAX == 20.0
    assert reliability_from_cv(5.0) == "super_high"
    assert reliability_from_cv(5.1) == "high"


def test_h6_clinical_composite_alias_in_ranking_and_means():
    from benchmark.judge import build_ranking
    from benchmark.schema import JudgeResult, ModelCallMeta, QuestionScore

    j = JudgeResult(
        blind_id="B1",
        candidate_key="chatgpt",
        question_scores=[
            QuestionScore(question_id="diagnosis", score=80.0, rationale="ok")
        ],
        weighted_accuracy=80.0,
        coverage_score=80.0,
        quality_score=80.0,
        discipline_score=80.0,
        judge_model="deepseek/deepseek-r1",
        judge_meta=ModelCallMeta(model="deepseek/deepseek-r1", provider="openrouter"),
        status="valid",
    )
    rows = build_ranking([j])
    assert rows[0]["accuracy"] == 80.0
    assert rows[0]["clinical_composite"] == 80.0
    assert rows[0]["clinical_composite"] == rows[0]["accuracy"]


def test_h6_clinical_composite_mean_alias_in_summary():
    from benchmark.report import summarize_runs
    from benchmark.schema import RunArtifact

    arts = []
    for i, acc in enumerate((70.0, 80.0, 90.0, 75.0, 85.0), start=1):
        arts.append(
            RunArtifact(
                run_id=f"r{i}",
                case_id="caseC",
                started_at="2026-01-01T00:00:00Z",
                finished_at="2026-01-01T00:01:00Z",
                n_index=i,
                batch_id="b1",
                cohort_id="cohort-test",
                ranking=[
                    {
                        "key": "chatgpt",
                        "accuracy": acc,
                        "status": "ok",
                        "rank": 1,
                    }
                ],
            )
        )
    summary = summarize_runs(arts, min_valid_for_ranking=5)
    assert summary.ranking_mean
    row = next(r for r in summary.ranking_mean if r["key"] == "chatgpt")
    assert "accuracy_mean" in row
    assert "clinical_composite_mean" in row
    assert row["clinical_composite_mean"] == row["accuracy_mean"]


def test_c4_rebuild_n_copy_is_exploratory_mean_std_only():
    """Rebuild N UI/i18n must not claim powered study / p-value / significatività."""
    from lib.i18n import _STRINGS

    banned = re.compile(
        r"powered\s+stud|p-value|significativ",
        re.IGNORECASE,
    )
    for lang in ("en", "it"):
        help_txt = _STRINGS[lang]["bench.rebuild_n_help"]
        cap = _STRINGS[lang]["comp.rebuild_caption"]
        assert "mean±std" in help_txt or "mean±std" in cap or "mean" in help_txt
        assert "explorator" in help_txt.lower() or "esplorativ" in help_txt.lower()
        assert not banned.search(help_txt), help_txt
        assert not banned.search(cap), cap

    home = (ROOT / "app.py").read_text(encoding="utf-8")
    structured = (ROOT / "pages" / "structured_graded.py").read_text(encoding="utf-8")
    for src in (home, structured):
        # Selectbox path uses exploratory mean±std framing for larger N.
        assert "exploratory mean±std" in src
        assert "bench.rebuild_n_help" in src


def test_h8_structured_critical_copy_uses_t():
    structured = (ROOT / "pages" / "structured_graded.py").read_text(encoding="utf-8")
    for key in (
        "struct.track_caption",
        "struct.judge_caption",
        "struct.confirm_locked",
        "struct.need_confirm",
        "struct.confirm_success",
    ):
        assert key in structured
    from lib.i18n import t

    it_track = t("struct.track_caption", "it")
    assert "OpenRouter" in it_track or "API" in it_track
    it_judge = t("struct.judge_caption", "it")
    assert "calibrat" in it_judge.lower() or "giudice" in it_judge.lower()


def test_fence_public_copy_brand_cooccurs_with_api():
    """Consumer brand names in public README/DEPLOY must sit near API/OpenRouter."""
    for rel in ("README.md", "DEPLOY.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for brand in ("ChatGPT", "Claude", "Gemini"):
            if brand not in text:
                continue
            for m in re.finditer(re.escape(brand), text):
                window = text[max(0, m.start() - 120) : m.end() + 120]
                assert re.search(r"API|OpenRouter", window), (
                    f"{rel}: {brand} without API/OpenRouter nearby: {window!r}"
                )


def test_fence_no_adversarial_audit_docs_in_tree():
    docs = ROOT / "docs"
    if docs.is_dir():
        bad = list(docs.glob("*adversarial*")) + list(docs.glob("*audit*"))
        # Allow non-adversarial docs; block adversarial-audit style names.
        bad = [p for p in bad if "adversarial" in p.name.lower()]
        assert not bad, bad


@pytest.mark.skipif(
    os.environ.get("SKIP_APPTEST") == "1",
    reason="explicit skip",
)
def test_m7_apptest_home_smoke_offline():
    """Offline AppTest: home loads with boot skipped; rebuild widgets present."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"AppTest unavailable: {exc}")

    # Avoid cloud strip / network during smoke.
    os.environ.pop("STREAMLIT_CLOUD", None)
    os.environ.pop("STREAMLIT_SHARING_MODE", None)
    os.environ["OPENROUTER_API_KEY"] = ""

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45)
    at.session_state["boot_welcome_done"] = True
    at.session_state["boot_step"] = "done"
    at.session_state["qvac_sdk_ack"] = True
    at.session_state["key_dialog_shown"] = True
    at.session_state["qvac_dialog_shown"] = True
    at.run()

    # Soft: no uncaught exception; rebuild N control or label reachable.
    assert not at.exception, at.exception
    labels = " ".join(str(x.label) for x in list(at.selectbox) + list(at.button))
    body = " ".join(str(getattr(x, "value", "") or getattr(x, "label", "")) for x in at.markdown)
    blob = (labels + " " + body).lower()
    assert (
        "rebuild" in blob
        or "n scored" in blob
        or "average over" in blob
        or "media su" in blob
        or at.selectbox
    )
