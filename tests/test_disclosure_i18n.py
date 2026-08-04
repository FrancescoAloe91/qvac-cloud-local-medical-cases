"""Loud disclosure package: i18n key parity + helper smoke tests."""

from __future__ import annotations

from pathlib import Path

from lib.disclosure import (
    DEFAULT_ROSTER_VERSION,
    honesty_block_html,
    rebuild_scan_honesty_html,
    scope_label,
    screenshot_footer_html,
    screenshot_share_checklist_html,
    short_cohort,
)
from lib.guide_overlays import guides_always_available_html, sidebar_guides_block_html
from lib.i18n import _STRINGS

ROOT = Path(__file__).resolve().parents[1]


DISCLOSURE_KEYS = [
    "disclosure.title",
    "disclosure.bullet_ref",
    "disclosure.bullet_api",
    "disclosure.bullet_judge",
    "disclosure.bullet_exploratory",
    "disclosure.bullet_rebuild",
    "disclosure.bullet_zero",
    "disclosure.bullet_ops",
    "disclosure.bullet_scope",
    "disclosure.bullet_confirm",
    "disclosure.bullet_validity",
    "disclosure.bullet_never_pool",
    "disclosure.bullet_comp_scoring",
    "disclosure.bullet_comp_copy",
    "disclosure.bullet_comp_provenance",
    "disclosure.bullet_structured",
    "disclosure.banner_auto_freeze",
    "disclosure.screenshot_checklist",
    "disclosure.rebuild_scan_line",
    "disclosure.bullet_roster",
    "disclosure.bullet_recovery",
    "disclosure.scope_same",
    "disclosure.scope_portfolio",
    "disclosure.scope_balanced",
    "disclosure.scope_comprehension",
    "disclosure.scope_structured",
    "disclosure.footer_mean_std",
    "disclosure.footer_n_default",
    "disclosure.footer_scope",
    "disclosure.footer_roster",
    "disclosure.footer_cohort",
    "disclosure.footer_exploratory",
    "disclosure.footer_gloss",
    "disclosure.confirm_new_cohort",
    "disclosure.rebuild_scope_loud",
    "comp.lock_checkbox",
    "comp.lock_btn",
    "comp.lock_ok",
    "comp.lock_success",
    "comp.track_caption",
    "comp.need_lock",
    "comp.rebuild_caption",
    "comp.ranking_pack_caption",
    "comp.rebuild_public_claim",
    "comp.same_key_warning",
    "bench.ranking_mean_label",
    "bench.ranking_mean_local_label",
    "bench.rebuild_customs_warning",
    "bench.rebuild_n_help",
    "struct.track_caption",
    "struct.judge_caption",
    "struct.confirm_success",
    "struct.confirm_fail",
    "struct.confirm_locked",
    "struct.need_confirm",
    "guide.setup_btn",
    "guide.rank_btn",
    "guide.setup_title",
    "guide.rank_title",
    "guide.hint",
    "guide.close",
    "guide.setup_need_h",
    "guide.setup_need_1",
    "guide.setup_need_2",
    "guide.setup_need_3",
    "guide.setup_steps_h",
    "guide.setup_step_1",
    "guide.setup_step_2",
    "guide.setup_step_3",
    "guide.setup_leave",
    "guide.setup_ready",
    "guide.setup_status",
    "guide.setup_browser",
]


def test_disclosure_keys_exist_in_en_and_it():
    en = _STRINGS["en"]
    it = _STRINGS["it"]
    for key in DISCLOSURE_KEYS:
        assert key in en, f"missing EN key {key}"
        assert key in it, f"missing IT key {key}"
        assert en[key].strip(), f"empty EN {key}"
        assert it[key].strip(), f"empty IT {key}"


def test_pack_ranking_and_same_key_copy():
    """Public claim = pack 1–10; same API key shares History vault."""
    en = _STRINGS["en"]
    it = _STRINGS["it"]
    pack_en = en["comp.ranking_pack_caption"].lower()
    assert "pack case 1–10" in pack_en or "pack case 1-10" in pack_en
    assert "custom" in pack_en and "leaderboard" in pack_en
    assert "official" not in pack_en
    claim_en = en["comp.rebuild_public_claim"].lower()
    assert "pack case 1–10" in claim_en or "pack case 1-10" in claim_en
    assert "custom" in claim_en
    assert "warning" in claim_en
    warn_en = en["bench.rebuild_customs_warning"].lower()
    assert "custom" in warn_en
    assert "1–10" in en["bench.rebuild_customs_warning"] or "1-10" in warn_en
    assert "hard-filtered" in warn_en
    rank_en = en["bench.ranking_mean_label"].lower()
    assert "ranking" in rank_en and "indicative" in rank_en
    assert "official" not in rank_en
    assert "indicativa" in it["bench.ranking_mean_label"].lower()
    key_en = en["comp.same_key_warning"].lower()
    assert "same openrouter key" in key_en
    assert "history" in key_en
    assert "custom" in key_en
    assert "case pack 1–10" in it["comp.ranking_pack_caption"].lower() or (
        "1–10" in it["comp.ranking_pack_caption"]
    )
    assert "openrouter" in it["comp.same_key_warning"].lower()
    # Soft rebuild help mentions public pack claim without changing math.
    assert "pack case 1–10" in en["bench.rebuild_n_help"].lower() or (
        "1–10" in en["bench.rebuild_n_help"]
    )
    assert "pack case 1–10" in en["bench.rebuild_btn_help_balanced"].lower() or (
        "1–10" in en["bench.rebuild_btn_help_balanced"]
    )


def test_honesty_block_includes_core_phrases():
    html = honesty_block_html(
        lang="en",
        roster_n=DEFAULT_ROSTER_VERSION,
        scope="same_case",
        cohort_id="abcdef0123456789",
    )
    assert "honesty-block" in html
    assert "reference answers" in html.lower()
    assert "OpenRouter" in html
    assert "DeepSeek R1" in html
    assert "only successful scores" in html
    assert "crush the average" in html
    assert "refusal" in html.lower()
    assert "MedGemma" in html and "~2/109" in html
    assert "failures/n/a" in html.lower() and "table" in html.lower()
    assert "reliability chart" not in html.lower()
    assert "Same-case" in html
    assert "abcdef01" in html
    assert str(DEFAULT_ROSTER_VERSION) in html
    it_html = honesty_block_html(
        lang="it",
        roster_n=DEFAULT_ROSTER_VERSION,
        scope="same_case",
        cohort_id="abcdef0123456789",
    )
    assert "failures/n/a" in it_html.lower() and "tabella" in it_html.lower()
    assert "grafico di" not in it_html.lower()
    # Comprehension extras stay plain-language + sharp validity.
    comp = honesty_block_html(lang="en", scope="comprehension", roster_n=9)
    assert "Not medical validity" in comp
    assert "Never mix tracks" in comp
    assert "gold_raw" not in comp
    assert "reference checklist" in comp.lower()
    assert "case text or locked reference" in html.lower()


def test_rebuild_scan_honesty_banner():
    banner = rebuild_scan_honesty_html(
        [
            {"n_scored": 8, "n_zero": 1, "n_technical_na": 1, "n_seen": 10},
            {"n_scored": 7, "n_zero": 0, "n_technical_na": 2, "n_seen": 9},
        ],
        lang="en",
    )
    assert "rebuild-scan-banner" in banner
    assert "scored 15/19" in banner
    assert "zeros 1" in banner
    assert "N/A 3" in banner


def test_guide_overlays_split_live_vs_rebuild_and_i18n():
    en = guides_always_available_html(lang="en")
    assert "Live Multi" in en
    assert "Rebuild mean" in en
    assert "successful scores only" in en
    # Live may mention partial; Rebuild section must say no partial on mean.
    assert "no partial badge on the\nmean ranking" in en or "no partial badge" in en.lower()
    assert "What you need for on-device MedPsy" in en
    it = guides_always_available_html(lang="it")
    assert "Media Rebuild" in it or "Come funziona" in it
    assert "Cosa serve per MedPsy on-device" in it
    side_it = sidebar_guides_block_html(lang="it")
    assert "Guida setup" in side_it
    assert "Come funziona la classifica" in side_it
    assert 'aria-label="Chiudi"' in it
    assert 'aria-label="Close"' in en


def test_screenshot_share_checklist_and_auto_freeze_copy():
    en = screenshot_share_checklist_html(lang="en")
    assert "screenshot-share-checklist" in en
    assert "honesty" in en.lower() or "footer" in en.lower()
    assert "Do not crop" in en or "crop" in en.lower()
    banner = _STRINGS["en"]["disclosure.banner_auto_freeze"]
    assert "Multi×all ≠ manual Lock" in banner or "≠" in banner
    assert "auto-locked" in banner.lower() or "auto-locked" in banner


def test_copy_matches_cohort_behavior_no_stale_confirm_equals_new_set():
    """UI/docs must not claim Confirm alone always mints a new cohort."""
    forbidden = (
        "New Confirm = new cohort",
        "Confirming again starts a new comparison set",
        "new Freeze/Confirm = new cohort",
        "Freeze/Confirm = new cohort",
    )
    paths = [
        ROOT / "PRESENTATION.md",
        ROOT / "README.md",
        ROOT / "docs" / "x-post-template.md",
        ROOT / "pages" / "structured_graded.py",
        ROOT / "lib" / "i18n.py",
        ROOT / "app.py",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for phrase in forbidden:
            assert phrase not in text, f"{path.name} still contains: {phrase}"
    # Honest phrasing present.
    i18n_en = _STRINGS["en"]["disclosure.confirm_new_cohort"]
    assert "only if case text or locked claims change" in i18n_en.lower()


def test_screenshot_footer_is_plain_and_compact():
    foot = screenshot_footer_html(
        lang="en",
        scope="portfolio",
        roster_n=9,
        cohort_id="deadbeef99",
        n_label="N=5 successful scores",
    )
    assert "screenshot-footer" in foot
    assert "mean±std" in foot
    assert "Portfolio" in foot
    assert "roster=v9" in foot
    assert "deadbeef" in foot
    assert "screenshot-footer-gloss" in foot
    assert "average ± spread" in foot
    assert "<script" not in foot.lower()


def test_scope_and_short_cohort_helpers():
    assert scope_label("portfolio", "en") == "Portfolio"
    assert scope_label("same_case", "en") == "Same-case"
    assert short_cohort("abcdefghijklmnop") == "abcdefgh"
    assert short_cohort("") == ""
    assert short_cohort(None) == ""
