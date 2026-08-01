"""Loud disclosure package: i18n key parity + helper smoke tests."""

from __future__ import annotations

from lib.disclosure import (
    DEFAULT_ROSTER_VERSION,
    honesty_block_html,
    scope_label,
    screenshot_footer_html,
    short_cohort,
)
from lib.i18n import _STRINGS


DISCLOSURE_KEYS = [
    "disclosure.title",
    "disclosure.bullet_ref",
    "disclosure.bullet_api",
    "disclosure.bullet_judge",
    "disclosure.bullet_exploratory",
    "disclosure.bullet_rebuild",
    "disclosure.bullet_scope",
    "disclosure.bullet_confirm",
    "disclosure.bullet_roster",
    "disclosure.bullet_recovery",
    "disclosure.scope_same",
    "disclosure.scope_portfolio",
    "disclosure.footer_mean_std",
    "disclosure.footer_n_default",
    "disclosure.footer_scope",
    "disclosure.footer_roster",
    "disclosure.footer_cohort",
    "disclosure.footer_exploratory",
    "disclosure.confirm_new_cohort",
    "disclosure.rebuild_scope_loud",
]


def test_disclosure_keys_exist_in_en_and_it():
    en = _STRINGS["en"]
    it = _STRINGS["it"]
    for key in DISCLOSURE_KEYS:
        assert key in en, f"missing EN key {key}"
        assert key in it, f"missing IT key {key}"
        assert en[key].strip(), f"empty EN {key}"
        assert it[key].strip(), f"empty IT {key}"


def test_honesty_block_includes_core_phrases():
    html = honesty_block_html(
        lang="en",
        roster_n=DEFAULT_ROSTER_VERSION,
        scope="same_case",
        cohort_id="abcdef0123456789",
    )
    assert "honesty-block" in html
    assert "Reference-relative" in html or "reference-relative" in html.lower()
    assert "OpenRouter" in html
    assert "DeepSeek R1" in html
    assert "scored-only" in html
    assert "treated like N/A" in html
    assert "MedGemma" in html and "~2/105" in html
    assert "Same-case" in html
    assert "abcdef01" in html
    assert str(DEFAULT_ROSTER_VERSION) in html


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
    assert "<script" not in foot.lower()


def test_scope_and_short_cohort_helpers():
    assert scope_label("portfolio", "en") == "Portfolio"
    assert scope_label("same_case", "en") == "Same-case"
    assert short_cohort("abcdefghijklmnop") == "abcdefgh"
    assert short_cohort("") == ""
    assert short_cohort(None) == ""
