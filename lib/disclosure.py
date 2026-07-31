"""Loud honesty / screenshot disclosure helpers (UI copy only — no scoring)."""

from __future__ import annotations

import html
from typing import List, Optional

from lib.i18n import t

DEFAULT_ROSTER_VERSION = 9


def short_cohort(cohort_id: Optional[str], *, n: int = 8) -> str:
    """First n hex chars of a cohort id, or empty."""
    cid = (cohort_id or "").strip()
    if not cid:
        return ""
    return cid[:n]


def scope_label(scope: str, lang: Optional[str] = None) -> str:
    """Human label for Same-case vs Portfolio."""
    if str(scope or "").strip().lower() == "portfolio":
        return t("disclosure.scope_portfolio", lang)
    return t("disclosure.scope_same", lang)


def honesty_block_html(
    *,
    lang: Optional[str] = None,
    cohort_id: Optional[str] = None,
    roster_n: int = DEFAULT_ROSTER_VERSION,
    scope: Optional[str] = None,
) -> str:
    """Hard-to-crop honesty box for hero / Rebuild / ranking panels."""
    cohort = short_cohort(cohort_id)
    confirm = t("disclosure.bullet_confirm", lang)
    if cohort:
        confirm = f"{confirm} · cohort <code>{html.escape(cohort)}</code>"

    lines: List[str] = [
        t("disclosure.bullet_ref", lang),
        t("disclosure.bullet_api", lang),
        t("disclosure.bullet_judge", lang),
        t("disclosure.bullet_exploratory", lang),
        t("disclosure.bullet_rebuild", lang),
    ]
    if scope:
        lines.append(
            f"<b>{html.escape(t('disclosure.bullet_scope', lang))}</b> — "
            f"{html.escape(scope_label(scope, lang))}"
        )
    lines.extend(
        [
            confirm,
            t("disclosure.bullet_roster", lang, n=roster_n),
            t("disclosure.bullet_recovery", lang),
        ]
    )
    items = "".join(f"<li>{line}</li>" for line in lines)
    title = html.escape(t("disclosure.title", lang))
    return (
        '<div class="honesty-block" role="note">'
        f'<div class="honesty-block-title">{title}</div>'
        f'<ul class="honesty-block-list">{items}</ul>'
        "</div>"
    )


def screenshot_footer_html(
    *,
    lang: Optional[str] = None,
    scope: str = "same_case",
    roster_n: int = DEFAULT_ROSTER_VERSION,
    cohort_id: Optional[str] = None,
    n_label: Optional[str] = None,
    extra: Optional[str] = None,
) -> str:
    """Compact footer under mean tables/charts (screenshot-friendly)."""
    cohort = short_cohort(cohort_id) or "—"
    n_bit = n_label or t("disclosure.footer_n_default", lang)
    bits = [
        t("disclosure.footer_mean_std", lang),
        n_bit,
        f"{t('disclosure.footer_scope', lang)}={scope_label(scope, lang)}",
        f"{t('disclosure.footer_roster', lang)}=v{int(roster_n)}",
        f"{t('disclosure.footer_cohort', lang)}={cohort}",
        t("disclosure.footer_exploratory", lang),
    ]
    if extra:
        bits.append(str(extra))
    line = " · ".join(bits)
    return (
        '<div class="screenshot-footer">'
        f"{html.escape(line)}"
        "</div>"
    )
