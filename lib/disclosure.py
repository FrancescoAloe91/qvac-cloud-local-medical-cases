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
    """Human label for Same-case / Portfolio / Balanced-cases / Comprehension."""
    s = str(scope or "").strip().lower()
    if s == "portfolio":
        return t("disclosure.scope_portfolio", lang)
    if s in {"balanced_cases", "balanced", "round_robin"}:
        return t("disclosure.scope_balanced", lang)
    if s in {"comprehension", "discursive", "beta", "beta_comprehension"}:
        return "Comprehension (discursive)"
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
        t("disclosure.bullet_zero", lang),
        t("disclosure.bullet_ops", lang),
    ]
    if scope:
        lines.append(
            f"<b>{html.escape(t('disclosure.bullet_scope', lang))}</b> — "
            f"{html.escape(scope_label(scope, lang))}"
        )
        scope_l = str(scope).strip().lower()
        if scope_l in {"comprehension", "discursive", "beta", "beta_comprehension"}:
            lines.append(
                "<b>Exercise only</b> — not a medical device · not clinical "
                "advice · not medical validity · not an official MedPsy blog eval."
            )
            lines.append(
                "<b>Comprehension gold</b> — scored against curated Q1–A5 "
                "<code>gold_raw</code> (reference prose is the narrative twin, "
                "not the claim contract)."
            )
            lines.append(
                "<b>Photocopy caveat</b> — unmarked free-form notes may be "
                "copied into all five sections; dimensions are not fully "
                "independent answers."
            )
            lines.append(
                "<b>Pack provenance</b> — Case 1–10 acute ED fixtures assembled "
                "with Cursor and/or adapted from public internet teaching "
                "material · not validated EHR charts · not general clinical IQ."
            )
        elif scope_l in {"structured", "graded", "a1a5", "structured_a1a5"}:
            lines.append(
                "<b>Structured track</b> — optional secondary · rigid A1–A5 "
                "slots · never pool with Comprehension History / Rebuild."
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
    pack_revision: Optional[int] = None,
    protocol_id: Optional[str] = None,
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
    if protocol_id:
        bits.append(f"protocol={html.escape(str(protocol_id))}")
    if pack_revision is not None:
        bits.append(f"pack_rev={int(pack_revision)}")
    if extra:
        bits.append(str(extra))
    line = " · ".join(bits)
    return (
        '<div class="screenshot-footer">'
        f"{html.escape(line)}"
        "</div>"
    )
