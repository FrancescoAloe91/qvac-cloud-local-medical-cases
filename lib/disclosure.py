"""Loud honesty / screenshot disclosure helpers (UI copy only — no scoring)."""

from __future__ import annotations

import html
from typing import Any, Iterable, List, Optional

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
        return t("disclosure.scope_comprehension", lang)
    if s in {"structured", "graded", "a1a5", "structured_a1a5"}:
        return t("disclosure.scope_structured", lang)
    return t("disclosure.scope_same", lang)


def rebuild_scan_honesty_html(
    ops_rows: Optional[Iterable[Any]] = None,
    *,
    lang: Optional[str] = None,
) -> str:
    """Loud one-liner under Rebuild mean chart: scored vs zeros/N/A excluded."""
    scored = zero = na = seen = 0
    for r in ops_rows or []:
        if not isinstance(r, dict):
            continue
        scored += int(r.get("n_scored") or 0)
        zero += int(r.get("n_zero") or 0)
        na += int(r.get("n_technical_na") or 0)
        seen += int(r.get("n_seen") or 0)
    if seen <= 0 and scored <= 0:
        return ""
    line = t(
        "disclosure.rebuild_scan_line",
        lang,
        scored=scored,
        seen=seen,
        zero=zero,
        na=na,
    )
    return (
        '<div class="rebuild-scan-banner" role="note" '
        'style="margin:0.35rem 0 0.75rem;padding:0.45rem 0.65rem;'
        "border:1px solid #f59e0b;border-radius:6px;background:#422006;"
        'color:#fde68a;font-size:0.82rem;font-weight:600">'
        f"{html.escape(line)}"
        "</div>"
    )


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
        lines.append(t("disclosure.bullet_validity", lang))
        lines.append(t("disclosure.bullet_never_pool", lang))
        if scope_l in {"comprehension", "discursive", "beta", "beta_comprehension"}:
            lines.append(t("disclosure.bullet_comp_scoring", lang))
            lines.append(t("disclosure.bullet_comp_copy", lang))
            lines.append(t("disclosure.bullet_comp_provenance", lang))
        elif scope_l in {"structured", "graded", "a1a5", "structured_a1a5"}:
            lines.append(t("disclosure.bullet_structured", lang))
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
    pack_revision_label: Optional[str] = None,
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
        bits.append(f"protocol={str(protocol_id)}")
    if pack_revision_label:
        bits.append(f"pack={str(pack_revision_label)}")
    elif pack_revision is not None:
        bits.append(f"pack={int(pack_revision)}")
    if extra:
        bits.append(str(extra))
    line = " · ".join(bits)
    gloss = html.escape(t("disclosure.footer_gloss", lang))
    return (
        '<div class="screenshot-footer">'
        f"{html.escape(line)}"
        f'<div class="screenshot-footer-gloss">{gloss}</div>'
        "</div>"
    )
