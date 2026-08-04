"""Shared Streamlit stream panel HTML helpers (graded + Beta)."""

from __future__ import annotations

import html
from typing import Optional

from lib.i18n import t


def kpi_line(meta: dict, text: str = "") -> str:
    parts = []
    if meta.get("ttft_s") is not None:
        parts.append(f"TTFT {meta['ttft_s']}s")
    if meta.get("tps") is not None:
        parts.append(f"TPS {meta['tps']}")
    if meta.get("latency_s") is not None:
        parts.append(f"Latency {meta['latency_s']}s")
    cost = meta.get("cost_usd")
    if cost is not None:
        parts.append(f"${cost:.4f}" if cost else "$0")
    body = (text or "").strip()
    words = len(body.split()) if body else 0
    toks = int(meta.get("completion_tokens") or 0)
    if toks <= 0 and words:
        toks = max(1, int(round(words * 1.3)))
    if words:
        parts.append(f"{words} words")
    if toks:
        parts.append(f"{toks} tok")
    return " · ".join(parts) if parts else "—"


def kpi_live_line(ttft_s, elapsed_s, tps_live) -> str:
    parts = []
    if ttft_s is not None:
        parts.append(f"TTFT {ttft_s}s")
    if tps_live is not None:
        parts.append(f"~{tps_live} TPS")
    if elapsed_s is not None:
        parts.append(f"{elapsed_s}s")
    return " · ".join(parts) if parts else "streaming…"


def status_pill(kind: str, text: str) -> str:
    return f'<span class="status-pill {kind}">{html.escape(text)}</span>'


def stream_uid(panel_id: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in (panel_id or "ans"))[:32]


def stream_shell_html(
    *,
    title: str = "Answer",
    panel_id: str = "ans",
    lang: Optional[str] = None,
) -> str:
    """Fullscreen chrome only — render once; never remount during token stream.

    Client-side checkbox + ``.fs-overlay`` (see ``dashboard_portal.js``): open/close
    does not Streamlit-rerun, so Multi×all collect/judge keep running.
    """
    tid = html.escape(title or "Answer")
    uid = stream_uid(panel_id)
    open_lab = html.escape(t("stream.fullscreen", lang))
    open_title = html.escape(t("stream.fullscreen_title", lang))
    close_lab = html.escape(t("stream.close", lang))
    return f"""
<div class="stream-wrap" data-panel="{uid}">
  <div class="stream-toolbar">
    <label class="stream-fs-lab" for="fs_{uid}" title="{open_title}">{open_lab}</label>
  </div>
  <input type="checkbox" id="fs_{uid}" class="fs-ck" autocomplete="off" />
  <div class="fs-overlay" hidden style="display:none !important;visibility:hidden !important">
    <div class="fs-card">
      <div class="fs-bar">
        <span>{tid}</span>
        <button type="button" class="fs-close" data-fs="fs_{uid}" title="{close_lab}" aria-label="{close_lab}">✕</button>
      </div>
      <div class="fs-scroll">
        <pre class="fs-pre"></pre>
      </div>
    </div>
  </div>
</div>
"""


def stream_body_html(
    text: str,
    live: bool = False,
    *,
    panel_id: str = "ans",
) -> str:
    """Fixed-height answer box — relies on dashboard.css ``.stream-out`` (no inline fight)."""
    caret = '<span class="caret"></span>' if live else ""
    body = html.escape(text or "")
    uid = stream_uid(panel_id)
    return (
        f'<div class="stream-out" data-panel="{uid}" id="sout_{uid}">'
        f"{body}{caret}</div>"
    )


def stream_html(
    text: str,
    live: bool = False,
    *,
    title: str = "Answer",
    panel_id: str = "ans",
    lang: Optional[str] = None,
) -> str:
    """Shell + body (idle / final paint). Prefer shell once + body updates while live."""
    return (
        stream_shell_html(title=title, panel_id=panel_id, lang=lang)
        + stream_body_html(text, live=live, panel_id=panel_id)
    )
