"""Helpers for multi-run progressive KPI UI (Automated Benchmark)."""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional


_SHORT = {
    "chatgpt": "ChatGPT",
    "claude": "Claude",
    "gemini": "Gemini",
    "qvac": "QVAC",
}


def short_model(key: str) -> str:
    return _SHORT.get(key, key)


def snapshot_from_artifact(art: Any) -> Dict[str, Any]:
    """Lightweight per-run snapshot for session state + tabs."""
    ranking = []
    for r in art.ranking or []:
        ranking.append(
            {
                "key": r.get("key"),
                "rank": r.get("rank"),
                "accuracy": r.get("accuracy"),
                "label": r.get("label") or r.get("key"),
                "status": r.get("status", "ok"),
                "ttft_s": r.get("ttft_s"),
                "tps": r.get("tps"),
                "cost_usd": r.get("cost_usd"),
            }
        )
    dims = []
    for j in art.judgments or []:
        by_q = {qs.question_id: qs.score for qs in (j.question_scores or [])}
        dims.append(
            {
                "key": j.candidate_key,
                "weighted": j.weighted_accuracy,
                "scores": by_q,
            }
        )
    return {
        "run_id": art.run_id,
        "n_index": art.n_index,
        "case_id": art.case_id,
        "total_cost_usd": art.total_cost_usd,
        "ranking": ranking,
        "dimensions": dims,
        "notes": art.notes or "",
    }


def mean_placeholder_html(*, n_done: int, n_total: int) -> str:
    """Empty mean chart card until all runs finish."""
    return f"""
<div style="border:1px dashed #334155;border-radius:12px;padding:1.1rem 1.25rem;
  background:linear-gradient(180deg,#0f172a,#111827);margin:0.4rem 0 0.8rem 0;">
  <div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;
    color:#64748b;font-weight:600;margin-bottom:0.35rem;">Multi-run mean KPIs</div>
  <div style="font-size:1.15rem;font-weight:650;color:#e2e8f0;margin-bottom:0.25rem;">
    Waiting for all runs…
  </div>
  <div style="color:#94a3b8;font-size:0.88rem;line-height:1.45;">
    Completed <b style="color:#fbbf24">{n_done}</b> / {n_total}.
    Mean accuracy, ±std and CV% reliability appear here when the batch finishes.
    Use the <b>Run tabs</b> below for each finished run.
  </div>
</div>
"""


def _ranking_table_html(ranking: List[Dict[str, Any]]) -> str:
    rows = sorted(ranking or [], key=lambda r: int(r.get("rank") or 99))
    if not rows:
        return "<p style='color:#94a3b8'>No ranking.</p>"
    body = []
    for r in rows:
        body.append(
            "<tr>"
            f"<td style='padding:0.3rem 0.45rem;border-bottom:1px solid #1e293b'>#{r.get('rank')}</td>"
            f"<td style='padding:0.3rem 0.45rem;border-bottom:1px solid #1e293b'>"
            f"{html.escape(short_model(str(r.get('key'))))}</td>"
            f"<td style='padding:0.3rem 0.45rem;border-bottom:1px solid #1e293b;font-weight:700;"
            f"color:#fbbf24'>{float(r.get('accuracy') or 0):.1f}%</td>"
            "</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:0.88rem;color:#e2e8f0'>"
        "<thead><tr style='color:#94a3b8;text-align:left'>"
        "<th style='padding:0.3rem 0.45rem'>#</th>"
        "<th style='padding:0.3rem 0.45rem'>Model</th>"
        "<th style='padding:0.3rem 0.45rem'>Acc</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def progressive_multi_panel_html(
    completed: List[Dict[str, Any]],
    *,
    n_total: int,
    batch_done: bool = False,
) -> str:
    """
    Live multi-run results strip: empty mean card + clickable run tabs with HTML modals.
    Works during the blocking collect/judge loop (no Streamlit widget click needed).
    """
    n_done = len(completed)
    mean_block = (
        ""
        if batch_done
        else mean_placeholder_html(n_done=n_done, n_total=n_total)
    )

    if not completed:
        tabs = (
            '<div style="color:#64748b;font-size:0.85rem;margin:0.25rem 0 0.6rem;">'
            "Run tabs appear here as soon as run 1 finishes.</div>"
        )
        return mean_block + tabs

    chips = []
    modals = []
    for snap in completed:
        i = int(snap.get("n_index") or 0)
        mid = f"mrun_modal_{i}_{html.escape(str(snap.get('run_id') or i)[:10])}"
        ranking = snap.get("ranking") or []
        top = "—"
        if ranking:
            best = min(ranking, key=lambda r: int(r.get("rank") or 99))
            top = (
                f"{short_model(str(best.get('key')))} "
                f"{float(best.get('accuracy') or 0):.0f}%"
            )
        chips.append(
            f'<button type="button" onclick="document.getElementById(\'{mid}\').style.display=\'flex\'" '
            f'style="display:inline-flex;align-items:center;gap:0.35rem;cursor:pointer;'
            f'padding:0.4rem 0.75rem;border-radius:999px;margin:0.15rem 0.3rem 0.15rem 0;'
            f'background:#1e293b;border:1px solid #fbbf24;color:#e2e8f0;font-size:0.84rem;">'
            f'<b style="color:#fbbf24">Run {i}</b>'
            f'<span style="color:#64748b">·</span>'
            f'<span>{html.escape(top)}</span>'
            f'<span style="color:#94a3b8;font-size:0.75rem">open</span></button>'
        )
        modals.append(
            f'<div id="{mid}" style="display:none;position:fixed;inset:0;z-index:100000;'
            f'align-items:center;justify-content:center;background:rgba(2,6,23,0.82);">'
            f'<div style="width:min(440px,96%);max-height:90%;overflow:auto;background:#0f172a;'
            f'border:1px solid #334155;border-radius:16px;padding:1.1rem 1.25rem;'
            f'box-shadow:0 16px 40px rgba(0,0,0,0.5);">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-bottom:0.55rem;">'
            f'<div style="font-weight:700;color:#f8fafc;font-size:1.1rem;">Run {i} KPIs</div>'
            f'<button type="button" onclick="document.getElementById(\'{mid}\').style.display=\'none\'" '
            f'style="border:0;background:#334155;color:#e2e8f0;border-radius:8px;padding:0.35rem 0.7rem;'
            f'cursor:pointer;">Close</button></div>'
            f'{_ranking_table_html(ranking)}'
            f'<div style="color:#64748b;font-size:0.78rem;margin-top:0.55rem;">'
            f'Cost ${float(snap.get("total_cost_usd") or 0):.4f} · full detail after batch ends'
            f'</div></div></div>'
        )

    pending = n_total - n_done
    pend = (
        f'<span style="color:#64748b;font-size:0.8rem;margin-left:0.35rem;">'
        f"· {pending} still running below…</span>"
        if pending > 0
        else ""
    )
    tabs = (
        '<div style="margin:0.15rem 0 0.65rem;">'
        '<div style="font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;'
        'color:#64748b;font-weight:600;margin-bottom:0.35rem;">Finished runs · click to open</div>'
        + "".join(chips)
        + pend
        + "".join(modals)
        + "</div>"
    )
    return mean_block + tabs


def client_toast_run_done(run_i: int, n_total: int, ranking: List[Dict[str, Any]]) -> str:
    """In-panel completion card (iframe-safe). Pair with st.toast for app-level notice."""
    leader = "—"
    if ranking:
        best = min(ranking, key=lambda r: int(r.get("rank") or 99))
        leader = (
            f"{short_model(str(best.get('key')))} · "
            f"{float(best.get('accuracy') or 0):.1f}%"
        )
    rid = f"mtoast_{run_i}_{n_total}"
    return f"""
<div id="{rid}" style="border:1px solid #fbbf24;border-radius:14px;padding:0.9rem 1rem;
  background:linear-gradient(135deg,#1c1917,#0f172a);margin:0.35rem 0 0.75rem 0;">
  <div style="font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase;color:#a8a29e;
    font-weight:600;">Multi-run progress</div>
  <div style="font-size:1.15rem;font-weight:700;color:#fef3c7;margin:0.25rem 0;">
    Run {run_i}/{n_total} complete
  </div>
  <div style="color:#d6d3d1;font-size:0.88rem;line-height:1.4;margin-bottom:0.65rem;">
    Leader: <b style="color:#fbbf24">{html.escape(leader)}</b> ·
    Click the <b>Run {run_i}</b> tab above for KPIs · mean chart waits until all runs finish.
  </div>
  <button type="button" onclick="document.getElementById('{rid}').remove()"
    style="padding:0.45rem 1rem;border:0;border-radius:9px;cursor:pointer;
    background:#fbbf24;color:#0f172a;font-weight:700;font-size:0.88rem;">OK</button>
</div>
"""


def reliability_badge(level: str) -> str:
    colors = {
        "super_high": ("#064e3b", "#6ee7b7", "Super High"),
        "high": ("#14532d", "#86efac", "High"),
        "medium": ("#713f12", "#fde047", "Medium"),
        "low": ("#9a3412", "#fdba74", "Low"),
        "very_low": ("#7f1d1d", "#fca5a5", "Very Low"),
    }
    bg, fg, lab = colors.get(level, ("#1e293b", "#94a3b8", level or "—"))
    return (
        f'<span style="display:inline-block;padding:0.12rem 0.45rem;border-radius:999px;'
        f'background:{bg};color:{fg};font-size:0.72rem;font-weight:700;">{lab}</span>'
    )
