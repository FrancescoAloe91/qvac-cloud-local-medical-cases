"""Helpers for multi-run progressive KPI UI (Automated Benchmark)."""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

from lib.model_labels import full_model_label, name_and_version


def short_model(key: str) -> str:
    """Full Name · Version (kept name for call-site compatibility)."""
    return full_model_label(key or "")


def model_name_version(key: str) -> tuple:
    return name_and_version(key or "")


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
                "ram_mb": r.get("ram_mb"),
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


_BAR_COLORS = {
    "chatgpt": "#10a37f",
    "claude": "#d97706",
    "gemini": "#8ab4f8",
    "local_gemma": "#a855f7",
    "local_llama": "#3b82f6",
    "local_phi": "#0ea5e9",
    "qvac": "#00d09c",
    "qvac_1_7b": "#34d399",
    "qvac_4b_q8": "#2dd4bf",
}


def _bar_color(key: str) -> str:
    return _BAR_COLORS.get(key, "#94a3b8")


def accuracy_histogram_html(
    rows: List[Dict[str, Any]],
    *,
    score_field: str = "accuracy",
    highlight_key: Optional[str] = None,
    include_pending: bool = False,
) -> str:
    """Pure-HTML horizontal histogram (safe inside live Streamlit HTML / modals)."""
    scored: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    for r in rows or []:
        st = str(r.get("status") or "scored")
        if r.get(score_field) is not None and st in ("scored", "failed", "ok", ""):
            scored.append(r)
        elif include_pending:
            pending.append(r)

    scored = sorted(
        scored,
        key=lambda r: float(r.get(score_field) or 0),
        reverse=True,
    )
    if not scored and not pending:
        return (
            '<div class="rank-hist empty">No scores yet — bars appear as '
            "each judge finishes.</div>"
        )

    bars = []
    for i, r in enumerate(scored, 1):
        key = str(r.get("key") or "")
        nm, ver = name_and_version(
            key, label=r.get("label"), model=r.get("model")
        )
        acc = float(r.get(score_field) or 0)
        failed = str(r.get("status")) == "failed"
        width = max(0.0, min(100.0, acc))
        flash = " hist-bar-flash" if highlight_key and key == highlight_key else ""
        color = "#7f1d1d" if failed else _bar_color(key)
        num = "—" if failed else f"{acc:.1f}%"
        bars.append(
            f'<div class="hist-row{flash}">'
            f'<div class="hist-label">'
            f'<span class="hist-rank"><span class="rank-prov-tag">Prov.</span> {i}</span>'
            f'<span class="hist-name">{html.escape(nm)}</span>'
            f'<span class="hist-ver">{html.escape(ver)}</span></div>'
            f'<div class="hist-track">'
            f'<div class="hist-fill" style="width:{width:.1f}%;background:{color}"></div>'
            f"</div>"
            f'<div class="hist-num">{num}</div>'
            f"</div>"
        )

    for r in pending:
        key = str(r.get("key") or "")
        nm, ver = name_and_version(
            key, label=r.get("label"), model=r.get("model")
        )
        bars.append(
            f'<div class="hist-row pending">'
            f'<div class="hist-label"><span class="hist-rank">—</span>'
            f'<span class="hist-name">{html.escape(nm)}</span>'
            f'<span class="hist-ver">{html.escape(ver)}</span></div>'
            f'<div class="hist-track"><div class="hist-fill pending-fill"></div></div>'
            f'<div class="hist-num hist-wait">…</div>'
            f"</div>"
        )

    return '<div class="rank-hist">' + "".join(bars) + "</div>"


def live_judging_board_html(
    entries: Dict[str, Dict[str, Any]],
    *,
    highlight_key: Optional[str] = None,
    title: str = "Live judging · collect order + provisional ranking",
) -> str:
    """Live board during DeepSeek judging.

    Left table stays **FIFO** (collect → judge queue order): rows never reorder;
    they only update status/score in place when a judge finishes.

    Right histogram is the **provisional ranking**: scored rows sort high→low and
    slide as new scores arrive. ``highlight_key`` gets a 2s glow (latest arrival).
    """
    all_rows: List[Dict[str, Any]] = []
    for key, raw in (entries or {}).items():
        row = dict(raw or {})
        row["key"] = key
        all_rows.append(row)

    # Left = fixed collect order
    fifo = sorted(
        all_rows,
        key=lambda r: (
            int(r.get("queue_i") or 10_000),
            str(r.get("label") or r.get("key") or "").lower(),
        ),
    )
    scored = [
        r
        for r in all_rows
        if str(r.get("status") or "") in ("scored", "failed")
        and r.get("accuracy") is not None
    ]
    scored_keys = {str(r.get("key") or "") for r in scored}
    pending = [r for r in all_rows if str(r.get("key") or "") not in scored_keys]

    body: List[str] = []
    for r in fifo:
        key = str(r.get("key") or "")
        nm, ver = name_and_version(
            key, label=r.get("label"), model=r.get("model")
        )
        qi = int(r.get("queue_i") or 0) or "—"
        st = str(r.get("status") or "pending")
        flash = " rank-row-flash" if highlight_key and key == highlight_key else ""
        if st in ("scored", "failed") and r.get("accuracy") is not None:
            acc = float(r.get("accuracy") or 0)
            failed = st == "failed"
            if failed:
                acc_col = (
                    "<span class='rank-live-acc-num fail'>—</span>"
                    "<span class='rank-live-note'>failed</span>"
                )
            else:
                acc_col = (
                    f"<span class='rank-live-acc-num'>{acc:.1f}</span>"
                    f"<span class='rank-live-acc-unit'>%</span>"
                )
            body.append(
                f"<tr class='rank-live-row scored{flash}' data-key='{html.escape(key)}'>"
                f"<td class='rank-live-pos'>{qi}</td>"
                f"<td class='rank-live-name'>{html.escape(nm)}"
                f"<div class='rank-live-ver'>{html.escape(ver)}</div></td>"
                f"<td class='rank-live-acc'>{acc_col}</td>"
                f"</tr>"
            )
        else:
            waiting = (
                "judging…"
                if st in ("pending", "queued", "retry", "judging")
                else ("collecting…" if st == "collecting" else st)
            )
            body.append(
                f"<tr class='rank-live-row pending' data-key='{html.escape(key)}'>"
                f"<td class='rank-live-pos' style='color:#64748b'>{qi}</td>"
                f"<td class='rank-live-name' style='opacity:.85'>{html.escape(nm)}"
                f"<div class='rank-live-ver'>{html.escape(ver)}</div></td>"
                f"<td class='rank-live-acc'><span class='rank-live-wait'>"
                f"{html.escape(waiting)}</span></td>"
                f"</tr>"
            )

    if not body:
        return (
            f'<div class="rank-live-board">'
            f'<div class="rank-live-title">{html.escape(title)}</div>'
            f'<div class="rank-live-empty">Waiting for first collect → judge…</div></div>'
        )

    n_scored = len(scored)
    table = (
        "<table class='rank-live-table'><thead><tr>"
        "<th>#</th><th>Model</th><th>Score</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )
    # Right = dynamic ranking (scored high→low; pending at bottom)
    hist = accuracy_histogram_html(
        scored + pending,
        highlight_key=highlight_key,
        include_pending=True,
    )
    return (
        f'<div class="rank-live-board">'
        f'<div class="rank-live-title">{html.escape(title)}</div>'
        f'<div class="rank-live-sub">{n_scored} scored · left = collect order (fixed) · '
        f"right = <b>Prov.</b> ranking (slides as scores arrive)"
        f"{' · latest highlighted' if highlight_key else ''}</div>"
        f'<div class="rank-live-grid">'
        f'<div class="rank-live-col table-col">'
        f'<div class="rank-live-hist-cap">Judge queue · FIFO</div>{table}</div>'
        f'<div class="rank-live-col hist-col">'
        f'<div class="rank-live-hist-cap">Provisional ranking · histogram</div>{hist}</div>'
        f"</div></div>"
    )


def _ranking_table_html(ranking: List[Dict[str, Any]]) -> str:
    rows = sorted(
        ranking or [],
        key=lambda r: float(r.get("accuracy") or 0),
        reverse=True,
    )
    if not rows:
        return "<p style='color:#94a3b8'>No ranking.</p>"
    body = []
    for i, r in enumerate(rows, 1):
        nm, ver = name_and_version(
            str(r.get("key") or ""),
            label=r.get("label"),
            model=r.get("model"),
        )
        acc = float(r.get("accuracy") or 0)
        body.append(
            "<tr>"
            f"<td style='padding:0.3rem 0.45rem;border-bottom:1px solid #1e293b'>#{i}</td>"
            f"<td style='padding:0.3rem 0.45rem;border-bottom:1px solid #1e293b'>"
            f"{html.escape(nm)}</td>"
            f"<td style='padding:0.3rem 0.45rem;border-bottom:1px solid #1e293b;"
            f"color:#94a3b8;font-size:0.8rem'>{html.escape(ver)}</td>"
            f"<td style='padding:0.3rem 0.45rem;border-bottom:1px solid #1e293b;font-weight:800;"
            f"color:#fbbf24;font-size:1.05rem;text-align:right'>{acc:.1f}%</td>"
            "</tr>"
        )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:0.88rem;color:#e2e8f0'>"
        "<thead><tr style='color:#94a3b8;text-align:left'>"
        "<th style='padding:0.3rem 0.45rem'>#</th>"
        "<th style='padding:0.3rem 0.45rem'>Name</th>"
        "<th style='padding:0.3rem 0.45rem'>Version</th>"
        "<th style='padding:0.3rem 0.45rem;text-align:right'>Score</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _run_summary_body_html(ranking: List[Dict[str, Any]]) -> str:
    """Table + histogram for finished-run modal."""
    rows = sorted(
        ranking or [],
        key=lambda r: float(r.get("accuracy") or 0),
        reverse=True,
    )
    return (
        '<div class="rank-live-grid run-summary-grid">'
        f'<div class="rank-live-col table-col">{_ranking_table_html(rows)}</div>'
        f'<div class="rank-live-col hist-col">'
        f'<div class="rank-live-hist-cap">Accuracy % · histogram</div>'
        f"{accuracy_histogram_html(rows)}"
        f"</div></div>"
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
            best = max(ranking, key=lambda r: float(r.get("accuracy") or 0))
            top = (
                f"{short_model(str(best.get('key')))} "
                f"{float(best.get('accuracy') or 0):.1f}%"
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
            f'<div style="width:min(720px,96%);max-height:92%;overflow:auto;background:#0f172a;'
            f'border:1px solid #334155;border-radius:16px;padding:1.1rem 1.25rem;'
            f'box-shadow:0 16px 40px rgba(0,0,0,0.5);">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-bottom:0.55rem;">'
            f'<div style="font-weight:700;color:#f8fafc;font-size:1.1rem;">'
            f"Run {i} · table + histogram</div>"
            f'<button type="button" onclick="document.getElementById(\'{mid}\').style.display=\'none\'" '
            f'style="border:0;background:#334155;color:#e2e8f0;border-radius:8px;padding:0.35rem 0.7rem;'
            f'cursor:pointer;">Close</button></div>'
            f"{_run_summary_body_html(ranking)}"
            f'<div style="color:#64748b;font-size:0.78rem;margin-top:0.65rem;">'
            f'Cost ${float(snap.get("total_cost_usd") or 0):.4f} · full detail after batch ends'
            f"</div></div></div>"
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
        best = max(ranking, key=lambda r: float(r.get("accuracy") or 0))
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
