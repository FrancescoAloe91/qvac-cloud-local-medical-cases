"""Shared Run clock panel (sidebar dock + live iframe tick)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components


def _fmt_s_min(seconds: float) -> str:
    """Primary seconds; compact minutes in parentheses (Italian comma), e.g. 150s (2,5m)."""
    s = max(0, int(round(float(seconds))))
    m = s / 60.0
    m_txt = f"{m:.1f}".replace(".", ",")
    return f'{s}s<span class="t-min"> ({m_txt}m)</span>'


def _fmt_s_plain(seconds: float) -> str:
    """Compact seconds for per-run strips (no HTML minutes)."""
    return f"{max(0, int(round(float(seconds))))}s"


def _per_run_timer_rows(per_run: list | None) -> str:
    """HTML rows: full wall time per Multi run (run1 … runN)."""
    rows = list(per_run or [])
    if len(rows) < 2:
        return ""
    bits = []
    for p in rows:
        ri = int(p.get("run") or 0)
        tot = _fmt_s_plain(p.get("total_s") or 0)
        c = _fmt_s_plain(p.get("collect_s") or 0)
        j = _fmt_s_plain(p.get("judge_s") or 0)
        bits.append(f"#{ri} {tot} <span class='t-min'>(c{c}+j{j})</span>")
    return (
        f'<div class="t-per"><span class="lab">per run</span>'
        f'<span class="val">{" · ".join(bits)}</span></div>'
    )


def _run_timer_idle(last: dict | None = None) -> str:
    """Always-visible sidebar clock (idle or last finished timings) — static HTML."""
    last = last or {}
    total = last.get("total_s")
    if total is None:
        return """
<div class="run-timer-panel idle">
  <div class="t-title">Run clock</div>
  <div class="t-big">0s<span class="t-min"> (0,0m)</span></div>
  <div class="t-row"><span class="lab">collect</span><span class="val">0s<span class="t-min"> (0,0m)</span></span></div>
  <div class="t-row"><span class="lab">judge</span><span class="val">0s<span class="t-min"> (0,0m)</span></span></div>
  <hr class="t-sep"/>
  <span class="phase">Ready · starts when you click RUN</span>
</div>
"""
    n = int(last.get("n") or 1)
    this_row = ""
    if n > 1 and last.get("last_run_s") is not None:
        this_row = (
            f'<div class="t-row"><span class="lab">last run</span>'
            f'<span class="val">{_fmt_s_min(last["last_run_s"])}</span></div>'
        )
    per_html = _per_run_timer_rows(last.get("per_run"))
    c_s = int(last.get("collect_s") or 0)
    j_s = int(last.get("judge_s") or 0)
    tot_i = int(total)
    overlap_note = ""
    if c_s + j_s > tot_i + 1:
        overlap_note = " · collect∥judge overlap"
    return f"""
<div class="run-timer-panel idle">
  <div class="t-title">Run clock · last</div>
  <div class="t-big">{_fmt_s_min(total)}</div>
  {this_row}
  {per_html}
  <div class="t-row"><span class="lab">collect</span><span class="val">{_fmt_s_min(c_s)}</span></div>
  <div class="t-row"><span class="lab">judge</span><span class="val">{_fmt_s_min(j_s)}</span></div>
  <hr class="t-sep"/>
  <span class="phase">Done · final scores ready{overlap_note}</span>
</div>
"""


_TIMER_IFRAME_CSS = """
html,body{margin:0;padding:0;background:transparent;}
.run-timer-panel{
  background:linear-gradient(165deg,#1c1917 0%,#0f172a 55%,#111827 100%);
  border:1px solid #f59e0b;color:#fde68a;
  font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
  padding:0.7rem 0.75rem 0.65rem;border-radius:12px;
  box-shadow:0 10px 28px rgba(0,0,0,0.35);
}
.run-timer-panel .t-title{
  font-size:0.72rem;font-weight:600;letter-spacing:0.08em;
  text-transform:uppercase;color:#94a3b8;margin:0 0 0.5rem 0;
}
.run-timer-panel .t-big{
  font-size:1.95rem;font-weight:700;line-height:1.1;color:#fbbf24;margin:0 0 0.45rem 0;
  white-space:nowrap;
}
.run-timer-panel .t-min{
  font-size:0.52em;font-weight:500;opacity:0.75;letter-spacing:-0.02em;white-space:nowrap;
}
.run-timer-panel .t-row{
  display:flex;justify-content:space-between;align-items:baseline;gap:0.35rem;
  font-size:0.95rem;font-weight:600;margin:0.18rem 0;color:#fde68a;flex-wrap:nowrap;
}
.run-timer-panel .t-row .lab{color:#94a3b8;font-weight:500;font-size:0.82rem;flex-shrink:0;}
.run-timer-panel .t-row .val{
  font-variant-numeric:tabular-nums;color:#fef3c7;white-space:nowrap;flex-shrink:0;
}
.run-timer-panel .t-row .val .t-min{font-size:0.68em;}
.run-timer-panel .t-row.active .val{color:#fbbf24;}
.run-timer-panel .t-per{
  display:flex;flex-direction:column;gap:0.15rem;margin:0.25rem 0 0.35rem 0;
  font-size:0.72rem;font-weight:500;color:#cbd5e1;line-height:1.35;
}
.run-timer-panel .t-per .lab{
  font-size:0.68rem;letter-spacing:0.06em;text-transform:uppercase;color:#64748b;font-weight:600;
}
.run-timer-panel .t-per .val{
  font-variant-numeric:tabular-nums;color:#e2e8f0;word-break:break-word;white-space:normal;
}
.run-timer-panel .t-sep{border:0;border-top:1px solid #334155;margin:0.45rem 0;}
.run-timer-panel .phase{
  display:block;font-size:0.74rem;font-weight:500;color:#cbd5e1;margin-top:0.4rem;line-height:1.3;
}
.run-timer-panel.idle .t-big{color:#64748b;}
.run-timer-panel.idle .phase{color:#64748b;}
"""


def _flash_collect_done(*, n_answers: int = 0) -> None:
    """Non-blocking notice after collect; OK closes it, else auto-hides while judge runs."""
    try:
        st.toast("Collect done — DeepSeek judge starting…", icon="✅")
    except Exception:
        pass
    components.html(
        f"""
<div id="collect-done-flash" style="
  font-family: system-ui, sans-serif;
  background: #14532d; color: #dcfce7;
  border: 1px solid #22c55e; border-radius: 10px;
  padding: 0.65rem 0.8rem; display: flex; align-items: center;
  justify-content: space-between; gap: 0.75rem;
">
  <span style="font-size:0.85rem;font-weight:600;line-height:1.3;">
    Collect done · {int(n_answers)} answer(s) ready · judging continues below
  </span>
  <button type="button" onclick="this.parentElement.remove()" style="
    cursor:pointer;border:0;border-radius:8px;padding:0.35rem 0.7rem;
    background:#22c55e;color:#052e16;font-weight:700;font-size:0.78rem;
  ">OK</button>
</div>
<script>
setTimeout(function() {{
  var el = document.getElementById('collect-done-flash');
  if (el) el.remove();
}}, 5000);
</script>
""",
        height=64,
    )


def _paint_run_timer(
    slot,
    inner_html: str,
    *,
    height: int = 160,
    live: bool | None = None,
    multi: bool = False,
    per_run_n: int = 0,
) -> None:
    """
    Idle / stopped: plain HTML docked at bottom of left sidebar (no iframe).
    Live ticking: iframe for setInterval. Multi-run needs extra height for the
    "this run" / per-run rows so the clock bottom is not clipped.
    """
    if live is None:
        live = "setInterval" in (inner_html or "")
    n_pr = max(0, int(per_run_n or 0))
    # Multi / overlap / per-run strip need taller dock
    if multi or n_pr > 1 or 't-per' in (inner_html or "") or 'style="display:flex"' in (
        inner_html or ""
    ):
        extra = 18 * max(0, min(n_pr, 10) - 1) if n_pr > 1 else 0
        height = max(int(height or 200), 210 + extra)
        height = min(height, 320)
    else:
        height = min(int(height or 160), 178)
    docked = f'<div class="sidebar-timer-dock">{inner_html}</div>'
    with slot:
        if live:
            doc = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<style>{_TIMER_IFRAME_CSS}"
                "html,body{margin:0;padding:0;overflow:visible !important;height:auto;}"
                ".run-timer-panel{margin:0;}</style>"
                f"</head><body>{inner_html}</body></html>"
            )
            st.markdown(
                '<div class="sidebar-timer-dock" aria-hidden="true"></div>',
                unsafe_allow_html=True,
            )
            components.html(doc, height=height, scrolling=False)
        else:
            st.markdown(docked, unsafe_allow_html=True)


def _run_timer_live(
    phase: str,
    *,
    n_runs: int = 1,
    elapsed_total: float = 0.0,
    elapsed_this: float = 0.0,
    collect_base: int = 0,
    judge_base: int = 0,
    bucket: str = "collect",
) -> str:
    """Live ticking panel. Baselines from Python survive iframe remounts.

    ``bucket``:
      - collect — only collect ticks
      - judge — only judge ticks
      - both — collect∥judge overlap (pipelined DeepSeek)
      - other — freeze phase counters; total/this still tick
    """
    phase_js = json.dumps(phase)
    bkt = bucket if bucket in ("collect", "judge", "both", "other") else "other"
    bucket_js = json.dumps(bkt)
    multi = int(n_runs) > 1
    this_display = "flex" if multi else "none"
    et = max(0, int(round(float(elapsed_total))))
    eh = max(0, int(round(float(elapsed_this))))
    cb = max(0, int(round(float(collect_base))))
    jb = max(0, int(round(float(judge_base))))
    return f"""
<div class="run-timer-panel">
  <div class="t-title">Run clock</div>
  <div class="t-big"><span id="t-total">{_fmt_s_min(et)}</span></div>
  <div class="t-row" style="display:{this_display}">
    <span class="lab">this run</span><span class="val"><span id="t-this">{_fmt_s_min(eh)}</span></span>
  </div>
  <div class="t-row" id="row-c"><span class="lab">collect</span><span class="val"><span id="t-collect">{_fmt_s_min(cb)}</span></span></div>
  <div class="t-row" id="row-j"><span class="lab">judge</span><span class="val"><span id="t-judge">{_fmt_s_min(jb)}</span></span></div>
  <hr class="t-sep"/>
  <span class="phase" id="t-phase"></span>
</div>
<script>
(function() {{
  var phaseEl = document.getElementById('t-phase');
  if (phaseEl) phaseEl.textContent = {phase_js};
  var totEl = document.getElementById('t-total');
  var thisEl = document.getElementById('t-this');
  var colEl = document.getElementById('t-collect');
  var judEl = document.getElementById('t-judge');
  var rowC = document.getElementById('row-c');
  var rowJ = document.getElementById('row-j');
  var bucket = {bucket_js};
  if (rowC) rowC.classList.toggle('active', bucket === 'collect' || bucket === 'both');
  if (rowJ) rowJ.classList.toggle('active', bucket === 'judge' || bucket === 'both');
  var paintAt = Date.now();
  var baseTotal = {et};
  var baseThis = {eh};
  var cBase = {cb};
  var jBase = {jb};
  function fmt(s) {{
    var m = (s / 60).toFixed(1).replace('.', ',');
    return s + 's<span class="t-min"> (' + m + 'm)</span>';
  }}
  function paint() {{
    var add = Math.floor((Date.now() - paintAt) / 1000);
    var totalS = baseTotal + add;
    var thisS = baseThis + add;
    var tickC = (bucket === 'collect' || bucket === 'both');
    var tickJ = (bucket === 'judge' || bucket === 'both');
    var collectS = cBase + (tickC ? add : 0);
    var judgeS = jBase + (tickJ ? add : 0);
    if (totEl) totEl.innerHTML = fmt(totalS);
    if (thisEl) thisEl.innerHTML = fmt(thisS);
    if (colEl) colEl.innerHTML = fmt(collectS);
    if (judEl) judEl.innerHTML = fmt(judgeS);
  }}
  paint();
  setInterval(paint, 250);
}})();
</script>
"""


def _run_timer_stop(
    total_s: int,
    *,
    this_s: int | None = None,
    n_runs: int = 1,
    collect_s: int = 0,
    judge_s: int = 0,
    per_run: list | None = None,
    title: str = "Run clock · done",
    phase: str = "Done · final scores ready",
) -> str:
    dual = int(n_runs) > 1
    this_show = int(this_s) if this_s is not None else int(total_s)
    this_row = ""
    if dual:
        this_row = (
            f'<div class="t-row"><span class="lab">last run</span>'
            f'<span class="val">{_fmt_s_min(this_show)}</span></div>'
        )
    per_html = _per_run_timer_rows(per_run)
    c_s = int(collect_s)
    j_s = int(judge_s)
    tot_i = int(total_s)
    phase_out = phase
    if c_s + j_s > tot_i + 1 and "overlap" not in (phase or "").lower():
        phase_out = f"{phase} · collect∥judge overlap"
    return f"""
<div class="run-timer-panel">
  <div class="t-title">{title}</div>
  <div class="t-big">{_fmt_s_min(total_s)}</div>
  {this_row}
  {per_html}
  <div class="t-row"><span class="lab">collect</span><span class="val">{_fmt_s_min(c_s)}</span></div>
  <div class="t-row"><span class="lab">judge</span><span class="val">{_fmt_s_min(j_s)}</span></div>
  <hr class="t-sep"/>
  <span class="phase">{phase_out}</span>
</div>
"""
