"""Helpers for multi-run progressive KPI UI (Automated Benchmark)."""

from __future__ import annotations

import html
import json
import time
from typing import Any, Callable, Dict, List, Optional

from lib.model_labels import (
    filter_current_roster_rows,
    full_model_label,
    name_and_version,
    rerank_rows,
)

# #region agent log
_DEBUG_LOG_PATH = (
    "/Users/m1/QVAC vs Cloud LLMs - Health Test/.cursor/debug-a76cc5.log"
)


def _agent_dbg(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "a76cc5",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data,
                        "timestamp": int(time.time() * 1000),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    except Exception:
        pass


# #endregion


def na_failure_label(status: str, reason: str) -> str:
    """Compact UI label for technical N/A; include missing section ids when known."""
    low = (reason or "").lower()
    st = (status or "").lower()
    if (
        st == "candidate_partial"
        or "missing required sections" in low
        or "partial candidate" in low
    ):
        detail = ""
        marker = "missing required sections:"
        if marker in low:
            idx = low.find(marker)
            detail = (reason or "")[idx + len(marker) :].strip()
            if len(detail) > 48:
                detail = detail[:45].rstrip(", ") + "…"
        return (
            f"N/A · missing sections ({detail})"
            if detail
            else "N/A · missing sections"
        )
    if st == "candidate_empty" or "empty answer" in low:
        return "N/A · empty"
    if st == "collect_failed" or "candidate error" in low:
        return "N/A · collect error"
    return "N/A · technical"


class LiveJudgingBoard:
    """FIFO judge queue + provisional ranking board (same UX as graded Multi).

    Wire ``on_progress`` to ``PipelinedJudge``, call ``ensure_queued`` when a
    candidate finishes collect, and ``paint`` / progress bar update themselves.
    """

    def __init__(
        self,
        *,
        title: str = "Live judging · collect order + provisional ranking",
        label_by_key: Optional[Dict[str, str]] = None,
        status_boxes: Optional[Dict[str, Any]] = None,
        status_pill_fn: Optional[Callable[[str, str], str]] = None,
    ) -> None:
        self.title = title
        self.label_by_key = dict(label_by_key or {})
        self.status_boxes = status_boxes or {}
        self.status_pill_fn = status_pill_fn
        self.board: Dict[str, Dict[str, Any]] = {}
        self.started: set[str] = set()
        self.highlight: Optional[str] = None
        self.queue_i = 0
        self.board_slot: Any = None
        self.progress_slot: Any = None
        self.judge_status: Any = None

    def bind(
        self,
        *,
        board_slot: Any,
        progress_slot: Any = None,
        judge_status: Any = None,
    ) -> "LiveJudgingBoard":
        self.board_slot = board_slot
        self.progress_slot = progress_slot
        self.judge_status = judge_status
        return self

    def paint(self) -> None:
        if self.board_slot is None:
            return
        self.board_slot.markdown(
            live_judging_board_html(
                self.board,
                highlight_key=self.highlight,
                title=self.title,
            ),
            unsafe_allow_html=True,
        )

    def ensure_queued(self, key: str, label: Optional[str] = None) -> None:
        """Append to FIFO board as soon as collect finishes (before DeepSeek returns)."""
        if not key or key in self.started:
            return
        self.started.add(key)
        self.queue_i = int(self.queue_i) + 1
        self.board[key] = {
            "label": label or self.label_by_key.get(key) or key,
            "status": "judging",
            "accuracy": None,
            "queue_i": self.queue_i,
            "progress_pct": 10,
            "progress_label": "queued",
            "elapsed_s": 0,
        }
        self.paint()

    def _set_status_pill(self, key: str, kind: str, text: str) -> None:
        box = self.status_boxes.get(key) if self.status_boxes else None
        if box is None or self.status_pill_fn is None:
            return
        box.markdown(self.status_pill_fn(kind, text), unsafe_allow_html=True)

    def on_progress(self, evt: dict) -> None:
        phase = evt.get("phase")
        key = str(evt.get("key") or "")
        name = self.label_by_key.get(key) or evt.get("label") or key
        done_n = int(evt.get("done") or 0)
        tot = int(evt.get("total") or max(1, done_n))

        if phase == "queued" and key and key not in self.started:
            self.started.add(key)
            if key not in self.board or self.board[key].get("status") not in (
                "judging",
                "scored",
                "failed",
            ):
                self.queue_i = int(self.queue_i) + 1
                qi = self.queue_i
            else:
                qi = (self.board.get(key) or {}).get("queue_i") or self.queue_i
            self.board[key] = {
                "label": name,
                "status": "judging",
                "accuracy": None,
                "queue_i": qi,
                "progress_pct": int(evt.get("percent") or 10),
                "progress_label": str(evt.get("stage") or "queued"),
                "elapsed_s": float(evt.get("elapsed_s") or 0),
            }
            self.paint()
        elif phase == "progress" and key:
            prev = self.board.get(key) or {}
            if prev.get("status") == "scored" or (
                prev.get("status") == "failed" and not evt.get("active_attempt")
            ):
                pass
            else:
                self.board[key] = {
                    **prev,
                    "label": name,
                    "status": "judging",
                    "accuracy": None,
                    "progress_pct": int(evt.get("percent") or 10),
                    "progress_label": str(evt.get("stage") or "judging"),
                    "elapsed_s": float(evt.get("elapsed_s") or 0),
                }
                self.paint()
        elif phase == "retry" and key:
            prev = self.board.get(key) or {}
            if prev.get("status") == "scored" or (
                prev.get("status") == "failed" and not evt.get("active_attempt")
            ):
                pass
            else:
                self.board[key] = {
                    **prev,
                    "label": name,
                    "status": "judging",
                    "accuracy": None,
                    "progress_pct": int(evt.get("percent") or 75),
                    "progress_label": str(evt.get("stage") or "corrective retry"),
                    "elapsed_s": float(evt.get("elapsed_s") or 0),
                }
                self.paint()
        elif phase in ("done", "retry_done") and key:
            prev_q = (self.board.get(key) or {}).get("queue_i")
            if evt.get("failed"):
                reason = str(
                    evt.get("failure_reason")
                    or evt.get("note")
                    or evt.get("status")
                    or ""
                )
                status = str(evt.get("status") or "").lower()
                na_label = na_failure_label(status, reason)
                self.board[key] = {
                    "label": name,
                    "status": "failed",
                    "accuracy": None,
                    "queue_i": prev_q,
                    "progress_pct": 100,
                    "progress_label": "complete",
                    "elapsed_s": float(evt.get("elapsed_s") or 0),
                }
                self._set_status_pill(key, "err", na_label)
            else:
                acc = float(evt.get("accuracy") or 0)
                self.board[key] = {
                    "label": name,
                    "status": "scored",
                    "accuracy": acc,
                    "coverage": evt.get("coverage"),
                    "quality": evt.get("quality"),
                    "discipline": evt.get("discipline"),
                    "queue_i": prev_q,
                    "progress_pct": 100,
                    "progress_label": "complete",
                    "elapsed_s": float(evt.get("elapsed_s") or 0),
                }
                self._set_status_pill(key, "done", f"Judged · {acc:.0f}%")
            self.highlight = key
            self.paint()
            if self.judge_status is not None:
                try:
                    self.judge_status.update(
                        label=f"DeepSeek R1 · {done_n}/{tot} scored · pipelined",
                        state="running",
                    )
                except Exception:
                    pass

        if self.progress_slot is not None:
            self.progress_slot.progress(
                min(1.0, done_n / max(1, tot)),
                text=f"Judge · {done_n}/{tot} (overlap with collect)",
            )


def short_model(key: str) -> str:
    """Full Name · Version (kept name for call-site compatibility)."""
    return full_model_label(key or "")


def model_name_version(key: str) -> tuple:
    return name_and_version(key or "")


def ops_reliability_has_scan_data(ops_rows: Optional[list]) -> bool:
    """True when Rebuild fill-N ops rows include any scan-window observations."""
    return sum(int(r.get("n_seen") or 0) for r in (ops_rows or [])) > 0


def paint_rebuild_ops_reliability_panels(
    st_mod: Any,
    ops_rows: list,
    *,
    n_per_model_cap: Any = None,
    chart_key: str = "rebuild_ops_chart",
    table_footer_html: Optional[str] = None,
    chart_footer_html: Optional[str] = None,
) -> bool:
    """Render Rebuild Failures/N/A table only (no fourth ops chart KPI).

    Shared by Structured and Comprehension rebuild mean dialogs.
    ``chart_key`` / ``chart_footer_html`` kept for call-site compat (unused).
    ``st_mod`` is the Streamlit module (or a test double). Returns True when
    the table was painted.
    """
    rows = list(ops_rows or [])
    if not ops_reliability_has_scan_data(rows):
        return False

    _ = (n_per_model_cap, chart_key, chart_footer_html)
    st_mod.markdown("##### Failures / N/A · relative % (like live Multi Failed %)")
    st_mod.caption(
        "Honesty view of the Rebuild fill-N scan window — same role as the "
        "Failed column on the live Multi ranking table: counts + relative % "
        "of exact Clinical Composite == 0 and technical N/A. Excluded from "
        "the scored-only mean above — not a clinical ranking."
    )
    st_mod.markdown(
        ops_reliability_table_html(rows),
        unsafe_allow_html=True,
    )
    if table_footer_html:
        st_mod.markdown(table_footer_html, unsafe_allow_html=True)
    return True


def ops_reliability_table_html(ops_rows: list) -> str:
    """Dashboard-styled failures / N/A table with relative % (Rebuild honesty).

    Mirrors the live Multi Failed column idea (counts + %), but for the Rebuild
    fill-N scan window: scored vs exact-zero vs technical N/A. Not a clinical
    ranking.
    """
    rows = [
        r
        for r in filter_current_roster_rows(ops_rows or [])
        if int(r.get("n_seen") or 0) > 0
    ]
    if not rows:
        return (
            "<div style='font-size:0.85rem;color:#94a3b8;margin:0.35rem 0 0.75rem'>"
            "No scan-window observations for failures / N/A.</div>"
        )
    _td = "padding:0.45rem 0.55rem;border-bottom:1px solid #1e293b"
    body = []
    for r in rows:
        nm, ver = name_and_version(
            str(r.get("key") or ""),
            label=r.get("label"),
            model=r.get("model"),
        )
        n_seen = int(r.get("n_seen") or 0)
        n_scored = int(r.get("n_scored") or 0)
        n_zero = int(r.get("n_zero") or 0)
        n_na = int(r.get("n_technical_na") or 0)
        n_excl = int(r.get("n_excluded") or (n_zero + n_na))
        try:
            pct_scored = float(r.get("pct_scored") or 0)
            pct_zero = float(r.get("pct_zero") or 0)
            pct_na = float(r.get("pct_technical_na") or 0)
            pct_excl = float(r.get("pct_excluded") or (pct_zero + pct_na))
        except (TypeError, ValueError):
            pct_scored = pct_zero = pct_na = pct_excl = 0.0
        excl_color = "#fca5a5" if n_excl else "#94a3b8"
        na_color = "#fca5a5" if n_na else "#94a3b8"
        zero_color = "#fbbf24" if n_zero else "#94a3b8"
        body.append(
            "<tr>"
            f"<td style='{_td};font-weight:600;color:#e2e8f0'>{html.escape(nm)}</td>"
            f"<td style='{_td};color:#94a3b8'>{html.escape(ver)}</td>"
            f"<td style='{_td};text-align:right;color:#86efac;font-variant-numeric:tabular-nums'>"
            f"{n_scored} ({pct_scored:.0f}%)</td>"
            f"<td style='{_td};text-align:right;color:{zero_color};font-variant-numeric:tabular-nums'>"
            f"{n_zero} ({pct_zero:.0f}%)</td>"
            f"<td style='{_td};text-align:right;color:{na_color};font-variant-numeric:tabular-nums'>"
            f"{n_na} ({pct_na:.0f}%)</td>"
            f"<td style='{_td};text-align:right;color:{excl_color};font-weight:700;"
            f"font-variant-numeric:tabular-nums'>"
            f"{n_excl} ({pct_excl:.0f}%)</td>"
            f"<td style='{_td};text-align:right;color:#e2e8f0;font-variant-numeric:tabular-nums'>"
            f"{n_seen}</td>"
            "</tr>"
        )
    footer = (
        "<b>Failed / excluded %</b> = exact Clinical Composite == 0 + technical N/A "
        "in the Rebuild fill-N scan window (same idea as live Multi Failed %) — "
        "not a clinical zero. These observations are excluded from the scored-only "
        "mean above. Chart below stacks scored / zero / N/A as relative % of "
        "n seen per model."
    )
    return (
        "<div style='overflow-x:auto;margin:0.35rem 0 0.75rem;border:1px solid #334155;"
        "border-radius:12px;background:#0f172a'>"
        "<table style='width:100%;border-collapse:collapse;color:#e2e8f0;font-size:0.9rem'>"
        "<thead><tr style='color:#94a3b8;text-align:left;font-size:0.75rem;"
        "letter-spacing:0.04em;text-transform:uppercase'>"
        "<th style='padding:0.55rem'>Name</th>"
        "<th style='padding:0.55rem'>Version</th>"
        "<th style='padding:0.55rem;text-align:right'>Scored</th>"
        "<th style='padding:0.55rem;text-align:right'>Exact zero</th>"
        "<th style='padding:0.55rem;text-align:right'>Technical N/A</th>"
        "<th style='padding:0.55rem;text-align:right'>Failed / excluded</th>"
        "<th style='padding:0.55rem;text-align:right'>n seen</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
        "<div style='font-size:0.78rem;color:#94a3b8;margin-bottom:0.5rem'>"
        f"{footer}</div>"
    )


def _is_na_rank_row(row: Dict[str, Any]) -> bool:
    status = str(row.get("status") or "ok").lower()
    if status in {"n/a", "na", "failed"}:
        return True
    return row.get("accuracy") is None and status != "ok"


def snapshot_from_artifact(art: Any) -> Dict[str, Any]:
    """Lightweight per-run snapshot for session state + tabs."""
    ranking = []
    for r in art.ranking or []:
        status = str(r.get("status") or "ok")
        failed = _is_na_rank_row(r)
        ranking.append(
            {
                "key": r.get("key"),
                "rank": None if failed else r.get("rank"),
                "accuracy": None if failed else r.get("accuracy"),
                "label": r.get("label") or r.get("key"),
                "status": "n/a" if failed else status,
                "status_note": r.get("status_note") or "",
                "ttft_s": r.get("ttft_s"),
                "tps": r.get("tps"),
                "cost_usd": r.get("cost_usd"),
                "ram_mb": r.get("ram_mb"),
            }
        )
    dims = []
    for j in art.judgments or []:
        by_q = {qs.question_id: qs.score for qs in (j.question_scores or [])}
        status = str(getattr(j, "status", None) or "valid")
        failed = status != "valid"
        dims.append(
            {
                "key": j.candidate_key,
                "weighted": None if failed else j.weighted_accuracy,
                "status": "n/a" if failed else "ok",
                "scores": {} if failed else by_q,
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
    Mean Clinical Composite Score, ±std and CV% appear when the batch finishes.
    Use the <b>Run tabs</b> below for each finished run.
  </div>
</div>
"""


def finished_multi_progress(
    completed: List[Dict[str, Any]],
    *,
    n_total: int,
    paths: Optional[List[str]] = None,
    aborted_early: bool = False,
) -> Dict[str, Any]:
    """Session payload when a multi-run batch ends (success, abort, or cancel).

    Always sets ``batch_done=True`` so the progressive strip never stays forever
    on "Waiting for all runs…" after an early abort with fewer than 2 artifacts.
    """
    done = list(completed or [])
    return {
        "completed": done,
        "n_total": int(n_total),
        "batch_done": True,
        "aborted_early": bool(aborted_early),
        "completed_runs": len(done),
        "requested_runs": int(n_total),
        "paths": list(paths or []),
    }


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


def _is_terminal_failure_row(row: Dict[str, Any]) -> bool:
    """Technical N/A / failed board row — never treat as in-flight or 0%."""
    st = str(row.get("status") or "").lower()
    return st in {"failed", "n/a", "na"}


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
        if _is_terminal_failure_row(r):
            scored.append(r)
        elif r.get(score_field) is not None and st in ("scored", "ok", ""):
            scored.append(r)
        elif include_pending:
            pending.append(r)

    scored = sorted(
        scored,
        key=lambda r: (
            1 if _is_terminal_failure_row(r) else 0,
            -float(r.get(score_field) or 0)
            if r.get(score_field) is not None
            and not _is_terminal_failure_row(r)
            else 0.0,
        ),
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
        failed = _is_terminal_failure_row(r)
        acc = float(r.get(score_field) or 0) if not failed else 0.0
        width = 0.0 if failed else max(0.0, min(100.0, acc))
        flash = " hist-bar-flash" if highlight_key and key == highlight_key else ""
        color = "#7f1d1d" if failed else _bar_color(key)
        num = "N/A" if failed else f"{acc:.1f}%"
        rank_cell = "—" if failed else str(i)
        bars.append(
            f'<div class="hist-row{flash}">'
            f'<div class="hist-label">'
            f'<span class="hist-rank"><span class="rank-prov-tag">Prov.</span> {rank_cell}</span>'
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
        if (
            str(r.get("status") or "") == "scored" and r.get("accuracy") is not None
        )
        or _is_terminal_failure_row(r)
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
        progress_pct = max(0, min(100, int(r.get("progress_pct") or 0)))
        elapsed_s = max(0, int(float(r.get("elapsed_s") or 0)))
        flash = " rank-row-flash" if highlight_key and key == highlight_key else ""
        # Terminal N/A must render even when accuracy is None (app sets failed + None).
        if _is_terminal_failure_row(r) or (
            st == "scored" and r.get("accuracy") is not None
        ):
            failed = _is_terminal_failure_row(r)
            if failed:
                acc_col = (
                    "<span class='rank-live-acc-num fail'>—</span>"
                    "<span class='rank-live-note'>N/A · technical</span>"
                )
            else:
                acc = float(r.get("accuracy") or 0)
                subscales = ""
                if all(
                    r.get(component) is not None
                    for component in ("coverage", "quality", "discipline")
                ):
                    subscales = (
                        "<span class='rank-live-note'>"
                        f"C {float(r['coverage']):.0f} · "
                        f"Q {float(r['quality']):.0f} · "
                        f"D {float(r['discipline']):.0f}"
                        "</span>"
                    )
                acc_col = (
                    f"<span class='rank-live-acc-num'>{acc:.1f}</span>"
                    f"<span class='rank-live-acc-unit'>%</span>"
                    f"{subscales}"
                )
            acc_col += (
                f"<span class='rank-live-note'>100% complete · {elapsed_s}s</span>"
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
            progress_label = str(r.get("progress_label") or waiting)
            progress = (
                "<div class='rank-live-progress-meta'>"
                f"<span>{progress_pct}%</span><span>{html.escape(progress_label)}</span>"
                "</div>"
                "<div class='rank-live-progress-track'>"
                f"<span style='width:{progress_pct}%'></span></div>"
                f"<div class='rank-live-note'>{elapsed_s}s elapsed</div>"
            )
            body.append(
                f"<tr class='rank-live-row pending' data-key='{html.escape(key)}'>"
                f"<td class='rank-live-pos' style='color:#64748b'>{qi}</td>"
                f"<td class='rank-live-name' style='opacity:.85'>{html.escape(nm)}"
                f"<div class='rank-live-ver'>{html.escape(ver)}</div></td>"
                f"<td class='rank-live-acc'>{progress}</td>"
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
        "<th>#</th><th>Model</th><th>Progress / score</th>"
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
        f"{' · latest highlighted' if highlight_key else ''}"
        " · progress % = completed pipeline stages, not ETA</div>"
        f'<div class="rank-live-grid">'
        f'<div class="rank-live-col table-col">'
        f'<div class="rank-live-hist-cap">Judge queue · FIFO</div>{table}</div>'
        f'<div class="rank-live-col hist-col">'
        f'<div class="rank-live-hist-cap">Provisional Clinical Composite Score</div>{hist}</div>'
        f"</div></div>"
    )


def _partial_chip_html() -> str:
    return (
        "<span style='display:inline-block;margin-left:0.3rem;padding:0.05rem 0.35rem;"
        "border-radius:999px;font-size:0.65rem;font-weight:700;letter-spacing:0.04em;"
        "text-transform:uppercase;color:#78350f;background:#fbbf24;"
        "border:1px solid #f59e0b;vertical-align:middle'>partial</span>"
    )


def _ranking_table_html(ranking: List[Dict[str, Any]]) -> str:
    rows = sorted(
        ranking or [],
        key=lambda r: (
            0 if not _is_na_rank_row(r) else 1,
            -float(r.get("accuracy") if r.get("accuracy") is not None else -1),
        ),
    )
    if not rows:
        return "<p style='color:#94a3b8'>No ranking.</p>"
    n_failed = sum(1 for r in rows if _is_na_rank_row(r))
    body = []
    for r in rows:
        nm, ver = name_and_version(
            str(r.get("key") or ""),
            label=r.get("label"),
            model=r.get("model"),
        )
        failed = _is_na_rank_row(r)
        is_partial = bool(r.get("partial")) or failed
        row_bg = "background:rgba(251,191,36,0.10);" if is_partial else ""
        if failed:
            rank_cell = f"— · {_partial_chip_html()}"
            score_cell = (
                "<span style='color:#f87171;font-weight:700'>N/A</span>"
                "<div style='color:#94a3b8;font-size:0.72rem;font-weight:500'>"
                "technical</div>"
            )
        else:
            rank_n = r.get("rank") or "—"
            rank_cell = (
                f"#{rank_n} · {_partial_chip_html()}"
                if is_partial
                else f"#{rank_n}"
            )
            acc = float(r.get("accuracy") or 0)
            score_cell = f"{acc:.1f}%"
        body.append(
            f"<tr style='{row_bg}'>"
            f"<td style='padding:0.3rem 0.45rem;border-bottom:1px solid #1e293b'>"
            f"{rank_cell}</td>"
            f"<td style='padding:0.3rem 0.45rem;border-bottom:1px solid #1e293b'>"
            f"{html.escape(nm)}</td>"
            f"<td style='padding:0.3rem 0.45rem;border-bottom:1px solid #1e293b;"
            f"color:#94a3b8;font-size:0.8rem'>{html.escape(ver)}</td>"
            f"<td style='padding:0.3rem 0.45rem;border-bottom:1px solid #1e293b;font-weight:800;"
            f"color:#fbbf24;font-size:1.05rem;text-align:right'>{score_cell}</td>"
            "</tr>"
        )
    banner = ""
    if n_failed:
        banner = (
            "<div style='margin:0 0 0.45rem;padding:0.4rem 0.55rem;border-radius:8px;"
            "border:1px solid #f59e0b;background:rgba(251,191,36,0.12);"
            "color:#fde68a;font-size:0.78rem'>"
            f"{_partial_chip_html()} "
            "<b style='color:#fbbf24'>Partial run.</b> "
            "Scored models keep their ranks; technical N/A stay listed "
            "(not clinical 0%).</div>"
        )
    return (
        banner
        + "<table style='width:100%;border-collapse:collapse;font-size:0.88rem;color:#e2e8f0'>"
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
        key=lambda r: (
            0 if not _is_na_rank_row(r) else 1,
            -float(r.get("accuracy") if r.get("accuracy") is not None else -1),
        ),
    )
    return (
        '<div class="rank-live-grid run-summary-grid">'
        f'<div class="rank-live-col table-col">{_ranking_table_html(rows)}</div>'
        f'<div class="rank-live-col hist-col">'
        f'<div class="rank-live-hist-cap">Clinical Composite Score</div>'
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
        valid = [r for r in ranking if not _is_na_rank_row(r)]
        if valid:
            best = max(valid, key=lambda r: float(r.get("accuracy") or 0))
            top = (
                f"{short_model(str(best.get('key')))} "
                f"{float(best.get('accuracy') or 0):.1f}%"
            )
        tab_label = str(snap.get("tab_label") or "").strip() or f"Run {i}"
        modal_title = (
            str(snap.get("modal_title") or "").strip()
            or f"{tab_label} · table + histogram"
        )
        chips.append(
            f'<button type="button" onclick="document.getElementById(\'{mid}\').style.display=\'flex\'" '
            f'style="display:inline-flex;align-items:center;gap:0.35rem;cursor:pointer;'
            f'padding:0.4rem 0.75rem;border-radius:999px;margin:0.15rem 0.3rem 0.15rem 0;'
            f'background:#1e293b;border:1px solid #fbbf24;color:#e2e8f0;font-size:0.84rem;">'
            f'<b style="color:#fbbf24">{html.escape(tab_label)}</b>'
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
            f"{html.escape(modal_title)}</div>"
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
        if pending > 0 and not batch_done
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
    valid = [r for r in ranking or [] if not _is_na_rank_row(r)]
    if valid:
        best = max(valid, key=lambda r: float(r.get("accuracy") or 0))
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


# CV% band colors — same thresholds as benchmark.report.reliability_from_cv
# Super High ≤5 · High ≤10 · Medium ≤15 · Low ≤20 · else Very Low
RELIABILITY_BAND_COLORS = {
    "super_high": ("#064e3b", "#6ee7b7", "Super High"),
    "high": ("#14532d", "#86efac", "High"),
    "medium": ("#713f12", "#fde047", "Medium"),
    "low": ("#9a3412", "#fdba74", "Low"),
    "very_low": ("#7f1d1d", "#fca5a5", "Very Low"),
}
_RELIABILITY_FALLBACK = ("#1e293b", "#94a3b8", "—")


def reliability_band_colors(level: str) -> tuple:
    """Return (bg, fg, label) for a CV reliability band key."""
    key = (level or "").strip().lower()
    if key in RELIABILITY_BAND_COLORS:
        return RELIABILITY_BAND_COLORS[key]
    return (*_RELIABILITY_FALLBACK[:2], (level or "—").replace("_", " ").title() or "—")


def reliability_band_from_cv(cv_pct: Optional[float]) -> str:
    """Map CV% → band key; None / non-finite → empty (neutral style)."""
    if cv_pct is None:
        return ""
    try:
        from benchmark.report import reliability_from_cv

        return reliability_from_cv(float(cv_pct))
    except (TypeError, ValueError):
        return ""


def reliability_badge(level: str) -> str:
    bg, fg, lab = reliability_band_colors(level)
    return (
        f'<span style="display:inline-block;padding:0.12rem 0.45rem;border-radius:999px;'
        f'background:{bg};color:{fg};font-size:0.72rem;font-weight:700;">{lab}</span>'
    )


def cv_reliability_cells_html(
    cv_pct: Optional[float],
    *,
    td_style: str = "padding:0.45rem 0.55rem;border-bottom:1px solid #1e293b",
) -> tuple:
    """CV% <td> + Reliability badge HTML, both tinted with the legend band colors.

    Returns (cv_td_html, badge_html, band_key).
    """
    band = reliability_band_from_cv(cv_pct)
    bg, fg, lab = reliability_band_colors(band)
    badge = (
        f'<span style="display:inline-block;min-width:4.2rem;text-align:center;'
        f'padding:0.2rem 0.45rem;border-radius:999px;background:{bg};color:{fg};'
        f'font-size:0.72rem;font-weight:800;letter-spacing:0.04em;">'
        f"{html.escape(lab.upper())}</span>"
    )
    if band and cv_pct is not None:
        cv_td = (
            f"<td style='{td_style};background:{bg};color:{fg};font-weight:700;"
            f"text-align:right;border-radius:6px'>{float(cv_pct):.1f}%</td>"
        )
    else:
        cv_td = f"<td style='{td_style};color:#64748b;text-align:right'>—</td>"
    return cv_td, badge, band


def _partial_badge_html() -> str:
    """Compact English badge for incomplete mean / single-run coverage."""
    return (
        "<span style='display:inline-block;margin-left:0.35rem;padding:0.08rem 0.4rem;"
        "border-radius:999px;font-size:0.68rem;font-weight:700;letter-spacing:0.04em;"
        "text-transform:uppercase;color:#78350f;background:#fbbf24;"
        "border:1px solid #f59e0b;vertical-align:middle'>partial</span>"
    )


def _score_bar_cell_html(
    value,
    *,
    bar_from: str = "#38bdf8",
    bar_to: str = "#7dd3fc",
    label_color: str = "#e2e8f0",
    bold: bool = False,
) -> str:
    """Conditional bar + label for a 0–100 Clinical Composite-style score."""
    if value is None:
        return "<span style='color:#94a3b8'>N/A</span>"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "<span style='color:#94a3b8'>N/A</span>"
    pct = max(0.0, min(100.0, val))
    weight = "700" if bold else "500"
    size = "1.05rem" if bold else "0.85rem"
    return (
        "<div style='display:flex;align-items:center;gap:0.45rem;min-width:8.2rem'>"
        "<div style='flex:1;height:0.55rem;border-radius:999px;background:#1e293b;"
        "overflow:hidden;border:1px solid #334155'>"
        f"<div style='width:{pct:.1f}%;height:100%;background:linear-gradient("
        f"90deg,{bar_from},{bar_to});border-radius:999px'></div></div>"
        f"<span style='color:{label_color};font-weight:{weight};font-size:{size};"
        f"font-variant-numeric:tabular-nums;min-width:2.8rem;text-align:right'>"
        f"{val:.1f}%</span>"
        "</div>"
    )


def _reliability_table_html(
    ranking_mean: list,
    *,
    successful_only: bool = False,
    rank_by: str = "mean",
) -> str:
    """Mean ranking table: CV% + Reliability cells tinted by CV band (legend).

    Shared by graded dashboard and Beta comprehension so Rebuild / Multi mean
    boards stay pixel-identical (bars, CV colors, C/Q/D, footer legend).

    ``successful_only`` (Rebuild mean): no yellow partial badge/banner, no Failed
    column theater — pool is already clean successful scored observations.
    ``rank_by``: ``mean`` (default) or ``median`` — updates # order only.
    """
    score_field = "median" if str(rank_by).lower() == "median" else "accuracy_mean"
    _n_in = len(ranking_mean or [])
    ranking_mean = rerank_rows(
        filter_current_roster_rows(ranking_mean or []),
        score_field=score_field,
    )
    _n_after_roster = len(ranking_mean)
    if successful_only:
        ranking_mean = [
            r
            for r in ranking_mean
            if r.get("eligible", True)
            and r.get("accuracy_mean") is not None
            and r.get("rank") is not None
        ]
    _n_after_clean = len(ranking_mean)
    _td = "padding:0.45rem 0.55rem;border-bottom:1px solid #1e293b"
    n_partial = (
        0
        if successful_only
        else sum(1 for r in ranking_mean if r.get("partial"))
    )

    def _fmt_pct(value, *, digits: int = 1, suffix: str = "%") -> str:
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.{digits}f}{suffix}"
        except (TypeError, ValueError):
            return "N/A"

    rows_html = []
    for r in ranking_mean:
        raw_cv = r.get("cv_pct")
        try:
            cv_val = float(raw_cv) if raw_cv is not None else None
        except (TypeError, ValueError):
            cv_val = None
        cv_cell, badge, _band = cv_reliability_cells_html(cv_val, td_style=_td)
        nm, ver = name_and_version(
            str(r.get("key") or ""),
            label=r.get("label"),
            model=r.get("model"),
        )
        n_runs = int(r.get("n_runs") or r.get("n") or 0)
        n_req = int(r.get("n_requested") or n_runs)
        n_failed = 0 if successful_only else int(r.get("n_failed") or 0)
        try:
            fail_pct = (
                0.0
                if successful_only
                else 100.0 * float(r.get("failure_rate") or 0)
            )
        except (TypeError, ValueError):
            fail_pct = 0.0
        rank = r.get("rank")
        is_partial = (not successful_only) and bool(r.get("partial"))
        if rank is None:
            rank_cell = "—"
        elif is_partial:
            rank_cell = f"#{rank} · {_partial_badge_html()}"
        else:
            rank_cell = f"#{rank}"
        mean = r.get("accuracy_mean")
        cov = r.get("coverage_mean")
        qual = r.get("quality_mean")
        disc = r.get("discipline_mean")
        if cov is None and qual is None and disc is None:
            cqd = "N/A"
        else:
            cqd = (
                f"{_fmt_pct(cov, digits=0, suffix='')}/"
                f"{_fmt_pct(qual, digits=0, suffix='')}/"
                f"{_fmt_pct(disc, digits=0, suffix='')}"
            )
        std = r.get("std")
        med = r.get("median")
        mn = r.get("min")
        mx = r.get("max")
        range_cell = (
            "N/A"
            if mn is None or mx is None
            else f"{float(mn):.0f}–{float(mx):.0f}"
        )
        fail_color = "#fca5a5" if n_failed else "#94a3b8"
        row_bg = (
            "background:rgba(251,191,36,0.10);" if is_partial else ""
        )
        mean_color = "#f59e0b" if is_partial else "#fbbf24"
        fail_cell = (
            ""
            if successful_only
            else (
                f"<td style='{_td};color:{fail_color};text-align:right'>"
                f"{n_failed} ({fail_pct:.0f}%)</td>"
            )
        )
        runs_cell = (
            f"<td style='{_td};font-weight:700;color:#e2e8f0;text-align:right'>"
            f"{n_runs}</td>"
            if successful_only
            else (
                f"<td style='{_td};font-weight:700;color:#e2e8f0;text-align:right'>"
                f"{n_runs}"
                f"<span style='color:#64748b;font-weight:500;font-size:0.8rem'>"
                f"/{n_req}</span></td>"
            )
        )
        med_bar = _score_bar_cell_html(
            med,
            bar_from="#38bdf8",
            bar_to="#7dd3fc",
            label_color="#bae6fd",
            bold=False,
        )
        rows_html.append(
            f"<tr style='{row_bg}'>"
            f"<td style='{_td}'>{rank_cell}</td>"
            f"<td style='{_td};font-weight:600'>{html.escape(nm)}</td>"
            f"<td style='{_td};color:#cbd5e1;font-size:0.85rem'>"
            f"{html.escape(ver)}</td>"
            f"<td style='{_td}'>"
            f"{_score_bar_cell_html(mean, bar_from='#f59e0b' if is_partial else '#fbbf24', bar_to='#fde68a' if is_partial else '#fcd34d', label_color=mean_color, bold=True)}"
            f"</td>"
            f"<td style='{_td};color:#cbd5e1'>{cqd}</td>"
            f"<td style='{_td};color:#cbd5e1'>"
            f"{'—' if std is None else f'± {float(std):.1f}'}</td>"
            f"{cv_cell}"
            f"<td style='{_td}'>{badge}</td>"
            f"<td style='{_td}'>{med_bar}</td>"
            f"<td style='{_td};color:#64748b;font-size:0.85rem'>"
            f"{range_cell}</td>"
            f"{runs_cell}"
            f"{fail_cell}"
            "</tr>"
        )
    banner = ""
    if n_partial:
        banner = (
            "<div style='margin:0.35rem 0 0.55rem;padding:0.55rem 0.75rem;"
            "border-radius:10px;border:1px solid #f59e0b;"
            "background:rgba(251,191,36,0.12);color:#fde68a;font-size:0.85rem'>"
            f"{_partial_badge_html()} "
            "<b style='color:#fbbf24'>Partial results in this mean window.</b> "
            "Models with technical N/A stay ranked by the mean of scored runs "
            f"({n_partial} model"
            f"{'s' if n_partial != 1 else ''}). "
            "Failed % is not a clinical zero.</div>"
        )
    fail_th = (
        ""
        if successful_only
        else "<th style='padding:0.55rem;text-align:right'>Failed</th>"
    )
    runs_th = (
        "<th style='padding:0.55rem;text-align:right'>n scored</th>"
        if successful_only
        else "<th style='padding:0.55rem;text-align:right'>Runs</th>"
    )
    footer = (
        (
            f"{reliability_badge('super_high')} CV ≤ 5% &nbsp; "
            f"{reliability_badge('high')} CV ≤ 10% &nbsp; "
            f"{reliability_badge('medium')} CV ≤ 15% &nbsp; "
            f"{reliability_badge('low')} CV ≤ 20% &nbsp; "
            f"{reliability_badge('very_low')} CV &gt; 20% &nbsp;·&nbsp; "
            "cell color = CV band · lower CV = stabler mean · "
            "<b>CV band ≠ clinical validation</b> · "
            "<b>n scored</b> = last ≤N error-free non-zero scored runs per model "
            "(technical N/A and exact-zero composites skipped; older successful "
            "History used) · Failed%/zeros live in the separate ops reliability "
            "chart below · models with only failures are omitted · "
            "N=5 exploratory · ~10 better for CV eye-check · 20–50 diminishing · 100 max · "
            "<b>C/Q/D</b> = coverage / quality / discipline "
            "(quality is independent of coverage; a high board % can still have low C)"
        )
        if successful_only
        else (
            f"{reliability_badge('super_high')} CV ≤ 5% &nbsp; "
            f"{reliability_badge('high')} CV ≤ 10% &nbsp; "
            f"{reliability_badge('medium')} CV ≤ 15% &nbsp; "
            f"{reliability_badge('low')} CV ≤ 20% &nbsp; "
            f"{reliability_badge('very_low')} CV &gt; 20% &nbsp;·&nbsp; "
            "cell color = CV band · lower CV = stabler mean · "
            "<b>CV band ≠ clinical validation</b> · "
            "<b>Runs</b> = scored / requested for that model "
            "(can differ across models) · "
            "<b>Failed</b> = technical N/A rate "
            "(collect / judge / timeout / partial / empty) — not clinical 0% · "
            "<b>partial</b> = ranked by mean of scored runs despite incomplete coverage · "
            "unranked rows (#—) have zero scored observations · "
            "≤N non-zero scored obs/model; N/A and exact-zero skipped, older scored used · "
            "N=5 exploratory · ~10 better for CV eye-check · 20–50 diminishing · 100 max · "
            "<b>C/Q/D</b> = coverage / quality / discipline "
            "(quality is independent of coverage; a high board % can still have low C)"
        )
    )
    out = (
        banner
        + "<div style='overflow-x:auto;margin:0.35rem 0 0.75rem;border:1px solid #334155;"
        "border-radius:12px;background:#0f172a'>"
        "<table style='width:100%;border-collapse:collapse;color:#e2e8f0;font-size:0.9rem'>"
        "<thead><tr style='color:#94a3b8;text-align:left;font-size:0.75rem;"
        "letter-spacing:0.04em;text-transform:uppercase'>"
        "<th style='padding:0.55rem'>#</th><th style='padding:0.55rem'>Name</th>"
        "<th style='padding:0.55rem'>Version</th>"
        "<th style='padding:0.55rem'>Clin. Composite</th>"
        "<th style='padding:0.55rem'>C/Q/D</th><th style='padding:0.55rem'>± Std</th>"
        "<th style='padding:0.55rem'>CV %</th><th style='padding:0.55rem'>Reliability</th>"
        "<th style='padding:0.55rem'>Median</th><th style='padding:0.55rem'>Min–Max</th>"
        f"{runs_th}"
        f"{fail_th}"
        "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>"
        "<div style='font-size:0.78rem;color:#94a3b8;margin-bottom:0.5rem'>"
        f"{footer}</div>"
    )
    # #region agent log
    _r0 = ranking_mean[0] if ranking_mean else {}
    _agent_dbg(
        "H3",
        "lib/benchmark_multi_ui.py:_reliability_table_html",
        "reliability_table_html built",
        {
            "n_in": _n_in,
            "n_after_roster": _n_after_roster,
            "n_after_clean": _n_after_clean,
            "successful_only": bool(successful_only),
            "rank_by": str(rank_by),
            "n_body_rows": len(rows_html),
            "has_linear_gradient": "linear-gradient" in out,
            "gradient_count": out.count("linear-gradient"),
            "has_reliability_th": "Reliability" in out,
            "has_super_high_token": "SUPER HIGH" in out.upper(),
            "has_cv_band_bg": ("background:#064e3b" in out or "background:#14532d" in out
                               or "background:#713f12" in out or "background:#9a3412" in out
                               or "background:#7f1d1d" in out),
            "row0": {
                "key": _r0.get("key"),
                "accuracy_mean": _r0.get("accuracy_mean"),
                "median": _r0.get("median"),
                "cv_pct": _r0.get("cv_pct"),
            },
            "html_len": len(out),
        },
    )
    # #endregion
    return out


# Public alias — graded app.py and Beta pages should import this name.
reliability_table_html = _reliability_table_html
