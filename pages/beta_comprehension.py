"""Comprehension (discursive) — free-form collect with isolated History / Rebuild.

UI label: Comprehension. Internal protocol ids stay ``beta_comprehension`` /
``beta-comprehension-v1`` so existing History / Multi artifacts keep pooling.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

from benchmark.beta_pack import (
    auto_freeze_beta_slot,
    count_beta_runs_by_slot,
    list_beta_slots,
    load_beta_pack,
)
from benchmark.beta_prompts import (
    beta_candidate_messages,
    beta_candidate_system,
    beta_candidate_user,
    parse_beta_candidate_answers,
)
from benchmark.beta_protocol import (
    CASE_ID,
    PROMPT_VERSION,
    PROTOCOL_ID,
    SCORING_VERSION as BETA_SV,
)
from benchmark.cases_loader import load_case
from benchmark.config import load_models_config
from benchmark.gold import (
    cohort_id as build_cohort_id,
    confirmed_gold,
    load_confirmed_gold,
    try_extract_qna_sections,
)
from benchmark.judge import PipelinedJudge, build_ranking
from benchmark.qvac_variants import merge_roster, panel_rows_for_roster
from benchmark.report import (
    artifacts_for_case,
    rebuild_balanced_cases_from_history,
    rebuild_multi_from_history,
    rebuild_portfolio_from_history,
)
from benchmark.runner import build_run_artifact, iter_collect_live, prepare_run
from benchmark.schema import ConfirmedGold, utc_now_iso
from benchmark.workspace import scoped_artifacts_dir
from benchmark.config import is_usable_openrouter_key
from datetime import datetime, timezone
from lib.benchmark_multi_ui import (
    LiveJudgingBoard,
    client_toast_run_done,
    finished_multi_progress,
    live_judging_board_html,
    paint_rebuild_ops_reliability_panels,
    progressive_multi_panel_html,
    reliability_table_html,
    snapshot_from_artifact,
)
from lib.charts import fig_judge_accuracy_bars, fig_judge_mean_accuracy_bars
from lib.disclosure import screenshot_footer_html
from lib.run_store import LocalRunStore
from lib.run_timer import (
    _flash_collect_done,
    _paint_run_timer,
    _run_timer_idle,
    _run_timer_live,
    _run_timer_stop,
)
from lib.stream_panels import (
    kpi_line,
    kpi_live_line,
    status_pill,
    stream_body_html,
    stream_shell_html,
)


def _paint_beta_multi_progress(
    slot,
    completed: list,
    *,
    n_total: int,
    batch_done: bool = False,
    toast_html: str = "",
    height: int = 240,
) -> None:
    """Same progressive Multi strip as graded dashboard (iframe for clickable tabs)."""
    body = progressive_multi_panel_html(
        completed, n_total=n_total, batch_done=batch_done
    ) + (toast_html or "")
    h = height + (180 if toast_html else 0)
    slot.empty()
    with slot.container():
        components.html(
            f"""<!doctype html><html><head><meta charset="utf-8"/>
<style>
  body {{ margin:0; background:transparent;
    font-family: ui-sans-serif, system-ui, sans-serif; }}
</style></head><body>{body}</body></html>""",
            height=h,
            scrolling=True,
        )

try:
    from benchmark import qvac_bridge
except Exception:  # pragma: no cover
    qvac_bridge = None  # type: ignore

st.set_page_config(
    page_title="Comprehension · discursive",
    page_icon="β",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.status-pill{display:inline-block;padding:0.15rem 0.45rem;border-radius:999px;
font-size:0.75rem;font-weight:600}
.status-pill.run{background:#1e3a5f;color:#93c5fd}
.status-pill.done{background:#14532d;color:#86efac}
.status-pill.err{background:#450a0a;color:#fca5a5}
.kpi-row{color:#94a3b8;font-size:0.78rem;margin:0.2rem 0 0.35rem}
.phase-banner{padding:0.45rem 0.7rem;border-radius:8px;background:#1e293b;
color:#e2e8f0;font-size:0.85rem;margin:0.4rem 0}
/* Comprehension case pickers — button row (not toggles); selected = amber */
div[class*="st-key-beta_case_btn_"] button[kind="primary"],
div[class*="st-key-beta_case_btn_"] button[data-testid="baseButton-primary"] {
  background:#fbbf24 !important;background-color:#fbbf24 !important;
  border:2px solid #d97706 !important;color:#78350f !important;font-weight:750 !important;
}
div[class*="st-key-beta_case_btn_"] button[kind="primary"] p,
div[class*="st-key-beta_case_btn_"] button[kind="primary"] span,
div[class*="st-key-beta_case_btn_"] button[data-testid="baseButton-primary"] p,
div[class*="st-key-beta_case_btn_"] button[data-testid="baseButton-primary"] span {
  color:#78350f !important;white-space:pre-line;line-height:1.25;font-size:0.82rem;
}
div[class*="st-key-beta_case_btn_"] button[kind="secondary"],
div[class*="st-key-beta_case_btn_"] button[data-testid="baseButton-secondary"] {
  background:#1e293b !important;border:1px solid #64748b !important;
}
div[class*="st-key-beta_case_btn_"] button[kind="secondary"] p,
div[class*="st-key-beta_case_btn_"] button[kind="secondary"] span,
div[class*="st-key-beta_case_btn_"] button[data-testid="baseButton-secondary"] p,
div[class*="st-key-beta_case_btn_"] button[data-testid="baseButton-secondary"] span {
  white-space:pre-line;line-height:1.25;font-size:0.82rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# Same rank-live / provisional board chrome as graded dashboard.
_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "dashboard.css"
if _ASSETS.is_file():
    _css = _ASSETS.read_text(encoding="utf-8")
    _css_js = _css.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    components.html(
        f"""
        <script>
        (function(){{
          var doc;
          try {{ doc = window.parent && window.parent.document ? window.parent.document : document; }}
          catch (e) {{ doc = document; }}
          var s = doc.getElementById('qvac-beta-dashboard-css');
          if (!s) {{
            s = doc.createElement('style');
            s.id = 'qvac-beta-dashboard-css';
            doc.head.appendChild(s);
          }}
          s.textContent = `{_css_js}`;
        }})();
        </script>
        """,
        height=0,
    )

# --- workspace (same owner scope as graded home when key is in session) ---
_session_key = (st.session_state.get("or_key_session") or "").strip()
if _session_key and not is_usable_openrouter_key(_session_key):
    st.session_state.pop("or_key_session", None)
    _session_key = ""
has_key = is_usable_openrouter_key(_session_key)
WORKSPACE_DIR = scoped_artifacts_dir(_session_key)
RUN_STORE = LocalRunStore(WORKSPACE_DIR)


def _persist(artifact) -> None:
    RUN_STORE.persist_artifact(artifact)


def _frozen_to_gold(frozen: Dict[str, Any]) -> ConfirmedGold:
    keep = {
        k: v
        for k, v in frozen.items()
        if k
        in {
            "raw_text",
            "sections",
            "confirmed_at",
            "extraction_model",
            "extraction_prompt_version",
            "extraction_cost_usd",
        }
    }
    return load_confirmed_gold(keep)


def _clear_beta_mean_popup() -> None:
    st.session_state.pop("show_beta_mean_popup", None)


def _normalize_beta_rebuild(built: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(built or {})
    sum_obj = out.get("summary")
    if sum_obj is not None and hasattr(sum_obj, "model_dump"):
        out["summary"] = sum_obj.model_dump()
    return out


def _arm_beta_mean_popup(built: Dict[str, Any]) -> None:
    """Store rebuild payload and open the mean KPI dialog (graded parity)."""
    built = _normalize_beta_rebuild(built)
    st.session_state["beta_rebuild_result"] = built
    if built.get("ok"):
        st.session_state["show_beta_mean_popup"] = True
        try:
            from benchmark.schema import MultiRunSummary as _MRS

            raw = built.get("summary")
            summary = _MRS.model_validate(raw) if isinstance(raw, dict) else raw
            st.session_state["beta_last_multi_summary"] = summary
        except Exception:
            pass
    # #region agent log
    try:
        import json as _json_arm
        import time as _time_arm

        with open(
            "/Users/m1/QVAC vs Cloud LLMs - Health Test/.cursor/debug-a76cc5.log",
            "a",
            encoding="utf-8",
        ) as _df_arm:
            _df_arm.write(
                _json_arm.dumps(
                    {
                        "sessionId": "a76cc5",
                        "hypothesisId": "H6",
                        "location": "pages/beta_comprehension.py:_arm_beta_mean_popup",
                        "message": "armed Beta mean popup",
                        "data": {
                            "ok": bool(built.get("ok")),
                            "scope": str(built.get("scope") or ""),
                            "show_popup": bool(
                                st.session_state.get("show_beta_mean_popup")
                            ),
                            "has_ops": bool(built.get("ops_reliability")),
                            "n_cap": built.get("n_per_model_cap"),
                        },
                        "timestamp": int(_time_arm.time() * 1000),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion


def _beta_rebuild_summary(payload: Dict[str, Any]):
    from benchmark.schema import MultiRunSummary as _MRS

    raw = payload.get("summary")
    if raw is None:
        return None
    if isinstance(raw, dict):
        try:
            return _MRS.model_validate(raw)
        except Exception:
            return None
    return raw


def _paint_beta_rebuild_mean_body(
    payload: Dict[str, Any],
    *,
    key_prefix: str,
    rank_key: str,
) -> None:
    """Full Rebuild mean KPIs: chart → ranking (bars+CV) → Failures/N/A → ops chart."""
    summary = _beta_rebuild_summary(payload)
    mean_rows = list(getattr(summary, "ranking_mean", None) or []) if summary else []
    rb_scope = str(payload.get("scope") or "same_case")
    rb_cohort = str(payload.get("cohort_id") or "")
    rb_cap = payload.get("n_per_model_cap") or "?"
    rb_n_label = f"N≤{rb_cap} scored/model scan"
    if mean_rows:
        rb_clean = bool(payload.get("successful_only", True))
        try:
            rb_n_cap = int(payload.get("n_per_model_cap") or 0)
        except (TypeError, ValueError):
            rb_n_cap = 0
        rank_by = "mean"
        if rb_n_cap >= 30:
            rank_pick = st.radio(
                "Rank by",
                options=["mean", "median"],
                horizontal=True,
                key=rank_key,
                format_func=lambda s: (
                    "Mean (default)" if s == "mean" else "Median"
                ),
                help=(
                    "Available when N≥30. Reorders chart bars and table # "
                    "together. Bars still show mean ±1 std; ◆ is median."
                ),
            )
            rank_by = str(rank_pick or "mean")
        st.markdown("##### Chart (mean %; whiskers = ±1 std)")
        st.plotly_chart(
            fig_judge_mean_accuracy_bars(
                mean_rows,
                hide_partial_labels=rb_clean,
                compact=True,
                rank_by=rank_by,
            ),
            use_container_width=True,
            key=f"{key_prefix}_mean_chart",
        )
        st.markdown("##### Ranking table")
        rb_html = reliability_table_html(
            mean_rows,
            successful_only=rb_clean,
            rank_by=rank_by,
        )
        # #region agent log
        try:
            import json as _json_dbg
            import time as _time_dbg

            with open(
                "/Users/m1/QVAC vs Cloud LLMs - Health Test/.cursor/debug-a76cc5.log",
                "a",
                encoding="utf-8",
            ) as _df:
                _df.write(
                    _json_dbg.dumps(
                        {
                            "sessionId": "a76cc5",
                            "hypothesisId": "H1",
                            "location": "pages/beta_comprehension.py:_paint_beta_rebuild_mean_body",
                            "message": "Beta Rebuild mean ranking paint",
                            "data": {
                                "path": key_prefix,
                                "n_mean_rows": len(mean_rows),
                                "successful_only": bool(rb_clean),
                                "rank_by": str(rank_by),
                                "render": "st.html" if hasattr(st, "html") else "st.markdown",
                                "html_len": len(rb_html),
                                "has_linear_gradient": "linear-gradient" in rb_html,
                                "gradient_count": rb_html.count("linear-gradient"),
                                "has_cv_band_bg": (
                                    "background:#064e3b" in rb_html
                                    or "background:#14532d" in rb_html
                                    or "background:#713f12" in rb_html
                                ),
                                "keys": [str(r.get("key")) for r in mean_rows[:6]],
                            },
                            "timestamp": int(_time_dbg.time() * 1000),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        # st.html preserves nested bar/badge styles better than markdown on some pages.
        if hasattr(st, "html"):
            st.html(rb_html)
        else:
            st.markdown(rb_html, unsafe_allow_html=True)
        st.markdown(
            screenshot_footer_html(
                scope=rb_scope,
                cohort_id=rb_cohort,
                n_label=rb_n_label,
                extra=(
                    "scored-only · exact Clinical Composite == 0 treated like N/A"
                    if rb_clean
                    else "mean ranking"
                ),
            ),
            unsafe_allow_html=True,
        )
    paint_rebuild_ops_reliability_panels(
        st,
        list(payload.get("ops_reliability") or []),
        n_per_model_cap=rb_cap,
        chart_key=f"{key_prefix}_ops_chart",
        table_footer_html=screenshot_footer_html(
            scope=rb_scope,
            cohort_id=rb_cohort,
            n_label=rb_n_label,
            extra="failures/N/A % · scan window · not clinical mean",
        ),
        chart_footer_html=screenshot_footer_html(
            scope=rb_scope,
            cohort_id=rb_cohort,
            n_label=rb_n_label,
            extra="ops reliability · zeros+N/A · not clinical mean",
        ),
    )
    st.caption(
        f"Available poolable · {payload.get('available', '—')} · "
        f"cohort `{str(payload.get('cohort_id') or '')[:12]}…`"
    )


@st.dialog(
    "Rebuild mean · Comprehension · offline · $0",
    width="large",
    on_dismiss=_clear_beta_mean_popup,
)
def beta_mean_rebuild_dialog() -> None:
    """Popup: full mean KPIs after Multi×all / Rebuild (same role as graded)."""
    payload = st.session_state.get("beta_rebuild_result") or {}
    if not payload.get("ok"):
        st.error(payload.get("reason") or "Nothing to show.")
        if st.button(
            "Close", type="primary", use_container_width=True, key="beta_hm_dlg_err"
        ):
            _clear_beta_mean_popup()
            st.rerun()
        return
    summary = _beta_rebuild_summary(payload)
    if summary is None:
        st.error("Summary invalid.")
        if st.button(
            "Close", type="primary", use_container_width=True, key="beta_hm_dlg_bad"
        ):
            _clear_beta_mean_popup()
            st.rerun()
        return
    scope = str(payload.get("scope") or "same_case")
    st.success(
        f"**Comprehension mean ready** · scope `{scope}` · "
        f"N≤{payload.get('n_per_model_cap') or '?'} scored/model · **$0 API**"
    )
    _paint_beta_rebuild_mean_body(
        payload,
        key_prefix="beta_hm_dlg",
        rank_key="beta_hm_dlg_rank_by",
    )
    if st.button(
        "Close", type="primary", use_container_width=True, key="beta_hm_dlg_close"
    ):
        _clear_beta_mean_popup()
        st.rerun()


# Open mean popup when armed (never mid-run — ✕ must not abort collect).
if st.session_state.get("show_beta_mean_popup") and not st.session_state.get(
    "beta_running"
):
    beta_mean_rebuild_dialog()


st.markdown(
    """
<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.35rem">
  <span style="font-size:0.7rem;font-weight:800;letter-spacing:0.08em;padding:0.25rem 0.55rem;
  border-radius:999px;background:#fbbf24;color:#78350f;border:1px solid #f59e0b">COMPREHENSION</span>
  <span style="color:#94a3b8;font-size:0.9rem">Discursive free-form track · Single / Multi · isolated KPIs</span>
</div>
""",
    unsafe_allow_html=True,
)
st.caption(
    f"Free-form candidates · same Clinical Composite dimensions · protocol `{PROTOCOL_ID}` · "
    f"case_id `{CASE_ID}` · **never pooled** with graded A1–A5 Rebuild. "
    "Exploratory amateur bench — not medical validation."
)

with st.sidebar:
    st.caption("Legacy URL · prefer **Comprehension home** for new sessions.")
    st.page_link("app.py", label="← Comprehension · home", icon="🏠")
    st.page_link(
        "pages/structured_graded.py",
        label="Structured · A1–A5",
        icon="📋",
        help="Rigid slot Q&A · separate KPIs",
    )
    st.markdown("**OpenRouter**")
    if has_key:
        st.success("Key OK · session")
        st.caption("Reload clears the key (BYOK).")
    else:
        st.warning("No key · paste on home or below")
        key_in = st.text_input("OPENROUTER_API_KEY", type="password", key="beta_or_key")
        if key_in and is_usable_openrouter_key(key_in):
            st.session_state["or_key_session"] = key_in.strip()
            st.rerun()
    sidecar_up = bool(qvac_bridge and (qvac_bridge.reachable() or qvac_bridge.available()))
    st.caption("QVAC sidecar · " + ("up" if sidecar_up else "offline"))
    st.code(PROTOCOL_ID, language=None)
    st.caption(
        "MedPsy 4B often sits longer on judge **corrective retry** "
        "(DeepSeek re-check, not QVAC hung)."
    )
    # LAST widget in left column = Run clock (same dock as graded home)
    st.markdown('<div class="sidebar-timer-spacer"></div>', unsafe_allow_html=True)
    timer_slot = st.empty()
    _beta_pending = st.session_state.get("beta_confirmed_run") or {}
    if _beta_pending or st.session_state.get("beta_running"):
        _pn = max(1, int((_beta_pending or {}).get("n") or 1))
        _paint_run_timer(
            timer_slot,
            _run_timer_live(
                "Starting…",
                n_runs=_pn,
                elapsed_total=0,
                elapsed_this=0,
                collect_base=0,
                judge_base=0,
                bucket="collect",
            ),
            height=210 if _pn > 1 else 168,
            live=True,
            multi=_pn > 1,
        )
    else:
        _last_tm = st.session_state.get("beta_last_run_timings") or {}
        _pr = list(_last_tm.get("per_run") or [])
        _paint_run_timer(
            timer_slot,
            _run_timer_idle(_last_tm),
            live=False,
            multi=int(_last_tm.get("n") or 1) > 1,
            per_run_n=len(_pr),
        )

try:
    pack = load_beta_pack()
    slots = list_beta_slots(pack)
except Exception as exc:
    st.error(f"Could not load Comprehension pack: {exc}")
    st.stop()

if not slots:
    st.warning("Comprehension pack has no usable cases.")
    st.stop()

# --- case pickers (buttons + run counts; NOT toggles — Single/Multi on selected) ---
_beta_arts = [a for _, a in RUN_STORE.list_artifacts()]
_beta_run_counts = count_beta_runs_by_slot(_beta_arts)
if "beta_active_case_slot" not in st.session_state:
    st.session_state["beta_active_case_slot"] = int(slots[0]["slot"])
_active_slot = int(st.session_state.get("beta_active_case_slot") or slots[0]["slot"])
if not any(int(s["slot"]) == _active_slot for s in slots):
    _active_slot = int(slots[0]["slot"])
    st.session_state["beta_active_case_slot"] = _active_slot

st.markdown(
    '<div class="sec-label">Comprehension cases · select one for Single / Multi ×N '
    "(Multi×all still runs every pack case)</div>",
    unsafe_allow_html=True,
)
_chunk = 4
for _off in range(0, len(slots), _chunk):
    _row = slots[_off : _off + _chunk]
    _cols = st.columns(len(_row), gap="small")
    for _ci, _slot_entry in enumerate(_row):
        _sid = int(_slot_entry["slot"])
        _n_runs = int(_beta_run_counts.get(_sid, 0))
        _title = str(_slot_entry.get("title") or f"Case {_sid}")
        _short = _title if len(_title) <= 36 else (_title[:34].rstrip() + "…")
        _runs_label = f"{_n_runs} run" if _n_runs == 1 else f"{_n_runs} runs"
        _btn_label = f"Case {_sid} · {_short}\n({_runs_label})"
        _is_sel = _sid == _active_slot
        with _cols[_ci]:
            if st.button(
                _btn_label,
                key=f"beta_case_btn_{_sid}",
                use_container_width=True,
                type="primary" if _is_sel else "secondary",
                help=(
                    f"{_title} · {_runs_label} saved "
                    "(includes Multi×all rounds that hit this case)"
                ),
            ):
                st.session_state["beta_active_case_slot"] = _sid
                st.rerun()

case_row = next(
    (s for s in slots if int(s["slot"]) == _active_slot),
    slots[0],
)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Box 1 · Presentation**")
    stem = st.text_area(
        "stem",
        value=case_row["stem"],
        height=260,
        key=f"beta_stem_{case_row['slot']}",
        label_visibility="collapsed",
    )
with c2:
    st.markdown("**Box 2 · Reference prose (confirm)**")
    prose = st.text_area(
        "prose",
        value=case_row["reference_prose"],
        height=260,
        key=f"beta_prose_{case_row['slot']}",
        label_visibility="collapsed",
    )

confirm = st.checkbox(
    "I confirm this undivided prose is the Comprehension scoring reference for this case.",
    key=f"beta_confirm_box_{case_row['slot']}",
)
if st.button("Freeze Comprehension reference", type="primary", disabled=not confirm):
    gold_raw = (case_row.get("gold_raw") or "").strip()
    sections = try_extract_qna_sections(gold_raw) if gold_raw else None
    if sections is None:
        st.error("Pack gold_raw scaffold missing for this slot.")
    else:
        gold = confirmed_gold(
            raw_text=gold_raw,
            sections=sections,
            extraction_model="beta-pack-local-qna-scaffold",
        )
        payload = gold.model_dump()
        payload.update(
            {
                "beta_reference_prose": prose.strip(),
                "beta_stem": stem.strip(),
                "protocol_id": PROTOCOL_ID,
                "scoring_version": BETA_SV,
                "case_slot": case_row["slot"],
                "case_title": case_row["title"],
            }
        )
        st.session_state["beta_confirmed_gold"] = payload
        st.success(f"Frozen · Case {case_row['slot']} · `{PROTOCOL_ID}`")

frozen = st.session_state.get("beta_confirmed_gold")
frozen_ok = (
    isinstance(frozen, dict)
    and frozen.get("case_slot") == case_row["slot"]
    and str(frozen.get("scoring_version") or "") == BETA_SV
)
if frozen_ok:
    st.success(
        f"Active Comprehension contract · Case {case_row['slot']} · "
        f"`{frozen.get('scoring_version')}`"
    )
else:
    st.info("Freeze a Comprehension reference before Single / Multi.")

# --- roster (beta-prefixed keys) ---
st.markdown("### Models")
cfg = load_models_config()
sidecar_up = bool(qvac_bridge and (qvac_bridge.reachable() or qvac_bridge.available()))
b1, b2, b3, b4 = st.columns(4)
with b1:
    include_cloud = st.toggle("Cloud", value=True, key="beta_include_cloud")
with b2:
    include_medpsy = st.toggle("MedPsy", value=True, key="beta_include_medpsy")
with b3:
    include_generic = st.toggle("Generic local", value=True, key="beta_include_generic")
with b4:
    include_medical = st.toggle("Medical peers", value=True, key="beta_include_medical")

benchmark_track = st.radio(
    "Track",
    options=["controlled", "strict_controlled", "native_defaults"],
    horizontal=True,
    key="beta_benchmark_track",
)
n_multi = st.number_input("Multi N", min_value=1, max_value=30, value=5, key="beta_n_multi")

_eff_medpsy = bool(include_medpsy) and sidecar_up
_eff_generic = bool(include_generic) and sidecar_up
_eff_medical = bool(include_medical) and sidecar_up
roster = merge_roster(
    list(cfg.get("candidates") or []) if include_cloud else [],
    triple_qvac=_eff_medpsy,
    include_qvac=_eff_medpsy,
    include_local_peers=_eff_generic,
    include_medical_peers=_eff_medical,
)
roster = [
    c for c in roster if c.get("provider") != "qvac" or c.get("gguf_ready", True)
]
st.caption(
    "Active: "
    + ", ".join(c.get("label") or c.get("key") for c in roster)
    if roster
    else "No models selected / ready."
)

with st.expander("Candidate prompt (free-form)", expanded=False):
    st.code(beta_candidate_system(), language=None)
    st.code(beta_candidate_user(stem=stem), language=None)

_can_launch = bool(
    roster and (has_key or _eff_medpsy or _eff_generic or _eff_medical)
)
_n_pack_cases = len(slots)
st.caption(
    f"**Multi ×N** = N repeats on the selected case (same progressive tabs as graded). "
    f"**Multi×all cases** = round-robin Case 1→{_n_pack_cases}, then again "
    f"(N passes) · e.g. N=2 → {_n_pack_cases * 2} rounds "
    f"(R1=Case1 … R{_n_pack_cases}=Case{_n_pack_cases} · "
    f"R{_n_pack_cases + 1}=Case1…) · auto-confirm pack gold · "
    "finished rounds stay visible in the strip below."
)
r1, r2, r3, r4 = st.columns(4)
with r1:
    if st.button(
        "Single run",
        type="primary",
        disabled=not (frozen_ok and _can_launch),
        use_container_width=True,
        help="Requires Freeze on the selected case.",
    ):
        st.session_state["beta_confirmed_run"] = {"n": 1, "multi_case": False}
        st.rerun()
with r2:
    if st.button(
        f"Multi ×{int(n_multi)}",
        disabled=not (frozen_ok and _can_launch),
        use_container_width=True,
        help="N repeats on the selected frozen case only.",
    ):
        st.session_state["beta_confirmed_run"] = {
            "n": int(n_multi),
            "multi_case": False,
        }
        st.rerun()
with r3:
    if st.button(
        f"Multi×all cases · {_n_pack_cases}×{int(n_multi)}",
        disabled=not _can_launch,
        use_container_width=True,
        type="primary",
        help=(
            f"Round-robin Case 1→{_n_pack_cases} for N={int(n_multi)} passes "
            f"({_n_pack_cases * int(n_multi)} rounds). "
            "Auto-confirms each pack reference. Progressive tabs under the live panels."
        ),
    ):
        st.session_state["beta_confirmed_run"] = {
            "n": int(n_multi),
            "multi_case": True,
        }
        st.rerun()
with r4:
    if st.button("Stop / clear pending", use_container_width=True):
        st.session_state.pop("beta_confirmed_run", None)
        st.session_state.pop("beta_running", None)
        st.session_state.pop("beta_multi_progress", None)
        st.rerun()

# --- execute ---
run_cfg = st.session_state.pop("beta_confirmed_run", None)
_multi_case = bool(run_cfg and run_cfg.get("multi_case"))
_ready = bool(run_cfg and roster and (_multi_case or frozen_ok))
if _ready:
    st.session_state["beta_running"] = True
    st.session_state.pop("beta_multi_progress", None)
    n_runs = int(run_cfg.get("n") or 1)
    cases_plan: List[Dict[str, Any]]
    if _multi_case:
        cases_plan = list(slots)
        st.info(
            f"**Multi×all cases** · round-robin Case 1→{len(cases_plan)} × "
            f"{n_runs} pass(es) = **{len(cases_plan) * n_runs}** rounds · "
            "progressive tabs below (same strip as graded Multi) · exploratory Comprehension track only."
        )
    else:
        cases_plan = [{**case_row, "_frozen": frozen}]

    prep = prepare_run(
        "caseC",
        skip_qvac=not _eff_medpsy,
        include_local_peers=_eff_generic,
        include_medical_peers=_eff_medical,
        triple_qvac=_eff_medpsy,
    )
    by_key = {c["key"]: c for c in prep["candidates_cfg"]}
    candidates_cfg = [by_key[c["key"]] for c in roster if c["key"] in by_key]
    if not candidates_cfg:
        st.error("No overlapping candidates between UI roster and prepare_run.")
        st.session_state.pop("beta_running", None)
        st.stop()
    blind_map = prep["blind_map"]
    for c in candidates_cfg:
        blind_map.setdefault(c["key"], f"B{c['key'][:4]}")

    model_config = {
        "candidates": [
            {"key": c["key"], "model": c.get("model"), "provider": c.get("provider")}
            for c in candidates_cfg
        ],
        "judge": prep["cfg"].get("judge") or {},
    }
    judge_cfg = prep["cfg"].get("judge") or {}
    judge_model = str(judge_cfg.get("model") or "")
    judge_temp = float(judge_cfg.get("temperature", 0) or 0)
    verifier_model = str(judge_cfg.get("verifier_model") or "")
    judge_providers = list(judge_cfg.get("allowed_providers") or [])
    base_case = load_case("caseC")
    label_by_key = {
        c["key"]: str(c.get("label") or c["key"]) for c in candidates_cfg
    }

    # Round plan: Multi×all = Case1→K then again (N passes); else N on one case.
    rounds: List[tuple] = []
    if _multi_case:
        ri = 0
        for pass_i in range(1, n_runs + 1):
            for case_entry in cases_plan:
                ri += 1
                rounds.append((ri, pass_i, case_entry))
    else:
        for run_i in range(1, n_runs + 1):
            rounds.append((run_i, run_i, cases_plan[0]))
    total_rounds = len(rounds)

    # Progressive strip ABOVE live panels (like graded Multi).
    multi_progress_slot = st.empty()
    if total_rounds > 1:
        _paint_beta_multi_progress(
            multi_progress_slot, [], n_total=total_rounds, batch_done=False, height=140
        )

    phase = st.empty()
    phase.markdown(
        f'<div class="phase-banner">Comprehension · preparing {total_rounds} round(s) · '
        f"`{PROTOCOL_ID}`</div>",
        unsafe_allow_html=True,
    )

    # Stream panels once — reused every round (same as graded Multi).
    status_boxes: Dict[str, Any] = {}
    kpi_boxes: Dict[str, Any] = {}
    text_boxes: Dict[str, Any] = {}
    for row in panel_rows_for_roster(candidates_cfg):
        cols = st.columns(len(row))
        for col, cand in zip(cols, row):
            key = cand["key"]
            with col:
                st.markdown(f"**{cand.get('label') or key}**")
                status_boxes[key] = st.empty()
                status_boxes[key].markdown(
                    status_pill("run", "Waiting…"), unsafe_allow_html=True
                )
                kpi_boxes[key] = st.empty()
                st.markdown(
                    stream_shell_html(
                        title=str(cand.get("label") or key), panel_id=key
                    ),
                    unsafe_allow_html=True,
                )
                text_boxes[key] = st.empty()
                text_boxes[key].markdown(
                    stream_body_html("", live=False, panel_id=key),
                    unsafe_allow_html=True,
                )

    live_host = st.empty()
    all_artifacts: List[Any] = []
    completed_snaps: List[Dict[str, Any]] = []
    last_cohort = None
    last_ranking = None
    t_run0 = time.time()
    collect_s_acc = 0.0
    judge_s_acc = 0.0
    per_run_timings: List[Dict[str, Any]] = []

    def _beta_paint_timer(
        phase_label: str,
        *,
        elapsed_this: float = 0.0,
        collect_base: float = 0.0,
        judge_base: float = 0.0,
        bucket: str = "collect",
    ) -> None:
        _paint_run_timer(
            timer_slot,
            _run_timer_live(
                phase_label,
                n_runs=total_rounds,
                elapsed_total=time.time() - t_run0,
                elapsed_this=elapsed_this,
                collect_base=int(round(collect_base)),
                judge_base=int(round(judge_base)),
                bucket=bucket,
            ),
            height=230 if total_rounds > 1 else 178,
            live=True,
            multi=total_rounds > 1,
        )

    _beta_paint_timer(
        f"Comprehension · round 1/{total_rounds} · collect",
        bucket="collect",
    )

    for round_i, pass_i, case_entry in rounds:
        try:
            if _multi_case:
                frozen_payload = auto_freeze_beta_slot(case_entry)
                st.session_state["beta_confirmed_gold"] = frozen_payload
            else:
                frozen_payload = case_entry.get("_frozen") or frozen
            gold_obj = _frozen_to_gold(frozen_payload)
        except Exception as exc:
            st.error(f"Skip Case {case_entry.get('slot')}: {exc}")
            continue

        gold_json = gold_obj.model_dump_json()
        live_stem = str(
            frozen_payload.get("beta_stem") or case_entry.get("stem") or ""
        ).strip()
        case_title = str(
            frozen_payload.get("case_title") or case_entry.get("title") or CASE_ID
        )
        case_slot = case_entry.get("slot")
        case_obj = base_case.model_copy(
            update={"id": CASE_ID, "stem": live_stem, "title": case_title}
        )
        messages = beta_candidate_messages(stem=live_stem)
        cohort = build_cohort_id(
            case_stem=live_stem,
            gold=gold_obj,
            prompt_version=PROMPT_VERSION,
            model_config=model_config,
            benchmark_track=benchmark_track,
            scoring_version=BETA_SV,
        )
        last_cohort = cohort

        tab_label = (
            f"R{round_i} · Case {case_slot}"
            if _multi_case
            else f"Run {round_i}"
        )
        phase.markdown(
            f'<div class="phase-banner">Comprehension · {tab_label} · {case_title[:40]} · '
            f"round {round_i}/{total_rounds}"
            + (f" · pass {pass_i}/{n_runs}" if _multi_case else "")
            + f" · `{PROTOCOL_ID}`</div>",
            unsafe_allow_html=True,
        )

        # Reset stream chrome for this round.
        bufs: Dict[str, str] = {}
        tok_n: Dict[str, int] = {}
        last_paint: Dict[str, float] = {}
        for key in status_boxes:
            status_boxes[key].markdown(
                status_pill("run", "Waiting…"), unsafe_allow_html=True
            )
            kpi_boxes[key].markdown("", unsafe_allow_html=True)
            text_boxes[key].markdown(
                stream_body_html("", live=False, panel_id=key),
                unsafe_allow_html=True,
            )

        live_board = LiveJudgingBoard(
            title=f"Live judging · Comprehension · {tab_label}",
            label_by_key=label_by_key,
            status_boxes=status_boxes,
            status_pill_fn=status_pill,
        )
        last_pipe_poll = 0.0
        t_run_i0 = time.time()
        t_collect0 = t_run_i0
        t_j0: Optional[float] = None
        _beta_paint_timer(
            f"{tab_label} · collect",
            elapsed_this=0.0,
            collect_base=collect_s_acc,
            judge_base=judge_s_acc,
            bucket="collect",
        )

        with live_host.container():
            judge_status_ctx = st.status(
                f"DeepSeek R1 · pipelined · {tab_label} · "
                f"round {round_i}/{total_rounds}",
                expanded=True,
            )
            judge_status = judge_status_ctx.__enter__()
            progress_slot = st.empty()
            board_slot = st.empty()
            live_board.bind(
                board_slot=board_slot,
                progress_slot=progress_slot,
                judge_status=judge_status,
            )
            progress_slot.progress(0.0, text="Judge · waiting for first answer…")
            live_board.paint()

            pipe = PipelinedJudge(
                case_obj,
                judge_model,
                temperature=judge_temp,
                gold_reference=gold_json,
                expected_total=len(candidates_cfg),
                max_workers=min(8, max(2, len(candidates_cfg))),
                on_progress=live_board.on_progress,
                api_key=st.session_state.get("or_key_session"),
                verifier_model=verifier_model,
                run_scope=f"beta-{case_slot}-{round_i}",
                benchmark_track=benchmark_track,
                judge_allowed_providers=judge_providers,
            )
            collected: List[Any] = []
            judgments: List[Any] = []
            try:
                for evt in iter_collect_live(
                    case_obj,
                    candidates_cfg,
                    blind_map,
                    benchmark_track=benchmark_track,
                    api_key=st.session_state.get("or_key_session"),
                    messages=messages,
                    answer_parser=parse_beta_candidate_answers,
                    allow_format_repair=False,
                ):
                    if evt.get("type") == "token":
                        key = evt["key"]
                        bufs[key] = bufs.get(key, "") + (evt.get("delta") or "")
                        tok_n[key] = tok_n.get(key, 0) + 1
                        now = time.time()
                        if (
                            tok_n[key] == 1
                            or tok_n[key] % 8 == 0
                            or (now - last_paint.get(key, 0.0)) >= 0.25
                        ):
                            last_paint[key] = now
                            text_boxes[key].markdown(
                                stream_body_html(bufs[key], live=True, panel_id=key),
                                unsafe_allow_html=True,
                            )
                            kpi_boxes[key].markdown(
                                f'<p class="kpi-row">{kpi_live_line(evt.get("ttft_s"), evt.get("elapsed_s"), evt.get("tps_live"))}</p>',
                                unsafe_allow_html=True,
                            )
                        if (now - last_pipe_poll) >= 0.45:
                            last_pipe_poll = now
                            pipe.poll()
                            if pipe.submitted:
                                phase.markdown(
                                    f'<div class="phase-banner">Comprehension · {tab_label} · '
                                    f"round {round_i}/{total_rounds} · collect + judge "
                                    f"{pipe.done_count}/{pipe.total}"
                                    + (
                                        f" · {pipe.pending_count} in flight"
                                        if pipe.pending_count
                                        else ""
                                    )
                                    + f" · `{PROTOCOL_ID}`</div>",
                                    unsafe_allow_html=True,
                                )
                                _j_so_far = (
                                    (now - t_j0) if t_j0 is not None else 0.0
                                )
                                _beta_paint_timer(
                                    f"{tab_label} · collect∥judge",
                                    elapsed_this=now - t_run_i0,
                                    collect_base=collect_s_acc
                                    + (now - t_collect0),
                                    judge_base=judge_s_acc + _j_so_far,
                                    bucket="both",
                                )
                    elif evt.get("type") == "retry":
                        key = evt["key"]
                        bufs[key] = ""
                        status_boxes[key].markdown(
                            status_pill("run", "Retry…"), unsafe_allow_html=True
                        )
                    elif evt.get("type") == "done":
                        cand = evt["candidate"]
                        collected.append(cand)
                        err = bool(cand.meta.error)
                        status_boxes[cand.candidate_key].markdown(
                            status_pill(
                                "err" if err else "done",
                                "Done · judge queued"
                                if not err
                                else f"Error: {str(cand.meta.error)[:50]}",
                            ),
                            unsafe_allow_html=True,
                        )
                        text = (
                            cand.raw_response
                            or bufs.get(cand.candidate_key)
                            or "(empty)"
                        )
                        kpi_boxes[cand.candidate_key].markdown(
                            f'<p class="kpi-row">{kpi_line(cand.meta.model_dump(), text)}</p>',
                            unsafe_allow_html=True,
                        )
                        text_boxes[cand.candidate_key].markdown(
                            stream_body_html(
                                text, live=False, panel_id=cand.candidate_key
                            ),
                            unsafe_allow_html=True,
                        )
                        if cand.candidate_key:
                            live_board.ensure_queued(
                                cand.candidate_key,
                                label_by_key.get(cand.candidate_key)
                                or cand.display_label
                                or cand.label
                                or cand.candidate_key,
                            )
                            if (
                                not err
                                and (cand.raw_response or "").strip()
                                and t_j0 is None
                            ):
                                t_j0 = time.time()
                            pipe.submit(cand)
                            pipe.poll()

                t_collect_end = time.time()
                run_collect_s = t_collect_end - t_collect0
                collect_s_acc += run_collect_s
                phase.markdown(
                    f'<div class="phase-banner">Comprehension · {tab_label} · '
                    f"round {round_i}/{total_rounds} · Collect done · finishing judge "
                    f"{pipe.done_count}/{pipe.total}… · `{PROTOCOL_ID}`</div>",
                    unsafe_allow_html=True,
                )
                if pipe.done_count < pipe.submitted:
                    _flash_collect_done(n_answers=len(collected))
                _j_so_far = (
                    (time.time() - t_j0) if t_j0 is not None else 0.0
                )
                _beta_paint_timer(
                    f"{tab_label} · DeepSeek R1 tail",
                    elapsed_this=time.time() - t_run_i0,
                    collect_base=collect_s_acc,
                    judge_base=judge_s_acc + _j_so_far,
                    bucket="judge",
                )
                try:
                    pipe.set_expected_total(pipe.submitted or len(collected))
                    judgments = pipe.finalize()
                except Exception:
                    judgments = []
                t_round_end = time.time()
                run_judge_s = (
                    (t_round_end - t_j0) if t_j0 is not None else 0.0
                )
                judge_s_acc += run_judge_s
                per_run_timings.append(
                    {
                        "run": round_i,
                        "total_s": t_round_end - t_run_i0,
                        "collect_s": run_collect_s,
                        "judge_s": run_judge_s,
                        "label": tab_label,
                    }
                )
            finally:
                try:
                    pipe.close(cancel_pending=False)
                except Exception:
                    pass
                try:
                    judge_status_ctx.__exit__(None, None, None)
                except Exception:
                    pass

        ranking = build_ranking(judgments) if judgments else []
        last_ranking = ranking
        by_cand = {c.candidate_key: c for c in collected}
        collected = [by_cand[c["key"]] for c in candidates_cfg if c["key"] in by_cand]

        for j in judgments:
            key = j.candidate_key
            if key not in status_boxes:
                continue
            if j.status == "valid":
                acc = float(getattr(j, "weighted_accuracy", None) or 0)
                status_boxes[key].markdown(
                    status_pill("done", f"Judged · {acc:.0f}%"),
                    unsafe_allow_html=True,
                )
            else:
                status_boxes[key].markdown(
                    status_pill("err", f"N/A · {j.status}"),
                    unsafe_allow_html=True,
                )
            prev = live_board.board.get(key) or {}
            if j.status == "valid":
                live_board.board[key] = {
                    **prev,
                    "label": label_by_key.get(key) or key,
                    "status": "scored",
                    "accuracy": float(j.weighted_accuracy or 0),
                    "coverage": j.coverage_score,
                    "quality": j.quality_score,
                    "discipline": j.discipline_score,
                    "progress_pct": 100,
                    "progress_label": "complete",
                }
            else:
                live_board.board[key] = {
                    **prev,
                    "label": label_by_key.get(key) or key,
                    "status": "failed",
                    "accuracy": None,
                    "progress_pct": 100,
                    "progress_label": "complete",
                }
        # Keep final board visible under streams for this round.
        with live_host.container():
            st.caption(f"Judge board · {tab_label} (latest round)")
            st.markdown(
                live_judging_board_html(
                    live_board.board,
                    highlight_key=live_board.highlight,
                    title=f"Live judging · Comprehension · {tab_label}",
                ),
                unsafe_allow_html=True,
            )

        artifact = build_run_artifact(
            config_snapshot={"beta": True, "protocol": PROTOCOL_ID, **model_config},
            run_id=f"beta-{uuid.uuid4().hex[:12]}",
            case_id=CASE_ID,
            started_at=datetime.fromtimestamp(t_run_i0, tz=timezone.utc).isoformat(),
            finished_at=utc_now_iso(),
            n_index=round_i,
            candidates=collected,
            judgments=judgments,
            ranking=ranking,
            cohort_id=cohort,
            scoring_version=BETA_SV,
            prompt_version=PROMPT_VERSION,
            benchmark_track=benchmark_track,
            models_config={
                **model_config,
                "gold_reference": gold_json,
                "case_stem": live_stem,
                "beta_reference_prose": frozen_payload.get("beta_reference_prose"),
                "beta_case_slot": case_slot,
                "beta_pass_i": pass_i,
                "beta_round_i": round_i,
                "beta_rounds_total": total_rounds,
                "beta_multi_case_batch": bool(_multi_case),
            },
            run_status=(
                "complete"
                if judgments and all(j.status == "valid" for j in judgments)
                else "partial"
            ),
        )
        _persist(artifact)
        all_artifacts.append(artifact)

        if total_rounds > 1:
            snap = snapshot_from_artifact(artifact)
            snap["tab_label"] = tab_label
            snap["modal_title"] = (
                f"{tab_label} · {case_title[:48]} · table + histogram"
            )
            completed_snaps.append(snap)
            st.session_state["beta_multi_progress"] = {
                "completed": list(completed_snaps),
                "n_total": total_rounds,
                "batch_done": False,
            }
            _paint_beta_multi_progress(
                multi_progress_slot,
                completed_snaps,
                n_total=total_rounds,
                batch_done=False,
                toast_html=client_toast_run_done(round_i, total_rounds, ranking),
                height=320,
            )
            st.toast(f"{tab_label} complete · {round_i}/{total_rounds}", icon="✅")

    if last_cohort:
        st.session_state["beta_last_cohort"] = last_cohort
    if last_ranking is not None and (total_rounds == 1 or not _multi_case):
        st.session_state["beta_last_ranking"] = last_ranking

    if total_rounds > 1:
        st.session_state["beta_multi_progress"] = finished_multi_progress(
            completed_snaps, n_total=total_rounds
        )
        _paint_beta_multi_progress(
            multi_progress_slot,
            completed_snaps,
            n_total=total_rounds,
            batch_done=True,
            height=280,
        )

    if not _multi_case and total_rounds == 1 and last_ranking is not None:
        st.plotly_chart(
            fig_judge_accuracy_bars(last_ranking),
            use_container_width=True,
        )
        st.session_state["beta_last_ranking"] = last_ranking
        st.session_state["beta_last_cohort"] = last_cohort

    t_end = time.time()
    total_s = int(round(t_end - t_run0))
    last_this_s = (
        int(round(per_run_timings[-1]["total_s"])) if per_run_timings else total_s
    )
    st.session_state["beta_last_run_timings"] = {
        "total_s": total_s,
        "last_run_s": last_this_s,
        "n": total_rounds,
        "collect_s": int(round(collect_s_acc)),
        "judge_s": int(round(judge_s_acc)),
        "per_run": list(per_run_timings),
    }
    _paint_run_timer(
        timer_slot,
        _run_timer_stop(
            total_s,
            this_s=last_this_s,
            n_runs=total_rounds,
            collect_s=int(round(collect_s_acc)),
            judge_s=int(round(judge_s_acc)),
            per_run=per_run_timings,
            title="Run clock · Comprehension done",
            phase=(
                f"Done · {len(all_artifacts)}/{total_rounds} rounds"
                + (
                    " · collect∥judge overlap"
                    if collect_s_acc + judge_s_acc > total_s + 1
                    else ""
                )
            ),
        ),
        live=False,
        multi=total_rounds > 1,
        per_run_n=len(per_run_timings),
    )

    st.session_state.pop("beta_running", None)

    # Multi×all previously skipped mean KPIs. Mirror graded: offline rebuild + popup.
    _open_mean_popup = False
    if len(all_artifacts) > 1:
        _model_ids = [c["key"] for c in roster] if roster else None
        _n_cap = max(1, int(n_runs) if _multi_case else len(all_artifacts))
        if _multi_case:
            _built = rebuild_balanced_cases_from_history(
                WORKSPACE_DIR,
                n=_n_cap,
                scoring_version=BETA_SV,
                track=benchmark_track,
                model_ids=_model_ids,
            )
        else:
            _built = rebuild_multi_from_history(
                WORKSPACE_DIR,
                CASE_ID,
                n=_n_cap,
                cohort_id=last_cohort,
                model_ids=_model_ids,
                scoring_version=BETA_SV,
            )
        # #region agent log
        try:
            import json as _json_fin
            import time as _time_fin

            with open(
                "/Users/m1/QVAC vs Cloud LLMs - Health Test/.cursor/debug-a76cc5.log",
                "a",
                encoding="utf-8",
            ) as _df_fin:
                _df_fin.write(
                    _json_fin.dumps(
                        {
                            "sessionId": "a76cc5",
                            "hypothesisId": "H6",
                            "location": "pages/beta_comprehension.py:multi_finish",
                            "message": "Beta multi finish auto-rebuild",
                            "data": {
                                "multi_case": bool(_multi_case),
                                "n_artifacts": len(all_artifacts),
                                "n_cap": _n_cap,
                                "built_ok": bool((_built or {}).get("ok")),
                                "scope": str((_built or {}).get("scope") or ""),
                            },
                            "timestamp": int(_time_fin.time() * 1000),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        if (_built or {}).get("ok"):
            _arm_beta_mean_popup(_built)
            _open_mean_popup = True
        else:
            st.warning(
                (_built or {}).get("reason")
                or "Mean rebuild after Multi failed — use Rebuild Comprehension mean below."
            )

    if _multi_case:
        phase.markdown(
            f'<div class="phase-banner">Comprehension Multi×all finished · '
            f"{len(all_artifacts)}/{total_rounds} rounds · "
            f"{len(cases_plan)} cases × {n_runs} passes · `{BETA_SV}` · "
            "mean KPI popup armed</div>",
            unsafe_allow_html=True,
        )
        st.success(
            f"Saved **{len(all_artifacts)}** Comprehension rounds "
            f"(Case1→{len(cases_plan)} × {n_runs}). "
            "Opening mean KPI popup (chart · ranking · Failures/N/A)."
        )
    else:
        phase.markdown(
            f'<div class="phase-banner">Comprehension finished · {n_runs} run(s) saved under '
            f"`{CASE_ID}` / `{BETA_SV}`</div>",
            unsafe_allow_html=True,
        )
    if _open_mean_popup:
        st.rerun()

# --- persist progressive Multi strip across reruns (like graded) ---
_beta_prog = st.session_state.get("beta_multi_progress") or {}
if _beta_prog.get("completed") is not None and not st.session_state.get("beta_running"):
    st.markdown("### Comprehension Multi progress")
    _paint_beta_multi_progress(
        st.empty(),
        list(_beta_prog.get("completed") or []),
        n_total=int(_beta_prog.get("n_total") or 1),
        batch_done=bool(_beta_prog.get("batch_done")),
        height=280 if _beta_prog.get("completed") else 140,
    )

# --- last KPIs ---
st.markdown("### Comprehension KPIs (this session)")
_last_rank = st.session_state.get("beta_last_ranking")
_last_sum = st.session_state.get("beta_last_multi_summary")
if _last_sum is not None:
    mean_rows = list(getattr(_last_sum, "ranking_mean", None) or [])
    if mean_rows:
        st.markdown("##### Ranking table")
        # #region agent log
        _sess_html = reliability_table_html(mean_rows)
        try:
            import json as _json_dbg2
            import time as _time_dbg2

            with open(
                "/Users/m1/QVAC vs Cloud LLMs - Health Test/.cursor/debug-a76cc5.log",
                "a",
                encoding="utf-8",
            ) as _df2:
                _df2.write(
                    _json_dbg2.dumps(
                        {
                            "sessionId": "a76cc5",
                            "hypothesisId": "H2",
                            "location": "pages/beta_comprehension.py:session_kpi_paint",
                            "message": "Beta session KPI ranking paint",
                            "data": {
                                "path": "session_kpi",
                                "n_mean_rows": len(mean_rows),
                                "render": "st.markdown",
                                "html_len": len(_sess_html),
                                "has_linear_gradient": "linear-gradient" in _sess_html,
                                "gradient_count": _sess_html.count("linear-gradient"),
                            },
                            "timestamp": int(_time_dbg2.time() * 1000),
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        if hasattr(st, "html"):
            st.html(_sess_html)
        else:
            st.markdown(_sess_html, unsafe_allow_html=True)
        st.plotly_chart(
            fig_judge_mean_accuracy_bars(mean_rows, hide_partial_labels=True),
            use_container_width=True,
        )
elif _last_rank:
    st.plotly_chart(
        fig_judge_accuracy_bars(_last_rank),
        use_container_width=True,
    )
else:
    st.caption("No Comprehension run in this session yet.")

# --- Comprehension Rebuild (isolated from graded) ---
st.markdown("### Rebuild mean · Comprehension only")
st.caption(
    f"Pools only `scoring_version={BETA_SV}` · case `{CASE_ID}` · $0 offline."
)
_bn = st.selectbox(
    "N scored / model",
    options=[5, 10, 20, 30, 50, 100],
    index=0,
    key="beta_rebuild_n",
)
_scope = st.radio(
    "Scope",
    options=["same_case", "portfolio", "balanced_cases"],
    horizontal=True,
    key="beta_rebuild_scope",
    format_func=lambda s: (
        "Same Comprehension case"
        if s == "same_case"
        else (
            "Portfolio (newest-N / model)"
            if s == "portfolio"
            else "Balanced cases (Case1→K round-robin)"
        )
    ),
)
if st.button("Rebuild Comprehension mean", key="beta_rebuild_btn", type="primary"):
    model_ids = [c["key"] for c in roster] if roster else None
    if _scope == "portfolio":
        built = rebuild_portfolio_from_history(
            WORKSPACE_DIR,
            n=int(_bn),
            scoring_version=BETA_SV,
            track=benchmark_track,
            model_ids=model_ids,
        )
    elif _scope == "balanced_cases":
        # Beta pack slots are Case 1…7; rebuild helper discovers stem order.
        built = rebuild_balanced_cases_from_history(
            WORKSPACE_DIR,
            n=int(_bn),
            scoring_version=BETA_SV,
            track=benchmark_track,
            model_ids=model_ids,
        )
    else:
        cohort = st.session_state.get("beta_last_cohort")
        built = rebuild_multi_from_history(
            WORKSPACE_DIR,
            CASE_ID,
            n=int(_bn),
            cohort_id=cohort,
            model_ids=model_ids,
            scoring_version=BETA_SV,
        )
    if not (built or {}).get("ok"):
        st.session_state["beta_rebuild_result"] = built or {}
        st.warning((built or {}).get("reason") or "Rebuild failed")
    else:
        _arm_beta_mean_popup(built)
        st.rerun()

_rb = st.session_state.get("beta_rebuild_result") or {}
_rb_ok = bool(_rb.get("ok") and _rb.get("summary") is not None)
if _rb_ok:
    _prev_n = _rb.get("n_used")
    if _prev_n is None:
        _sum_prev = _beta_rebuild_summary(_rb)
        _prev_n = getattr(_sum_prev, "n", "?") if _sum_prev is not None else "?"
    _reopen = (
        f"Re-open mean popup · N={_prev_n} · "
        f"{_rb.get('scope') or 'mean'} · $0"
    )
    if st.button(_reopen, key="beta_rebuild_reopen"):
        st.session_state["show_beta_mean_popup"] = True
        st.rerun()
elif isinstance(_rb, dict) and _rb and not _rb.get("ok"):
    st.warning(_rb.get("reason") or "Rebuild failed")

_hist = artifacts_for_case(WORKSPACE_DIR, CASE_ID, limit=20)
_beta_hist = [
    a for _, a in _hist if str(a.scoring_version or "") == BETA_SV
]
st.caption(f"Comprehension History for `{CASE_ID}`: {len(_beta_hist)} recent artifact(s) in workspace.")
