"""Comprehension (discursive) — **main** Streamlit entry (home).

Free-form clinical collect with isolated History / Rebuild.
Wire protocol ``comprehension-v1`` / ``case_id=comprehension`` (legacy
``beta-*`` stamps still pool via dual-read aliases).

Rigid A1–A5 live collect lives on ``pages/structured_graded.py``
(**Structured · A1–A5**) — KPIs never pool across tracks.
"""

from __future__ import annotations

import hashlib
import html
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

from benchmark.beta_pack import (
    SOFT_MAX_BETA_SLOTS,
    auto_freeze_beta_slot,
    beta_case_slot_of,
    count_beta_runs_by_slot,
    custom_slots_ready_for_multi_all,
    delete_beta_artifacts_for_slot,
    delete_beta_custom_slot,
    is_beta_artifact,
    is_photocopy_custom_gold,
    list_beta_slots,
    load_beta_custom_state,
    load_beta_pack,
    merge_beta_slots,
    open_new_beta_case_slot,
    resolve_beta_gold_raw,
    save_beta_custom_state,
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
    scoring_versions_equivalent,
)
from benchmark.runner import (
    build_run_artifact,
    estimate_cost_breakdown,
    iter_collect_live,
    prepare_run,
)
from benchmark.schema import ConfirmedGold, utc_now_iso
from benchmark.workspace import scoped_artifacts_dir
from benchmark.config import is_usable_openrouter_key
from datetime import datetime, timezone
from lib.deployment import (
    capture_and_strip_openrouter_env,
    is_local_install,
    is_streamlit_cloud,
)
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
from lib.boot_welcome import run_boot_dialogs
from lib.charts import fig_judge_accuracy_bars, fig_judge_mean_accuracy_bars
from lib.disclosure import (
    honesty_block_html,
    rebuild_scan_honesty_html,
    screenshot_footer_html,
    screenshot_share_checklist_html,
)
from lib.guide_overlays import guides_always_available_html
from lib.i18n import t
from lib.track_sidebar import render_guides_and_protocol, render_tracks_block
from lib.run_store import HostedRunStore, LocalRunStore
from lib.run_timer import (
    _flash_collect_done,
    _paint_run_timer,
    _run_timer_idle,
    _run_timer_live,
    _run_timer_stop,
)
from lib.spend_confirm import (
    fmt_cost_multi,
    fmt_cost_single,
    render_spend_confirm_card,
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
    page_title="Comprehension · Cloud & local medical LLMs",
    page_icon="🩺",
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
.status-pill.ready{background:#1e293b;color:#cbd5e1}
.status-pill.skip{background:#334155;color:#94a3b8}
.kpi-row{color:#94a3b8;font-size:0.78rem;margin:0.2rem 0 0.35rem}
.phase-banner{padding:0.45rem 0.7rem;border-radius:8px;background:#1e293b;
color:#e2e8f0;font-size:0.85rem;margin:0.4rem 0}
/* Comprehension case pickers — NEW CASE = green; selected case = yellow */
div[class*="st-key-beta_case_new_btn"] button,
div[class*="st-key-beta_case_new_btn"] button[kind="primary"],
div[class*="st-key-beta_case_new_btn"] button[data-testid="baseButton-primary"],
div[class*="st-key-beta_case_new_btn"] button[kind="secondary"],
div[class*="st-key-beta_case_new_btn"] button[data-testid="baseButton-secondary"] {
  background:#16a34a !important;background-color:#16a34a !important;
  border:2px solid #15803d !important;color:#f0fdf4 !important;font-weight:750 !important;
}
div[class*="st-key-beta_case_new_btn"] button p,
div[class*="st-key-beta_case_new_btn"] button span {
  color:#f0fdf4 !important;
}
div[class*="st-key-beta_case_btn_"] button[kind="primary"],
div[class*="st-key-beta_case_btn_"] button[data-testid="baseButton-primary"] {
  background:#facc15 !important;background-color:#facc15 !important;
  border:2px solid #ca8a04 !important;color:#1c1917 !important;font-weight:750 !important;
}
div[class*="st-key-beta_case_btn_"] button[kind="primary"] p,
div[class*="st-key-beta_case_btn_"] button[kind="primary"] span,
div[class*="st-key-beta_case_btn_"] button[data-testid="baseButton-primary"] p,
div[class*="st-key-beta_case_btn_"] button[data-testid="baseButton-primary"] span {
  color:#1c1917 !important;white-space:pre-line;line-height:1.25;font-size:0.82rem;
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

# Same rank-live / provisional board chrome + guide portal as Structured.
# Portal JS is required: overlays ship with inline display:none !important, so
# CSS :checked alone cannot open Setup / How ranking from sidebar labels.
_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_ASSETS = _ASSETS_DIR / "dashboard.css"
if _ASSETS.is_file():
    _css = _ASSETS.read_text(encoding="utf-8")
    _css_js = _css.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    _css_inject = f"""
(function(){{
  var doc;
  try {{ doc = window.parent && window.parent.document ? window.parent.document : document; }}
  catch (e) {{ doc = document; }}
  var s = doc.getElementById('qvac-comprehension-dashboard-css');
  if (!s) {{
    s = doc.createElement('style');
    s.id = 'qvac-comprehension-dashboard-css';
    doc.head.appendChild(s);
  }}
  s.textContent = `{_css_js}`;
}})();
"""
    _portal_path = _ASSETS_DIR / "dashboard_portal.js"
    _portal_js = _portal_path.read_text(encoding="utf-8") if _portal_path.is_file() else ""
    components.html(
        f"<script>{_css_inject}\n{_portal_js}</script>",
        height=0,
        width=0,
    )

# --- API key: Cloud strip parity with Structured (no silent host-key spend) ---
# Load local .env only into missing env vars; never treat truncated placeholders
# as real keys. On Streamlit Cloud, strip process-wide OPENROUTER_API_KEY so
# openrouter_api_key() cannot fall back to a shared host secret.
_ROOT = Path(__file__).resolve().parent
_env_path = _ROOT / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        if not _k:
            continue
        if _k == "OPENROUTER_API_KEY" and not is_usable_openrouter_key(_v):
            continue
        if _k not in os.environ:
            os.environ[_k] = _v

_server_env_key = capture_and_strip_openrouter_env()
if (
    (not is_streamlit_cloud())
    and is_local_install()
    and is_usable_openrouter_key(_server_env_key)
    and not st.session_state.get("or_key_session")
):
    st.session_state["or_key_session"] = _server_env_key

# --- workspace (same owner scope when key is in session) ---
_session_key = (st.session_state.get("or_key_session") or "").strip()
if _session_key and not is_usable_openrouter_key(_session_key):
    st.session_state.pop("or_key_session", None)
    _session_key = ""
has_key = is_usable_openrouter_key(_session_key)
# Cloud + no key: per-browser ephemeral owner — never shared `_local_no_key`.
_cloud_ephemeral: Optional[str] = None
if is_streamlit_cloud() and not has_key:
    if "_cloud_anon_ws" not in st.session_state:
        st.session_state["_cloud_anon_ws"] = str(uuid.uuid4())
    _cloud_ephemeral = str(st.session_state["_cloud_anon_ws"])
WORKSPACE_DIR = scoped_artifacts_dir(
    _session_key, cloud_ephemeral_id=_cloud_ephemeral
)
# Cloud Comprehension: session-memory only (no plaintext run JSON on host FS).
# Local: keep LocalRunStore so History/means under artifacts/owners/ unchanged.
if is_streamlit_cloud():
    RUN_STORE = HostedRunStore(
        memory=list(st.session_state.get("_comp_artifacts_memory") or []),
        memory_setter=lambda arts: st.session_state.__setitem__(
            "_comp_artifacts_memory", arts
        ),
        summaries=list(st.session_state.get("_comp_summaries_memory") or []),
        summaries_setter=lambda s: st.session_state.__setitem__(
            "_comp_summaries_memory", s
        ),
    )
else:
    RUN_STORE = LocalRunStore(WORKSPACE_DIR)


def _comp_preloaded() -> List[Any]:
    """Newest-first artifacts for Rebuild/History when HostedRunStore is active."""
    return [a for _, a in RUN_STORE.list_artifacts()]

# Boot: API key every session · QVAC SDK OK remembered locally (not mid-run / spend).
_qvac_online = bool(qvac_bridge and qvac_bridge.reachable())
_qvac_loaded = bool(qvac_bridge and qvac_bridge.available())
_beta_busy = bool(
    st.session_state.get("beta_running") or st.session_state.get("beta_confirmed_run")
)
_beta_pending_spend = bool(
    st.session_state.get("beta_pending_run")
    and not st.session_state.get("beta_confirmed_run")
)
run_boot_dialogs(
    qvac_online=_qvac_online,
    qvac_loaded=_qvac_loaded,
    running=_beta_busy,
    pending_spend=_beta_pending_spend,
    other_dialog_open=bool(st.session_state.get("show_beta_mean_popup")),
)


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
        st.caption(
            t(
                "comp.zero_mean_policy",
                str(st.session_state.get("lang") or "en"),
            )
        )
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
        _ui_lang = str(st.session_state.get("lang") or "en")
        _exec_notes = [
            o
            for o in list(getattr(summary, "outliers", None) or [])
            if "execution_cohort" in str(o).lower()
        ]
        if _exec_notes:
            st.warning(
                t("comp.exec_cohort_banner", _ui_lang)
                + " "
                + " · ".join(str(o) for o in _exec_notes[:2])
            )
        _scan_banner = rebuild_scan_honesty_html(
            list(payload.get("ops_reliability") or []),
            lang=_ui_lang,
        )
        if _scan_banner:
            st.markdown(_scan_banner, unsafe_allow_html=True)
        st.markdown("##### Ranking table")
        rb_html = reliability_table_html(
            mean_rows,
            successful_only=rb_clean,
            rank_by=rank_by,
        )
        # st.html preserves nested bar/badge styles better than markdown on some pages.
        if hasattr(st, "html"):
            st.html(rb_html)
        else:
            st.markdown(rb_html, unsafe_allow_html=True)
        st.markdown(
            screenshot_footer_html(
                lang=_ui_lang,
                scope=rb_scope,
                cohort_id=rb_cohort,
                n_label=rb_n_label,
                protocol_id=PROTOCOL_ID,
                pack_revision_label=str(
                    payload.get("pack_revision_label")
                    or (
                        st.session_state.get("beta_confirmed_gold") or {}
                    ).get("pack_revision")
                    or _pack_revision
                ),
                extra=(
                    "Comprehension · scored-only · exact Clinical Composite == 0 "
                    "treated like N/A"
                    if rb_clean
                    else "Comprehension · mean ranking"
                ),
            ),
            unsafe_allow_html=True,
        )
    _ui_lang = str(st.session_state.get("lang") or "en")
    _ops_pack = str(
        payload.get("pack_revision_label")
        or (st.session_state.get("beta_confirmed_gold") or {}).get("pack_revision")
        or _pack_revision
    )
    paint_rebuild_ops_reliability_panels(
        st,
        list(payload.get("ops_reliability") or []),
        n_per_model_cap=rb_cap,
        chart_key=f"{key_prefix}_ops_chart",
        table_footer_html=screenshot_footer_html(
            lang=_ui_lang,
            scope=rb_scope,
            cohort_id=rb_cohort,
            n_label=rb_n_label,
            protocol_id=PROTOCOL_ID,
            pack_revision_label=_ops_pack,
            extra="Comprehension · failures/N/A % · not clinical mean",
        ),
        chart_footer_html=screenshot_footer_html(
            lang=_ui_lang,
            scope=rb_scope,
            cohort_id=rb_cohort,
            n_label=rb_n_label,
            protocol_id=PROTOCOL_ID,
            pack_revision_label=_ops_pack,
            extra="Comprehension · ops reliability · zeros+N/A · not clinical mean",
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
    _dlg_lang = str(st.session_state.get("lang") or "en")
    st.markdown(
        screenshot_share_checklist_html(lang=_dlg_lang),
        unsafe_allow_html=True,
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
<div class="demo-hero" style="margin:0 0 0.35rem 0">Cloud &amp; local medical LLMs · Comprehension</div>
<div style="display:flex;align-items:center;gap:0.75rem;flex-wrap:wrap;margin-bottom:0.5rem">
  <span style="font-size:0.7rem;font-weight:800;letter-spacing:0.08em;padding:0.25rem 0.55rem;
  border-radius:999px;background:#fbbf24;color:#78350f;border:1px solid #f59e0b">COMPREHENSION · DEFAULT</span>
  <span style="color:#94a3b8;font-size:0.95rem">Discursive free-form · main track · isolated KPIs</span>
</div>
<p class="demo-sub" style="margin:0 0 0.75rem 0">
QVAC / local medical peers on your machine · cloud models via OpenRouter API ·
an AI judge scores answers · hobby comparison — not medical advice and not a medical device.
</p>
""",
    unsafe_allow_html=True,
)
_guide_lang = str(st.session_state.get("lang") or "en")
st.markdown(
    honesty_block_html(
        lang=_guide_lang,
        scope="comprehension",
        roster_n=9,
    ),
    unsafe_allow_html=True,
)
st.caption(
    t("comp.track_caption", _guide_lang)
    + f" · *Advanced: protocol `{PROTOCOL_ID}`*"
)
_qvac_guide_status = (
    "ready — MedPsy will be included"
    if _qvac_loaded
    else (
        "sidecar online · MedPsy not loaded"
        if _qvac_online
        else "sidecar offline — start it to include on-device"
    )
)
if hasattr(st, "html"):
    st.html(
        guides_always_available_html(
            qvac_status_line=_qvac_guide_status,
            lang=_guide_lang,
        )
    )
else:
    st.markdown(
        guides_always_available_html(
            qvac_status_line=_qvac_guide_status,
            lang=_guide_lang,
        ),
        unsafe_allow_html=True,
    )

with st.sidebar:
    render_tracks_block(active="comprehension")
    st.markdown("**OpenRouter**")
    if has_key:
        st.success("Key OK · session")
        st.caption("Reload clears the key (BYOK).")
    else:
        st.warning("No key · use boot dialog or paste below")
    key_in = st.text_input(
        "OPENROUTER_API_KEY",
        type="password",
        key="beta_or_key",
        placeholder="sk-or-v1-…",
        label_visibility="collapsed",
        help="Full sk-or-v1-… from openrouter.ai/keys — session only",
    )
    if key_in and is_usable_openrouter_key(key_in):
        st.session_state["or_key_session"] = key_in.strip()
        st.rerun()
    elif key_in:
        st.error("Key truncated / too short")
    st.caption(t("comp.same_key_warning", _guide_lang))
    st.markdown("**QVAC · MedPsy**")
    sidecar_up = bool(qvac_bridge and (qvac_bridge.reachable() or qvac_bridge.available()))
    if _qvac_loaded:
        st.success("Ready · on-device")
    elif _qvac_online:
        st.warning("Sidecar up · MedPsy not loaded")
    else:
        st.error("Offline")
    st.caption("QVAC sidecar · " + ("up" if sidecar_up else "offline"))
    render_guides_and_protocol(
        protocol_id=PROTOCOL_ID,
        lang=_guide_lang,
        active_track="comprehension",
        extra_caption=(
            "Rebuild below averages only this track’s saved runs. "
            "MedPsy may look idle while the AI judge double-checks an answer — "
            "that is normal, not a hung local model."
        ),
    )
    # LAST widget in left column = Run clock (same dock as Structured)
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
    pack_slots = list_beta_slots(pack)
except Exception as exc:
    st.error(f"Could not load Comprehension pack: {exc}")
    st.stop()

if not pack_slots:
    st.warning("Comprehension pack has no usable cases.")
    st.stop()

_pack_revision = int(pack.get("revision") or 0)
_pack_title = str(pack.get("title") or "Comprehension pack")
st.caption(
    f"**{_pack_title}** · pack version **{_pack_revision}** · "
    f"{len(pack_slots)} ready-made emergency-style cases. "
    f"· *Advanced: `{PROTOCOL_ID}`*"
)
st.caption(t("comp.ranking_pack_caption", _guide_lang))

# Custom cases (NEW CASE) — local: session + owner-workspace disk (API-key scoped).
# Cloud: session-only (never hydrate/persist shared host drafts).
# Pack JSON (cases 1–K) is never mutated.
if is_streamlit_cloud():
    if "beta_custom_drafts" not in st.session_state:
        st.session_state["beta_custom_drafts"] = {}
    if "beta_locked_custom_slots" not in st.session_state:
        st.session_state["beta_locked_custom_slots"] = []
else:
    _ws_fp = str(WORKSPACE_DIR.resolve())
    if st.session_state.get("beta_custom_workspace_fp") != _ws_fp:
        _disk_drafts, _disk_locked = load_beta_custom_state(WORKSPACE_DIR)
        st.session_state["beta_custom_drafts"] = _disk_drafts
        st.session_state["beta_locked_custom_slots"] = list(_disk_locked)
        st.session_state["beta_custom_workspace_fp"] = _ws_fp
    if "beta_custom_drafts" not in st.session_state:
        st.session_state["beta_custom_drafts"] = {}
    if "beta_locked_custom_slots" not in st.session_state:
        st.session_state["beta_locked_custom_slots"] = []


def _persist_beta_customs() -> None:
    # Cloud: customs stay in session_state only — no host-disk draft file.
    if is_streamlit_cloud():
        return
    save_beta_custom_state(
        WORKSPACE_DIR,
        st.session_state.get("beta_custom_drafts") or {},
        st.session_state.get("beta_locked_custom_slots") or [],
    )


slots = merge_beta_slots(pack_slots, st.session_state.get("beta_custom_drafts") or {})
_pack_slot_ids = {int(s["slot"]) for s in pack_slots}
_slots_locked = bool(
    st.session_state.get("beta_running") or st.session_state.get("beta_confirmed_run")
)
_locked_custom_ids = [
    int(x) for x in (st.session_state.get("beta_locked_custom_slots") or [])
]
_multi_all_extra = custom_slots_ready_for_multi_all(slots, _locked_custom_ids)
_n_multi_all_cases = len(pack_slots) + len(_multi_all_extra)

# --- case pickers (NEW CASE + buttons + run counts; NOT toggles) ---
_beta_arts = [a for _, a in RUN_STORE.list_artifacts()]
_beta_run_counts = count_beta_runs_by_slot(_beta_arts)
if "beta_active_case_slot" not in st.session_state:
    st.session_state["beta_active_case_slot"] = int(slots[0]["slot"])
_active_slot = int(st.session_state.get("beta_active_case_slot") or slots[0]["slot"])
if not any(int(s["slot"]) == _active_slot for s in slots):
    _active_slot = int(slots[0]["slot"])
    st.session_state["beta_active_case_slot"] = _active_slot

st.markdown(
    '<div class="sec-label">Cases · pick one for a single run or Multi ×N '
    "(Multi×all walks pack cases + Lock-ed custom cases)</div>",
    unsafe_allow_html=True,
)

# NEW CASE (green) first, then Case 1…N in rows of 4.
_base_row = [s for s in slots if int(s["slot"]) <= 4]
_rest_slots = [s for s in slots if int(s["slot"]) > 4]
_row1_cols = st.columns([1.25] + [1] * max(1, len(_base_row)), gap="small")
with _row1_cols[0]:
    if st.button(
        "New case",
        key="beta_case_new_btn",
        use_container_width=True,
        disabled=_slots_locked,
        help=(
            "Open a blank custom case after the pack. "
            "Paste the case story and your reference answer, then lock them. "
            "Does not change the built-in pack."
        ),
        type="primary",
    ):
        if not _slots_locked:
            try:
                _new_idx, _new_drafts = open_new_beta_case_slot(
                    slots,
                    custom_drafts=st.session_state.get("beta_custom_drafts") or {},
                )
            except ValueError:
                st.session_state["_beta_case_flash"] = "full"
            else:
                st.session_state["beta_custom_drafts"] = _new_drafts
                st.session_state["beta_active_case_slot"] = int(_new_idx)
                st.session_state["_beta_case_flash"] = "empty"
                _persist_beta_customs()
            st.rerun()
for _ci, _slot_entry in enumerate(_base_row):
    _sid = int(_slot_entry["slot"])
    _n_runs = int(_beta_run_counts.get(_sid, 0))
    _title = str(_slot_entry.get("title") or f"Case {_sid}")
    _short = _title if len(_title) <= 28 else (_title[:26].rstrip() + "…")
    _runs_label = f"{_n_runs} run" if _n_runs == 1 else f"{_n_runs} runs"
    _btn_label = f"Case {_sid} · {_short}\n({_runs_label})"
    with _row1_cols[_ci + 1]:
        if st.button(
            _btn_label,
            key=f"beta_case_btn_{_sid}",
            use_container_width=True,
            type="primary" if _sid == _active_slot else "secondary",
            disabled=_slots_locked,
            help=f"{_title} · {_runs_label} saved",
        ):
            if not _slots_locked:
                st.session_state["beta_active_case_slot"] = _sid
                st.rerun()
for _off in range(0, len(_rest_slots), 4):
    _row = _rest_slots[_off : _off + 4]
    _cols = st.columns(len(_row), gap="small")
    for _ci, _slot_entry in enumerate(_row):
        _sid = int(_slot_entry["slot"])
        _n_runs = int(_beta_run_counts.get(_sid, 0))
        _title = str(_slot_entry.get("title") or f"Case {_sid}")
        _short = _title if len(_title) <= 36 else (_title[:34].rstrip() + "…")
        _runs_label = f"{_n_runs} run" if _n_runs == 1 else f"{_n_runs} runs"
        _custom_tag = " · custom" if _slot_entry.get("custom") else ""
        _btn_label = f"Case {_sid} · {_short}\n({_runs_label}{_custom_tag})"
        with _cols[_ci]:
            if st.button(
                _btn_label,
                key=f"beta_case_btn_{_sid}",
                use_container_width=True,
                type="primary" if _sid == _active_slot else "secondary",
                disabled=_slots_locked,
                help=(
                    f"{_title} · {_runs_label} saved"
                    + (" · session custom (not pack)" if _slot_entry.get("custom") else "")
                ),
            ):
                if not _slots_locked:
                    st.session_state["beta_active_case_slot"] = _sid
                    st.rerun()

_flash = st.session_state.pop("_beta_case_flash", None)
if _flash == "full":
    st.warning(f"Reached soft limit of {SOFT_MAX_BETA_SLOTS} Comprehension cases.")
elif _flash == "empty":
    st.info(
        f"**Case {_active_slot}** is empty — paste the case story and your "
        "reference answer, then lock them for scoring."
    )

case_row = next(
    (s for s in slots if int(s["slot"]) == _active_slot),
    slots[0],
)
_is_custom_case = bool(case_row.get("custom"))

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Box 1 · Case story**")
    stem = st.text_area(
        "stem",
        value=case_row["stem"],
        height=260,
        key=f"beta_stem_{case_row['slot']}",
        label_visibility="collapsed",
        disabled=_slots_locked,
    )
with c2:
    st.markdown("**Box 2 · Your reference answer**")
    prose = st.text_area(
        "prose",
        value=case_row["reference_prose"],
        height=260,
        key=f"beta_prose_{case_row['slot']}",
        label_visibility="collapsed",
        disabled=_slots_locked,
    )

# Persist custom draft edits in session (pack slots stay read-source of truth).
if _is_custom_case and not _slots_locked:
    _drafts = dict(st.session_state.get("beta_custom_drafts") or {})
    _drafts[int(case_row["slot"])] = {
        "title": str(case_row.get("title") or f"Custom case {case_row['slot']}"),
        "stem": stem or "",
        "reference_prose": prose or "",
        "gold_raw": str(case_row.get("gold_raw") or ""),
    }
    st.session_state["beta_custom_drafts"] = _drafts
    _persist_beta_customs()
    case_row = {**case_row, "stem": stem or "", "reference_prose": prose or ""}

if _is_custom_case:
    st.caption(t("comp.custom_multi_caption", _guide_lang))
    if is_photocopy_custom_gold(
        {**case_row, "stem": stem or "", "reference_prose": prose or ""}
    ):
        st.warning(t("comp.photocopy_gold_warning", _guide_lang))
    _del_sid = int(case_row["slot"])
    _del_pending = int(st.session_state.get("beta_delete_custom_pending") or 0)
    if _del_pending == _del_sid:
        st.warning(
            f"**{t('comp.delete_custom_confirm_title', _guide_lang, n=_del_sid)}**\n\n"
            + t("comp.delete_custom_confirm_body", _guide_lang, n=_del_sid)
        )
        _dy, _dn = st.columns(2)
        with _dy:
            if st.button(
                t("comp.delete_custom_yes", _guide_lang, n=_del_sid),
                key=f"beta_delete_custom_yes_{_del_sid}",
                type="primary",
                use_container_width=True,
                disabled=_slots_locked,
            ):
                if _slots_locked:
                    st.session_state["_beta_delete_flash"] = ("busy", _del_sid, 0)
                else:
                    try:
                        _new_drafts = delete_beta_custom_slot(
                            _del_sid,
                            pack_slots=pack_slots,
                            custom_drafts=st.session_state.get("beta_custom_drafts")
                            or {},
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        if getattr(RUN_STORE, "writes_plaintext", True):
                            _n_wiped = delete_beta_artifacts_for_slot(
                                WORKSPACE_DIR, _del_sid
                            )
                        else:
                            _mem = list(
                                st.session_state.get("_comp_artifacts_memory") or []
                            )
                            _kept = [
                                a
                                for a in _mem
                                if beta_case_slot_of(a) != _del_sid
                            ]
                            _n_wiped = len(_mem) - len(_kept)
                            st.session_state["_comp_artifacts_memory"] = _kept
                        st.session_state["beta_custom_drafts"] = _new_drafts
                        st.session_state["beta_locked_custom_slots"] = [
                            int(x)
                            for x in (
                                st.session_state.get("beta_locked_custom_slots") or []
                            )
                            if int(x) != _del_sid
                        ]
                        _persist_beta_customs()
                        _fg = st.session_state.get("beta_confirmed_gold") or {}
                        if (
                            isinstance(_fg, dict)
                            and int(_fg.get("case_slot") or 0) == _del_sid
                        ):
                            st.session_state.pop("beta_confirmed_gold", None)
                        for _wk in (
                            f"beta_stem_{_del_sid}",
                            f"beta_prose_{_del_sid}",
                            f"beta_confirm_box_{_del_sid}",
                        ):
                            st.session_state.pop(_wk, None)
                        for _rk in (
                            "beta_rebuild_result",
                            "beta_last_cohort",
                            "beta_live_outputs",
                        ):
                            st.session_state.pop(_rk, None)
                        if int(
                            st.session_state.get("beta_active_case_slot") or 0
                        ) == _del_sid:
                            st.session_state["beta_active_case_slot"] = int(
                                pack_slots[0]["slot"]
                            )
                        st.session_state.pop("beta_delete_custom_pending", None)
                        st.session_state["_beta_delete_flash"] = (
                            "done",
                            _del_sid,
                            _n_wiped,
                        )
                        st.rerun()
        with _dn:
            if st.button(
                t("comp.delete_custom_cancel", _guide_lang),
                key=f"beta_delete_custom_no_{_del_sid}",
                use_container_width=True,
            ):
                st.session_state.pop("beta_delete_custom_pending", None)
                st.rerun()
    else:
        if st.button(
            t("comp.delete_custom_btn", _guide_lang),
            key=f"beta_delete_custom_btn_{_del_sid}",
            use_container_width=True,
            disabled=_slots_locked,
            help=t("comp.delete_custom_help", _guide_lang),
        ):
            if _slots_locked:
                st.session_state["_beta_delete_flash"] = ("busy", _del_sid, 0)
            else:
                st.session_state["beta_delete_custom_pending"] = _del_sid
                st.rerun()
else:
    st.caption(
        "Scoring uses the curated reference checklist for this pack case. "
        "The long text above is the readable story — the judge scores against "
        "the checklist, not that story alone."
    )

_del_flash = st.session_state.pop("_beta_delete_flash", None)
if isinstance(_del_flash, tuple) and _del_flash:
    if _del_flash[0] == "busy":
        st.warning(t("comp.delete_custom_busy", _guide_lang))
    elif _del_flash[0] == "done":
        st.success(
            t(
                "comp.delete_custom_done",
                _guide_lang,
                n=int(_del_flash[1]),
                runs=int(_del_flash[2]),
            )
        )

confirm = st.checkbox(
    t("comp.lock_checkbox", _guide_lang),
    key=f"beta_confirm_box_{case_row['slot']}",
    disabled=_slots_locked,
)
if st.button(
    t("comp.lock_btn", _guide_lang),
    type="primary",
    disabled=not confirm or _slots_locked,
):
    try:
        gold_raw = resolve_beta_gold_raw(
            {**case_row, "stem": stem or "", "reference_prose": prose or ""}
        )
    except ValueError as exc:
        st.error(str(exc))
        gold_raw = ""
    sections = try_extract_qna_sections(gold_raw) if gold_raw else None
    if sections is None:
        st.error("Could not build gold QnA scaffold for this slot.")
    else:
        gold = confirmed_gold(
            raw_text=gold_raw,
            sections=sections,
            extraction_model=(
                "comprehension-custom-prose-scaffold"
                if _is_custom_case
                else "comprehension-pack-local-qna-scaffold"
            ),
        )
        n_claims = sum(len(s.claims) for s in sections.values())
        _fp = hashlib.sha256(
            f"{stem.strip()}\n{gold_raw}\n{PROTOCOL_ID}".encode("utf-8")
        ).hexdigest()[:12]
        payload = gold.model_dump()
        payload.update(
            {
                "reference_prose": prose.strip(),
                "beta_reference_prose": prose.strip(),
                "stem": stem.strip(),
                "beta_stem": stem.strip(),
                "protocol_id": PROTOCOL_ID,
                "scoring_version": BETA_SV,
                "case_slot": case_row["slot"],
                "case_title": case_row["title"],
                "custom_case": _is_custom_case,
                "pack_revision": _pack_revision,
                "gold_claim_count": n_claims,
                "gold_fingerprint": _fp,
            }
        )
        st.session_state["beta_confirmed_gold"] = payload
        if _is_custom_case:
            _locked_set = {
                int(x)
                for x in (st.session_state.get("beta_locked_custom_slots") or [])
            }
            _locked_set.add(int(case_row["slot"]))
            st.session_state["beta_locked_custom_slots"] = sorted(_locked_set)
            _persist_beta_customs()
        st.success(
            t(
                "comp.lock_success",
                _guide_lang,
                n=case_row["slot"],
                claims=n_claims,
            )
        )
        st.warning(t("disclosure.confirm_new_cohort", _guide_lang, hash=""))

# Re-hydrate Lock for persisted custom slots (same API-key workspace).
if (
    _is_custom_case
    and int(case_row["slot"]) in set(_locked_custom_ids)
    and (stem or "").strip()
    and (prose or "").strip()
):
    _cur_fr = st.session_state.get("beta_confirmed_gold")
    _need_restore = not (
        isinstance(_cur_fr, dict)
        and int(_cur_fr.get("case_slot") or 0) == int(case_row["slot"])
        and scoring_versions_equivalent(
            str(_cur_fr.get("scoring_version") or ""), BETA_SV
        )
    )
    if _need_restore:
        try:
            _restored = auto_freeze_beta_slot(
                {
                    **case_row,
                    "stem": (stem or "").strip(),
                    "reference_prose": (prose or "").strip(),
                }
            )
            _restored["auto_confirmed"] = False
            _restored["custom_case"] = True
            _restored["pack_revision"] = _pack_revision
            st.session_state["beta_confirmed_gold"] = _restored
        except Exception:
            pass

frozen = st.session_state.get("beta_confirmed_gold")
frozen_ok = (
    isinstance(frozen, dict)
    and frozen.get("case_slot") == case_row["slot"]
    and scoring_versions_equivalent(
        str(frozen.get("scoring_version") or ""), BETA_SV
    )
)
if frozen_ok:
    st.success(
        t(
            "comp.lock_ok",
            _guide_lang,
            n=case_row["slot"],
            claims=int(frozen.get("gold_claim_count") or 0),
        )
    )
    st.caption(
        f"*Advanced · pack v{frozen.get('pack_revision') or _pack_revision} · "
        f"id `{frozen.get('gold_fingerprint') or '—'}` · "
        f"`{frozen.get('scoring_version')}`*"
    )
    st.caption(t("disclosure.confirm_new_cohort", _guide_lang, hash=""))
else:
    st.info(t("comp.need_lock", _guide_lang))

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
_n_pack_cases = len(pack_slots)
# Credible OpenRouter forecast (same helpers as Structured) — gold already frozen.
_live_case = load_case("caseC").model_copy(
    update={"id": CASE_ID, "stem": (stem or "").strip(), "title": str(case_row.get("title") or CASE_ID)}
)
_gold_for_est = ""
if frozen_ok and isinstance(frozen, dict):
    try:
        _gold_for_est = _frozen_to_gold(frozen).model_dump_json()
    except Exception:
        _gold_for_est = str(frozen.get("raw_text") or "")
_hist_for_cost = [a for a in _beta_arts if is_beta_artifact(a)]
_cost_kwargs = dict(
    include_qvac=_eff_medpsy,
    gold_reference=_gold_for_est,
    triple_qvac=_eff_medpsy,
    include_local_peers=_eff_generic,
    include_medical_peers=_eff_medical,
    include_extractor=False,
    extraction_cost_usd=0.0,
    history_artifacts=_hist_for_cost,
    local_only=not bool(include_cloud),
)
_bd_single = estimate_cost_breakdown(cfg, _live_case, n=1, **_cost_kwargs)
_bd_multi = estimate_cost_breakdown(cfg, _live_case, n=int(n_multi), **_cost_kwargs)
_bd_all = estimate_cost_breakdown(
    cfg, _live_case, n=max(1, _n_multi_all_cases * int(n_multi)), **_cost_kwargs
)

st.caption(
    f"**Multi ×N** repeats the selected case N times. "
    f"**Multi×all cases** walks pack Case 1→{_n_pack_cases}"
    + (
        f" + {len(_multi_all_extra)} Lock-ed custom"
        if _multi_all_extra
        else ""
    )
    + f" ({_n_multi_all_cases} cases), then again "
    f"(N passes) · e.g. N=2 → {_n_multi_all_cases * 2} rounds · pack "
    "references lock automatically · customs need Lock first · finished "
    "rounds stay visible below. "
    "**New launches** ask you to OK the estimated cost first."
)
st.caption(t("comp.ranking_pack_caption", _guide_lang))
show_cost_forecast = st.toggle(
    "Show OpenRouter cost forecast",
    value=bool(st.session_state.get("show_cost_forecast", True)),
    key="beta_show_cost_forecast",
    help="Pre-run forecast is a rough estimate (often over). "
    "Billed truth = OpenRouter usage.",
)
# Keep Structured + Comprehension toggles aligned for the shared spend card copy.
st.session_state["show_cost_forecast"] = bool(show_cost_forecast)

r1, r2, r3, r4 = st.columns(4)
with r1:
    single_clicked = st.button(
        "Single run",
        type="secondary",
        disabled=not (frozen_ok and _can_launch) or _slots_locked,
        use_container_width=True,
        help="Requires a locked reference · then cost OK / Yes before streams.",
    )
    if show_cost_forecast:
        st.markdown(fmt_cost_single(_bd_single), unsafe_allow_html=True)
with r2:
    multi_clicked = st.button(
        f"Multi ×{int(n_multi)}",
        type="primary",
        disabled=not (frozen_ok and _can_launch) or _slots_locked,
        use_container_width=True,
        help="N repeats on the selected locked case · cost OK required.",
    )
    if show_cost_forecast:
        st.markdown(fmt_cost_multi(_bd_multi, int(n_multi)), unsafe_allow_html=True)
with r3:
    multi_all_clicked = st.button(
        f"Multi×all cases · {_n_multi_all_cases}×{int(n_multi)}",
        disabled=not _can_launch or _slots_locked,
        use_container_width=True,
        type="secondary",
        help=(
            f"Round-robin pack Case 1→{_n_pack_cases}"
            + (
                f" + {len(_multi_all_extra)} Lock-ed custom"
                if _multi_all_extra
                else ""
            )
            + f" for N={int(n_multi)} passes "
            f"({_n_multi_all_cases * int(n_multi)} rounds). "
            "Empty/unlocked customs excluded. Cost OK required."
        ),
    )
    if show_cost_forecast:
        st.markdown(
            fmt_cost_multi(_bd_all, max(1, _n_multi_all_cases * int(n_multi))),
            unsafe_allow_html=True,
        )
with r4:
    if st.button("Stop / clear pending", use_container_width=True):
        for _k in (
            "beta_confirmed_run",
            "beta_pending_run",
            "beta_running",
            "beta_multi_progress",
        ):
            st.session_state.pop(_k, None)
        st.rerun()

# Arm spend gate (never start streams until Yes).
if single_clicked:
    st.session_state["beta_pending_run"] = {
        "n": 1,
        "rounds": 1,
        "multi_case": False,
        "mode": "full" if include_cloud else "local_only",
        "est": float(_bd_single.get("total_usd") or 0),
        "est_hi": float(
            _bd_single.get("total_usd_upper") or _bd_single.get("total_usd") or 0
        ),
    }
    st.rerun()
if multi_clicked:
    st.session_state["beta_pending_run"] = {
        "n": int(n_multi),
        "rounds": int(n_multi),
        "multi_case": False,
        "mode": "full" if include_cloud else "local_only",
        "est": float(_bd_multi.get("total_usd_for_n") or 0),
        "est_hi": float(
            _bd_multi.get("total_usd_upper_for_n")
            or _bd_multi.get("total_usd_for_n")
            or 0
        ),
    }
    st.rerun()
if multi_all_clicked:
    _rounds_all = max(1, _n_multi_all_cases * int(n_multi))
    st.session_state["beta_pending_run"] = {
        "n": int(n_multi),
        "rounds": _rounds_all,
        "multi_case": True,
        "n_pack": int(_n_pack_cases),
        "n_custom": int(len(_multi_all_extra)),
        "n_cases": int(_n_multi_all_cases),
        "mode": "full" if include_cloud else "local_only",
        "est": float(_bd_all.get("total_usd_for_n") or 0),
        "est_hi": float(
            _bd_all.get("total_usd_upper_for_n") or _bd_all.get("total_usd_for_n") or 0
        ),
    }
    st.rerun()

# Inline cost OK / Yes — same pattern as Structured (never st.dialog).
if (
    st.session_state.get("beta_pending_run")
    and not st.session_state.get("beta_confirmed_run")
    and not st.session_state.get("beta_running")
):
    render_spend_confirm_card(
        pending_key="beta_pending_run",
        confirmed_key="beta_confirmed_run",
        has_key=has_key,
        track_label="Comprehension",
    )
    st.stop()

# --- execute (only after Yes · start run) ---
run_cfg = st.session_state.pop("beta_confirmed_run", None)
_multi_case = bool(run_cfg and run_cfg.get("multi_case"))
_ready = bool(run_cfg and roster and (_multi_case or frozen_ok))
if _ready:
    st.session_state["beta_running"] = True
    st.session_state.pop("beta_pending_run", None)
    st.session_state.pop("beta_multi_progress", None)
    n_runs = int(run_cfg.get("n") or 1)
    cases_plan: List[Dict[str, Any]]
    if _multi_case:
        # Re-merge drafts at launch so Lock-ed customs match current session.
        _launch_slots = merge_beta_slots(
            pack_slots, st.session_state.get("beta_custom_drafts") or {}
        )
        _launch_locked = [
            int(x) for x in (st.session_state.get("beta_locked_custom_slots") or [])
        ]
        _launch_customs = custom_slots_ready_for_multi_all(
            _launch_slots, _launch_locked
        )
        cases_plan = list(pack_slots) + _launch_customs
        _n_pack_plan = len(pack_slots)
        _n_custom_plan = len(_launch_customs)
        st.info(
            f"**Multi×all cases** · round-robin pack 1→{_n_pack_plan}"
            + (
                f" + {_n_custom_plan} Lock-ed custom"
                if _n_custom_plan
                else ""
            )
            + f" × {n_runs} pass(es) = **{len(cases_plan) * n_runs}** rounds · "
            "progressive tabs below (same strip as graded Multi) · exploratory "
            "Comprehension track only."
        )
        st.warning(t("disclosure.banner_auto_freeze", _guide_lang))
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
    verifier_providers = list(judge_cfg.get("verifier_allowed_providers") or [])
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

    st.markdown(
        '<div class="sec-label">Live responses</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Model streams first · Multi progress / KPI / totals / Rebuild stay below "
        "these boxes. "
        + t("stream.live_caption_fs", str(st.session_state.get("lang") or "en"))
        + " Same order as Structured."
    )

    phase = st.empty()
    phase.markdown(
        f'<div class="phase-banner">Comprehension · preparing {total_rounds} round(s) · '
        f"`{PROTOCOL_ID}`</div>",
        unsafe_allow_html=True,
    )

    # Stream panels once — panel-card stack avoids column overlap during Multi.
    status_boxes: Dict[str, Any] = {}
    kpi_boxes: Dict[str, Any] = {}
    text_boxes: Dict[str, Any] = {}
    for row in panel_rows_for_roster(candidates_cfg):
        cols = st.columns(len(row) or 1, gap="small")
        for col, cand in zip(cols, row):
            key = cand["key"]
            _raw_color = str(cand.get("color") or "#64748b")
            color = (
                _raw_color
                if re.fullmatch(r"#[0-9A-Fa-f]{3,8}", _raw_color)
                else "#64748b"
            )
            label = str(cand.get("label") or key)
            model_bit = str(cand.get("model") or "")
            with col:
                st.markdown(
                    f'<div class="panel-card" style="border-top-color:{color}">'
                    f'<p class="live-head">{html.escape(label)}</p>'
                    f'<p class="live-meta">{html.escape(model_bit)}</p></div>',
                    unsafe_allow_html=True,
                )
                status_boxes[key] = st.empty()
                status_boxes[key].markdown(
                    status_pill("run", "Waiting…"), unsafe_allow_html=True
                )
                kpi_boxes[key] = st.empty()
                kpi_boxes[key].markdown(
                    '<div class="kpi-slot"></div>', unsafe_allow_html=True
                )
                shell_slot = st.empty()
                _shell = stream_shell_html(
                    title=label,
                    panel_id=key,
                    lang=str(st.session_state.get("lang") or "en"),
                )
                if hasattr(shell_slot, "html"):
                    shell_slot.html(_shell)
                else:
                    shell_slot.markdown(
                        _shell,
                        unsafe_allow_html=True,
                    )
                text_boxes[key] = st.empty()
                text_boxes[key].markdown(
                    stream_body_html("", live=False, panel_id=key),
                    unsafe_allow_html=True,
                )

    # Progressive Multi strip BELOW live panels (Live → Multi → KPI → Rebuild).
    multi_progress_slot = st.empty()
    if total_rounds > 1:
        _paint_beta_multi_progress(
            multi_progress_slot, [], n_total=total_rounds, batch_done=False, height=140
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
        _art_pack_rev = int(
            frozen_payload.get("pack_revision") or _pack_revision or 0
        )
        cohort = build_cohort_id(
            case_stem=live_stem,
            gold=gold_obj,
            prompt_version=PROMPT_VERSION,
            model_config=model_config,
            benchmark_track=benchmark_track,
            scoring_version=BETA_SV,
            pack_revision=_art_pack_rev or None,
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
                verifier_allowed_providers=verifier_providers,
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
            config_snapshot={
                "comprehension": True,
                "protocol": PROTOCOL_ID,
                **model_config,
            },
            run_id=f"comp-{uuid.uuid4().hex[:12]}",
            case_id=CASE_ID,
            started_at=datetime.fromtimestamp(t_run_i0, tz=timezone.utc).isoformat(),
            finished_at=utc_now_iso(),
            n_index=round_i,
            candidates=collected,
            judgments=judgments,
            ranking=ranking,
            cohort_id=cohort,
            scoring_version=BETA_SV,
            pack_revision=_art_pack_rev or None,
            prompt_version=PROMPT_VERSION,
            benchmark_track=benchmark_track,
            models_config={
                **model_config,
                "gold_reference": gold_json,
                "case_stem": live_stem,
                "reference_prose": frozen_payload.get("reference_prose")
                or frozen_payload.get("beta_reference_prose"),
                "comprehension_case_slot": case_slot,
                # Dual-write so older slot counters still see History.
                "beta_case_slot": case_slot,
                "comprehension_pass_i": pass_i,
                "comprehension_round_i": round_i,
                "comprehension_rounds_total": total_rounds,
                "comprehension_multi_case_batch": bool(_multi_case),
                "pack_revision": _art_pack_rev,
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
            key="comp_bar_live_single",
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
        _pre_mem = (
            None if getattr(RUN_STORE, "writes_plaintext", True) else _comp_preloaded()
        )
        if _multi_case:
            st.session_state["beta_rebuild_scope"] = "balanced_cases"
            _built = rebuild_balanced_cases_from_history(
                WORKSPACE_DIR,
                n=_n_cap,
                scoring_version=BETA_SV,
                track=benchmark_track,
                model_ids=_model_ids,
                preloaded=_pre_mem,
            )
        else:
            _built = rebuild_multi_from_history(
                WORKSPACE_DIR,
                CASE_ID,
                n=_n_cap,
                cohort_id=last_cohort,
                model_ids=_model_ids,
                scoring_version=BETA_SV,
                preloaded=_pre_mem,
            )
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
    # Persist last stream texts so idle layout keeps boxes ABOVE KPI / Rebuild.
    try:
        _snap: Dict[str, Any] = {}
        for _ck, _cand in [(c["key"], c) for c in candidates_cfg]:
            _snap[_ck] = {
                "label": str(_cand.get("label") or _ck),
                "model": str(_cand.get("model") or ""),
                "color": _cand.get("color") or "#64748b",
                "text": bufs.get(_ck, "") if "bufs" in dir() else "",
                "status": "Done",
            }
        # Prefer final collected answers when available.
        for _cand_obj in collected if "collected" in dir() else []:
            _ck = getattr(_cand_obj, "candidate_key", None)
            if not _ck:
                continue
            _snap.setdefault(_ck, {})
            _snap[_ck]["text"] = str(getattr(_cand_obj, "raw_text", None) or _snap[_ck].get("text") or "")
            _snap[_ck]["status"] = "Done"
        if _snap:
            st.session_state["beta_live_outputs"] = _snap
    except Exception:
        pass

    if _open_mean_popup:
        st.rerun()

# --- Idle: keep LLM output boxes above KPI / Rebuild (sequential UX) ---
if not st.session_state.get("beta_running") and not _ready:
    _saved_out = st.session_state.get("beta_live_outputs") or {}
    if _saved_out or roster:
        st.markdown(
            '<div class="sec-label">Live responses</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Last streams stay here · KPI / totals / Rebuild are below. "
            + t("stream.live_caption_fs", str(st.session_state.get("lang") or "en"))
            + " Same order as Structured."
        )
        _idle_rows = panel_rows_for_roster(roster) if roster else []
        if not _idle_rows and _saved_out:
            # Fallback: one row from saved keys.
            _idle_rows = [
                [
                    {
                        "key": k,
                        "label": v.get("label") or k,
                        "model": v.get("model") or "",
                        "color": v.get("color") or "#64748b",
                    }
                    for k, v in _saved_out.items()
                ]
            ]
        for _row in _idle_rows:
            _cols = st.columns(len(_row) or 1, gap="small")
            for _col, _cand in zip(_cols, _row):
                _key = _cand["key"]
                _prev = _saved_out.get(_key) or {}
                _label = html.escape(
                    str(_cand.get("label") or _prev.get("label") or _key)
                )
                _raw_c = str(
                    _cand.get("color") or _prev.get("color") or "#64748b"
                )
                _color = (
                    _raw_c
                    if re.fullmatch(r"#[0-9A-Fa-f]{3,8}", _raw_c)
                    else "#64748b"
                )
                _model_bit = html.escape(
                    str(_cand.get("model") or _prev.get("model") or "")
                )
                with _col:
                    st.markdown(
                        f'<div class="panel-card" style="border-top-color:{_color}">'
                        f'<p class="live-head">{_label}</p>'
                        f'<p class="live-meta">{_model_bit}</p></div>',
                        unsafe_allow_html=True,
                    )
                    if _prev.get("text") is not None:
                        st.markdown(
                            status_pill("done", str(_prev.get("status") or "Done")),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            status_pill("ready", "Ready"),
                            unsafe_allow_html=True,
                        )
                    _idle_shell = stream_shell_html(
                        title=str(
                            _cand.get("label") or _prev.get("label") or _key
                        ),
                        panel_id=_key,
                        lang=str(st.session_state.get("lang") or "en"),
                    )
                    _idle_shell_slot = st.empty()
                    if hasattr(_idle_shell_slot, "html"):
                        _idle_shell_slot.html(_idle_shell)
                    else:
                        _idle_shell_slot.markdown(
                            _idle_shell, unsafe_allow_html=True
                        )
                    st.markdown(
                        stream_body_html(
                            str(_prev.get("text") or "")
                            if _prev.get("text") is not None
                            else "",
                            live=False,
                            panel_id=_key,
                        ),
                        unsafe_allow_html=True,
                    )

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

# --- last KPIs (always BELOW live response boxes) ---
st.markdown("### Comprehension KPIs (this session)")
_last_rank = st.session_state.get("beta_last_ranking")
_last_sum = st.session_state.get("beta_last_multi_summary")
if _last_sum is not None:
    mean_rows = list(getattr(_last_sum, "ranking_mean", None) or [])
    if mean_rows:
        st.markdown("##### Ranking table")
        _sess_html = reliability_table_html(mean_rows)
        if hasattr(st, "html"):
            st.html(_sess_html)
        else:
            st.markdown(_sess_html, unsafe_allow_html=True)
        st.plotly_chart(
            fig_judge_mean_accuracy_bars(mean_rows, hide_partial_labels=True),
            use_container_width=True,
            key="comp_bar_kpi_mean",
        )
elif _last_rank:
    st.plotly_chart(
        fig_judge_accuracy_bars(_last_rank),
        use_container_width=True,
        key="comp_bar_kpi_last",
    )
else:
    st.caption("No Comprehension run in this session yet.")

# --- Comprehension Rebuild (isolated from graded) ---
st.markdown("### Rebuild average · Comprehension only")
st.caption(
    t("comp.rebuild_caption", _guide_lang)
    + f" · *Advanced · pack v{_pack_revision}*"
)
st.caption(t("comp.rebuild_public_claim", _guide_lang))
_bn = st.selectbox(
    t("bench.rebuild_n_label", _guide_lang),
    options=[5, 10, 20, 30, 50, 70, 100],
    index=0,
    key="beta_rebuild_n",
    format_func=lambda n: (
        f"≤{n}"
        + (" · exploratory" if n == 5 else "")
        + (" · steadier mean±std" if n == 10 else "")
        + (" · exploratory mean±std" if n in (20, 30, 50, 70, 100) else "")
    ),
    help=t("bench.rebuild_n_help", _guide_lang),
)
if "beta_rebuild_scope" not in st.session_state:
    st.session_state["beta_rebuild_scope"] = "balanced_cases"
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
    _pre_mem = (
        None if getattr(RUN_STORE, "writes_plaintext", True) else _comp_preloaded()
    )
    if _scope == "portfolio":
        built = rebuild_portfolio_from_history(
            WORKSPACE_DIR,
            n=int(_bn),
            scoring_version=BETA_SV,
            track=benchmark_track,
            model_ids=model_ids,
            pack_revision=_pack_revision,
            current_pack_revision=_pack_revision,
            preloaded=_pre_mem,
        )
    elif _scope == "balanced_cases":
        # Comprehension pack slots are Case 1…K; rebuild helper discovers stem order.
        built = rebuild_balanced_cases_from_history(
            WORKSPACE_DIR,
            n=int(_bn),
            scoring_version=BETA_SV,
            track=benchmark_track,
            model_ids=model_ids,
            pack_revision=_pack_revision,
            current_pack_revision=_pack_revision,
            preloaded=_pre_mem,
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
            pack_revision=_pack_revision,
            current_pack_revision=_pack_revision,
            preloaded=_pre_mem,
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

_hist = artifacts_for_case(
    WORKSPACE_DIR,
    CASE_ID,
    limit=20,
    preloaded=(
        None if getattr(RUN_STORE, "writes_plaintext", True) else _comp_preloaded()
    ),
)
_beta_hist = [
    a
    for _, a in _hist
    if scoring_versions_equivalent(str(a.scoring_version or ""), BETA_SV)
]
_hist_note = (
    "this browser session"
    if not getattr(RUN_STORE, "writes_plaintext", True)
    else "this workspace"
)
st.caption(
    f"Comprehension History · {len(_beta_hist)} recent saved run(s) in {_hist_note}. "
    f"· *Advanced · `{BETA_SV}`*"
)
st.caption(t("comp.same_key_warning", _guide_lang))
st.caption(t("comp.ranking_pack_caption", _guide_lang))
