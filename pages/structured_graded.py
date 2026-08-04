"""Structured · A1–A5 graded track (secondary page).

Rigid slot Q&A collect. History / Rebuild KPIs stay isolated from Comprehension
(``comprehension-v1``). Main entry is ``app.py`` (Comprehension home).
"""

from __future__ import annotations

import hashlib
import html
import os
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

# This file lives under pages/ — project root is the parent directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

import streamlit.components.v1 as components

from benchmark import openrouter
from benchmark.case_slots import (
    BASE_CASE_SLOTS,
    SOFT_MAX_CASE_SLOTS,
    bind_stem_to_slot,
    count_distinct_stem_keys,
    empty_slot,
    ensure_owner_slots,
    filter_artifacts_for_slot,
    load_default_pack_meta,
    load_pack_revision,
    open_new_case_slot,
    save_bindings,
    slot_label_for_artifact,
    validate_gold_for_restore,
)
from benchmark.cases_loader import case_display_name, load_case
from benchmark.config import is_usable_openrouter_key, load_models_config
from benchmark.gold import (
    LOCAL_QNA_EXTRACTOR_MODEL,
    SCORING_VERSION,
    SECTION_IDS,
    extract_with_chat,
    format_prepare_error,
    gold_json,
    is_strict_track,
    load_confirmed_gold,
    looks_like_qna_reference,
    source_quote_is_verbatim,
    track_ui_routing_blurb,
    uses_controlled_sampling,
)
from benchmark.gold import (
    cohort_id as build_cohort_id,
)
from benchmark.gold import (
    confirmed_gold as build_confirmed_gold,
)
from benchmark.judge import (
    PipelinedJudge,
    abandon_all_pipelines,
    build_ranking,
    explain_run_scores,
    systemic_judge_failure,
)
from benchmark.prompts import (
    candidate_system,
    candidate_user,
    local_chat_messages,
    missing_section_ids,
    parse_candidate_answers,
)
from benchmark.qvac_bridge import available as qvac_available
from benchmark.qvac_bridge import ensure_sidecar as qvac_ensure_sidecar
from benchmark.qvac_bridge import health as qvac_health
from benchmark.qvac_bridge import iter_tokens as qvac_iter_tokens
from benchmark.qvac_bridge import load_model as qvac_load_model
from benchmark.qvac_bridge import reachable as qvac_reachable
from benchmark.qvac_variants import (
    is_on_device_key,
    is_qvac_key,
    medical_peers_ready,
    merge_roster,
    panel_rows_for_roster,
)
from benchmark.report import (
    artifacts_for_case,
    find_case_family_cohorts,
    is_mean_poolable_run,
    list_portfolio_runs,
    load_artifact,
    planned_on_device_model_contract,
    rebuild_model_ids,
    rebuild_multi_from_history,
    rebuild_balanced_cases_from_history,
    rebuild_portfolio_from_history,
    reliability_caption,
    scoring_versions_equivalent,
    summarize_multi_batch,
    write_artifact,
)
from benchmark.run_control import cancel_run, finish_run, is_cancelled, start_run
from benchmark.runner import (
    _validate_judge_separation,
    build_run_artifact,
    estimate_cost_breakdown,
    is_retryable_local_error,
    iter_collect_live,
    maybe_retry_candidate,
    prepare_run,
)
from benchmark.schema import (
    CandidateAnswer,
    Case,
    GoldSection,
    ModelCallMeta,
    RunArtifact,
    utc_now_iso,
)
from benchmark.workspace import (
    assert_path_in_workspace,
    maybe_claim_legacy_root_artifacts,
    owner_id_for_current_key,
    scoped_artifacts_dir,
    short_owner_label,
)
from lib.benchmark_multi_ui import (
    client_toast_run_done,
    finished_multi_progress,
    live_judging_board_html,
    paint_rebuild_ops_reliability_panels,
    progressive_multi_panel_html,
    reliability_table_html as _reliability_table_html,
    short_model,
    snapshot_from_artifact,
)
from lib.charts import (
    fig_judge_accuracy_bars,
    fig_judge_mean_accuracy_bars,
)
from lib.deployment import (
    capture_and_strip_openrouter_env,
    is_local_install,
    is_streamlit_cloud,
)
from lib.disclosure import (
    DEFAULT_ROSTER_VERSION,
    honesty_block_html,
    rebuild_scan_honesty_html,
    scope_label,
    screenshot_footer_html,
    screenshot_share_checklist_html,
    short_cohort,
)
from lib.boot_welcome import init_boot_state, run_boot_dialogs
from lib.guide_overlays import guides_always_available_html
from lib.i18n import t
from lib.stream_panels import (
    stream_body_html as _stream_body_html_shared,
    stream_shell_html as _stream_shell_html_shared,
)
from lib.spend_confirm import (
    fmt_cost_multi as _fmt_cost_multi,
    fmt_cost_single as _fmt_cost_single,
    render_spend_confirm_card,
)
from lib.track_sidebar import render_guides_and_protocol, render_tracks_block
from lib.model_labels import (
    OPTIONAL_LEGACY_SLOT_KEYS,
    filter_current_roster_rows,
    name_and_version,
    rerank_rows,
)
from lib.secure_account_store import (
    AccountSession,
)
from lib.secure_account_store import (
    configured as account_store_configured,
)
from lib.secure_account_store import (
    list_artifacts as account_list_artifacts,
)
from lib.secure_account_store import (
    load_openrouter_key as account_load_key,
)
from lib.secure_account_store import (
    save_artifact as account_save_artifact,
)
from lib.secure_account_store import (
    save_openrouter_key as account_save_key,
)
from lib.ui_prefs import load_qvac_sdk_ack


def _nv(key, *, label=None, model=None):
    """(Name, Version) for ranking tables / charts."""
    return name_and_version(str(key or ""), label=label, model=model)


def _current_ranking(rows, *, score_field: str = "accuracy"):
    """Drop legacy models; rewrite ranks for the active 9-model roster."""
    return rerank_rows(
        filter_current_roster_rows(rows),
        score_field=score_field,
    )


def _mean_rows_to_last_ranking(ranking_mean):
    """Persist mean summary as last_ranking (current roster + per-model Runs)."""
    out = []
    for r in _current_ranking(ranking_mean or [], score_field="accuracy_mean"):
        if r.get("eligible") is False or r.get("rank") is None:
            continue
        if r.get("accuracy_mean") is None:
            continue
        out.append(
            {
                "key": r["key"],
                "rank": r["rank"],
                "accuracy": r["accuracy_mean"],
                "label": short_model(str(r["key"])),
                "status": "partial" if r.get("partial") else "ok",
                "partial": bool(r.get("partial")),
                "std": r.get("std"),
                "cv_pct": r.get("cv_pct"),
                "n_runs": int(r.get("n_runs") or r.get("n") or 0),
                "n_requested": int(r.get("n_requested") or r.get("n_runs") or 0),
                "n_failed": int(r.get("n_failed") or 0),
            }
        )
    return out

st.set_page_config(
    page_title="Structured (legacy / advanced) · Cloud & local medical LLMs",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

LIVE_BOX_H = 168
if "_run_scope" not in st.session_state:
    st.session_state["_run_scope"] = uuid.uuid4().hex


def _finish_scope_run() -> None:
    run_id = st.session_state.pop("_active_run_id", None)
    if run_id:
        finish_run(st.session_state["_run_scope"], str(run_id))

# --- .env ---
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if not k:
            continue
        # Never treat a truncated OpenRouter placeholder as a real key
        if k == "OPENROUTER_API_KEY" and not is_usable_openrouter_key(v):
            continue
        if k not in os.environ:
            os.environ[k] = v

# --- API key: never share one Streamlit Secret / .env key with every visitor ---
# On Streamlit Cloud, strip any process-wide OPENROUTER_API_KEY so visitor B cannot
# silently spend visitor A's credits. Prefill comes only from an authenticated
# encrypted account vault (or this browser session). Local .env remains developer-only.
_server_env_key = capture_and_strip_openrouter_env()

# Streamlit 1.50 DOMPurify empties <style> from st.html (event container stays blank).
# Inject CSS + portal into parent.document.head via components iframe instead.
_ASSETS_DIR = ROOT / "assets"
_dashboard_css = (_ASSETS_DIR / "dashboard.css").read_text(encoding="utf-8")
_css_js_literal = (
    _dashboard_css.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
)
_css_inject = f"""
(function(){{
  var doc;
  try {{ doc = window.parent && window.parent.document ? window.parent.document : document; }}
  catch (e) {{ doc = document; }}
  var s = doc.getElementById('qvac-dashboard-css');
  if (!s) {{
    s = doc.createElement('style');
    s.id = 'qvac-dashboard-css';
    doc.head.appendChild(s);
  }}
  s.textContent = `{_css_js_literal}`;
  if (!doc.getElementById('qvac-fonts')) {{
    var l = doc.createElement('link');
    l.id = 'qvac-fonts';
    l.rel = 'stylesheet';
    l.href = 'https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap';
    doc.head.appendChild(l);
  }}
}})();
"""
_portal_js = (_ASSETS_DIR / "dashboard_portal.js").read_text(encoding="utf-8")
components.html(
    f"<script>{_css_inject}\n{_portal_js}</script>",
    height=0,
    width=0,
)


cfg = load_models_config()
judge_cfg = cfg.get("judge") or {}

# Keep the real QVAC SDK sidecar alive for this session (not a mock).
# Spawn quickly; do not freeze the UI for a long warm-load wait.
if "qvac_ensure_tried" not in st.session_state:
    st.session_state.qvac_ensure_tried = False
if not qvac_available() and not st.session_state.qvac_ensure_tried:
    with st.spinner("Starting on-device MedPsy via QVAC SDK…"):
        ensured = qvac_ensure_sidecar(wait_s=12.0, start_if_down=True)
    st.session_state.qvac_ensure_tried = True
    st.session_state["qvac_ensure_error"] = ensured.get("ensure_error") or ensured.get(
        "lastError"
    )
    # If still loading, allow another pass next run
    if ensured.get("modelLoaded"):
        st.rerun()
    elif qvac_reachable():
        st.session_state.qvac_ensure_tried = False

qvac_ok = qvac_available()  # MedPsy loaded
qvac_up = qvac_reachable()  # sidecar HTTP up (may still be loading)
# Sidecar up is enough — /load can hot-swap GGUFs before generate.
qvac_run_ok = bool(qvac_ok or qvac_up)

# Restore only from an authenticated encrypted vault; never from IP identity.
_account_session = st.session_state.get("account_session")
if (
    account_store_configured()
    and isinstance(_account_session, AccountSession)
    and not st.session_state.get("_account_key_loaded")
):
    try:
        _account_key = account_load_key(_account_session)
        if is_usable_openrouter_key(_account_key):
            st.session_state["or_key_session"] = _account_key
        st.session_state["_account_key_loaded"] = True
    except Exception as _account_exc:
        st.session_state["_account_key_error"] = str(_account_exc)

# Local install may prefill .env; cloud keys must come from the authenticated/session vault.
if (
    (not is_streamlit_cloud())
    and is_local_install()
    and is_usable_openrouter_key(_server_env_key)
    and not st.session_state.get("or_key_session")
):
    st.session_state["or_key_session"] = _server_env_key

_session_key = (st.session_state.get("or_key_session") or "").strip()
if _session_key and not is_usable_openrouter_key(_session_key):
    st.session_state.pop("or_key_session", None)
    _session_key = ""
has_key = is_usable_openrouter_key(_session_key)

# Local: key-scoped (or `_local_no_key`). Cloud without key/account: per-browser
# ephemeral owner — never shared hash of "anonymous" / `_local_no_key`.
_account_uid = (
    _account_session.user_id
    if isinstance(_account_session, AccountSession)
    else None
)
_cloud_ephemeral = None
if is_streamlit_cloud() and not has_key and not _account_uid:
    if "_cloud_anon_ws" not in st.session_state:
        st.session_state["_cloud_anon_ws"] = str(uuid.uuid4())
    _cloud_ephemeral = str(st.session_state["_cloud_anon_ws"])
WORKSPACE_DIR = scoped_artifacts_dir(
    _session_key,
    account_user_id=_account_uid,
    cloud_ephemeral_id=_cloud_ephemeral,
)
# Encrypted Supabase path (Cloud + Auth). Any Cloud path skips host plaintext.
_HOSTED_ENCRYPTED = bool(
    is_streamlit_cloud()
    and account_store_configured()
    and isinstance(_account_session, AccountSession)
)
_HOSTED_NO_PLAINTEXT = bool(is_streamlit_cloud())
_moved_legacy = maybe_claim_legacy_root_artifacts()
if _moved_legacy and not st.session_state.get("_legacy_artifacts_toast"):
    st.session_state["_legacy_artifacts_toast"] = True
    try:
        st.toast(
            f"Restored {_moved_legacy} earlier run(s) into your History "
            "(Custom Case / Demo).",
            icon="📂",
        )
    except Exception:
        pass
# Refresh workspace path after claim (same dir, now populated)
WORKSPACE_DIR = scoped_artifacts_dir(
    _session_key,
    account_user_id=_account_uid,
    cloud_ephemeral_id=_cloud_ephemeral,
)
if (
    account_store_configured()
    and isinstance(_account_session, AccountSession)
    and not st.session_state.get("_account_artifacts_synced")
):
    try:
        _synced = []
        for _cloud_row in account_list_artifacts(_account_session, limit=200):
            _synced.append(_cloud_row["artifact"])
            # Local + Supabase may mirror to disk; Cloud never writes plaintext.
            if not _HOSTED_NO_PLAINTEXT:
                write_artifact(_cloud_row["artifact"], WORKSPACE_DIR)
        st.session_state["_account_artifacts_memory"] = _synced
        st.session_state["_account_artifacts_synced"] = True
    except Exception as _sync_exc:
        st.session_state["_account_sync_error"] = str(_sync_exc)

from lib.run_store import HostedRunStore, LocalRunStore

# Cloud Structured: session-memory only (no plaintext run JSON on host FS).
# With Supabase Auth, also encrypt to account vault. Local: LocalRunStore.
if is_streamlit_cloud():
    _hosted_kwargs = dict(
        memory=list(st.session_state.get("_account_artifacts_memory") or []),
        memory_setter=lambda arts: st.session_state.__setitem__(
            "_account_artifacts_memory", arts
        ),
        summaries=list(st.session_state.get("_account_summaries_memory") or []),
        summaries_setter=lambda s: st.session_state.__setitem__(
            "_account_summaries_memory", s
        ),
    )
    if _HOSTED_ENCRYPTED:
        _hosted_kwargs["account_session"] = _account_session
        _hosted_kwargs["save_cloud"] = account_save_artifact
        _hosted_kwargs["error_setter"] = lambda msg: st.session_state.__setitem__(
            "_hosted_cloud_save_error", msg
        )
    RUN_STORE = HostedRunStore(**_hosted_kwargs)
else:
    RUN_STORE = LocalRunStore(WORKSPACE_DIR)

_hosted_save_err = st.session_state.get("_hosted_cloud_save_error")
if _hosted_save_err:
    st.warning(
        "Encrypted cloud save failed — run kept in session memory "
        f"(no plaintext fallback): {_hosted_save_err}"
    )


def _persist_run_artifact(artifact, workspace: Path):
    """Persist via LocalRunStore or HostedRunStore (no plaintext on hosted)."""
    return RUN_STORE.persist_artifact(artifact)


def _preloaded_artifacts():
    return [a for _, a in RUN_STORE.list_artifacts()]


def _persist_summary(summary):
    return RUN_STORE.persist_summary(summary)

# --- Startup dialogs ---
# API key: every browser refresh starts a new session → key prompt again (BYOK).
# QVAC SDK status: ask once, then remember locally (.ui_prefs.json) so reload
# does not force a second OK after the key dialog.
init_boot_state()
if "qvac_sdk_ack" not in st.session_state:
    st.session_state.qvac_sdk_ack = load_qvac_sdk_ack()

# Clear sticky run flag when idle (never clear while a confirmed/pending run exists).
if (
    not st.session_state.get("confirmed_run")
    and not st.session_state.get("pending_run")
    and not st.session_state.get("benchmark_running")
):
    pass  # idle


def _hard_abort_run(*, flash: bool = True) -> None:
    """Kill pending/active run state and wipe live panels. History on disk stays."""
    # Best-effort: cancel queued DeepSeek futures (in-flight HTTP may still finish).
    cancelled_snapshots = []
    try:
        cancel_run(st.session_state["_run_scope"])
        cancelled_snapshots = abandon_all_pipelines(st.session_state["_run_scope"])
    except Exception:
        pass
    for _snapshot in cancelled_snapshots:
        try:
            _case = _snapshot["case"]
            _candidates = list(_snapshot.get("candidates") or [])
            _judgments = list(_snapshot.get("judgments") or [])
            if not _candidates:
                continue
            _judge_model = str(_snapshot.get("judge_model") or "")
            _total_cost = sum(
                float(candidate.meta.cost_usd or 0.0) for candidate in _candidates
            ) + sum(
                float(judgment.judge_meta.cost_usd or 0.0)
                for judgment in _judgments
            )
            _artifact = build_run_artifact(
                config_snapshot={"judge": {"model": _judge_model}},
                run_id=f"{_case.id}-{uuid.uuid4().hex[:10]}",
                case_id=_case.id,
                started_at=str(_snapshot.get("started_at") or utc_now_iso()),
                finished_at=utc_now_iso(),
                batch_id=uuid.uuid4().hex,
                n_index=1,
                models_config={
                    "mode": "cancelled",
                    "judge": {"model": _judge_model},
                    "gold_reference": _snapshot.get("gold_reference") or "",
                    "candidates": [
                        {
                            "key": candidate.candidate_key,
                            "model": candidate.meta.model,
                        }
                        for candidate in _candidates
                    ],
                },
                candidates=_candidates,
                judgments=_judgments,
                ranking=build_ranking(_judgments),
                total_cost_usd=round(_total_cost, 6),
                notes="Cancelled by user; submitted candidate KPIs retained.",
                run_status="cancelled",
                benchmark_track=str(
                    _snapshot.get("benchmark_track") or "controlled"
                ),
                # Prefer restored cohort only. Do not reuse _active_cohort_id from a
                # prior completed run — that stamp could mis-label a cancelled
                # mid-batch artifact. Empty cohort_id stays non-poolable.
                cohort_id=str(
                    st.session_state.get("_restored_cohort_id") or ""
                ),
            )
            _persist_run_artifact(_artifact, WORKSPACE_DIR)
            if account_store_configured() and isinstance(
                st.session_state.get("account_session"), AccountSession
            ):
                account_save_artifact(
                    st.session_state["account_session"],
                    _artifact,
                )
        except Exception:
            # Cancellation itself must not be blocked by persistence failure.
            pass
    for k in (
        "confirmed_run",
        "pending_run",
        "benchmark_running",
        "live_outputs",
        "multi_progress",
        "show_run_done",
        "kpi_dialog_armed",
        "multi_run_popup_path",
        "history_popup_path",
        "show_history_mean_popup",
        "last_ranking",
        "last_judgments",
        "last_cost_rows",
        "show_last_run_costs",
        "last_multi_summary",
        "last_multi_paths",
        "last_explain",
        "inline_run_path",
        "inline_run_kind",
    ):
        st.session_state.pop(k, None)
    _finish_scope_run()
    st.session_state["benchmark_running"] = False
    if flash:
        st.session_state["run_aborted_flash"] = True


def _mask_api_key(key: str) -> str:
    """Show start + end only (middle hidden) for video / confirmation UI."""
    k = (key or "").strip()
    if not k:
        return "(none)"
    if len(k) <= 16:
        return "•" * min(len(k), 12)
    return f"{k[:10]}…{'•' * 8}…{k[-4:]}"


def _client_guide_overlay(uid: str, title: str, body_html: str) -> str:
    """Fullscreen guide overlay toggled by <label for=uid> — no Streamlit rerun."""
    u = html.escape(uid)
    t = html.escape(title)
    return f"""
<input type="checkbox" id="{u}" class="fs-ck" autocomplete="off" />
<div class="fs-overlay" hidden style="display:none !important;visibility:hidden !important">
  <div class="fs-card">
    <div class="fs-bar">
      <span>{t}</span>
      <button type="button" class="fs-close" data-fs="{u}" title="Close" aria-label="Close">✕</button>
    </div>
    <div class="guide-body">{body_html}</div>
  </div>
</div>
"""


def _guides_always_available_html(*, qvac_status_line: str = "") -> str:
    """Compat wrapper — shared implementation in ``lib/guide_overlays``."""
    return guides_always_available_html(
        qvac_status_line=qvac_status_line,
        lang=_ui_lang(),
    )


QVAC_SETUP_GUIDE = """
### What you need for on-device MedPsy

- **QVAC software** running locally (the `sidecar/` folder)  
- **MedPsy model file** in `models/` (GPU/Metal preferred when available)  
- **Node.js 22+** to run the sidecar  

### Setup after cloning

1. Install Node.js 22+ from https://nodejs.org/  
2. Put the MedPsy model file in `models/`  
3. From the **project folder**, in a second terminal:

```bash
./scripts/setup_qvac_sidecar.sh
cd sidecar && npm start
```

4. Leave that terminal open, then refresh this page.

When the sidecar is running, MedPsy is included (on your machine, $0 API).  
If GPU load fails on some Macs, it retries on CPU automatically.
"""


def _clear_all_kpi_popups() -> None:
    """Drop KPI dialog flags + inline saved-run panel."""
    for k in (
        "history_popup_path",
        "multi_run_popup_path",
        "show_run_done",
        "show_history_mean_popup",
        "kpi_dialog_armed",
        "history_path",
        "inline_run_path",
        "inline_run_kind",
    ):
        st.session_state.pop(k, None)


def _open_saved_run_inline(path: str, *, kind: str = "history") -> None:
    """Show a saved run as an in-page panel — never a st.dialog (✕ aborts runs)."""
    if st.session_state.get("benchmark_running") or st.session_state.get("confirmed_run"):
        return
    st.session_state["inline_run_path"] = str(path)
    st.session_state["inline_run_kind"] = kind
    for k in (
        "kpi_dialog_armed",
        "history_popup_path",
        "multi_run_popup_path",
        "show_run_done",
    ):
        st.session_state.pop(k, None)


def _arm_kpi_dialog(kind: str, *, path: str | None = None) -> None:
    """History / per-run detail → inline panel. Rebuild mean may still use a dialog."""
    if st.session_state.get("benchmark_running") or st.session_state.get("confirmed_run"):
        return
    if kind in ("history", "multi_run") and path:
        _open_saved_run_inline(path, kind=kind)
        return
    st.session_state["kpi_dialog_armed"] = kind
    if kind == "rebuild":
        st.session_state["show_history_mean_popup"] = True
    elif kind == "run_done":
        st.session_state["show_run_done"] = True


def _on_case_fields_edit() -> None:
    """Blur/edit on Step 1 / Step 2 — never show KPI popups."""
    _clear_all_kpi_popups()


def _on_rebuild_n_pick_change() -> None:
    """Changing 'Average over N runs' must not open any KPI popup."""
    _clear_all_kpi_popups()


def _ui_lang() -> str:
    return str(st.session_state.get("lang") or "en")


def _source_fingerprint(case_stem: str, reference_raw: str) -> str:
    return hashlib.sha256(
        f"{(case_stem or '').strip()}\n---\n{(reference_raw or '').strip()}".encode()
    ).hexdigest()


def _restore_confirmed_gold_contract(
    *,
    gold_reference_json: str,
    case_stem_saved: str,
    cohort_id: str = "",
) -> None:
    """Load an exact prior confirmed gold into session (resume-by-restore).

    Does not fuzzy-match claims. Treats restore as a fresh user confirmation of
    that saved contract (new confirmed_at; cohort hash excludes timestamp).
    """
    gold = load_confirmed_gold(gold_reference_json)
    gold = gold.model_copy(update={"confirmed_at": utc_now_iso()})
    effective = gold_json(gold)
    stem = (case_stem_saved or "").strip() or str(
        st.session_state.get("demo_case_stem") or ""
    ).strip()
    st.session_state["demo_gold_ref"] = gold.raw_text
    if stem:
        st.session_state["demo_case_stem"] = stem
    st.session_state["_confirmed_gold_json"] = effective
    st.session_state["_gold_confirmed_at"] = gold.confirmed_at
    sections_dump = {
        sid: gold.sections[sid].model_dump() for sid in SECTION_IDS
    }
    st.session_state["_prepared_gold_sections"] = sections_dump
    st.session_state["_gold_sections"] = sections_dump
    st.session_state["_prepared_extraction_meta"] = {
        "model": gold.extraction_model or "",
        "cost_usd": float(gold.extraction_cost_usd or 0.0),
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    st.session_state["_prepared_extraction_cost"] = float(
        gold.extraction_cost_usd or 0.0
    )
    if cohort_id:
        st.session_state["_restored_cohort_id"] = cohort_id
    else:
        st.session_state.pop("_restored_cohort_id", None)
    for _wk in list(st.session_state.keys()):
        if str(_wk).startswith("prep_sum_") or str(_wk).startswith("prep_q_"):
            st.session_state.pop(_wk, None)
    st.session_state["_gold_source_fingerprint"] = _source_fingerprint(
        stem, gold.raw_text
    )
    st.session_state["_persist_case_stem"] = stem
    st.session_state["_persist_gold_ref"] = gold.raw_text


def _clear_case_editor_state() -> None:
    """Blank stem/reference + prepared/confirmed gold (empty Case slot)."""
    st.session_state["demo_case_stem"] = ""
    st.session_state["demo_gold_ref"] = ""
    st.session_state["_persist_case_stem"] = ""
    st.session_state["_persist_gold_ref"] = ""
    for _gk in (
        "_confirmed_gold_json",
        "_gold_sections",
        "_prepared_gold_sections",
        "_prepared_extraction_meta",
        "_prepared_extraction_cost",
        "_gold_confirmed_at",
        "_restored_cohort_id",
        "_prepare_error",
    ):
        st.session_state.pop(_gk, None)
    for _wk in list(st.session_state.keys()):
        if str(_wk).startswith("prep_sum_") or str(_wk).startswith("prep_q_"):
            st.session_state.pop(_wk, None)
    st.session_state["_gold_source_fingerprint"] = _source_fingerprint("", "")


def _select_case_slot(slot, *, as_new: bool = False) -> None:
    """Activate a Case slot and load stem+gold from History or drafts when filled."""
    st.session_state["active_case_slot"] = int(slot.index)
    if as_new or not slot.filled:
        _clear_case_editor_state()
        return
    if slot.gold_reference and validate_gold_for_restore(slot.gold_reference):
        _restore_confirmed_gold_contract(
            gold_reference_json=slot.gold_reference,
            case_stem_saved=slot.stem,
            cohort_id=slot.cohort_id or "",
        )
        return
    # Draft / freeform gold (Prepare + Confirm still required) or stem-only.
    stem = slot.stem or ""
    gold_raw = getattr(slot, "gold_raw", "") or ""
    st.session_state["demo_case_stem"] = stem
    st.session_state["demo_gold_ref"] = gold_raw
    st.session_state["_persist_case_stem"] = stem
    st.session_state["_persist_gold_ref"] = gold_raw
    for _gk in (
        "_confirmed_gold_json",
        "_gold_sections",
        "_prepared_gold_sections",
        "_prepared_extraction_meta",
        "_prepared_extraction_cost",
        "_gold_confirmed_at",
        "_restored_cohort_id",
        "_prepare_error",
    ):
        st.session_state.pop(_gk, None)
    for _wk in list(st.session_state.keys()):
        if str(_wk).startswith("prep_sum_") or str(_wk).startswith("prep_q_"):
            st.session_state.pop(_wk, None)
    st.session_state["_gold_source_fingerprint"] = _source_fingerprint(stem, gold_raw)


def _dlg_full_text(text: str) -> None:
    """Show full answer text in dialogs without clipping line starts/ends."""
    st.markdown(
        f'<pre class="dlg-pre">{html.escape(text or "")}</pre>',
        unsafe_allow_html=True,
    )


def _on_spend_confirm(pending: dict) -> None:
    """Persist stem/gold then arm confirmed_run (shared spend card callback)."""
    st.session_state["_persist_case_stem"] = (
        st.session_state.get("demo_case_stem") or ""
    )
    st.session_state["_persist_gold_ref"] = (
        st.session_state.get("demo_gold_ref") or ""
    )
    st.session_state["confirmed_run"] = pending


def _render_spend_confirm_card() -> None:
    """Thin wrapper — shared inline Yes/Cancel gate (never st.dialog)."""
    render_spend_confirm_card(
        pending_key="pending_run",
        confirmed_key="confirmed_run",
        has_key=has_key,
        track_label="Structured A1–A5",
        on_confirm=_on_spend_confirm,
    )


@st.dialog("QVAC SDK + MedPsy setup guide")
def qvac_setup_guide_dialog():
    st.markdown(QVAC_SETUP_GUIDE)
    st.markdown(
        f"**Status on this machine right now:** "
        f"{'ready — MedPsy will be included' if qvac_available() else ('sidecar online · MedPsy not loaded' if qvac_reachable() else 'sidecar offline — start it to include on-device')}"
    )
    if st.button("Close", type="primary", use_container_width=True):
        st.session_state["show_qvac_guide"] = False
        st.rerun()


def _remember_openrouter_key(key: str) -> None:
    """Activate a key in this Streamlit session; never mutate process-global env."""
    key = (key or "").strip()
    if not is_usable_openrouter_key(key):
        return
    st.session_state["or_key_session"] = key
    account = st.session_state.get("account_session")
    if account_store_configured() and isinstance(account, AccountSession):
        account_save_key(account, key)
        st.session_state["_account_key_remembered"] = True
    else:
        st.session_state["_account_key_remembered"] = False


def _fmt_ram_mb(ram_mb) -> str:
    """Human label for sidecar process-tree RSS (MB). Not VRAM / full mmap."""
    try:
        mb = float(ram_mb)
    except (TypeError, ValueError):
        return ""
    if mb >= 1024:
        return f"RAM(RSS) {mb / 1024:.1f} GB"
    if mb >= 100:
        return f"RAM(RSS) {mb:.0f} MB"
    return f"RAM(RSS) {mb:.1f} MB"


def _fmt_gguf_mb(gguf_mb) -> str:
    """On-disk GGUF size (MB)."""
    try:
        mb = float(gguf_mb)
    except (TypeError, ValueError):
        return ""
    if mb >= 1024:
        return f"GGUF {mb / 1024:.1f} GB"
    if mb >= 100:
        return f"GGUF {mb:.0f} MB"
    return f"GGUF {mb:.1f} MB"


def _render_saved_run_panel(path_str: str, *, key_prefix: str = "saved") -> None:
    """In-page review of a past artifact (sidebar History / Run tabs). No st.dialog."""
    hist = None
    if str(path_str).startswith("memory:"):
        rid = str(path_str).split(":", 1)[1]
        for _p, _a in RUN_STORE.list_artifacts():
            if _a.run_id == rid:
                hist = _a
                break
        if hist is None:
            st.error("That in-memory run is no longer available.")
            return
    else:
        _hp = Path(path_str)
        if not assert_path_in_workspace(_hp, WORKSPACE_DIR):
            st.error("That run is not in your private history (API key mismatch).")
            return
        try:
            hist = load_artifact(_hp)
        except Exception as exc:
            st.error(f"Could not load run: {exc}")
            return

    when = (hist.finished_at or hist.started_at or "")[:19].replace("T", " ")
    if hist.schema_version != "2.0" or not hist.cohort_id:
        st.warning(
            "Legacy experimental artifact · excluded from gold-only cohort rankings. "
            "Visible for audit only."
        )
    st.caption(
        f"{case_display_name(hist.case_id)} · run {hist.n_index} · {when} · "
        f"${hist.total_cost_usd:.4f} · `{Path(path_str).name}`"
    )
    _eff = (hist.reproducibility or {}).get("effective_judge") or ""
    _pri = (hist.reproducibility or {}).get("primary_judge") or ""
    if _eff:
        _judge_note = f"Effective judge · `{_eff}`"
        if _pri and _pri != _eff:
            _judge_note += f" (primary was `{_pri}`)"
        if (hist.reproducibility or {}).get("verifier_activated"):
            _judge_note += " · whole-run verifier active"
        st.caption(_judge_note)

    _hist_rank = _current_ranking(hist.ranking or [])
    _hist_cands = filter_current_roster_rows(
        hist.candidates or [], key_field="candidate_key"
    )
    _hist_judgments = filter_current_roster_rows(
        hist.judgments or [], key_field="candidate_key"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Case", case_display_name(hist.case_id))
    m2.metric("Models", str(len(_hist_rank) or len(_hist_cands)))
    m3.metric("Cost $", f"{hist.total_cost_usd:.3f}")

    if _hist_rank:
        st.plotly_chart(
            fig_judge_accuracy_bars(
                [
                    {
                        **r,
                        "label": next(
                            (
                                c.display_label or c.label
                                for c in _hist_cands
                                if c.candidate_key == r.get("key")
                            ),
                            r.get("key"),
                        ),
                    }
                    for r in _hist_rank
                ],
                height=220,
                title=f"Run {hist.n_index} · Clinical Composite Score",
            ),
            use_container_width=True,
            key=f"{key_prefix}_rank_chart_{hist.n_index}_{_hp.stem[-8:]}",
        )
        rows = []
        for r in _hist_rank:
            cand = next(
                (c for c in _hist_cands if c.candidate_key == r.get("key")),
                None,
            )
            nm, ver = _nv(
                r.get("key"),
                label=(cand.display_label or cand.label) if cand else r.get("label"),
                model=(cand.meta.model if cand and cand.meta else None)
                or r.get("model"),
            )
            na = str(r.get("status") or "ok") != "ok" or r.get("accuracy") is None
            rows.append(
                {
                    "#": None if na else r.get("rank"),
                    "Name": nm,
                    "Version": ver,
                    "Clinical Composite %": "N/A · technical" if na else r.get("accuracy"),
                    "TTFT": r.get("ttft_s"),
                    "TPS": r.get("tps"),
                    "RAM(RSS)": _fmt_ram_mb(r.get("ram_mb")) or "—",
                    "GGUF": _fmt_gguf_mb(r.get("gguf_mb")) or "—",
                    "$": r.get("cost_usd"),
                    "Runs": int(r.get("n_runs") or 1),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if _hist_judgments:
        q_ids = []
        for j in _hist_judgments:
            for qs in j.question_scores:
                if qs.question_id not in q_ids:
                    q_ids.append(qs.question_id)
        matrix = []
        for j in _hist_judgments:
            nm, ver = _nv(j.candidate_key)
            row = {"Name": nm, "Version": ver}
            failed = str(getattr(j, "status", "valid") or "valid") != "valid"
            by_q = {
                qs.question_id: qs.score for qs in j.question_scores
            } if not failed else {}
            for qid in q_ids:
                row[qid] = "N/A" if failed else by_q.get(qid)
            row["Clinical Composite %"] = (
                "N/A · technical" if failed else j.weighted_accuracy
            )
            row["Runs"] = 1
            matrix.append(row)
        st.markdown("**Scores by clinical dimension**")
        st.dataframe(pd.DataFrame(matrix), use_container_width=True, hide_index=True)

    name_by_key = {
        c.candidate_key: (c.display_label or c.label or c.candidate_key)
        for c in _hist_cands
    }
    score_by_key = {j.candidate_key: j for j in _hist_judgments}
    for c in _hist_cands:
        name = name_by_key.get(c.candidate_key, c.candidate_key)
        j = score_by_key.get(c.candidate_key)
        acc = f"{j.weighted_accuracy}%" if j else "—"
        with st.expander(f"{name} · {acc}", expanded=False):
            if c.meta and (
                c.meta.ttft_s is not None
                or c.meta.tps is not None
                or c.meta.ram_mb is not None
                or c.meta.gguf_mb is not None
            ):
                _ram = _fmt_ram_mb(c.meta.ram_mb) if c.meta.ram_mb is not None else ""
                _gguf = (
                    _fmt_gguf_mb(c.meta.gguf_mb) if c.meta.gguf_mb is not None else ""
                )
                st.caption(
                    f"TTFT {c.meta.ttft_s}s · TPS {c.meta.tps} · "
                    f"${c.meta.cost_usd or 0:.4f}"
                    + (f" · {_ram}" if _ram else "")
                    + (f" · {_gguf}" if _gguf else "")
                    + (f" · err {c.meta.error}" if c.meta.error else "")
                )
            if c.raw_response:
                st.markdown("**Full answer**")
                _dlg_full_text(c.raw_response)
            elif c.answers:
                for qid, ans in c.answers.items():
                    st.markdown(f"**{qid}**")
                    _dlg_full_text(ans or "")
            if j:
                st.markdown("**Judge**")
                for qs in j.question_scores:
                    st.caption(f"{qs.question_id}: {qs.score}/100 — {qs.rationale}")


@st.dialog(
    "Rebuild mean · offline · $0",
    width="large",
    on_dismiss=_clear_all_kpi_popups,
)
def history_mean_rebuild_dialog():
    """Popup: offline mean KPIs after rescoring saved runs with current formula."""
    from benchmark.schema import MultiRunSummary as _MRS

    payload = st.session_state.get("history_rebuild_result") or {}
    if not payload.get("ok"):
        st.error(payload.get("reason") or "Nothing to show.")
        if st.button("Close", type="primary", use_container_width=True, key="hm_dlg_err"):
            _clear_all_kpi_popups()
            st.rerun()
        return

    raw = payload.get("summary") or {}
    try:
        summary = _MRS.model_validate(raw) if isinstance(raw, dict) else raw
    except Exception as exc:
        st.error(f"Summary invalid: {exc}")
        if st.button("Close", type="primary", use_container_width=True, key="hm_dlg_bad"):
            _clear_all_kpi_popups()
            st.rerun()
        return

    _scope = str(payload.get("scope") or "same_case")
    _n_cases = int(payload.get("n_cases") or 1)
    if _scope == "portfolio":
        st.success(
            t(
                "bench.rebuild_portfolio_success",
                _ui_lang(),
                n=payload.get("n_used") or summary.n,
                cases=_n_cases,
            )
        )
        st.caption(t("bench.rebuild_portfolio_caption", _ui_lang()))
        _mr = payload.get("mean_rank") or {}
        if _mr:
            _mr_bits = ", ".join(
                f"{short_model(k)}≈{v}"
                for k, v in sorted(_mr.items(), key=lambda kv: kv[1])[:6]
            )
            st.caption(
                t("bench.rebuild_portfolio_mean_rank", _ui_lang(), ranks=_mr_bits)
            )
    elif _scope == "balanced_cases":
        st.success(
            t(
                "bench.rebuild_balanced_success",
                _ui_lang(),
                n=payload.get("n_used") or summary.n,
                cases=_n_cases,
            )
        )
        st.caption(t("bench.rebuild_balanced_caption", _ui_lang()))
        _mr = payload.get("mean_rank") or {}
        if _mr:
            _mr_bits = ", ".join(
                f"{short_model(k)}≈{v}"
                for k, v in sorted(_mr.items(), key=lambda kv: kv[1])[:6]
            )
            st.caption(
                t("bench.rebuild_portfolio_mean_rank", _ui_lang(), ranks=_mr_bits)
            )
    else:
        st.success(
            f"**{case_display_name(summary.case_id)}** · per-model successful N shown below · "
            f"same-cohort **reference-relative Clinical Composite Score** · "
            f"**$0 API** (no OpenRouter / DeepSeek calls)"
        )
    _rebuild_clean = bool(payload.get("successful_only", True))
    st.caption(reliability_caption(summary, successful_only=_rebuild_clean))
    _dlg_cohort = str(
        payload.get("cohort_id")
        or getattr(summary, "cohort_id", None)
        or st.session_state.get("_restored_cohort_id")
        or ""
    )
    _dlg_roster_n = int(
        payload.get("roster_n")
        or len(filter_current_roster_rows(summary.ranking_mean or []))
        or DEFAULT_ROSTER_VERSION
    )
    _dlg_pack_rev = str(
        payload.get("pack_revision_label")
        or st.session_state.get("_ui_pack_revision")
        or _pack_rev_meta
        or ""
    )
    st.markdown(
        honesty_block_html(
            lang=_ui_lang(),
            roster_n=_dlg_roster_n,
            scope=_scope,
            cohort_id=_dlg_cohort,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        screenshot_share_checklist_html(lang=_ui_lang()),
        unsafe_allow_html=True,
    )

    # Open at top of the dialog (Streamlit often restores prior scroll).
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;
          const root =
            doc.querySelector('[data-testid="stDialog"]') ||
            doc.querySelector('[role="dialog"]');
          if (!root) return;
          root.scrollTop = 0;
          root.querySelectorAll("*").forEach((n) => {
            try {
              const s = window.parent.getComputedStyle(n);
              if (
                (s.overflowY === "auto" || s.overflowY === "scroll") &&
                n.scrollHeight > n.clientHeight + 8
              ) {
                n.scrollTop = 0;
              }
            } catch (e) {}
          });
        })();
        </script>
        """,
        height=0,
    )

    try:
        _n_cap = int(payload.get("n_per_model_cap") or summary.n or 0)
    except (TypeError, ValueError):
        _n_cap = 0
    _rank_by = "mean"
    if _n_cap >= 30:
        _rank_pick = st.radio(
            "Rank by",
            options=["mean", "median"],
            horizontal=True,
            key="hm_dlg_rank_by",
            format_func=lambda s: (
                "Mean (default)" if s == "mean" else "Median"
            ),
            help=(
                "Available when N≥30. Reorders chart bars and table # together. "
                "Bars still show mean ±1 std; ◆ is median."
            ),
        )
        _rank_by = str(_rank_pick or "mean")

    if _scope == "portfolio":
        _chart_title = (
            f"Mean Clinical Composite · Portfolio · ≤{payload.get('n_per_model_cap') or '?'} successful/model · "
            f"{payload.get('n_used')} run docs · {payload.get('n_cases')} cases"
        )
    elif _scope == "balanced_cases":
        _chart_title = (
            f"Mean Clinical Composite · Balanced Case1→K · ≤{payload.get('n_per_model_cap') or '?'} successful/model · "
            f"{payload.get('n_used')} run docs · {payload.get('n_cases')} cases"
        )
    else:
        _chart_title = (
            f"Mean Clinical Composite Score · {case_display_name(summary.case_id)}"
        )
    if _rank_by == "median":
        _chart_title += " · ranked by median"

    st.markdown("##### Chart (mean %; whiskers = ±1 std)")
    st.plotly_chart(
        fig_judge_mean_accuracy_bars(
            summary.ranking_mean,
            title=_chart_title,
            height=200,
            hide_partial_labels=_rebuild_clean,
            rank_by=_rank_by,
            compact=True,
        ),
        use_container_width=True,
        key="hm_dlg_mean_chart",
    )
    _scan_banner = rebuild_scan_honesty_html(
        list(payload.get("ops_reliability") or []),
        lang=_ui_lang(),
    )
    if _scan_banner:
        st.markdown(_scan_banner, unsafe_allow_html=True)
    st.markdown(
        screenshot_footer_html(
            lang=_ui_lang(),
            scope=_scope,
            roster_n=_dlg_roster_n,
            cohort_id=_dlg_cohort,
            n_label=(
                f"N≤{payload.get('n_per_model_cap') or '?'} successful/model · "
                f"{payload.get('n_used') or summary.n} docs"
            ),
            pack_revision_label=_dlg_pack_rev or None,
            protocol_id=str(SCORING_VERSION),
            extra="mean±std whiskers · scored-only",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("##### Ranking table")
    st.markdown(
        _reliability_table_html(
            summary.ranking_mean,
            successful_only=_rebuild_clean,
            rank_by=_rank_by,
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        screenshot_footer_html(
            lang=_ui_lang(),
            scope=_scope,
            roster_n=_dlg_roster_n,
            cohort_id=_dlg_cohort,
            n_label=(
                f"N≤{payload.get('n_per_model_cap') or '?'} successful/model · "
                f"{payload.get('n_used') or summary.n} docs"
            ),
            pack_revision_label=_dlg_pack_rev or None,
            protocol_id=str(SCORING_VERSION),
            extra=(
                "scored-only · exact Clinical Composite == 0 treated like N/A"
            ),
        ),
        unsafe_allow_html=True,
    )

    _ops_rows = list(payload.get("ops_reliability") or [])
    _ops_n_label = (
        f"N≤{payload.get('n_per_model_cap') or '?'} scored/model scan"
    )
    paint_rebuild_ops_reliability_panels(
        st,
        _ops_rows,
        n_per_model_cap=payload.get("n_per_model_cap"),
        chart_key="hm_dlg_ops_table",
        table_footer_html=screenshot_footer_html(
            lang=_ui_lang(),
            scope=_scope,
            roster_n=_dlg_roster_n,
            cohort_id=_dlg_cohort,
            n_label=_ops_n_label,
            pack_revision_label=_dlg_pack_rev or None,
            protocol_id=str(SCORING_VERSION),
            extra="failures/N/A % · scan window · not clinical mean",
        ),
    )

    st.markdown("##### Paired complete-case sensitivity analysis")
    if summary.paired_ranking:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Rank": row.get("rank"),
                        "Model": short_model(str(row.get("key"))),
                        "Paired mean %": row.get("accuracy_mean"),
                        "Coverage %": row.get("coverage_mean"),
                        "Quality %": row.get("quality_mean"),
                        "Discipline %": row.get("discipline_mean"),
                        "Paired N": summary.paired_n,
                    }
                    for row in summary.paired_ranking
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption(
            f"Paired N={summary.paired_n}; at least 5 complete iterations are required. "
            "This sensitivity analysis never imputes missing scores."
        )

    with st.expander("Per-run Clinical Composite Score", expanded=False):
        pr_rows = []
        for pr in payload.get("per_run") or []:
            when = (pr.get("finished_at") or "")[:19].replace("T", " ")
            row = {"When": when, "run_id": (pr.get("run_id") or "")[:18]}
            for r in filter_current_roster_rows(pr.get("ranking") or []):
                row[short_model(str(r.get("key")))] = r.get("accuracy")
            pr_rows.append(row)
        if pr_rows:
            st.dataframe(pd.DataFrame(pr_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Used {payload.get('n_used')} of {payload.get('available')} saved runs · "
        f"{payload.get('formula')}"
    )
    if st.button("Close", type="primary", use_container_width=True, key="hm_dlg_close"):
        _clear_all_kpi_popups()
        st.rerun()


@st.dialog("How ranking works", width="large")
def scoring_guide_dialog():
    """Wide, shallow popup: formula + parameters side-by-side (not a deep expander)."""
    st.caption(
        "A blind AI judge (DeepSeek R1) scores answers against your locked reference. "
        "Technical failures count as N/A, not clinical zeros. Exact ties stay ties. "
        "You need a confirmed five-part reference first."
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Per-section score")
        st.code(
            "section ≈ 50% coverage + 35% clinical quality + 15% discipline",
            language=None,
        )
        st.markdown(
            """
| Part | Wt | Meaning |
|------|----|---------|
| **coverage** | 50% | How much of your reference checklist the answer covers |
| **quality** | 35% | Coherence, priorities, usefulness, caution |
| **discipline** | 15% | Penalties only for verified unsupported / contradictory / dangerous additions |
"""
        )
        st.markdown(
            "Matches need evidence from the model's answer. "
            "Risky additions the judge cannot verify are dropped (not auto-failed). "
            "Missing sections stay N/A. The judge is not human-calibrated."
        )
    with right:
        st.markdown("##### Final ranking %")
        st.code(
            "Overall score = weighted mix of the five sections\n"
            "exact ties keep the same rank · technical failures are N/A\n"
            "Multi ×N average = mean overall score ± spread",
            language=None,
        )
        st.markdown(
            """
| Piece | Role |
|-------|------|
| **Section weights** | Diagnosis 30% · Safety 25% · Plan 20% · Tests 15% · Urgency 10% |
| **Ties** | Same score → same rank |
| **Multi reliability** | Spread band · N=5 exploratory · ~10 runs better for stability |

**Flow:** same prompt → answers → blind judge → ranking.
"""
        )
        cid = st.session_state.get("case_pick")
        if cid:
            try:
                _c = load_case(cid)
                st.markdown(f"**This case (`{cid}`) — section weights**")
                bits = " · ".join(f"**{q.id}** {q.weight:.0%}" for q in _c.questions)
                st.markdown(bits)
            except Exception:
                pass

    if st.button("Close", type="primary", use_container_width=True, key="scoring_guide_close"):
        st.session_state["show_scoring_guide"] = False
        st.rerun()


# Prefer the setup guide when requested from the sidebar (so online users
# can always re-open the install steps they only see when the sidecar is offline).
_busy_boot = bool(
    st.session_state.get("benchmark_running") or st.session_state.get("confirmed_run")
)
# ONE dialog max per script run. KPI popups require kpi_dialog_armed (set only by
# Run-tab / History View / Rebuild-mean clicks) — N-picker / stem/gold clear the arm.
# NEVER open History/KPI dialogs while a run is in flight (✕ aborts the script).
_armed = st.session_state.get("kpi_dialog_armed")
_pending_spend = bool(
    st.session_state.get("pending_run") and not st.session_state.get("confirmed_run")
)
if _busy_boot:
    # Never open dialogs mid-collect (✕ would abort). Keep inline_run_path cleared too.
    st.session_state.pop("inline_run_path", None)
    st.session_state.pop("inline_run_kind", None)
    for k in (
        "history_popup_path",
        "multi_run_popup_path",
        "show_run_done",
        "show_history_mean_popup",
        "kpi_dialog_armed",
    ):
        st.session_state.pop(k, None)
elif not _pending_spend:
    # History / per-run detail are INLINE panels — never st.dialog here.
    if _armed == "rebuild" and st.session_state.get("show_history_mean_popup"):
        history_mean_rebuild_dialog()
    elif st.session_state.get("show_scoring_guide"):
        scoring_guide_dialog()
    elif st.session_state.get("show_qvac_guide"):
        qvac_setup_guide_dialog()
    elif not st.session_state.get("boot_welcome_done"):
        run_boot_dialogs(
            qvac_online=qvac_up,
            qvac_loaded=qvac_ok,
            pending_spend=_pending_spend,
            running=_busy_boot,
            other_dialog_open=bool(
                st.session_state.get("show_scoring_guide")
                or st.session_state.get("show_qvac_guide")
                or st.session_state.get("show_history_mean_popup")
            ),
            show_account=True,
        )
    else:
        if st.session_state.get("show_history_mean_popup") and not _armed:
            _clear_all_kpi_popups()

if st.session_state.get("or_key_session") and is_usable_openrouter_key(
    st.session_state["or_key_session"]
):
    has_key = True

st.markdown(
    '<p class="demo-hero">Cloud &amp; local medical LLMs · Structured '
    '<span style="font-size:0.65em;opacity:.75">'
    "(legacy / advanced · optional)</span></p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="demo-sub">Local MedPsy on your machine · your own OpenRouter key · '
    "an AI judge scores answers · hobby comparison — not medical advice.</p>",
    unsafe_allow_html=True,
)
st.caption(
    "Optional advanced track · KPIs here must not be pooled with Comprehension home."
)
st.markdown(
    """
<div class="steps-bar">
  <div class="step-pill"><b>Step 1</b> Paste anonymized case</div>
  <div class="step-pill"><b>Step 2</b> Prepare + confirm your reference</div>
  <div class="step-pill"><b>Step 3</b> Run the models</div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(
    honesty_block_html(
        lang=_ui_lang(),
        roster_n=DEFAULT_ROSTER_VERSION,
        scope="structured",
        cohort_id=st.session_state.get("_restored_cohort_id")
        or st.session_state.get("_active_cohort_id"),
    ),
    unsafe_allow_html=True,
)
st.caption(t("struct.track_caption", _ui_lang()))
st.caption(t("struct.judge_caption", _ui_lang()))
if is_streamlit_cloud() and not qvac_ok:
    st.caption(
        "Hosted demo · on-device QVAC/MedPsy needs a local sidecar — cloud roster only here."
    )

# Client-side guide overlays in main DOM (sidebar labels toggle via for=… — no run interrupt)
# Use st.html (not markdown) so hidden overlays are not sanitized into visible page text.
_qvac_guide_status = (
    "ready — MedPsy will be included"
    if qvac_ok
    else (
        "sidecar online · MedPsy not loaded"
        if qvac_up
        else "sidecar offline — start it to include on-device"
    )
)
st.html(
    guides_always_available_html(
        qvac_status_line=_qvac_guide_status,
        lang=_ui_lang(),
    )
)

# --- Sidebar: Tracks → key → QVAC → guides → (History later) → clock ---
with st.sidebar:
    render_tracks_block(active="structured")
    _run_busy = bool(
        st.session_state.get("benchmark_running")
        or st.session_state.get("confirmed_run")
        or st.session_state.get("pending_run")
    )
    if st.button(
        "STOP · abort run",
        key="hard_stop_btn",
        use_container_width=True,
        help="Abort pending/active run, clear live panels, reset KPI state. History stays.",
    ):
        _hard_abort_run(flash=True)
        st.rerun()
    if st.session_state.pop("run_aborted_flash", None):
        st.warning("Run aborted · panels cleared.")
    elif _run_busy:
        st.caption(
            "Run in progress / waiting confirm — STOP clears UI and cancels queued judges; "
            "an HTTP call already in flight may still finish once."
        )

    st.markdown("**OpenRouter**")
    if has_key:
        st.success("Key OK · cloud + R1")
    else:
        st.warning("No full key · Single/Multi off")
    st.caption("Session/account isolated · never shared through process environment")
    st.caption(t("comp.same_key_warning", _ui_lang()))
    if is_streamlit_cloud() and not account_store_configured():
        st.caption("Hosted without Supabase · session-only key/history (not durable).")
    key_in = st.text_input(
        "OPENROUTER_API_KEY",
        value="",
        type="password",
        help="Full sk-or-v1-… from openrouter.ai/keys — scoped to this session/account",
        placeholder="sk-or-v1-…",
        label_visibility="collapsed",
    )
    if key_in:
        if is_usable_openrouter_key(key_in):
            _remember_openrouter_key(key_in.strip())
            has_key = True
        else:
            st.error("Key truncated / too short")
    _signed_account = st.session_state.get("account_session")
    if isinstance(_signed_account, AccountSession):
        st.caption(f"Account · {_signed_account.email}")
        if st.button("Sign out account", use_container_width=True):
            for _key in (
                "account_session",
                "or_key_session",
                "_account_key_loaded",
                "_account_artifacts_synced",
            ):
                st.session_state.pop(_key, None)
            st.rerun()

    st.markdown("**QVAC · MedPsy**")
    if qvac_up:
        h = qvac_health()
        device = h.get("device")
        layers = h.get("gpu_layers")
        model_loaded = h.get("modelLoaded")
        last_err = (h.get("lastError") or "").strip()
        if h.get("error") and not model_loaded:
            st.warning(f"Health · {h.get('error')}")
        elif model_loaded is False:
            if last_err and ("libssl" in last_err.lower() or "openssl" in last_err.lower()):
                st.warning("Needs OpenSSL 3 · setup script")
            elif last_err:
                st.warning(f"Not loaded · {last_err[:80]}")
            else:
                st.warning("Loading MedPsy…")
        elif device is not None and layers is not None:
            st.success(f"Ready · {device} · L{layers}")
        elif device is not None:
            st.success(f"Ready · {device}")
        elif h.get("stream") is not True:
            st.warning("Old sidecar · restart")
        else:
            st.success("Ready")
    else:
        ensure_err = (st.session_state.get("qvac_ensure_error") or "").strip()
        st.error("Offline")
        if ensure_err:
            st.caption(ensure_err[:120])
        if st.button("Start MedPsy", type="primary", use_container_width=True):
            st.session_state.qvac_ensure_tried = False
            with st.spinner("Starting…"):
                ensured = qvac_ensure_sidecar(wait_s=90.0, start_if_down=True)
            st.session_state.qvac_ensure_tried = True
            st.session_state["qvac_ensure_error"] = ensured.get("ensure_error") or ensured.get(
                "lastError"
            )
            st.rerun()
    st.caption(f"Judge · {(judge_cfg.get('display_label') or judge_cfg.get('model') or 'R1')[:42]}")
    render_guides_and_protocol(
        protocol_id=str(SCORING_VERSION),
        lang=_ui_lang(),
        active_track="structured",
        extra_caption="History picker + Run clock appear lower in this column.",
    )

# --- Gold-only real-case workflow ---
case_id = "caseC"
preset = load_case(case_id)
is_custom_real = True

# Case slots (sticky stem_key bindings per API-key owner workspace).
# Base Case 1–5 always shown; New case opens next empty or grows to Case 6+.
_slot_arts = _preloaded_artifacts()
_case_slots, _case_bindings, _case_slot_count, _case_drafts = ensure_owner_slots(
    WORKSPACE_DIR,
    _slot_arts,
    session_bindings=st.session_state.get("_case_slot_bindings") or {},
    session_slot_count=st.session_state.get("_case_slot_count"),
    session_drafts=st.session_state.get("_case_slot_drafts") or {},
    persist=bool(getattr(RUN_STORE, "writes_plaintext", True)),
)
st.session_state["_case_slot_bindings"] = dict(_case_bindings)
st.session_state["_case_slot_count"] = int(_case_slot_count)
st.session_state["_case_slot_drafts"] = {
    int(i): dict(v) for i, v in (_case_drafts or {}).items()
}
if "active_case_slot" not in st.session_state:
    # Prefer first filled slot (migrated Case 1), else Case 1 empty.
    _first_filled = next((s.index for s in _case_slots if s.filled), 1)
    st.session_state["active_case_slot"] = int(_first_filled)
    _boot_slot = next(
        (s for s in _case_slots if s.index == st.session_state["active_case_slot"]),
        _case_slots[0],
    )
    if _boot_slot.filled and not (
        st.session_state.get("demo_case_stem") or st.session_state.get("_persist_case_stem")
    ):
        _select_case_slot(_boot_slot)

_active_slot_idx = int(st.session_state.get("active_case_slot") or 1)
_active_slot = next(
    (s for s in _case_slots if s.index == _active_slot_idx),
    _case_slots[0] if _case_slots else empty_slot(1),
)
_slots_locked = bool(
    st.session_state.get("benchmark_running") or st.session_state.get("confirmed_run")
)

# After a default-pack revision force-seed (e.g. Case 6–7 emergency-breadth seeds), reload the
# active editor when that slot was remapped — without interrupting a live run.
_pack_rev_now = load_pack_revision(WORKSPACE_DIR)
_pack_rev_meta, _force_seed_slots = load_default_pack_meta()
_pack_rev_seen = st.session_state.get("_ui_pack_revision")
if (
    not _slots_locked
    and _force_seed_slots
    and _pack_rev_now >= _pack_rev_meta
    and _pack_rev_seen != _pack_rev_now
    and int(_active_slot_idx) in _force_seed_slots
    and _active_slot.filled
):
    st.session_state["_ui_pack_revision"] = _pack_rev_now
    _select_case_slot(_active_slot)
    st.rerun()
st.session_state.setdefault("_ui_pack_revision", _pack_rev_now)

st.markdown(
    f'<div class="sec-label">{t("bench.case_slots_label", _ui_lang())}</div>',
    unsafe_allow_html=True,
)


def _render_case_slot_button(slot, *, active_idx: int, locked: bool) -> None:
    is_active = slot.index == active_idx
    btn_label = t("bench.case_slot_btn", _ui_lang(), n=slot.index)
    if slot.filled:
        btn_label += f" · {slot.run_count}"
    else:
        btn_label += f" · {t('bench.case_slot_empty', _ui_lang())}"
    if st.button(
        btn_label,
        key=f"case_slot_btn_{slot.index}",
        use_container_width=True,
        type="primary" if is_active else "secondary",
        disabled=locked,
        help=(
            f"{slot.stem[:120]}…"
            if slot.stem and len(slot.stem) > 120
            else (
                slot.stem
                or t("bench.case_slot_ready_empty", _ui_lang(), n=slot.index)
            )
        ),
    ):
        if not locked:
            _select_case_slot(slot)
            st.rerun()


# New case (green) first, then Case 1…N. Wrap after base row for readability.
_base_slots = [s for s in _case_slots if s.index <= BASE_CASE_SLOTS]
_extra_slots = [s for s in _case_slots if s.index > BASE_CASE_SLOTS]
_row1_cols = st.columns([1.25] + [1] * len(_base_slots), gap="small")
with _row1_cols[0]:
    if st.button(
        t("bench.new_case_btn", _ui_lang()),
        key="case_slot_new_btn",
        use_container_width=True,
        disabled=_slots_locked,
        help=t("bench.new_case_help", _ui_lang()),
    ):
        if not _slots_locked:
            try:
                _new_idx, _new_count = open_new_case_slot(
                    _case_slots, slot_count=int(_case_slot_count)
                )
            except ValueError:
                st.session_state["_case_slot_flash"] = "full"
            else:
                st.session_state["_case_slot_count"] = int(_new_count)
                if getattr(RUN_STORE, "writes_plaintext", True):
                    try:
                        save_bindings(
                            WORKSPACE_DIR,
                            st.session_state.get("_case_slot_bindings") or {},
                            slot_count=int(_new_count),
                        )
                    except OSError:
                        pass
                _target = next(
                    (s for s in _case_slots if s.index == _new_idx),
                    empty_slot(_new_idx),
                )
                _select_case_slot(_target, as_new=True)
                st.session_state["_case_slot_flash"] = "empty"
            st.rerun()
for _si, _slot in enumerate(_base_slots):
    with _row1_cols[_si + 1]:
        _render_case_slot_button(
            _slot, active_idx=_active_slot_idx, locked=_slots_locked
        )
if _extra_slots:
    # Case 6+ on following row(s), chunks of 6.
    for _off in range(0, len(_extra_slots), 6):
        _chunk = _extra_slots[_off : _off + 6]
        _extra_cols = st.columns([1] * len(_chunk), gap="small")
        for _ci, _slot in enumerate(_chunk):
            with _extra_cols[_ci]:
                _render_case_slot_button(
                    _slot, active_idx=_active_slot_idx, locked=_slots_locked
                )

_flash = st.session_state.pop("_case_slot_flash", None)
if _flash == "full":
    st.warning(t("bench.new_case_full", _ui_lang(), n=SOFT_MAX_CASE_SLOTS))
elif _flash == "empty":
    st.info(t("bench.case_slot_ready_empty", _ui_lang(), n=_active_slot_idx))
elif _active_slot.filled and st.session_state.get("_confirmed_gold_json"):
    st.caption(
        t("bench.case_slot_loaded", _ui_lang(), n=_active_slot.index)
        + (
            f" · {t('bench.case_slot_runs', _ui_lang(), n=_active_slot.run_count)}"
            if _active_slot.run_count
            else ""
        )
    )
else:
    st.caption(
        f"**Case {_active_slot_idx}** selected · History stays private to this "
        f"API key ({short_owner_label()}). Base Case 1–{BASE_CASE_SLOTS}; "
        f"New case adds Case {BASE_CASE_SLOTS + 1}+ when the base row is full."
    )

st.markdown('<div class="sec-label">Case + your reference answer</div>', unsafe_allow_html=True)
st.info(
    "Paste an anonymized case and your reference answer. The benchmark measures "
    "agreement with your reference; a wrong or incomplete reference makes the "
    "scores misleading."
)

if "demo_case_stem" not in st.session_state:
    st.session_state["demo_case_stem"] = st.session_state.get("_persist_case_stem", "")
if "demo_gold_ref" not in st.session_state:
    st.session_state["demo_gold_ref"] = st.session_state.get("_persist_gold_ref", "")

col_case, col_gold = st.columns([1, 1], gap="large")
with col_case:
    st.markdown('<div class="sec-label">1 · Clinical case</div>', unsafe_allow_html=True)
    case_stem = st.text_area(
        "case",
        height=180,
        key="demo_case_stem",
        label_visibility="collapsed",
        placeholder="Paste the anonymized symptoms, history, findings and context…",
        on_change=_on_case_fields_edit,
    )
    st.caption("No names, dates of birth, addresses, IDs or other identifying data.")
with col_gold:
    st.markdown(
        '<div class="sec-label">2 · Your reference answer (required)</div>',
        unsafe_allow_html=True,
    )
    gold_reference = st.text_area(
        "gold",
        height=180,
        placeholder=(
            "Write your reference diagnosis, tests, urgency, safety traps "
            "and initial plan in any order. Prepare turns this into checklist "
            "points; Confirm locks them before any model runs."
        ),
        key="demo_gold_ref",
        label_visibility="collapsed",
        on_change=_on_case_fields_edit,
    )
    st.caption(
        "Your locked reference is what models are scored against — "
        "not certified medical ground truth."
    )

st.session_state["_persist_case_stem"] = case_stem or ""
st.session_state["_persist_gold_ref"] = gold_reference or ""
# Fingerprint case + reference so editing either invalidates prepared/confirmed (H3).
raw_gold_fingerprint = _source_fingerprint(case_stem or "", gold_reference or "")
if st.session_state.get("_gold_source_fingerprint") != raw_gold_fingerprint:
    # Invalidating raw case/reference clears prepared + confirmed + edit widgets.
    for _gk in (
        "_confirmed_gold_json",
        "_gold_sections",
        "_prepared_gold_sections",
        "_prepared_extraction_meta",
        "_prepared_extraction_cost",
        "_gold_confirmed_at",
        "_restored_cohort_id",
        "_prepare_error",
    ):
        st.session_state.pop(_gk, None)
    for _wk in list(st.session_state.keys()):
        if str(_wk).startswith("prep_sum_") or str(_wk).startswith("prep_q_"):
            st.session_state.pop(_wk, None)
    st.session_state["_gold_source_fingerprint"] = raw_gold_fingerprint

gold_reference = (gold_reference or "").strip()

# Quiet resume: same case+raw reference family → restore exact prior confirmed gold
# (never auto-merges different claim-split cohorts).
_family_cohorts: list = []
if (case_stem or "").strip() and len(gold_reference) >= 40:
    try:
        _family_cohorts = find_case_family_cohorts(
            WORKSPACE_DIR,
            case_stem=case_stem or "",
            reference_raw=gold_reference,
            case_id=case_id,
        )
    except Exception:
        _family_cohorts = []
_family_run_total = sum(int(c.get("run_count") or 0) for c in _family_cohorts)
_current_confirmed = st.session_state.get("_confirmed_gold_json") or ""
_show_family_restore = bool(_family_cohorts) and not _current_confirmed
if _show_family_restore:
    _lang = _ui_lang()
    st.caption(
        t(
            "bench.family_found",
            _lang,
            n=_family_run_total,
            cohorts=len(_family_cohorts),
        )
    )
    _fam_labels = []
    _fam_by_label = {}
    for _fc in _family_cohorts:
        _when = str(_fc.get("latest_finished_at") or "")[:19].replace("T", " ")
        _lab = (
            f"{int(_fc.get('run_count') or 0)} runs · {_when} · "
            f"cohort {_fc.get('cohort_short') or ''}"
        )
        _fam_labels.append(_lab)
        _fam_by_label[_lab] = _fc
    _fam_pick = st.selectbox(
        t("bench.family_select", _lang),
        options=_fam_labels,
        key="family_restore_pick",
        label_visibility="collapsed",
    )
    if st.button(
        t("bench.family_restore_btn", _lang),
        use_container_width=True,
        key="family_restore_btn",
        help=t("bench.family_restore_help", _lang),
    ):
        _chosen = _fam_by_label.get(_fam_pick) or _family_cohorts[0]
        try:
            _restore_confirmed_gold_contract(
                gold_reference_json=str(_chosen.get("gold_reference") or ""),
                case_stem_saved=str(_chosen.get("case_stem") or case_stem or ""),
                cohort_id=str(_chosen.get("cohort_id") or ""),
            )
            st.success(t("bench.family_restore_ok", _lang))
            st.rerun()
        except Exception as _rex:
            st.error(t("bench.family_restore_fail", _lang, err=str(_rex)))

_prep_col, _conf_col = st.columns(2)
_qna_local_ok = looks_like_qna_reference(gold_reference)
with _prep_col:
    prepare_clicked = st.button(
        "Prepare reference",
        use_container_width=True,
        disabled=len(gold_reference) < 40
        or bool(st.session_state.get("benchmark_running")),
        help=(
            "Turn your reference text into checklist points you can review. "
            "Pre-formatted five-part answers extract locally (no API). "
            "Free-form text uses OpenRouter once."
        ),
    )
with _conf_col:
    confirm_clicked = st.button(
        "Confirm reference",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.get("_prepared_gold_sections")
        or bool(st.session_state.get("benchmark_running")),
        help=(
            "Locks the case text and reference checklist for scoring. "
            "Models unlock after this. Scores compare to this reference — "
            "not external medical truth."
        ),
    )

if prepare_clicked:
    st.session_state.pop("_prepare_error", None)
    extractor_model = os.environ.get(
        "BENCHMARK_GOLD_EXTRACTOR_MODEL", "openai/gpt-4o-mini"
    )
    has_or_key = is_usable_openrouter_key(st.session_state.get("or_key_session"))
    if not _qna_local_ok and not has_or_key:
        st.session_state["_prepare_error"] = format_prepare_error(
            ValueError(
                "An OpenRouter key is required to prepare free-form references "
                "(pre-formatted five-part answers can Prepare without a key)"
            )
        )
    else:
        try:
            _spinner = (
                "Extracting Q1–A5 claims locally…"
                if _qna_local_ok
                else "Extracting source-linked claims (once)…"
            )
            with st.spinner(_spinner):
                sections, extract_meta = extract_with_chat(
                    gold_reference,
                    model=extractor_model,
                    chat=openrouter.chat,
                    api_key=st.session_state.get("or_key_session"),
                )
            used_model = str(
                getattr(extract_meta, "model", None)
                or getattr(extract_meta, "requested_model", None)
                or extractor_model
            )
            st.session_state["_prepared_gold_sections"] = {
                sid: sections[sid].model_dump() for sid in SECTION_IDS
            }
            st.session_state["_prepared_extraction_meta"] = {
                "model": used_model,
                "cost_usd": float(getattr(extract_meta, "cost_usd", 0.0) or 0.0),
                "prompt_tokens": int(getattr(extract_meta, "prompt_tokens", 0) or 0),
                "completion_tokens": int(
                    getattr(extract_meta, "completion_tokens", 0) or 0
                ),
            }
            st.session_state["_prepared_extraction_cost"] = float(
                getattr(extract_meta, "cost_usd", 0.0) or 0.0
            )
            st.session_state.pop("_confirmed_gold_json", None)
            st.session_state.pop("_gold_confirmed_at", None)
            st.session_state.pop("_restored_cohort_id", None)
            st.session_state.pop("_prepare_error", None)
            if used_model == LOCAL_QNA_EXTRACTOR_MODEL:
                st.success(
                    "Reference prepared locally — review/edit the checklist points, "
                    "then Confirm."
                )
            else:
                st.success(
                    "Reference prepared — review/edit the checklist points, then Confirm."
                )
            st.rerun()
        except Exception as exc:
            st.session_state["_prepare_error"] = format_prepare_error(exc)

_prepare_err = st.session_state.get("_prepare_error")
if _prepare_err:
    st.error(f"Prepare failed: {_prepare_err}")
    _err_clear, _err_hint = st.columns([1, 3])
    with _err_clear:
        if st.button("Clear prepare error", use_container_width=True):
            st.session_state.pop("_prepare_error", None)
            st.rerun()
    with _err_hint:
        st.caption(
            "Retry Prepare after fixing the issue. "
            "Seeded Case 6/7 and pre-formatted New cases extract locally (no API). "
            "Free-form text needs a full OpenRouter key in the sidebar."
        )

# Editable prepared sections
_prepared = st.session_state.get("_prepared_gold_sections")
if isinstance(_prepared, dict) and _prepared:
    st.markdown(
        '<div class="sec-label">Prepared checklist (edit · each quote must stay '
        "a substring of your reference text)</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Add / split / delete / move points between sections — Confirm locks them "
        "for scoring. Section summaries are display-only (not scored). "
        "Each source quote must be copied from your reference text."
    )
    raw_norm_check = gold_reference or ""

    # Handle deferred claim mutations before widgets bind keys.
    _claim_action = st.session_state.pop("_gold_claim_action", None)
    if isinstance(_claim_action, dict) and _claim_action.get("op"):
        _op = str(_claim_action.get("op") or "")
        _sid = str(_claim_action.get("section") or "")
        _idx = int(_claim_action.get("index") or 0)
        _mut = {
            sid: {
                "summary": str((_prepared.get(sid) or {}).get("summary") or ""),
                "claims": [
                    dict(c) for c in list((_prepared.get(sid) or {}).get("claims") or [])
                ],
            }
            for sid in SECTION_IDS
        }
        if _op == "add" and _sid in _mut:
            _mut[_sid]["claims"].append(
                {
                    "id": f"{_sid}-new",
                    "text": "",
                    "source_quote": "",
                    "critical": False,
                }
            )
        elif _op == "delete" and _sid in _mut:
            claims = _mut[_sid]["claims"]
            if 0 <= _idx < len(claims):
                claims.pop(_idx)
        elif _op == "split" and _sid in _mut:
            claims = _mut[_sid]["claims"]
            if 0 <= _idx < len(claims):
                src = dict(claims[_idx])
                claims[_idx] = {
                    **src,
                    "source_quote": "",
                    "text": "",
                }
                claims.insert(
                    _idx + 1,
                    {
                        "id": f"{_sid}-split",
                        "text": "",
                        "source_quote": "",
                        "critical": False,
                    },
                )
        elif _op == "move" and _sid in _mut:
            _dest = str(_claim_action.get("dest") or "")
            claims = _mut[_sid]["claims"]
            if _dest in _mut and 0 <= _idx < len(claims) and _dest != _sid:
                moved = claims.pop(_idx)
                _mut[_dest]["claims"].append(moved)
        st.session_state["_prepared_gold_sections"] = _mut
        _prepared = _mut

    edited_sections: dict = {}
    quote_ok = True
    for section_id in SECTION_IDS:
        sec = _prepared.get(section_id) or {}
        with st.expander(f"{section_id}", expanded=section_id == "diagnosis"):
            summary = st.text_input(
                f"{section_id} summary (display-only)",
                value=str(sec.get("summary") or ""),
                key=f"prep_sum_{section_id}",
                help="Not scored and not part of cohort identity.",
            )
            claims_in = list(sec.get("claims") or [])
            claims_out = []
            for ci, claim in enumerate(claims_in):
                c1, c2, c3, c4 = st.columns([6, 1, 1, 2])
                with c1:
                    quote = st.text_area(
                        f"{section_id} claim {ci + 1} source_quote",
                        value=str(claim.get("source_quote") or ""),
                        key=f"prep_q_{section_id}_{ci}",
                        height=68,
                    )
                with c2:
                    if st.button(
                        "Del",
                        key=f"prep_del_{section_id}_{ci}",
                        help="Delete this claim",
                    ):
                        st.session_state["_gold_claim_action"] = {
                            "op": "delete",
                            "section": section_id,
                            "index": ci,
                        }
                        st.rerun()
                with c3:
                    if st.button(
                        "Split",
                        key=f"prep_split_{section_id}_{ci}",
                        help="Split into two claims (paste two verbatim quotes)",
                    ):
                        st.session_state["_gold_claim_action"] = {
                            "op": "split",
                            "section": section_id,
                            "index": ci,
                        }
                        st.rerun()
                with c4:
                    _move_opts = [s for s in SECTION_IDS if s != section_id]
                    _dest = st.selectbox(
                        "Move",
                        options=["—"] + _move_opts,
                        key=f"prep_mv_{section_id}_{ci}",
                        label_visibility="collapsed",
                    )
                    if _dest != "—":
                        if st.button(
                            f"→ {_dest[:4]}",
                            key=f"prep_mvgo_{section_id}_{ci}",
                        ):
                            st.session_state["_gold_claim_action"] = {
                                "op": "move",
                                "section": section_id,
                                "index": ci,
                                "dest": _dest,
                            }
                            st.rerun()
                if quote.strip() and not source_quote_is_verbatim(
                    raw_norm_check, quote
                ):
                    preview = quote.strip().replace("\n", " ")
                    if len(preview) > 100:
                        preview = preview[:97] + "..."
                    st.error(
                        f"{section_id}-{ci + 1}: source_quote must be a contiguous "
                        f"substring of the raw reference (paraphrase not allowed). "
                        f"Rejected: {preview!r}"
                    )
                    quote_ok = False
                claims_out.append(
                    {
                        "id": str(claim.get("id") or f"{section_id}-{ci + 1}"),
                        "text": quote.strip(),
                        "source_quote": quote.strip(),
                        "critical": False,
                    }
                )
            if st.button(
                f"+ Add claim to {section_id}",
                key=f"prep_add_{section_id}",
            ):
                st.session_state["_gold_claim_action"] = {
                    "op": "add",
                    "section": section_id,
                    "index": 0,
                }
                st.rerun()
            edited_sections[section_id] = {
                "summary": summary.strip(),
                "claims": [c for c in claims_out if c["source_quote"]],
            }
    st.session_state["_prepared_gold_sections"] = edited_sections

    if confirm_clicked:
        if not quote_ok:
            st.error("Fix source_quote errors before confirming.")
        else:
            try:
                extractor_model = (
                    (st.session_state.get("_prepared_extraction_meta") or {}).get(
                        "model"
                    )
                    or os.environ.get(
                        "BENCHMARK_GOLD_EXTRACTOR_MODEL", "openai/gpt-4o-mini"
                    )
                )
                extract_cost = float(
                    st.session_state.get("_prepared_extraction_cost") or 0.0
                )
                parsed = {
                    sid: GoldSection.model_validate(edited_sections[sid])
                    for sid in SECTION_IDS
                }
                contract = build_confirmed_gold(
                    raw_text=gold_reference,
                    sections=parsed,
                    extraction_model=str(extractor_model),
                    extraction_cost_usd=extract_cost,
                )
                effective = gold_json(contract)
                st.session_state["_confirmed_gold_json"] = effective
                st.session_state["_gold_confirmed_at"] = contract.confirmed_at
                st.session_state["_gold_sections"] = {
                    sid: contract.sections[sid].model_dump() for sid in SECTION_IDS
                }
                st.session_state.pop("_restored_cohort_id", None)
                # Bind this stem to the active Case slot (sticky per owner).
                try:
                    _bind_idx = int(st.session_state.get("active_case_slot") or 1)
                    _bound = bind_stem_to_slot(
                        st.session_state.get("_case_slot_bindings") or {},
                        slot_index=_bind_idx,
                        case_stem=str(
                            st.session_state.get("demo_case_stem") or case_stem or ""
                        ),
                    )
                    st.session_state["_case_slot_bindings"] = _bound
                    _bind_count = max(
                        int(st.session_state.get("_case_slot_count") or BASE_CASE_SLOTS),
                        _bind_idx,
                        BASE_CASE_SLOTS,
                    )
                    st.session_state["_case_slot_count"] = _bind_count
                    if getattr(RUN_STORE, "writes_plaintext", True):
                        save_bindings(
                            WORKSPACE_DIR, _bound, slot_count=_bind_count
                        )
                except Exception:
                    pass
                st.success(
                    t(
                        "struct.confirm_success",
                        _ui_lang(),
                        at=contract.confirmed_at,
                        cost=f"{extract_cost:.4f}",
                    )
                )
                st.warning(
                    t(
                        "disclosure.confirm_new_cohort",
                        _ui_lang(),
                        hash="",
                    )
                )
                st.rerun()
            except Exception as exc:
                st.error(t("struct.confirm_fail", _ui_lang(), err=str(exc)))

if st.session_state.get("_confirmed_gold_json"):
    st.caption(
        t(
            "struct.confirm_locked",
            _ui_lang(),
            at=(
                f" · {st.session_state.get('_gold_confirmed_at')}"
                if st.session_state.get("_gold_confirmed_at")
                else ""
            ),
        )
    )
    st.caption(
        t(
            "disclosure.confirm_new_cohort",
            _ui_lang(),
            hash=(
                f" · `{short_cohort(st.session_state.get('_restored_cohort_id'))}`"
                if st.session_state.get("_restored_cohort_id")
                else ""
            ),
        )
    )
else:
    st.caption(t("struct.need_confirm", _ui_lang()))

live_case = preset.model_copy(update={"stem": (case_stem or "").strip()})
effective_gold = st.session_state.get("_confirmed_gold_json", "")

# --- Models roster: default ≤9; optional legacy slots can grow to ≤12 ---
# Preset shortcuts apply BEFORE toggle widgets (click → next rerun).
def _clear_optional_legacy_slots() -> None:
    st.session_state.opt_legacy_local_gemma = False
    st.session_state.opt_legacy_local_llama = False
    st.session_state.opt_legacy_qvac_4b_q8 = False


if st.session_state.pop("_force_medical_on_device_only", False):
    st.session_state.include_cloud_models = False
    st.session_state.include_generic_peers = False
    st.session_state.include_medical_peers = True
    st.session_state.include_medpsy_models = True
    _clear_optional_legacy_slots()
if st.session_state.pop("_force_all_on_device", False):
    st.session_state.include_cloud_models = False
    st.session_state.include_generic_peers = True
    st.session_state.include_medical_peers = True
    st.session_state.include_medpsy_models = True
    _clear_optional_legacy_slots()
if st.session_state.pop("_force_full_roster", False):
    st.session_state.include_cloud_models = True
    st.session_state.include_generic_peers = True
    st.session_state.include_medical_peers = True
    st.session_state.include_medpsy_models = True
    _clear_optional_legacy_slots()
if st.session_state.pop("_force_cloud_only", False):
    st.session_state.include_cloud_models = True
    st.session_state.include_generic_peers = False
    st.session_state.include_medical_peers = False
    st.session_state.include_medpsy_models = False
    _clear_optional_legacy_slots()
# Migrate legacy triple_qvac_toggle → include_medpsy_models once.
if "include_medpsy_models" not in st.session_state:
    st.session_state.include_medpsy_models = bool(
        st.session_state.get("triple_qvac_toggle", True)
    )

sidecar_up = bool(qvac_ok or qvac_up)
include_qvac = sidecar_up  # legacy name: sidecar available for on-device loads
skip_qvac = False
_med_ready = medical_peers_ready()
if "include_medical_peers" not in st.session_state:
    st.session_state.include_medical_peers = bool(_med_ready and sidecar_up)
if "include_generic_peers" not in st.session_state:
    st.session_state.include_generic_peers = bool(sidecar_up)
for _opt_key in (
    "opt_legacy_local_gemma",
    "opt_legacy_local_llama",
    "opt_legacy_qvac_4b_q8",
):
    if _opt_key not in st.session_state:
        st.session_state[_opt_key] = False

st.markdown(
    '<div class="sec-label">Roster bands (default ≤9 · optional ≤12)</div>',
    unsafe_allow_html=True,
)
include_cloud = st.toggle(
    "Include 3 cloud API models · ChatGPT / Claude / Gemini (OpenRouter)",
    value=True,
    key="include_cloud_models",
    help="OpenRouter API routes — not ChatGPT/Claude/Gemini consumer web apps.",
)
include_medpsy = st.toggle(
    "Include QVAC MedPsy · 1.7B / 4B Q4",
    key="include_medpsy_models",
    disabled=not sidecar_up,
    help="Default MedPsy pair on the QVAC sidecar (serial load). "
    "4B Q8 is optional under Optional / legacy slots.",
)
include_generic = st.toggle(
    "Include generic local · Phi-3.5 mini",
    key="include_generic_peers",
    disabled=not sidecar_up,
    help="Default Band B slot (Phi). Gemma / Llama are optional legacy slots.",
)
include_medical = st.toggle(
    "Include 3 medical local LLMs · MedGemma / Med42 / UltraMedical",
    key="include_medical_peers",
    disabled=not sidecar_up,
    help="Medical-specialized open GGUFs (not MedPsy). "
    "./scripts/download_medical_peers.sh when missing.",
)
if sidecar_up and not _med_ready and include_medical:
    st.caption(
        "Medical GGUFs missing under `models/` — slots drop until you run "
        "`./scripts/download_medical_peers.sh` (or `./scripts/download_all_ggufs.sh`)."
    )
elif sidecar_up and not _med_ready:
    st.caption(
        "Medical local LLMs OFF (GGUFs not ready). "
        "`./scripts/download_medical_peers.sh` → then enable the toggle."
    )

with st.expander("Optional / legacy slots (off by default)", expanded=False):
    st.caption(
        "History and artifacts still resolve these labels. Re-enable to grow the "
        "live roster up to 12. GGUFs are not deleted."
    )
    st.toggle(
        "Gemma 2 2B · Band B legacy",
        key="opt_legacy_local_gemma",
        disabled=not sidecar_up,
        help="Requires generic local band ON.",
    )
    st.toggle(
        "Llama 3.2 3B · Band B legacy",
        key="opt_legacy_local_llama",
        disabled=not sidecar_up,
        help="Requires generic local band ON.",
    )
    st.toggle(
        "MedPsy 4B Q8 · MedPsy legacy",
        key="opt_legacy_qvac_4b_q8",
        disabled=not sidecar_up,
        help="Requires MedPsy band ON.",
    )

_preset_cols = st.columns(4, gap="small")
with _preset_cols[0]:
    if st.button(
        "Medical on-device only",
        key="preset_medical_on_device_btn",
        use_container_width=True,
        disabled=not sidecar_up,
        help="Cloud OFF · generic OFF · medical ON · MedPsy ON → 5 models "
        "(+ optional Q8 → 6).",
    ):
        st.session_state["_force_medical_on_device_only"] = True
        st.rerun()
with _preset_cols[1]:
    if st.button(
        "All on-device",
        key="preset_all_on_device_btn",
        use_container_width=True,
        disabled=not sidecar_up,
        help="Cloud OFF · Phi + medical + dual MedPsy → up to 6 "
        "(+ optional legacy → 9).",
    ):
        st.session_state["_force_all_on_device"] = True
        st.rerun()
with _preset_cols[2]:
    if st.button(
        "Full roster",
        key="preset_full_roster_btn",
        use_container_width=True,
        help="All four default bands ON → up to 9 models "
        "(+ Optional / legacy → ≤12).",
    ):
        st.session_state["_force_full_roster"] = True
        st.rerun()
with _preset_cols[3]:
    if st.button(
        "Cloud only",
        key="preset_cloud_only_btn",
        use_container_width=True,
        help="Cloud ON · all on-device bands OFF → 3 models.",
    ):
        st.session_state["_force_cloud_only"] = True
        st.rerun()

st.caption(
    "Presets set the four band toggles and clear optional/legacy slots. "
    "**Medical on-device only** = dual MedPsy + 3 medical (5). "
    "**All on-device** ≤6. **Full roster** ≤9 (+ optional ≤12). "
    "Generics ≠ medical band."
)
with st.expander("Advanced · generation settings", expanded=False):
    st.caption(
        "Best-effort controlled (temp 0.2 + preferred provider, fallbacks on) is the default. "
        "Strict controlled is opt-in (no fallback; route miss → N/A). "
        "Provider defaults is a separate cohort — never pooled with controlled. "
        "Declare the track in any public screenshot."
    )
    benchmark_track = st.radio(
        "Generation settings",
        options=["controlled", "strict_controlled", "native_defaults"],
        format_func=lambda value: (
            "Best-effort · temp 0.2 · preferred provider"
            if value == "controlled"
            else (
                "Strict controlled · no fallback (opt-in)"
                if value == "strict_controlled"
                else "Provider defaults · ecological"
            )
        ),
        horizontal=True,
        key="benchmark_track",
        label_visibility="collapsed",
    )
n_multi = st.number_input(
    "Multi N",
    min_value=1,
    max_value=30,
    value=5,
    help="Repeats for Multi run and Only local. "
    "N=5 exploratory (CV noisy) · ~10 pragmatic floor for eyeballing CV / rank stability · "
    "20–30 nicer means but diminishing returns for this anecdotal protocol · "
    "1 = single pass. Not clinical validation.",
)

# MedPsy band ON ⇒ default dual (1.7B + 4B Q4); Q8 only via optional legacy.
triple_qvac = bool(include_medpsy) and sidecar_up
_eff_triple = triple_qvac
_eff_generic = bool(include_generic) and sidecar_up
_eff_medical = bool(include_medical) and sidecar_up
_eff_medpsy = bool(include_medpsy) and sidecar_up
_optional_legacy_keys = []
if _eff_generic:
    if st.session_state.get("opt_legacy_local_gemma"):
        _optional_legacy_keys.append("local_gemma")
    if st.session_state.get("opt_legacy_local_llama"):
        _optional_legacy_keys.append("local_llama")
if _eff_medpsy and st.session_state.get("opt_legacy_qvac_4b_q8"):
    _optional_legacy_keys.append("qvac_4b_q8")

roster = merge_roster(
    list(cfg.get("candidates") or []) if include_cloud else [],
    triple_qvac=_eff_triple,
    include_qvac=_eff_medpsy,
    include_local_peers=_eff_generic,
    include_medical_peers=_eff_medical,
    optional_legacy_keys=_optional_legacy_keys,
)
# Drop on-device slots whose GGUF is missing (toggle stays on; chip shows ready set).
roster = [
    c
    for c in roster
    if c.get("provider") != "qvac" or c.get("gguf_ready", True)
]
n_models = len(roster)
_medical_on_device_only = (
    not include_cloud
    and not _eff_generic
    and _eff_medical
    and _eff_medpsy
)

with st.expander(
    f"Exact prompt (inference) — identical for all {n_models} models",
    expanded=False,
):
    st.markdown("**System**")
    st.code(candidate_system())
    st.markdown("**User**")
    st.code(candidate_user(live_case))

_band_bits = []
if include_cloud:
    _band_bits.append("Cloud")
if _eff_medpsy:
    _band_bits.append("MedPsy")
if _eff_generic:
    _band_bits.append("generic local")
if _eff_medical:
    _band_bits.append("medical local")
if _optional_legacy_keys:
    _band_bits.append(f"+{len(_optional_legacy_keys)} legacy")
_band_label = " + ".join(_band_bits) if _band_bits else "no slots"
st.markdown(
    f'<div class="sec-label">Models ({n_models} on the same case · {_band_label})</div>',
    unsafe_allow_html=True,
)
if _medical_on_device_only:
    st.caption(
        "**Medical on-device only** · dual MedPsy + 3 medical local · "
        + track_ui_routing_blurb(benchmark_track)
    )
else:
    st.caption(
        (
            "**Cloud** OpenRouter API · not ChatGPT/Claude/Gemini web · "
            if include_cloud
            else "**Cloud API excluded** · "
        )
        + ("**MedPsy** 1.7B/4B Q4 · " if _eff_medpsy else "")
        + (
            "**Generic local** Phi · "
            if _eff_generic
            else ""
        )
        + (
            "**Medical local** MedGemma/Med42/UltraMedical · "
            if _eff_medical
            else ""
        )
        + (
            f"**Optional legacy** {', '.join(_optional_legacy_keys)} · "
            if _optional_legacy_keys
            else ""
        )
        + track_ui_routing_blurb(benchmark_track)
    )
_chip_n = 3 if n_models >= 6 else min(3, max(1, n_models))
chip_cols = st.columns(_chip_n)
for i, c in enumerate(roster):
    color = c.get("color") or "#64748b"
    with chip_cols[i % _chip_n]:
        ready = c.get("gguf_ready", True)
        miss = "" if ready else " · GGUF missing"
        # UI-only site label — do not change models.yaml site (cohort hash).
        _site = (
            "OpenRouter API"
            if (c.get("provider") or "") == "openrouter"
            else (c.get("site") or "")
        )
        st.markdown(
            f'<div class="model-chip" style="background:{color}18;border-color:{color};color:{color}">'
            f'{c.get("label") or c.get("key")}{miss}'
            f'<span>{_site} · {c.get("model")}</span></div>',
            unsafe_allow_html=True,
        )
missing = [
    c.get("label") or c["key"]
    for c in roster
    if c.get("provider") == "qvac" and not c.get("gguf_ready")
]
if missing:
    st.warning(
        "Missing GGUF(s) under `models/`: "
        + ", ".join(missing)
        + ". MedPsy: `./scripts/download_medpsy_gguf.sh` · "
        "Band B: `./scripts/download_local_peers.sh` · "
        "Medical: `./scripts/download_medical_peers.sh`."
    )

_hist_for_cost = []
try:
    _hist_for_cost = [
        a
        for _, a in RUN_STORE.list_artifacts()[:60]
        if scoring_versions_equivalent(str(a.scoring_version or ""), SCORING_VERSION)
    ]
except Exception:
    _hist_for_cost = []
# Confirmed gold ⇒ extractor already billed; omit from pre-run forecast.
_extract_already = bool(st.session_state.get("_confirmed_gold_json"))
_cost_kwargs = dict(
    include_qvac=_eff_medpsy,
    gold_reference=effective_gold or gold_reference,
    triple_qvac=_eff_triple,
    include_local_peers=_eff_generic,
    include_medical_peers=_eff_medical,
    optional_legacy_keys=_optional_legacy_keys,
    include_extractor=not _extract_already,
    extraction_cost_usd=0.0 if _extract_already else None,
    history_artifacts=_hist_for_cost,
)

bd = estimate_cost_breakdown(cfg, live_case, n=1, **_cost_kwargs)
bd_multi = estimate_cost_breakdown(
    cfg, live_case, n=int(n_multi), **_cost_kwargs
)
bd_local_only = estimate_cost_breakdown(
    cfg,
    live_case,
    n=1,
    local_only=True,
    include_qvac=True,
    gold_reference=effective_gold or gold_reference,
    triple_qvac=True,
    include_local_peers=_eff_generic,
    include_medical_peers=_eff_medical,
    optional_legacy_keys=_optional_legacy_keys,
    include_extractor=not _extract_already,
    extraction_cost_usd=0.0 if _extract_already else None,
    history_artifacts=_hist_for_cost,
)
bd_local_only_multi = estimate_cost_breakdown(
    cfg,
    live_case,
    n=int(n_multi),
    local_only=True,
    include_qvac=True,
    gold_reference=effective_gold or gold_reference,
    triple_qvac=True,
    include_local_peers=_eff_generic,
    include_medical_peers=_eff_medical,
    optional_legacy_keys=_optional_legacy_keys,
    include_extractor=not _extract_already,
    extraction_cost_usd=0.0 if _extract_already else None,
    history_artifacts=_hist_for_cost,
)


st.markdown('<div class="sec-label">Step 3 · Run</div>', unsafe_allow_html=True)
selected_bd = bd
selected_bd_multi = bd_multi
selected_mode = "full" if include_cloud else "local_only"
selected_scope = f"{n_models} models" if include_cloud else f"{n_models} on-device"
selected_unavailable = (
    not has_key
    or n_models < 1
    or (not include_cloud and not qvac_run_ok)
    or not bool(effective_gold)
)
show_cost_forecast = st.toggle(
    "Show OpenRouter cost forecast",
    value=bool(st.session_state.get("show_cost_forecast", True)),
    key="show_cost_forecast",
    help="Pre-run forecast is a rough estimate (often over). "
    "Toggle off if you prefer not to see it. Billed truth = OpenRouter usage.",
)
_run_r1 = st.columns(2, gap="small")
with _run_r1[0]:
    single_clicked = st.button(
        f"Single run · {selected_scope}",
        type="secondary",
        use_container_width=True,
        disabled=selected_unavailable,
        help="Quick one-shot. For published-style comparison prefer Multi ×5.",
    )
    if show_cost_forecast:
        st.markdown(_fmt_cost_single(selected_bd), unsafe_allow_html=True)
        st.caption(
            "Rough estimate · often over · not the OpenRouter ledger "
            "(History-calibrated when comparable runs exist · usage is truth)."
        )
with _run_r1[1]:
    multi_clicked = st.button(
        f"Multi run ×{int(n_multi)} · {selected_scope}",
        type="primary",
        use_container_width=True,
        disabled=selected_unavailable,
        help="Mean/median/std across N runs (default 5 exploratory; ~10 steadier CV).",
    )
    if show_cost_forecast:
        st.markdown(_fmt_cost_multi(selected_bd_multi, int(n_multi)), unsafe_allow_html=True)
        st.caption(
            "Rough estimate · often over · not the OpenRouter ledger "
            "(History-calibrated when comparable runs exist · usage is truth)."
        )

_run_r2 = st.columns([1, 1], gap="small")
with _run_r2[0]:
    qvac_only_clicked = st.button(
        "Run MedPsy only · $0 collect",
        key="qvac_only_btn",
        use_container_width=True,
        disabled=not qvac_run_ok,
        help="Force MedPsy-only for this run (cloud/generic/medical skipped). "
        "Default dual 1.7B/4B Q4; Q8 if optional legacy is on. "
        "KPI compare always; clinical ranking via DeepSeek if an OpenRouter key "
        "is present (judge $ only).",
    )
    st.markdown(
        '<div class="cost-compact cost-multi run-cost-cell"><b>$0 collect</b> · MedPsy · '
        "KPI compare · judge only if key</div>",
        unsafe_allow_html=True,
    )
with _run_r2[1]:
    st.info(
        f"Roster now: **{n_models}** models (default ≤9 · optional ≤12). "
        "Use the four band toggles, Optional / legacy slots, or presets: "
        "**Medical on-device only** (5) · **All on-device** (≤6) · "
        "**Full roster** (≤9) · **Cloud only** (3)."
    )
st.caption(
    "**Minimum aggregate:** 5 valid runs per model (exploratory cohort); "
    "N/A does not block other models. "
    "Multi N: 5 = quick look · ~10 = better CV stability · 20+ = diminishing returns. "
    "Presets only set toggles — Single/Multi use the active roster. "
    "MedPsy only = MedPsy rehearsal without other bands."
)

# Confirm flow via session state (rerun → spend modal AFTER Step 1/2 widgets render)
if single_clicked:
    st.session_state["_persist_case_stem"] = st.session_state.get("demo_case_stem") or ""
    st.session_state["_persist_gold_ref"] = st.session_state.get("demo_gold_ref") or ""
    st.session_state["pending_run"] = {
        "n": 1,
        "est": selected_bd["total_usd"],
        "est_hi": selected_bd.get("total_usd_upper") or selected_bd["total_usd"],
        "mode": selected_mode,
    }
    st.rerun()
if multi_clicked:
    st.session_state["_persist_case_stem"] = st.session_state.get("demo_case_stem") or ""
    st.session_state["_persist_gold_ref"] = st.session_state.get("demo_gold_ref") or ""
    st.session_state["pending_run"] = {
        "n": int(n_multi),
        "est": selected_bd_multi["total_usd_for_n"],
        "est_hi": selected_bd_multi.get("total_usd_upper_for_n")
        or selected_bd_multi["total_usd_for_n"],
        "mode": selected_mode,
    }
    st.rerun()
if qvac_only_clicked:
    # Clear stale failed snapshot (e.g. old 404) so the live panels reset.
    st.session_state.pop("live_outputs", None)
    # Same spend Yes/Cancel gate as Single/Multi (judge path · collect ≈ $0).
    st.session_state["pending_run"] = {
        "n": 1,
        "rounds": 1,
        "est": float(bd_local_only.get("total_usd") or 0),
        "est_hi": float(
            bd_local_only.get("total_usd_upper") or bd_local_only.get("total_usd") or 0
        ),
        "mode": "qvac_only",
    }
    st.rerun()

# Spend confirm AFTER case/gold widgets (keeps widget keys alive).
# Inline card — never st.dialog (overlay ✕ used to kill in-flight runs).
if (
    st.session_state.get("pending_run")
    and not st.session_state.get("confirmed_run")
    and not st.session_state.get("benchmark_running")
):
    _render_spend_confirm_card()
    st.stop()


phase_slot = st.empty()


def _kpi_line(meta: dict, text: str = "") -> str:
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
    if meta.get("ram_mb") is not None:
        ram_lbl = _fmt_ram_mb(meta.get("ram_mb"))
        if ram_lbl:
            parts.append(ram_lbl)
    if meta.get("gguf_mb") is not None:
        gguf_lbl = _fmt_gguf_mb(meta.get("gguf_mb"))
        if gguf_lbl:
            parts.append(gguf_lbl)
    body = (text or "").strip()
    words = len(body.split()) if body else 0
    toks = int(meta.get("completion_tokens") or 0)
    if toks <= 0 and words:
        # Fallback when API did not return completion_tokens
        toks = max(1, int(round(words * 1.3)))
    if words:
        parts.append(f"{words} words")
    if toks:
        parts.append(f"{toks} tok")
    return " · ".join(parts) if parts else "—"


def _status_pill(kind: str, text: str) -> str:
    return f'<span class="status-pill {kind}">{text}</span>'


def _na_failure_label(status: str, reason: str) -> str:
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
            # Keep the label short for the status pill.
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


def _stream_shell_html(*, title: str = "Answer", panel_id: str = "ans") -> str:
    """Fullscreen chrome only — render once; never remount during token stream."""
    return _stream_shell_html_shared(
        title=title, panel_id=panel_id, lang=_ui_lang()
    )


def _stream_body_html(
    text: str,
    live: bool = False,
    *,
    panel_id: str = "ans",
) -> str:
    """Fixed-height answer box only — safe to remount every few tokens."""
    return _stream_body_html_shared(text, live=live, panel_id=panel_id)


def _stream_html(
    text: str,
    live: bool = False,
    *,
    title: str = "Answer",
    panel_id: str = "ans",
) -> str:
    """Shell + body (idle / final paint). Prefer shell once + body updates while live."""
    return (
        _stream_shell_html(title=title, panel_id=panel_id)
        + _stream_body_html(text, live=live, panel_id=panel_id)
    )


def _kpi_live_line(ttft_s, elapsed_s, tps_live) -> str:
    parts = []
    if ttft_s is not None:
        parts.append(f"TTFT {ttft_s}s")
    if tps_live is not None:
        parts.append(f"~{tps_live} TPS")
    if elapsed_s is not None:
        parts.append(f"{elapsed_s}s…")
    return " · ".join(parts) if parts else "streaming…"



from lib.run_timer import (
    _flash_collect_done,
    _paint_run_timer,
    _run_timer_idle,
    _run_timer_live,
    _run_timer_stop,
)


# --- Sidebar: History first, Run clock pinned at the very bottom of the left column ---
with st.sidebar:
    st.markdown("---")
    _hist_pairs = RUN_STORE.list_artifacts()[:12]
    st.caption(
        f"Private history · {short_owner_label()}"
        + (" · enter API key to unlock" if not has_key else "")
    )
    if _hist_pairs:
        st.markdown("**History**")
        _placeholder = "— select a run —"
        _opts = {_placeholder: None}
        _hist_bindings = st.session_state.get("_case_slot_bindings") or {}
        for p, art in _hist_pairs:
            try:
                when = (art.finished_at or art.started_at or "")[5:16].replace("T", " ")
                top = ""
                if art.ranking:
                    top = f" · {art.ranking[0].get('accuracy')}%"
                _slot_tag = slot_label_for_artifact(art, _hist_bindings)
                label = (
                    f"{_slot_tag} · {when} · "
                    f"${art.total_cost_usd:.2f}{top}"
                )
            except Exception:
                label = getattr(p, "stem", None) or art.run_id
            base = label
            n = 2
            while label in _opts:
                label = f"{base} ·{n}"
                n += 1
            _opts[label] = str(p) if p is not None else f"memory:{art.run_id}"
        st.session_state["_hist_sidebar_opts"] = _opts

        _hist_locked = bool(
            st.session_state.get("benchmark_running")
            or st.session_state.get("confirmed_run")
        )
        pick = st.selectbox(
            "Recent runs",
            list(_opts.keys()),
            label_visibility="collapsed",
            key="hist_sidebar_pick",
            disabled=_hist_locked,
        )
        sel_path = _opts.get(pick)
        if st.button(
            "View run results",
            use_container_width=True,
            disabled=not sel_path or _hist_locked,
            key="hist_sidebar_view",
            help="Show ranking + answers in the main page (no popup)",
        ):
            _open_saved_run_inline(sel_path, kind="history")
            st.rerun()
    else:
        st.caption("No runs in History yet for this key.")

    # LAST widget in left column = Run clock (clear gap above History / guides)
    st.markdown('<div class="sidebar-timer-spacer"></div>', unsafe_allow_html=True)
    timer_slot = st.empty()
    _pending = st.session_state.get("confirmed_run") or {}
    if _pending:
        _pn = int(_pending.get("n") or 1)
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
        _last_tm = st.session_state.get("last_run_timings") or {}
        _pr = list(_last_tm.get("per_run") or [])
        _paint_run_timer(
            timer_slot,
            _run_timer_idle(_last_tm),
            live=False,
            multi=int(_last_tm.get("n") or 1) > 1,
            per_run_n=len(_pr),
        )

# Visual order: Live responses → run/results → Rebuild mean → History
# Containers declare page order so Rebuild cannot sit above streams.
_live_zone = st.container()
_results_zone = st.container()
_rebuild_zone = st.container()

with _live_zone:
    # --- Live response panels (roster already built above for chips/cost) ---
    saved_outputs = st.session_state.get("live_outputs") or {}
    _run_pending = st.session_state.get("confirmed_run") or {}
    running_now = bool(_run_pending)
    qvac_only_now = _run_pending.get("mode") == "qvac_only"
    local_only_now = _run_pending.get("mode") == "local_only"

    st.markdown('<div class="sec-label">Live responses</div>', unsafe_allow_html=True)
    st.caption(
        "Same prompt for all models · shorter answers = early stop, not a smaller prompt. "
        + t("stream.live_caption_fs", _ui_lang())
        + (
            " · Grid 3×4 · on-device GGUFs load one after another."
            if n_models >= 12
            else (
                " · Grid 3×3 · on-device GGUFs load one after another."
                if n_models >= 9
                else (
                    " · Grid rows follow the active roster."
                    if n_models >= 5
                    else ""
                )
            )
        )
    )
    if any(c.get("provider") == "qvac" for c in roster):
        st.caption(
            "On-device KPI: **RAM(RSS)** = process-tree resident set (≠ VRAM/mmap) · "
            "**GGUF** = on-disk file size · Band B + MedPsy share one sidecar (serial load)."
        )

    shell_boxes, text_boxes, kpi_boxes, status_boxes = {}, {}, {}, {}
    _panel_rows = panel_rows_for_roster(roster)

    for row in _panel_rows:
        card_cols = st.columns(len(row) or 1)
        for i, c in enumerate(row):
            key = c["key"]
            prev = saved_outputs.get(key) or {}
            color = c.get("color") or "#64748b"
            label = c.get("display_label") or c.get("label") or key
            with card_cols[i]:
                # One fixed card: header + status + KPI + stream (shell once, body remounts)
                st.markdown(
                    f'<div class="panel-card" style="border-top-color: {color}">'
                    f'<p class="live-head">{html.escape(str(label))}</p>'
                    f'<p class="live-meta">{html.escape(str(c.get("model") or ""))}</p></div>',
                    unsafe_allow_html=True,
                )
                status_boxes[key] = st.empty()
                kpi_boxes[key] = st.empty()
                shell_boxes[key] = st.empty()
                text_boxes[key] = st.empty()

                # st.html keeps hidden fullscreen overlay out of the visible page flow
                shell_boxes[key].html(
                    _stream_shell_html(title=str(label), panel_id=key)
                )

                if running_now:
                    _skip_cloud_local = local_only_now and not is_on_device_key(key)
                    _skip_non_medpsy = qvac_only_now and not is_qvac_key(key)
                    if _skip_cloud_local or _skip_non_medpsy:
                        status_boxes[key].markdown(
                            _status_pill(
                                "skip",
                                "Skipped · only-local"
                                if _skip_cloud_local
                                else "Skipped · $0 rehearsal",
                            ),
                            unsafe_allow_html=True,
                        )
                        text_boxes[key].markdown(
                            _stream_body_html(
                                (
                                    f"Skipped — on-device bake-off ({n_models} models)"
                                    if _skip_cloud_local
                                    else "Skipped — MedPsy-only rehearsal"
                                ),
                                live=False,
                                panel_id=key,
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        status_boxes[key].markdown(
                            _status_pill(
                                "wait",
                                "Generating…"
                                if (
                                    (is_qvac_key(key) and qvac_only_now)
                                    or (is_on_device_key(key) and local_only_now)
                                )
                                else "Waiting…",
                            ),
                            unsafe_allow_html=True,
                        )
                        kpi_boxes[key].markdown(
                            '<div class="kpi-slot"></div>', unsafe_allow_html=True
                        )
                        text_boxes[key].markdown(
                            _stream_body_html("", live=True, panel_id=key),
                            unsafe_allow_html=True,
                        )
                elif prev.get("text") is not None:
                    status_msg = prev.get("status") or "Done"
                    kind = "err" if prev.get("error") else "done"
                    status_boxes[key].markdown(
                        _status_pill(kind, status_msg), unsafe_allow_html=True
                    )
                    if prev.get("kpi"):
                        kpi_boxes[key].markdown(
                            f'<div class="kpi-slot"><p class="kpi-row">{prev["kpi"]}</p></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        kpi_boxes[key].markdown(
                            '<div class="kpi-slot"></div>', unsafe_allow_html=True
                        )
                    text_boxes[key].markdown(
                        _stream_body_html(
                            prev.get("text") or "", live=False, panel_id=key
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    on_device_out = is_on_device_key(key) and not sidecar_up
                    if on_device_out:
                        status_boxes[key].markdown(
                            _status_pill("skip", "Sidecar offline"),
                            unsafe_allow_html=True,
                        )
                    else:
                        status_boxes[key].markdown(
                            _status_pill("ready", "Ready"),
                            unsafe_allow_html=True,
                        )
                    kpi_boxes[key].markdown(
                        '<div class="kpi-slot"></div>', unsafe_allow_html=True
                    )
                    text_boxes[key].markdown(
                        _stream_body_html("", live=False, panel_id=key),
                        unsafe_allow_html=True,
                    )

    def _struct_live_footer_html(
        *,
        cohort_id: str | None = None,
        n_label: str = "live · provisional",
        extra: str = "Structured · live provisional",
    ) -> str:
        """Screenshot footer for live / session Structured ranking surfaces."""
        return screenshot_footer_html(
            lang=_ui_lang(),
            scope="same_case",
            cohort_id=cohort_id
            or str(
                st.session_state.get("_active_cohort_id")
                or st.session_state.get("_restored_cohort_id")
                or ""
            )
            or None,
            n_label=n_label,
            protocol_id=str(SCORING_VERSION),
            pack_revision_label=str(
                st.session_state.get("_ui_pack_revision") or _pack_rev_now or ""
            )
            or None,
            extra=extra,
        )

    def _paint_multi_progress(
        slot,
        completed: list,
        *,
        n_total: int,
        batch_done: bool = False,
        toast_html: str = "",
        height: int = 220,
        footer_html: str = "",
    ) -> None:
        """Render multi progress in an iframe so onclick modals/toasts survive sanitizer."""
        foot = footer_html or _struct_live_footer_html(
            n_label="live Multi · per-run ranking",
            extra="Structured · progressive Multi",
        )
        body = progressive_multi_panel_html(
            completed,
            n_total=n_total,
            batch_done=batch_done,
            footer_html=foot,
        ) + (toast_html or "")
        # Extra height when toast is present
        h = height + (180 if toast_html else 0)
        slot.empty()
        with slot.container():
            components.html(
                f"""<!doctype html><html><head><meta charset="utf-8"/>
    <style>
      body {{ margin:0; background:transparent; font-family: ui-sans-serif, system-ui, sans-serif; }}
      .screenshot-footer {{
        margin:0.45rem 0 0.85rem; padding:0.55rem 0.7rem; border-radius:8px;
        border:2px solid #f59e0b; background:#1c1917; color:#fde68a;
        font-size:0.78rem; font-weight:600;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        line-height:1.4; word-break:break-word;
      }}
      .screenshot-footer-gloss {{
        margin-top:0.32rem; font-size:0.72rem; font-weight:500; color:#fbbf24;
      }}
    </style></head><body>{body}</body></html>""",
                height=h,
                scrolling=True,
            )


    # Progressive multi-run KPI strip (filled during / after Multi ×N)
    multi_progress_slot = st.empty()
    _multi_live = st.session_state.get("multi_progress") or {}
    if _multi_live.get("completed") is not None:
        _paint_multi_progress(
            multi_progress_slot,
            list(_multi_live.get("completed") or []),
            n_total=int(_multi_live.get("n_total") or 1),
            batch_done=bool(_multi_live.get("batch_done")),
            height=240 if _multi_live.get("completed") else 120,
        )

with _results_zone:
    # --- Execute confirmed run ---
    if st.session_state.get("confirmed_run"):
        run_cfg = st.session_state.pop("confirmed_run")
        run_mode = run_cfg.get("mode") or "full"
        _batch_id = uuid.uuid4().hex
        # Every run owns a fresh progress lifecycle; never repaint an older 1/N batch.
        for _stale_key in (
            "multi_progress",
            "last_multi_summary",
            "last_multi_paths",
            "show_history_mean_popup",
        ):
            st.session_state.pop(_stale_key, None)
        multi_progress_slot.empty()
        if len(gold_reference) < 40:
            st.error(
                "Add a sufficiently detailed reference answer before running "
                "(diagnosis, tests, urgency, safety and plan)."
            )
            st.stop()
        if not effective_gold:
            st.error(
                "Prepare and Confirm your reference before starting models."
            )
            st.stop()
        try:
            _confirmed = load_confirmed_gold(effective_gold)
            st.session_state["_extract_cost_usd"] = float(
                getattr(_confirmed, "extraction_cost_usd", 0.0) or 0.0
            )
        except Exception as exc:
            st.error(f"Confirmed gold JSON is invalid: {exc}")
            st.stop()
        _active_run_token = start_run(st.session_state["_run_scope"])
        st.session_state["_active_run_id"] = _active_run_token.run_id
        st.session_state["benchmark_running"] = True
        # No leftover dialogs from a previous run (avoids double-dialog crash at the end)
        for _dlg_k in (
            "show_run_done",
            "multi_run_popup_path",
            "history_popup_path",
            "show_history_mean_popup",
            "show_scoring_guide",
            "show_qvac_guide",
        ):
            st.session_state.pop(_dlg_k, None)
        st.session_state.pop("pending_run", None)  # never re-open confirm mid-run
        n_runs = int(run_cfg["n"])
        t_run0 = time.time()
        _paint_run_timer(
            timer_slot,
            _run_timer_live(
                "Starting…",
                n_runs=n_runs,
                elapsed_total=0,
                elapsed_this=0,
                collect_base=0,
                judge_base=0,
                bucket="collect",
            ),
            height=210 if n_runs > 1 else 168,
            multi=n_runs > 1,
        )

        def _abort_run(msg: str, *, phase: str = "Stopped · fix the issue and retry") -> None:
            try:
                cancel_run(st.session_state["_run_scope"])
                abandon_all_pipelines(st.session_state["_run_scope"])
            except Exception:
                pass
            _finish_scope_run()
            st.session_state["benchmark_running"] = False
            elapsed = int(round(time.time() - t_run0))
            _paint_run_timer(
                timer_slot,
                _run_timer_stop(
                    elapsed,
                    n_runs=n_runs,
                    collect_s=0,
                    judge_s=0,
                    title="Run clock · stopped",
                    phase=phase,
                ),
                height=220,
            )
            st.error(msg)
            st.stop()

        # ---- On-device collect: QVAC-only (MedPsy) or Only-local (6 GGUFs) ----
        if run_mode in ("qvac_only", "local_only"):
            _local_bakeoff = run_mode == "local_only"
            if not case_stem.strip():
                _abort_run("Clinical case is empty.")
            if not effective_gold.strip():
                _abort_run(
                    "Automatic reference setup is unavailable; retry the run."
                )
            if not qvac_run_ok:
                _abort_run(
                    "QVAC SDK sidecar offline — start it: `cd sidecar && npm start` "
                    "(requires OpenSSL 3: `brew install openssl@3`)."
                )
            if _local_bakeoff and not has_key:
                _abort_run(
                    "Only local needs an OpenRouter key for DeepSeek R1 judge "
                    "(collect stays $0 · you pay judge tokens only)."
                )

            if _local_bakeoff:
                # Respect active toggles (generic / medical / MedPsy) — not a fixed 6.
                local_slots = [c for c in roster if is_on_device_key(str(c.get("key") or ""))]
                for slot in local_slots:
                    k = slot["key"]
                    if k not in status_boxes:
                        _abort_run(
                            "UI panels missing for Only local — reload the page "
                            "after changing roster toggles."
                        )
            else:
                local_slots = [c for c in roster if is_qvac_key(c["key"])]
            if not local_slots:
                _abort_run("No on-device slots available.")

            n_local = max(1, int(n_runs)) if _local_bakeoff else 1
            _mode_title = (
                f"Only local ×{n_local} · {len(local_slots)} GGUFs · $0 collect"
                if _local_bakeoff
                else f"QVAC only · {len(local_slots)} MedPsy GGUF(s) sequential · $0"
            )
            phase_slot.markdown(
                f'<div class="phase-banner">{_mode_title}</div>',
                unsafe_allow_html=True,
            )
            _paint_run_timer(
                timer_slot,
                _run_timer_live(
                    "Only local · streaming" if _local_bakeoff else "QVAC only · streaming",
                    n_runs=n_local,
                    elapsed_total=time.time() - t_run0,
                    elapsed_this=time.time() - t_run0,
                    collect_base=0,
                    judge_base=0,
                    bucket="collect",
                ),
                height=210 if n_local > 1 else 168,
                multi=n_local > 1,
            )
            _active_local = {c["key"] for c in local_slots}
            for c in roster:
                if c["key"] not in _active_local:
                    status_boxes[c["key"]].markdown(
                        _status_pill("skip", "Skipped"), unsafe_allow_html=True
                    )

            _sys_p = candidate_system()
            _user_p = candidate_user(live_case)
            prompt = _sys_p + "\n\n" + _user_p
            _base_chat_msgs = [
                {"role": "system", "content": _sys_p},
                {"role": "user", "content": _user_p},
            ]
            import time as _time_live

            # Multi N for Only local; QVAC-only stays single-pass
            all_artifacts: list[RunArtifact] = []
            artifact_paths: list[str] = []
            completed_snaps: list = []
            collect_s_acc = 0.0
            judge_s_acc = 0.0
            per_run_timings: list[dict] = []
            last_ranking = None
            last_judgments: list = []
            last_collected: list = []
            last_ok_local: list[str] = []
            last_live_snap: dict = {}
            ranking = None
            judgments: list = []
            collected: list = []
            ok_local: list[str] = []
            judge_s = 0
            live_snap: dict = {}
            abort_multi = False

            if n_local > 1:
                st.session_state["multi_progress"] = {
                    "completed": [],
                    "n_total": n_local,
                    "batch_done": False,
                    "paths": [],
                }
                _paint_multi_progress(
                    multi_progress_slot, [], n_total=n_local, batch_done=False, height=120
                )

            try:
                for run_i in range(1, n_local + 1):
                    if is_cancelled(st.session_state["_run_scope"]):
                        st.warning("Run cancelled before the next iteration.")
                        break
                    _iteration_started = utc_now_iso()
                    t_run_i0 = _time_live.time()
                    if n_local > 1:
                        phase_slot.markdown(
                            f'<div class="phase-banner">Only local · Run {run_i}/{n_local} · '
                            f"collecting {len(local_slots)} GGUFs…</div>",
                            unsafe_allow_html=True,
                        )
                        for qkey in _active_local:
                            status_boxes[qkey].markdown(
                                _status_pill("wait", f"Run {run_i}/{n_local}…"),
                                unsafe_allow_html=True,
                            )
                            text_boxes[qkey].markdown(
                                _stream_body_html("", live=False, panel_id=qkey),
                                unsafe_allow_html=True,
                            )
                            kpi_boxes[qkey].markdown(
                                '<div class="kpi-slot"><p class="kpi-row">—</p></div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        phase_slot.markdown(
                            f'<div class="phase-banner">{_mode_title}</div>',
                            unsafe_allow_html=True,
                        )

                    _paint_run_timer(
                        timer_slot,
                        _run_timer_live(
                            (
                                f"Run {run_i}/{n_local} · collecting"
                                if n_local > 1
                                else (
                                    "Only local · streaming"
                                    if _local_bakeoff
                                    else "QVAC only · streaming"
                                )
                            ),
                            n_runs=n_local,
                            elapsed_total=time.time() - t_run0,
                            elapsed_this=0,
                            collect_base=collect_s_acc,
                            judge_base=judge_s_acc,
                            bucket="collect",
                        ),
                        height=210 if n_local > 1 else 168,
                        multi=n_local > 1,
                    )

                    live_snap = {
                        c["key"]: {
                            "text": "",
                            "status": "Skipped",
                            "error": False,
                            "kpi": "",
                        }
                        for c in roster
                    }

                    # Pipeline: DeepSeek starts as soon as each GGUF finishes (overlap with next loads).
                    pipe = None  # type: PipelinedJudge | None
                    judge_model = (cfg.get("judge") or {}).get("model", "deepseek/deepseek-r1")
                    judge_temp = float((cfg.get("judge") or {}).get("temperature", 0))
                    # Need ≥2 slots for a ranking; avoid paying R1 for a lone QVAC-only GGUF.
                    _pipe_on = bool(has_key) and len(local_slots) >= 2
                    t_j0 = None
                    collected = []
                    judgments = []
                    judge_status = None
                    judge_status_ctx = None
                    progress_slot = None
                    board_slot = None
                    started_keys: set[str] = set()
                    label_by_key: dict[str, str] = {}
                    submitted_local: set[str] = set()
                    lo_board: dict = {}
                    # Mutable bag — avoids nonlocal + annotated-assign pitfalls on Py3.9
                    lo_ui = {"highlight": None, "queue_i": 0, "blind_i": 0}

                    def _paint_lo_board() -> None:
                        if board_slot is None:
                            return
                        board_slot.markdown(
                            live_judging_board_html(
                                lo_board,
                                highlight_key=lo_ui["highlight"],
                                title="Live judging · local + MedPsy",
                            )
                            + _struct_live_footer_html(
                                n_label="live round · provisional",
                                extra="Structured · live provisional",
                            ),
                            unsafe_allow_html=True,
                        )

                    def _on_lo_progress(evt: dict) -> None:
                        if progress_slot is None:
                            return
                        phase = evt.get("phase")
                        key = str(evt.get("key") or "")
                        name = label_by_key.get(key) or evt.get("label") or key
                        done_n = int(evt.get("done") or 0)
                        tot = int(evt.get("total") or max(1, done_n))
                        if phase == "queued" and key not in started_keys:
                            started_keys.add(key)
                            lo_ui["queue_i"] = int(lo_ui["queue_i"]) + 1
                            lo_board[key] = {
                                "label": name,
                                "status": "judging",
                                "accuracy": None,
                                "queue_i": lo_ui["queue_i"],
                                "progress_pct": int(evt.get("percent") or 10),
                                "progress_label": str(evt.get("stage") or "queued"),
                                "elapsed_s": float(evt.get("elapsed_s") or 0),
                            }
                            _paint_lo_board()
                        elif phase == "progress" and key:
                            prev = lo_board.get(key) or {}
                            # Never regress a finished score; only reopen a failed row when
                            # a real new attempt starts (active_attempt / retry stages).
                            if prev.get("status") == "scored" or prev.get("status") == "failed" and not evt.get(
                                "active_attempt"
                            ):
                                pass
                            else:
                                lo_board[key] = {
                                    **prev,
                                    "label": name,
                                    "status": "judging",
                                    "accuracy": None,
                                    "progress_pct": int(evt.get("percent") or 10),
                                    "progress_label": str(evt.get("stage") or "judging"),
                                    "elapsed_s": float(evt.get("elapsed_s") or 0),
                                }
                                _paint_lo_board()
                        elif phase in ("done", "retry_done"):
                            prev_q = (lo_board.get(key) or {}).get("queue_i")
                            if evt.get("failed"):
                                reason = str(
                                    evt.get("failure_reason")
                                    or evt.get("note")
                                    or evt.get("status")
                                    or ""
                                )
                                status = str(evt.get("status") or "").lower()
                                na_label = _na_failure_label(status, reason)
                                lo_board[key] = {
                                    "label": name,
                                    "status": "failed",
                                    "accuracy": None,
                                    "queue_i": prev_q,
                                    "progress_pct": 100,
                                    "progress_label": "complete",
                                    "elapsed_s": float(evt.get("elapsed_s") or 0),
                                }
                                if key in status_boxes:
                                    status_boxes[key].markdown(
                                        _status_pill("err", na_label),
                                        unsafe_allow_html=True,
                                    )
                            else:
                                acc = float(evt.get("accuracy") or 0)
                                lo_board[key] = {
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
                                if key in status_boxes:
                                    status_boxes[key].markdown(
                                        _status_pill("done", f"Judged · {acc:.0f}%"),
                                        unsafe_allow_html=True,
                                    )
                            lo_ui["highlight"] = key
                            _paint_lo_board()
                            if judge_status is not None:
                                judge_status.update(
                                    label=f"DeepSeek R1 · {done_n}/{tot} scored"
                                    + (f" · run {run_i}" if n_local > 1 else "")
                                    + " · pipelined",
                                    state="running",
                                )
                        elif phase == "retry":
                            if key:
                                prev = lo_board.get(key) or {}
                                # Scored rows stay scored. Terminal N/A only reopens when
                                # the pipeline marks a real new attempt (live clock).
                                if prev.get("status") == "scored" or prev.get("status") == "failed" and not evt.get(
                                    "active_attempt"
                                ):
                                    pass
                                else:
                                    if key not in lo_board:
                                        lo_ui["queue_i"] = int(lo_ui["queue_i"]) + 1
                                    lo_board[key] = {
                                        "label": name,
                                        "status": "judging",
                                        "accuracy": None,
                                        "queue_i": prev.get("queue_i", lo_ui["queue_i"]),
                                        "progress_pct": int(evt.get("percent") or 75),
                                        "progress_label": str(
                                            evt.get("stage") or "corrective retry"
                                        ),
                                        "elapsed_s": float(evt.get("elapsed_s") or 0),
                                    }
                                    _paint_lo_board()
                        progress_slot.progress(
                            min(1.0, done_n / max(1, tot)),
                            text=f"Judge · {done_n}/{tot} (overlap with collect)",
                        )

                    def _poll_pipe() -> None:
                        if pipe is None:
                            return
                        pipe.poll()
                        if pipe.submitted:
                            phase_slot.markdown(
                                '<div class="phase-banner">'
                                + (
                                    f"Only local · Run {run_i}/{n_local} · "
                                    if n_local > 1 and _local_bakeoff
                                    else f'{"Only local" if _local_bakeoff else "QVAC-only"} · '
                                )
                                + f"collect + judge {pipe.done_count}/{pipe.total}"
                                + (f" · {pipe.pending_count} in flight" if pipe.pending_count else "")
                                + "</div>",
                                unsafe_allow_html=True,
                            )

                    if _pipe_on:
                        try:
                            _validate_judge_separation(
                                cfg if isinstance(cfg, dict) else {},
                                local_slots,
                            )
                        except ValueError as exc:
                            st.error(f"Judge separation check failed: {exc}")
                            st.stop()
                        pipe = PipelinedJudge(
                            live_case,
                            judge_model,
                            temperature=judge_temp,
                            gold_reference=effective_gold,
                            expected_total=len(local_slots),
                            max_workers=min(6, max(2, len(local_slots))),
                            on_progress=_on_lo_progress,
                            api_key=st.session_state.get("or_key_session"),
                            verifier_model=str(
                                ((cfg.get("judge") or {}) if isinstance(cfg, dict) else {}).get(
                                    "verifier_model"
                                )
                                or ""
                            ),
                            run_scope=st.session_state["_run_scope"],
                            benchmark_track=benchmark_track,
                            judge_allowed_providers=list(
                                ((cfg.get("judge") or {}) if isinstance(cfg, dict) else {}).get(
                                    "allowed_providers"
                                )
                                or []
                            ),
                            verifier_allowed_providers=list(
                                ((cfg.get("judge") or {}) if isinstance(cfg, dict) else {}).get(
                                    "verifier_allowed_providers"
                                )
                                or []
                            ),
                        )
                        judge_status_ctx = st.status(
                            "DeepSeek R1 · pipelined with collect · "
                            + (
                                f"run {run_i}/{n_local}"
                                if n_local > 1
                                else ("Only local" if _local_bakeoff else "QVAC-only")
                            ),
                            expanded=True,
                        )
                        judge_status = judge_status_ctx.__enter__()
                        progress_slot = st.empty()
                        board_slot = st.empty()
                        progress_slot.progress(0.0, text="Judge · waiting for first GGUF…")
                        _paint_lo_board()

                    def _terminalize_local(
                        slot_cfg: dict,
                        label_text: str,
                        body_text: str,
                        error_text,
                        meta_fields: dict,
                    ) -> None:
                        """Hand one roster slot to the judge, including failed ones.

                        A GGUF that never loaded or never streamed is still part of the
                        fixed cohort, so it has to arrive as an explicit N/A row instead
                        of quietly disappearing from the comparison.
                        """
                        if pipe is None:
                            return
                        key_ = str(slot_cfg.get("key") or "")
                        if not key_ or key_ in submitted_local:
                            return
                        lo_ui["blind_i"] = int(lo_ui["blind_i"]) + 1
                        blind_id_ = f"Candidate {lo_ui['blind_i']}"
                        cand_row = CandidateAnswer(
                            candidate_key=key_,
                            label=str(slot_cfg.get("label") or key_),
                            display_label=str(
                                label_text or slot_cfg.get("display_label") or key_
                            ),
                            vendor=str(slot_cfg.get("vendor") or "local"),
                            site=str(slot_cfg.get("site") or "local (QVAC SDK)"),
                            blind_id=blind_id_,
                            answers=(
                                parse_candidate_answers(live_case, body_text)
                                if body_text
                                else {}
                            ),
                            raw_response=body_text or "",
                            meta=ModelCallMeta(
                                model=str(
                                    meta_fields.get("model") or slot_cfg.get("model") or key_
                                ),
                                provider="qvac",
                                display_label=str(label_text or key_),
                                ttft_s=meta_fields.get("ttft_s"),
                                tps=meta_fields.get("tps"),
                                latency_s=meta_fields.get("latency_s"),
                                finish_reason=str(meta_fields.get("finish_reason") or ""),
                                completion_tokens=int(
                                    meta_fields.get("completion_tokens") or 0
                                ),
                                ram_mb=meta_fields.get("ram_mb"),
                                gguf_mb=meta_fields.get("gguf_mb"),
                                gguf_sha256=str(meta_fields.get("gguf_sha256") or ""),
                                device=str(meta_fields.get("device") or ""),
                                gpu_layers=meta_fields.get("gpu_layers"),
                                ctx_size=meta_fields.get("ctx_size"),
                                predict=meta_fields.get("predict"),
                                seed=meta_fields.get("seed"),
                                temperature=meta_fields.get("temperature"),
                                top_k=meta_fields.get("top_k"),
                                top_p=meta_fields.get("top_p"),
                                cost_usd=0.0,
                                error=error_text,
                                prior_attempts=list(meta_fields.get("prior_attempts") or []),
                                retry_count=int(meta_fields.get("retry_count") or 0),
                            ),
                        )
                        # Same format-repair / section recovery as the CLI collector.
                        # Always clear Recovering status (timeout / exception / success).
                        if not error_text and missing_section_ids(
                            live_case, cand_row.answers or {}
                        ):
                            if key_ in status_boxes:
                                status_boxes[key_].markdown(
                                    _status_pill("wait", "Recovering sections…"),
                                    unsafe_allow_html=True,
                                )
                            try:
                                cand_row = maybe_retry_candidate(
                                    live_case,
                                    cand_row,
                                    slot_cfg,
                                    blind_id_,
                                    benchmark_track=benchmark_track,
                                )
                            except Exception:
                                # Leave first-pass answers; missing sections stay N/A.
                                pass
                            finally:
                                if key_ in status_boxes:
                                    status_boxes[key_].markdown(
                                        _status_pill(
                                            "done",
                                            "Done · $0 · judge queued",
                                        ),
                                        unsafe_allow_html=True,
                                    )
                        submitted_local.add(key_)
                        label_by_key[key_] = cand_row.display_label or cand_row.label
                        if key_ not in started_keys:
                            lo_ui["queue_i"] = int(lo_ui["queue_i"]) + 1
                            lo_board[key_] = {
                                "label": label_by_key[key_],
                                "status": "judging",
                                "accuracy": None,
                                "queue_i": lo_ui["queue_i"],
                            }
                            _paint_lo_board()
                        pipe.submit(cand_row)

                    for slot in local_slots:
                        qkey = slot["key"]
                        qlabel = slot.get("display_label") or slot.get("label") or qkey
                        gguf = slot.get("gguf_path")
                        model_id = str(slot.get("model") or qkey)
                        # Match CLI runner: blind id is assigned in order; seed uses
                        # sha256(f"{blind_id}:{key}:{model_id}")[:8] % (2**31-1).
                        _next_blind = f"Candidate {int(lo_ui['blind_i']) + 1}"
                        _sampling: dict = {}
                        if uses_controlled_sampling(benchmark_track):
                            _sampling = {"temp": 0.2, "top_k": 20, "top_p": 0.95}
                            if is_strict_track(benchmark_track):
                                seed_basis = f"{_next_blind}:{qkey}:{model_id}"
                                _sampling["seed"] = int(
                                    hashlib.sha256(seed_basis.encode("utf-8")).hexdigest()[
                                        :8
                                    ],
                                    16,
                                ) % (2**31 - 1)
                        status_boxes[qkey].markdown(
                            _status_pill("wait", "Loading GGUF…" if gguf else "Streaming…"),
                            unsafe_allow_html=True,
                        )
                        _runtime_pin: dict = {
                            "seed": _sampling.get("seed"),
                            "temperature": _sampling.get("temp"),
                            "top_k": _sampling.get("top_k"),
                            "top_p": _sampling.get("top_p"),
                        }
                        if gguf:
                            loaded = qvac_load_model(gguf, sampling=_sampling)
                            if not loaded.get("ok"):
                                # A hot-swap can fail while the previous GGUF unloads;
                                # spend one free local retry before calling it N/A.
                                status_boxes[qkey].markdown(
                                    _status_pill("wait", "Reloading GGUF…"),
                                    unsafe_allow_html=True,
                                )
                                _time_live.sleep(1.5)
                                loaded = qvac_load_model(gguf, sampling=_sampling)
                            if not loaded.get("ok"):
                                err_load = str(loaded.get("error") or "load failed")[:80]
                                status_boxes[qkey].markdown(
                                    _status_pill("err", err_load), unsafe_allow_html=True
                                )
                                live_snap[qkey] = {
                                    "text": "",
                                    "status": err_load,
                                    "error": True,
                                    "kpi": "",
                                    "meta": {},
                                    "label": qlabel,
                                }
                                _terminalize_local(
                                    slot,
                                    str(qlabel),
                                    "",
                                    str(loaded.get("error") or "Failed to load GGUF"),
                                    {
                                        **_runtime_pin,
                                        "gguf_sha256": loaded.get("gguf_sha256") or "",
                                        "device": loaded.get("device") or "",
                                        "gpu_layers": loaded.get("gpu_layers"),
                                        "ctx_size": loaded.get("ctx_size"),
                                        "predict": loaded.get("predict"),
                                    },
                                )
                                _poll_pipe()
                                continue
                            for _rk in (
                                "device",
                                "gpu_layers",
                                "ctx_size",
                                "predict",
                                "gguf_sha256",
                            ):
                                if loaded.get(_rk) is not None and loaded.get(_rk) != "":
                                    _runtime_pin[_rk] = loaded.get(_rk)
                        status_boxes[qkey].markdown(
                            _status_pill("wait", "Streaming…"), unsafe_allow_html=True
                        )
                        stream_state = {"last_paint": 0.0, "last_pipe_poll": 0.0}
                        _chat_msgs = local_chat_messages(_base_chat_msgs, slot)
                        _slot_prompt = "\n\n".join(
                            str(m.get("content") or "") for m in _chat_msgs
                        ) or prompt

                        def _stream_local_once() -> tuple:
                            """Consume one on-device generation, painting tokens live."""
                            buf_ = ""
                            done_ = {}
                            err_ = None
                            n_tok_ = 0
                            t0_ = _time_live.time()
                            ttft_ = None
                            for evt in qvac_iter_tokens(
                                _slot_prompt,
                                messages=_chat_msgs,
                                sampling=_sampling or None,
                            ):
                                et = evt.get("type")
                                if et == "token":
                                    tok = evt.get("token") or ""
                                    if not tok:
                                        continue
                                    buf_ += tok
                                    n_tok_ += 1
                                    now = _time_live.time()
                                    if ttft_ is None:
                                        ttft_ = round(now - t0_, 2)
                                    elapsed = round(now - t0_, 2)
                                    gen_elapsed = max(elapsed - (ttft_ or 0), 0.001)
                                    tps_live = round(n_tok_ / gen_elapsed, 1)
                                    if (
                                        n_tok_ == 1
                                        or n_tok_ % 8 == 0
                                        or (now - stream_state["last_paint"]) >= 0.25
                                    ):
                                        stream_state["last_paint"] = now
                                        text_boxes[qkey].markdown(
                                            _stream_body_html(buf_, live=True, panel_id=qkey),
                                            unsafe_allow_html=True,
                                        )
                                        kpi_boxes[qkey].markdown(
                                            f'<div class="kpi-slot"><p class="kpi-row live">'
                                            f"{_kpi_live_line(ttft_, elapsed, tps_live)}"
                                            "</p></div>",
                                            unsafe_allow_html=True,
                                        )
                                    # Harvest finished DeepSeek calls while this GGUF streams
                                    if (
                                        pipe is not None
                                        and (now - stream_state["last_pipe_poll"]) >= 0.45
                                    ):
                                        stream_state["last_pipe_poll"] = now
                                        _poll_pipe()
                                elif et == "done":
                                    done_ = evt
                                    if evt.get("content"):
                                        buf_ = str(evt["content"])
                                    if evt.get("error") and not (buf_ or "").strip():
                                        err_ = str(evt["error"])
                                elif et == "error":
                                    err_ = str(evt.get("error") or "stream error")
                                    break
                            return buf_, done_, err_, n_tok_, ttft_

                        buf, done_meta, err_msg, n_tok, ttft_s = _stream_local_once()
                        # One free local re-stream: a sidecar worker or transport fault
                        # says nothing about the model's clinical ability.
                        if err_msg and not (buf or "").strip() and is_retryable_local_error(err_msg):
                            status_boxes[qkey].markdown(
                                _status_pill("wait", "Retrying local generation…"),
                                unsafe_allow_html=True,
                            )
                            _time_live.sleep(1.5)
                            buf, done_meta, err_msg, n_tok, ttft_s = _stream_local_once()

                        if err_msg:
                            low = err_msg.lower()
                            if "libssl" in low or "openssl" in low:
                                err_msg = (
                                    "QVAC SDK needs OpenSSL 3. "
                                    "Run: ./scripts/setup_qvac_sidecar.sh && cd sidecar && npm start"
                                )
                            elif "404" in err_msg or "not found" in low:
                                err_msg = (
                                    "Sidecar outdated / unreachable. "
                                    "Restart: cd sidecar && npm start"
                                )
                            elif "rpc" in low or "worker" in low:
                                err_msg = (
                                    "QVAC SDK worker failed to start. "
                                    "Run: ./scripts/setup_qvac_sidecar.sh && cd sidecar && npm start"
                                )

                        _body_chk = (buf or "").strip()
                        if (
                            not err_msg
                            and _body_chk
                            and len(_body_chk.split()) <= 2
                            and len(_body_chk) > 200
                        ):
                            err_msg = "Collapsed / non-text output (not scored)"

                        text_boxes[qkey].markdown(
                            _stream_body_html(buf or "(empty)", live=False, panel_id=qkey),
                            unsafe_allow_html=True,
                        )
                        if err_msg:
                            status_boxes[qkey].markdown(
                                _status_pill("err", err_msg[:80]), unsafe_allow_html=True
                            )
                            live_snap[qkey] = {
                                "text": buf or "",
                                "status": err_msg[:80],
                                "error": True,
                                "kpi": "",
                                "meta": {},
                                "label": qlabel,
                            }
                            _terminalize_local(
                                slot,
                                str(qlabel),
                                buf or "",
                                err_msg,
                                {
                                    **_runtime_pin,
                                    "model": slot.get("model") or qkey,
                                    "completion_tokens": done_meta.get("completion_tokens"),
                                    "finish_reason": done_meta.get("finish_reason"),
                                    "device": (
                                        done_meta.get("device")
                                        or _runtime_pin.get("device")
                                        or qvac_health().get("device")
                                        or ""
                                    ),
                                    "gpu_layers": (
                                        done_meta.get("gpu_layers")
                                        if done_meta.get("gpu_layers") is not None
                                        else _runtime_pin.get("gpu_layers")
                                    ),
                                    "ctx_size": (
                                        done_meta.get("ctx_size")
                                        if done_meta.get("ctx_size") is not None
                                        else _runtime_pin.get("ctx_size")
                                    ),
                                    "predict": (
                                        done_meta.get("predict")
                                        if done_meta.get("predict") is not None
                                        else _runtime_pin.get("predict")
                                    ),
                                    "gguf_sha256": (
                                        done_meta.get("gguf_sha256")
                                        or _runtime_pin.get("gguf_sha256")
                                        or ""
                                    ),
                                },
                            )
                        else:
                            status_boxes[qkey].markdown(
                                _status_pill(
                                    "done",
                                    "Done · $0 · judge queued" if pipe is not None else "Done · $0",
                                ),
                                unsafe_allow_html=True,
                            )
                            meta_done = {
                                "ttft_s": done_meta.get("ttft_s")
                                if done_meta.get("ttft_s") is not None
                                else ttft_s,
                                "tps": done_meta.get("tps"),
                                "latency_s": done_meta.get("latency_s"),
                                "cost_usd": 0,
                                "completion_tokens": done_meta.get("completion_tokens") or 0,
                                "ram_mb": done_meta.get("ram_mb"),
                                "gguf_mb": done_meta.get("gguf_mb"),
                                "finish_reason": done_meta.get("finish_reason"),
                                "device": (
                                    done_meta.get("device")
                                    or _runtime_pin.get("device")
                                    or qvac_health().get("device")
                                    or ""
                                ),
                                "gpu_layers": (
                                    done_meta.get("gpu_layers")
                                    if done_meta.get("gpu_layers") is not None
                                    else _runtime_pin.get("gpu_layers")
                                    if _runtime_pin.get("gpu_layers") is not None
                                    else qvac_health().get("gpu_layers")
                                ),
                                "ctx_size": (
                                    done_meta.get("ctx_size")
                                    if done_meta.get("ctx_size") is not None
                                    else _runtime_pin.get("ctx_size")
                                    if _runtime_pin.get("ctx_size") is not None
                                    else qvac_health().get("ctx_size")
                                ),
                                "predict": (
                                    done_meta.get("predict")
                                    if done_meta.get("predict") is not None
                                    else _runtime_pin.get("predict")
                                    if _runtime_pin.get("predict") is not None
                                    else qvac_health().get("predict")
                                ),
                                "seed": (
                                    done_meta.get("seed")
                                    if done_meta.get("seed") is not None
                                    else _runtime_pin.get("seed")
                                ),
                                "temperature": _runtime_pin.get("temperature"),
                                "top_k": _runtime_pin.get("top_k"),
                                "top_p": _runtime_pin.get("top_p"),
                                "gguf_sha256": (
                                    done_meta.get("gguf_sha256")
                                    or _runtime_pin.get("gguf_sha256")
                                    or ""
                                ),
                                "model": slot.get("model") or qkey,
                            }
                            kpi = _kpi_line(meta_done, buf)
                            kpi_full = f"{kpi} · device {meta_done['device'] or '?'}"
                            if meta_done.get("gpu_layers") is not None:
                                kpi_full += f" · L{meta_done['gpu_layers']}"
                            if meta_done.get("seed") is not None:
                                kpi_full += f" · seed {meta_done['seed']}"
                            kpi_boxes[qkey].markdown(
                                f'<div class="kpi-slot"><p class="kpi-row">{kpi_full}</p></div>',
                                unsafe_allow_html=True,
                            )
                            live_snap[qkey] = {
                                "text": buf or "",
                                "status": "Done · $0",
                                "error": False,
                                "kpi": kpi_full,
                                "meta": meta_done,
                                "label": qlabel,
                                "slot": slot,
                            }
                            # Kick DeepSeek while the next GGUF loads / streams
                            if pipe is not None:
                                body_ok = (buf or "").strip()
                                words = body_ok.split()
                                collapsed = len(words) <= 2 and len(body_ok) > 200
                                candidate_error = (
                                    str(meta_done.get("error") or "").strip()
                                    or (
                                        "Empty candidate output"
                                        if not body_ok
                                        else (
                                            "Unusable collapsed candidate output"
                                            if collapsed
                                            else ""
                                        )
                                    )
                                    or None
                                )
                                if qkey:
                                    if candidate_error is None and t_j0 is None:
                                        t_j0 = time.time()
                                        # Both phases tick while DeepSeek overlaps next GGUFs
                                        _el_this = t_j0 - t_run_i0
                                        _paint_run_timer(
                                            timer_slot,
                                            _run_timer_live(
                                                (
                                                    f"Run {run_i}/{n_local} · collect∥judge"
                                                    if n_local > 1
                                                    else "collect∥judge · pipelined"
                                                ),
                                                n_runs=n_local,
                                                elapsed_total=t_j0 - t_run0,
                                                elapsed_this=_el_this,
                                                collect_base=int(
                                                    round(collect_s_acc + _el_this)
                                                ),
                                                judge_base=int(round(judge_s_acc)),
                                                bucket="both",
                                            ),
                                            height=230 if n_local > 1 else 178,
                                            multi=n_local > 1,
                                        )
                                    # Collect → judging: append to bottom of live board (FIFO)
                                    _terminalize_local(
                                        slot, str(qlabel), buf, candidate_error, meta_done
                                    )
                        _poll_pipe()

                    t_collect_end = time.time()
                    run_collect_s = t_collect_end - t_run_i0
                    collect_s_acc += run_collect_s
                    st.session_state["live_outputs"] = live_snap
                    last_live_snap = live_snap

                    def _ok_local_text(k: str) -> bool:
                        snap = live_snap.get(k) or {}
                        if snap.get("error"):
                            return False
                        body = (snap.get("text") or "").strip()
                        if not body:
                            return False
                        words = body.split()
                        if len(words) <= 2 and len(body) > 200:
                            return False
                        return True

                    ok_local = [
                        k for k in live_snap if k in _active_local and _ok_local_text(k)
                    ]
                    last_ok_local = list(ok_local)

                    # KPI compare (single run only — multi shows mean after batch)
                    if n_local == 1 and len(ok_local) >= 2:
                        st.markdown(
                            f'<div class="sec-label">'
                            f'{"Only local" if _local_bakeoff else "QVAC-only"} · KPI compare'
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        kpi_rows = []
                        for k in ok_local:
                            snap = live_snap[k]
                            m = snap.get("meta") or {}
                            body = (snap.get("text") or "").strip()
                            nm, ver = _nv(
                                k, label=snap.get("label"), model=m.get("model")
                            )
                            kpi_rows.append(
                                {
                                    "Name": nm,
                                    "Version": ver,
                                    "TTFT s": m.get("ttft_s"),
                                    "TPS": m.get("tps"),
                                    "Latency s": m.get("latency_s"),
                                    "RAM(RSS)": _fmt_ram_mb(m.get("ram_mb")) or "—",
                                    "GGUF": _fmt_gguf_mb(m.get("gguf_mb")) or "—",
                                    "Words": len(body.split()) if body else 0,
                                    "Tok": int(m.get("completion_tokens") or 0),
                                }
                            )
                        st.dataframe(
                            pd.DataFrame(kpi_rows), use_container_width=True, hide_index=True
                        )
                        st.caption(
                            "Local collect · $0 API · same prompt · sequential GGUF · "
                            "judge overlaps collect when a key is present."
                        )

                    run_judge_s = 0.0
                    ranking = None
                    abort_multi = False
                    _want_judge = len(ok_local) >= 2 and (
                        has_key if not _local_bakeoff else True
                    )

                    if pipe is not None:
                        try:
                            # Shrink expected total to what we actually submitted
                            pipe.set_expected_total(pipe.submitted)
                            if pipe.submitted:
                                phase_slot.markdown(
                                    '<div class="phase-banner">'
                                    + (
                                        f"Only local · Run {run_i}/{n_local} · "
                                        if n_local > 1
                                        else f'{"Only local" if _local_bakeoff else "QVAC-only"} · '
                                    )
                                    + f"finishing judge {pipe.done_count}/{pipe.total}…</div>",
                                    unsafe_allow_html=True,
                                )
                                _judge_so_far = (
                                    (t_collect_end - t_j0) if t_j0 is not None else 0.0
                                )
                                _paint_run_timer(
                                    timer_slot,
                                    _run_timer_live(
                                        (
                                            f"Run {run_i}/{n_local} · DeepSeek tail"
                                            if n_local > 1
                                            else "DeepSeek · finishing overlap"
                                        ),
                                        n_runs=n_local,
                                        elapsed_total=time.time() - t_run0,
                                        elapsed_this=time.time() - t_run_i0,
                                        collect_base=int(round(collect_s_acc)),
                                        judge_base=int(round(judge_s_acc + _judge_so_far)),
                                        bucket="judge",
                                    ),
                                    height=230 if n_local > 1 else 178,
                                    multi=n_local > 1,
                                )
                                judgments = pipe.finalize()
                                collected = pipe.candidates
                            else:
                                judgments = []
                                collected = []
                            if judge_status is not None:
                                judge_status.update(
                                    label=f"DeepSeek R1 · {len(judgments)} done · pipelined"
                                    + (f" · run {run_i}" if n_local > 1 else ""),
                                    state="complete",
                                )
                            # Full judge wall (includes overlap with collect) — not tail-only
                            _t_j_end = time.time()
                            if t_j0 is not None:
                                run_judge_s = _t_j_end - t_j0
                            else:
                                run_judge_s = max(0.0, _t_j_end - t_collect_end)
                            judge_s_acc += run_judge_s
                            if not _want_judge:
                                # Had key but <2 usable answers — drop ranking path
                                judgments = judgments if len(collected) >= 2 else []
                                if len(ok_local) < 2:
                                    if n_local == 1 and _local_bakeoff:
                                        st.warning(
                                            "Fewer than 2 usable local answers — check GGUFs / sidecar "
                                            "(empty or collapsed outputs are excluded from ranking)."
                                        )
                                    elif n_local > 1:
                                        st.warning(
                                            f"Run {run_i}/{n_local}: fewer than 2 usable "
                                            "local answers — skipped."
                                        )
                        finally:
                            try:
                                pipe.close(cancel_pending=False)
                            except Exception:
                                pass
                            if judge_status_ctx is not None:
                                try:
                                    judge_status_ctx.__exit__(None, None, None)
                                except Exception:
                                    pass
                                judge_status_ctx = None
                    elif len(ok_local) >= 2 and not has_key:
                        if n_local == 1:
                            st.info(
                                "**KPI compare** above (local, $0). "
                                "Paste an OpenRouter key to run DeepSeek R1 "
                                "(cloud candidates stay skipped · judge tokens only)."
                            )
                    elif len(ok_local) < 2:
                        if n_local == 1:
                            st.session_state.pop("last_ranking", None)
                            if _local_bakeoff:
                                st.warning(
                                    "Fewer than 2 usable local answers — check GGUFs / sidecar "
                                    "(empty or collapsed outputs are excluded from ranking)."
                                )
                        else:
                            st.warning(
                                f"Run {run_i}/{n_local}: fewer than 2 usable local answers — skipped."
                            )

                    if judgments and len(collected) >= 2:
                        ranking = build_ranking(judgments)
                        by_meta = {c.candidate_key: c for c in collected}
                        for row in ranking:
                            c = by_meta.get(row["key"])
                            if c:
                                row["label"] = c.display_label or c.label
                                row["model"] = c.meta.model
                                if c.meta.ttft_s is not None:
                                    row["ttft_s"] = c.meta.ttft_s
                                if c.meta.tps is not None:
                                    row["tps"] = c.meta.tps
                                if c.meta.ram_mb is not None:
                                    row["ram_mb"] = c.meta.ram_mb
                                if c.meta.gguf_mb is not None:
                                    row["gguf_mb"] = c.meta.gguf_mb
                                row["cost_usd"] = 0.0
                        last_ranking = ranking
                        last_judgments = judgments
                        last_collected = collected

                        from benchmark.costing import cost_breakdown_for_run, run_cost_usd

                        _extract_fee = float(
                            getattr(
                                load_confirmed_gold(effective_gold),
                                "extraction_cost_usd",
                                0.0,
                            )
                            or 0.0
                        ) if effective_gold else 0.0
                        judge_cost = run_cost_usd(collected, judgments)
                        _local_cost_bd = cost_breakdown_for_run(
                            collected, judgments, extraction_cost_usd=_extract_fee
                        )
                        abort_multi = n_local > 1 and systemic_judge_failure(judgments)
                        notes = ""
                        if abort_multi:
                            notes = (
                                f"Only-local multi aborted after run {run_i}/{n_local}: "
                                "systemic judge failure. Remaining runs skipped."
                            )
                            st.warning(notes)
                        # Planned roster model ids — not per-run sidecar labels
                        # (meta.model can oscillate and split one Multi into mixed cohorts).
                        _local_model_contract = planned_on_device_model_contract(
                            local_slots
                        )
                        _local_cohort = build_cohort_id(
                            case_stem=case_stem,
                            gold=load_confirmed_gold(effective_gold),
                            prompt_version="gold-only-v1",
                            model_config={
                                "candidates": _local_model_contract,
                                "judge": cfg.get("judge") if isinstance(cfg, dict) else {},
                            },
                            benchmark_track=benchmark_track,
                            pack_revision=int(_pack_rev_now) or None,
                        )
                        st.session_state["_active_cohort_id"] = _local_cohort
                        artifact = build_run_artifact(
                            config_snapshot=cfg,
                            judge_temperature=judge_temp,
                            run_id=f"{case_id}-{uuid.uuid4().hex[:10]}",
                            case_id=case_id,
                            started_at=_iteration_started,
                            finished_at=utc_now_iso(),
                            n_index=run_i,
                            batch_id=_batch_id,
                            models_config={
                                "profile": (cfg.get("profile") if isinstance(cfg, dict) else None),
                                "mode": run_mode,
                                "pipelined_judge": True,
                                "candidates": [
                                    {
                                        "key": c.candidate_key,
                                        "label": c.label,
                                        "display_label": c.display_label,
                                        "model": c.meta.model,
                                    }
                                    for c in collected
                                ],
                                "judge": cfg.get("judge") if isinstance(cfg, dict) else None,
                                "gold_reference": effective_gold.strip() if effective_gold else "",
                                "case_stem": case_stem.strip(),
                                "owner_id": owner_id_for_current_key(
                                    st.session_state.get("or_key_session")
                                ),
                                "estimated_breakdown": (
                                    bd_local_only_multi
                                    if n_local > 1 and _local_bakeoff
                                    else bd_local_only
                                    if _local_bakeoff
                                    else None
                                ),
                                "pack_revision": int(_pack_rev_now),
                            },
                            candidates=collected,
                            judgments=judgments,
                            ranking=ranking,
                            total_cost_usd=round(judge_cost, 6),
                            cost_breakdown=_local_cost_bd,
                            notes=notes,
                            cohort_id=_local_cohort,
                            scoring_version=SCORING_VERSION,
                            pack_revision=int(_pack_rev_now) or None,
                            prompt_version="gold-only-v1",
                            benchmark_track=benchmark_track,
                            run_status=(
                                "cancelled"
                                if any(j.status == "cancelled" for j in judgments)
                                else (
                                    "complete"
                                    if all(j.status == "valid" for j in judgments)
                                    else "partial"
                                )
                            ),
                            reproducibility={
                                "benchmark_track": benchmark_track,
                                "candidate_temperature": (
                                    0.2
                                    if uses_controlled_sampling(benchmark_track)
                                    else None
                                ),
                                "judge_temperature": judge_temp,
                                "blind_map": {
                                    c.candidate_key: c.blind_id for c in collected
                                },
                            },
                        )

                        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
                        art_path = _persist_run_artifact(artifact, WORKSPACE_DIR)
                        _account = st.session_state.get("account_session")
                        if account_store_configured() and isinstance(_account, AccountSession):
                            try:
                                account_save_artifact(_account, artifact)
                            except Exception as exc:
                                st.warning(f"Encrypted cloud persistence failed: {exc}")
                        all_artifacts.append(artifact)
                        artifact_paths.append(str(art_path) if art_path else artifact.run_id)

                        if n_local > 1:
                            snap = snapshot_from_artifact(artifact)
                            completed_snaps.append(snap)
                            st.session_state["multi_progress"] = {
                                "completed": list(completed_snaps),
                                "n_total": n_local,
                                "batch_done": False,
                                "paths": list(artifact_paths),
                            }
                            _paint_multi_progress(
                                multi_progress_slot,
                                completed_snaps,
                                n_total=n_local,
                                batch_done=False,
                                toast_html=client_toast_run_done(run_i, n_local, ranking),
                                height=320,
                            )
                            leader = "—"
                            if ranking:
                                best = min(ranking, key=lambda r: int(r.get("rank") or 99))
                                leader = (
                                    f"{short_model(str(best.get('key')))} "
                                    f"{float(best.get('accuracy') or 0):.1f}%"
                                )
                            st.toast(
                                f"Only local · run {run_i}/{n_local} · leader {leader}",
                                icon="✅",
                            )
                        else:
                            # Single-run clinical ranking UI
                            ranking = _current_ranking(ranking)
                            for _r in ranking:
                                _r.setdefault("n_runs", 1)
                            st.session_state["last_ranking"] = ranking
                            st.session_state["last_judgments"] = [
                                j.model_dump()
                                for j in filter_current_roster_rows(
                                    judgments, key_field="candidate_key"
                                )
                            ]
                            st.session_state["show_last_run_costs"] = True
                            st.session_state["last_cost_rows"] = [
                                {
                                    "Key": c.candidate_key,
                                    "Model": c.meta.model,
                                    "$": 0.0,
                                    "TTFT": c.meta.ttft_s,
                                    "TPS": c.meta.tps,
                                    "RAM(RSS)": _fmt_ram_mb(c.meta.ram_mb) or "—",
                                    "GGUF": _fmt_gguf_mb(c.meta.gguf_mb) or "—",
                                }
                                for c in filter_current_roster_rows(
                                    collected, key_field="candidate_key"
                                )
                            ] + [
                                {
                                    "Key": "judge",
                                    "Model": judge_model,
                                    "$": round(judge_cost, 6),
                                    "TTFT": None,
                                    "TPS": None,
                                    "RAM(RSS)": "—",
                                    "GGUF": "—",
                                }
                            ]
                            st.markdown(
                                f'<div class="sec-label">'
                                f'{"Only local" if _local_bakeoff else "QVAC-only"} · clinical ranking'
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            st.plotly_chart(
                                fig_judge_accuracy_bars(ranking, height=280),
                                use_container_width=True,
                                key=(
                                    "rank_chart_local_only"
                                    if _local_bakeoff
                                    else "rank_chart_qvac_only"
                                ),
                            )
                            _rank_rows = []
                            _any_na = False
                            for r in ranking:
                                _nm, _ver = _nv(
                                    r.get("key"), label=r.get("label"), model=r.get("model")
                                )
                                _na = str(r.get("status") or "ok") != "ok" or r.get(
                                    "accuracy"
                                ) is None
                                _any_na = _any_na or _na
                                _rank = r.get("rank")
                                _rank_disp = (
                                    "— · partial"
                                    if _na
                                    else (
                                        f"#{_rank} · partial"
                                        if r.get("partial") and _rank is not None
                                        else _rank
                                    )
                                )
                                _rank_rows.append(
                                    {
                                        "#": _rank_disp,
                                        "Name": _nm,
                                        "Version": _ver,
                                        "Clinical Composite %": (
                                            "N/A" if _na else r.get("accuracy")
                                        ),
                                        "Status": (
                                            "partial"
                                            if _na or r.get("partial")
                                            else "ok"
                                        ),
                                        "TTFT": r.get("ttft_s"),
                                        "TPS": r.get("tps"),
                                        "RAM(RSS)": _fmt_ram_mb(r.get("ram_mb")) or "—",
                                        "GGUF": _fmt_gguf_mb(r.get("gguf_mb")) or "—",
                                        "Runs": int(r.get("n_runs") or 1),
                                    }
                                )
                            if _any_na:
                                st.markdown(
                                    '<div style="margin:0.25rem 0 0.5rem;padding:0.45rem 0.7rem;'
                                    "border-radius:8px;border:1px solid #f59e0b;"
                                    'background:rgba(251,191,36,0.12);color:#fde68a;font-size:0.85rem">'
                                    "<b style='color:#fbbf24'>partial</b> · technical N/A "
                                    "remain listed; scored models keep their ranks "
                                    "(not clinical 0%).</div>",
                                    unsafe_allow_html=True,
                                )
                            st.dataframe(
                                pd.DataFrame(_rank_rows),
                                use_container_width=True,
                                hide_index=True,
                            )

                    run_total_s = time.time() - t_run_i0
                    per_run_timings.append(
                        {
                            "run": run_i,
                            "collect_s": int(round(run_collect_s)),
                            "judge_s": int(round(run_judge_s)),
                            "total_s": int(round(run_total_s)),
                        }
                    )
                    if abort_multi:
                        break

            except Exception as exc:
                # Mirror full-Multi except: never leave multi_progress stuck at batch_done=false.
                if n_local > 1:
                    st.session_state["multi_progress"] = finished_multi_progress(
                        completed_snaps,
                        n_total=n_local,
                        paths=artifact_paths,
                        aborted_early=True,
                    )
                    try:
                        _paint_multi_progress(
                            multi_progress_slot,
                            completed_snaps,
                            n_total=n_local,
                            batch_done=True,
                            height=160,
                        )
                    except Exception:
                        pass
                try:
                    cancel_run(st.session_state["_run_scope"])
                    abandon_all_pipelines(st.session_state["_run_scope"])
                except Exception:
                    pass
                _finish_scope_run()
                st.session_state["benchmark_running"] = False
                st.error(
                    f"Local/QVAC multi-run failed: {type(exc).__name__}: {exc}"
                )
                st.stop()

            # ---- Batch wrap-up ----
            _finish_scope_run()
            st.session_state["benchmark_running"] = False
            st.session_state["live_outputs"] = last_live_snap or live_snap
            total_s = int(round(time.time() - t_run0))
            collect_s = int(round(collect_s_acc))
            judge_s = int(round(judge_s_acc))
            last_this_s = int(per_run_timings[-1]["total_s"]) if per_run_timings else total_s
            # Do NOT force collect+judge == total: pipelined overlap makes sum ≥ wall.
            _paint_run_timer(
                timer_slot,
                _run_timer_stop(
                    total_s,
                    this_s=last_this_s if n_local > 1 else None,
                    n_runs=n_local,
                    collect_s=collect_s,
                    judge_s=judge_s,
                    per_run=per_run_timings,
                ),
                height=220,
                multi=n_local > 1,
                per_run_n=len(per_run_timings),
            )
            st.session_state["last_run_timings"] = {
                "collect_s": collect_s,
                "judge_s": judge_s,
                "total_s": total_s,
                "last_run_s": last_this_s,
                "per_run": per_run_timings,
                "mode": run_mode,
                "n": n_local,
            }
            st.session_state["last_multi_n"] = n_local

            _done_label = "Only local" if _local_bakeoff else "QVAC-only"
            if n_local > 1:
                st.session_state["multi_progress"] = finished_multi_progress(
                    completed_snaps,
                    n_total=n_local,
                    paths=artifact_paths,
                    aborted_early=bool(abort_multi),
                )
                _paint_multi_progress(
                    multi_progress_slot,
                    completed_snaps,
                    n_total=n_local,
                    batch_done=True,
                    height=160,
                )
            if len(all_artifacts) > 1:
                phase_slot.markdown(
                    f'<div class="phase-banner">{_done_label} ×{len(all_artifacts)} done · '
                    f"mean ranking ready · judge wall {judge_s}s</div>",
                    unsafe_allow_html=True,
                )
                summary, _mean_warn = summarize_multi_batch(all_artifacts)
                if _mean_warn:
                    st.warning(_mean_warn)
                st.session_state["last_multi_paths"] = list(artifact_paths)
                st.session_state["multi_progress"] = finished_multi_progress(
                    completed_snaps,
                    n_total=n_local,
                    paths=artifact_paths,
                    aborted_early=bool(abort_multi),
                )
                _paint_multi_progress(
                    multi_progress_slot,
                    completed_snaps,
                    n_total=n_local,
                    batch_done=True,
                    height=160,
                )

                if summary is not None:
                    _persist_summary(summary)
                    st.session_state["last_multi_summary"] = summary.model_dump()

                    st.markdown(
                        f'<div class="sec-label">{t("bench.ranking_mean_local_label", _ui_lang())}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(reliability_caption(summary))
                    _lo_art = all_artifacts[-1] if all_artifacts else None
                    _lo_cohort = (
                        str(getattr(_lo_art, "cohort_id", None) or "")
                        if _lo_art is not None
                        else ""
                    )
                    _lo_roster_n = len(
                        filter_current_roster_rows(summary.ranking_mean or [])
                    ) or DEFAULT_ROSTER_VERSION
                    st.markdown(
                        honesty_block_html(
                            lang=_ui_lang(),
                            roster_n=_lo_roster_n,
                            scope="same_case",
                            cohort_id=_lo_cohort,
                        ),
                        unsafe_allow_html=True,
                    )
                    _lo_eff = (
                        ((_lo_art.reproducibility or {}).get("effective_judge") or "")
                        if _lo_art is not None
                        else ""
                    )
                    if _lo_eff:
                        st.caption(f"Effective judge (last run) · `{_lo_eff}`")
                    st.markdown("##### Ranking table")
                    st.markdown(
                        _reliability_table_html(summary.ranking_mean), unsafe_allow_html=True
                    )
                    st.markdown(
                        screenshot_footer_html(
                            lang=_ui_lang(),
                            scope="same_case",
                            roster_n=_lo_roster_n,
                            cohort_id=_lo_cohort,
                            n_label=f"N={summary.n} runs · successful scores",
                            protocol_id=str(SCORING_VERSION),
                            pack_revision_label=str(
                                st.session_state.get("_ui_pack_revision")
                                or _pack_rev_now
                                or ""
                            )
                            or None,
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown("##### Chart (mean %; whiskers = ±1 std)")
                    st.plotly_chart(
                        fig_judge_mean_accuracy_bars(
                            summary.ranking_mean,
                            title="Only local · mean Clinical Composite Score",
                            height=320,
                        ),
                        use_container_width=True,
                        key="rank_chart_local_only_mean",
                    )
                    st.markdown(
                        screenshot_footer_html(
                            lang=_ui_lang(),
                            scope="same_case",
                            roster_n=_lo_roster_n,
                            cohort_id=_lo_cohort,
                            n_label=f"N={summary.n} runs · successful scores",
                            protocol_id=str(SCORING_VERSION),
                            pack_revision_label=str(
                                st.session_state.get("_ui_pack_revision")
                                or _pack_rev_now
                                or ""
                            )
                            or None,
                            extra="mean±std whiskers",
                        ),
                        unsafe_allow_html=True,
                    )
                    if summary.outliers:
                        _exec_notes = [
                            o
                            for o in summary.outliers
                            if "execution_cohort" in o.lower()
                        ]
                        _other_notes = [
                            o for o in summary.outliers if o not in _exec_notes
                        ]
                        if _exec_notes:
                            st.warning(
                                "**Execution cohort varied** across pooled runs — "
                                "primary judge vs verifier/route may differ. "
                                + " · ".join(_exec_notes)
                                + " · mean still pools on cohort_id (requested recipe)."
                            )
                        if _other_notes:
                            st.caption("Notes · " + " · ".join(_other_notes[:4]))
                        _prior_bits = []
                        for art in all_artifacts:
                            for cand in art.candidates or []:
                                pa = list(getattr(cand.meta, "prior_attempts", None) or [])
                                if pa:
                                    _prior_bits.append(
                                        f"{cand.candidate_key}×{len(pa)}"
                                    )
                            for j in art.judgments or []:
                                pa = list(getattr(j, "prior_attempts", None) or [])
                                if pa:
                                    _prior_bits.append(
                                        f"judge:{j.candidate_key}×{len(pa)}"
                                    )
                        if _prior_bits:
                            st.caption(
                                "Retries kept prior attempts · "
                                + ", ".join(_prior_bits[:8])
                            )
                    import statistics as _stats_lo

                    _kpi_acc: dict[str, dict[str, list]] = {}
                    for art in all_artifacts:
                        for c in art.candidates or []:
                            k = c.candidate_key
                            bucket = _kpi_acc.setdefault(
                                k, {"ttft": [], "tps": [], "lat": [], "ram": [], "label": c.display_label or c.label}
                            )
                            if c.meta.ttft_s is not None:
                                bucket["ttft"].append(float(c.meta.ttft_s))
                            if c.meta.tps is not None:
                                bucket["tps"].append(float(c.meta.tps))
                            if c.meta.latency_s is not None:
                                bucket["lat"].append(float(c.meta.latency_s))
                            if c.meta.ram_mb is not None:
                                bucket["ram"].append(float(c.meta.ram_mb))
                    _kpi_mean_rows = []
                    for k, b in _kpi_acc.items():
                        nm, ver = _nv(k, label=b.get("label"))
                        def _m(vals):
                            return round(_stats_lo.fmean(vals), 2) if vals else None
                        _kpi_mean_rows.append(
                            {
                                "Name": nm,
                                "Version": ver,
                                "TTFT mean": _m(b["ttft"]),
                                "TPS mean": _m(b["tps"]),
                                "Latency mean": _m(b["lat"]),
                                "RAM mean": _fmt_ram_mb(_m(b["ram"])) if b["ram"] else "—",
                                "n": len(all_artifacts),
                            }
                        )
                    if _kpi_mean_rows:
                        st.markdown(
                            '<div class="sec-label">Only local · mean on-device KPIs</div>',
                            unsafe_allow_html=True,
                        )
                        st.dataframe(
                            pd.DataFrame(_kpi_mean_rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.caption(
                            f"$0 collect × {n_local} requested iterations · "
                            f"judge spend ≈ ${summary.total_cost_usd:.4f}"
                        )

                st.markdown(
                    '<div class="sec-label">Per-run detail · open a tab</div>',
                    unsafe_allow_html=True,
                )
                tab_cols = st.columns(min(len(artifact_paths), 5) or 1)
                for i, path in enumerate(artifact_paths):
                    snap = completed_snaps[i] if i < len(completed_snaps) else {}
                    ri = snap.get("n_index") or (i + 1)
                    top = "—"
                    ranking_snap = snap.get("ranking") or []
                    if ranking_snap:
                        best = min(ranking_snap, key=lambda r: int(r.get("rank") or 99))
                        top = (
                            f"{short_model(str(best.get('key')))} "
                            f"{float(best.get('accuracy') or 0):.0f}%"
                        )
                    with tab_cols[i % len(tab_cols)]:
                        if st.button(
                            f"Run {ri}\n{top}",
                            use_container_width=True,
                            key=f"lo_mrun_tab_btn_{ri}_{Path(path).stem[-6:]}",
                        ):
                            _arm_kpi_dialog("multi_run", path=path)
                            st.rerun()

                if summary is not None:
                    st.session_state["last_ranking"] = _mean_rows_to_last_ranking(
                        summary.ranking_mean
                    )
                if last_judgments:
                    st.session_state["last_judgments"] = [
                        j.model_dump() if hasattr(j, "model_dump") else j
                        for j in filter_current_roster_rows(
                            last_judgments, key_field="candidate_key"
                        )
                    ]
                if last_collected:
                    judge_model = (cfg.get("judge") or {}).get(
                        "model", "deepseek/deepseek-r1"
                    )
                    judge_cost = sum(
                        (j.judge_meta.cost_usd or 0) for j in last_judgments
                    ) if last_judgments else 0
                    st.session_state["last_cost_rows"] = [
                        {
                            "Key": c.candidate_key,
                            "Model": c.meta.model,
                            "$": 0.0,
                            "TTFT": c.meta.ttft_s,
                            "TPS": c.meta.tps,
                            "RAM(RSS)": _fmt_ram_mb(c.meta.ram_mb) or "—",
                            "GGUF": _fmt_gguf_mb(c.meta.gguf_mb) or "—",
                        }
                        for c in filter_current_roster_rows(
                            last_collected, key_field="candidate_key"
                        )
                    ] + [
                        {
                            "Key": "judge (last run)",
                            "Model": judge_model,
                            "$": round(judge_cost, 6),
                            "TTFT": None,
                            "TPS": None,
                            "RAM(RSS)": "—",
                            "GGUF": "—",
                        }
                    ]
                    st.session_state["show_last_run_costs"] = True

                if per_run_timings:
                    per_bits = " · ".join(
                        f"run{p['run']} {p['total_s']}s (c{p['collect_s']}+j{p['judge_s']})"
                        for p in per_run_timings
                    )
                    st.caption(
                        f"**Wall time** · collect {collect_s}s · judge {judge_s}s · "
                        f"**total {total_s}s** · {per_bits}"
                    )
            else:
                if abort_multi:
                    phase_slot.markdown(
                        f'<div class="phase-banner">{_done_label} aborted early · '
                        f"{len(all_artifacts)}/{n_local} run completed · "
                        "judge infrastructure unavailable</div>",
                        unsafe_allow_html=True,
                    )
                elif ranking:
                    phase_slot.markdown(
                        f'<div class="phase-banner">{_done_label} done · {len(ok_local)} local · '
                        f"KPI + clinical ranking · judge {judge_s}s</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    phase_slot.markdown(
                        f'<div class="phase-banner">{_done_label} done · {len(ok_local)} local · '
                        f"$0 collect · KPI {'compare' if len(ok_local) >= 2 else 'single'}</div>",
                        unsafe_allow_html=True,
                    )
                st.caption(
                    f"Wall time · collect {collect_s}s"
                    + (f" · judge {judge_s}s" if judge_s else "")
                    + " (matches sidebar Run clock)"
                )
            st.stop()

        if not is_usable_openrouter_key(st.session_state.get("or_key_session")):
            _abort_run(
                "OpenRouter API key missing or invalid (truncated placeholder). "
                "Paste the full key from https://openrouter.ai/keys in the sidebar."
            )
        if not case_stem.strip():
            _abort_run("Clinical case is empty.")
        if not effective_gold.strip():
            _abort_run(
                "Automatic reference setup is unavailable; retry the run."
            )

        try:
            prep = prepare_run(
                case_id,
                skip_qvac=not _eff_medpsy,
                require_qvac=False,
                triple_qvac=_eff_triple,
                include_local_peers=_eff_generic,
                include_medical_peers=_eff_medical,
                optional_legacy_keys=_optional_legacy_keys,
            )
        except RuntimeError as exc:
            _abort_run(str(exc))

        candidates_cfg = prep["candidates_cfg"]
        blind_map = prep["blind_map"]
        case_obj: Case = live_case
        judge_model = (prep["cfg"].get("judge") or {}).get("model", "deepseek/deepseek-r1")
        judge_label = (prep["cfg"].get("judge") or {}).get("display_label") or judge_model
        judge_temp = float((prep["cfg"].get("judge") or {}).get("temperature", 0))
        active_keys = {c["key"] for c in candidates_cfg}

        phase_slot.markdown(
            f'<div class="phase-banner">Calling {len(candidates_cfg)} models · '
            f"Judge {judge_label} · N={n_runs}</div>",
            unsafe_allow_html=True,
        )

        # Mark slots not in this run (e.g. QVAC offline) — same HTML box, no text_area
        for c in roster:
            if c["key"] not in active_keys:
                status_boxes[c["key"]].markdown(
                    _status_pill("skip", "Skipped"), unsafe_allow_html=True
                )
                text_boxes[c["key"]].markdown(
                    _stream_body_html(
                        "(start QVAC SDK sidecar to include MedPsy)",
                        live=False,
                        panel_id=c["key"],
                    ),
                    unsafe_allow_html=True,
                )

        all_artifacts = []
        completed_snaps: list[dict] = []
        artifact_paths: list[str] = []
        last_ranking = None
        last_judgments = []
        last_collected = []
        collect_s_acc = 0.0
        judge_s_acc = 0.0
        per_run_timings: list[dict] = []
        abort_multi = False
        live_snap: dict = {
            c["key"]: {
                "text": "(start QVAC SDK sidecar to include MedPsy)",
                "status": "Skipped",
                "error": False,
                "kpi": "",
            }
            for c in roster
            if c["key"] not in active_keys
        }
        if n_runs > 1:
            st.session_state["multi_progress"] = {
                "completed": [],
                "n_total": n_runs,
                "batch_done": False,
            }
            _paint_multi_progress(
                multi_progress_slot, [], n_total=n_runs, batch_done=False, height=120
            )

        try:
            for run_i in range(1, n_runs + 1):
                if is_cancelled(st.session_state["_run_scope"]):
                    st.warning("Run cancelled before the next iteration.")
                    break
                _iteration_started = utc_now_iso()
                phase_slot.markdown(
                    f'<div class="phase-banner">Run {run_i}/{n_runs} · Collecting answers…</div>',
                    unsafe_allow_html=True,
                )
                t_run_i0 = time.time()
                t_collect0 = time.time()
                _paint_run_timer(
                    timer_slot,
                    _run_timer_live(
                        f"Run {run_i}/{n_runs} · collecting",
                        n_runs=n_runs,
                        elapsed_total=t_collect0 - t_run0,
                        elapsed_this=0,
                        collect_base=int(round(collect_s_acc)),
                        judge_base=int(round(judge_s_acc)),
                        bucket="collect",
                    ),
                    height=210 if n_runs > 1 else 168,
                    multi=n_runs > 1,
                )
                collected = []
                bufs = {c["key"]: "" for c in candidates_cfg}
                tok_n = {c["key"]: 0 for c in candidates_cfg}
                label_live = {
                    c["key"]: (c.get("display_label") or c.get("label") or c["key"])
                    for c in candidates_cfg
                }
                last_paint_at: dict[str, float] = {}
                last_pipe_poll = 0.0
                t_j0_full = None  # first DeepSeek submit (overlap start)
                for c in candidates_cfg:
                    status_boxes[c["key"]].markdown(
                        _status_pill("wait", "Streaming…"), unsafe_allow_html=True
                    )
                    kpi_boxes[c["key"]].markdown(
                        '<div class="kpi-slot"></div>', unsafe_allow_html=True
                    )
                    # Shell already mounted once — only clear the fixed-height body
                    text_boxes[c["key"]].markdown(
                        _stream_body_html("", live=True, panel_id=c["key"]),
                        unsafe_allow_html=True,
                    )

                # Start DeepSeek as each model finishes (esp. while QVAC GGUFs still load).
                try:
                    _validate_judge_separation(
                        prep["cfg"] if isinstance(prep.get("cfg"), dict) else {},
                        candidates_cfg,
                    )
                except ValueError as exc:
                    st.error(f"Judge separation check failed: {exc}")
                    st.stop()
                full_pipe = PipelinedJudge(
                    case_obj,
                    judge_model,
                    temperature=judge_temp,
                    gold_reference=effective_gold,
                    expected_total=len(candidates_cfg),
                    max_workers=min(8, max(2, len(candidates_cfg))),
                    on_progress=None,  # wired below inside st.status
                    api_key=st.session_state.get("or_key_session"),
                    verifier_model=str(
                        (prep["cfg"].get("judge") or {}).get("verifier_model") or ""
                    ),
                    run_scope=st.session_state["_run_scope"],
                    benchmark_track=benchmark_track,
                    judge_allowed_providers=list(
                        (prep["cfg"].get("judge") or {}).get("allowed_providers") or []
                    ),
                    verifier_allowed_providers=list(
                        (prep["cfg"].get("judge") or {}).get(
                            "verifier_allowed_providers"
                        )
                        or []
                    ),
                )
                full_started: set[str] = set()
                full_progress_slot = None
                full_board_slot = None
                full_judge_status = None
                full_board: dict = {}
                full_ui = {"highlight": None, "queue_i": 0}

                def _paint_full_board() -> None:
                    if full_board_slot is None:
                        return
                    full_board_slot.markdown(
                        live_judging_board_html(
                            full_board,
                            highlight_key=full_ui["highlight"],
                            title="Live judging · cloud + local + MedPsy",
                        )
                        + _struct_live_footer_html(
                            n_label="live round · provisional",
                            extra="Structured · live provisional",
                        ),
                        unsafe_allow_html=True,
                    )

                def _on_full_progress(evt: dict) -> None:
                    if full_progress_slot is None:
                        return
                    phase = evt.get("phase")
                    key = str(evt.get("key") or "")
                    name = label_live.get(key) or evt.get("label") or key
                    done_n = int(evt.get("done") or 0)
                    tot = int(evt.get("total") or max(1, done_n))
                    if phase == "queued" and key not in full_started:
                        full_started.add(key)
                        if key not in full_board or full_board[key].get("status") not in (
                            "judging",
                            "scored",
                            "failed",
                        ):
                            full_ui["queue_i"] = int(full_ui["queue_i"]) + 1
                            qi = full_ui["queue_i"]
                        else:
                            qi = (full_board.get(key) or {}).get("queue_i") or full_ui[
                                "queue_i"
                            ]
                        full_board[key] = {
                            "label": name,
                            "status": "judging",
                            "accuracy": None,
                            "queue_i": qi,
                            "progress_pct": int(evt.get("percent") or 10),
                            "progress_label": str(evt.get("stage") or "queued"),
                            "elapsed_s": float(evt.get("elapsed_s") or 0),
                        }
                        _paint_full_board()
                    elif phase == "progress" and key:
                        prev = full_board.get(key) or {}
                        if prev.get("status") == "scored" or prev.get("status") == "failed" and not evt.get(
                            "active_attempt"
                        ):
                            pass
                        else:
                            full_board[key] = {
                                **prev,
                                "label": name,
                                "status": "judging",
                                "accuracy": None,
                                "progress_pct": int(evt.get("percent") or 10),
                                "progress_label": str(evt.get("stage") or "judging"),
                                "elapsed_s": float(evt.get("elapsed_s") or 0),
                            }
                            _paint_full_board()
                    elif phase == "retry" and key:
                        prev = full_board.get(key) or {}
                        if prev.get("status") == "scored" or prev.get("status") == "failed" and not evt.get(
                            "active_attempt"
                        ):
                            pass
                        else:
                            full_board[key] = {
                                **prev,
                                "label": name,
                                "status": "judging",
                                "accuracy": None,
                                "progress_pct": int(evt.get("percent") or 75),
                                "progress_label": str(
                                    evt.get("stage") or "corrective retry"
                                ),
                                "elapsed_s": float(evt.get("elapsed_s") or 0),
                            }
                            _paint_full_board()
                    elif phase in ("done", "retry_done"):
                        prev_q = (full_board.get(key) or {}).get("queue_i")
                        if evt.get("failed"):
                            reason = str(
                                evt.get("failure_reason")
                                or evt.get("note")
                                or evt.get("status")
                                or ""
                            )
                            status = str(evt.get("status") or "").lower()
                            na_label = _na_failure_label(status, reason)
                            full_board[key] = {
                                "label": name,
                                "status": "failed",
                                "accuracy": None,
                                "queue_i": prev_q,
                                "progress_pct": 100,
                                "progress_label": "complete",
                                "elapsed_s": float(evt.get("elapsed_s") or 0),
                            }
                            if key in status_boxes:
                                status_boxes[key].markdown(
                                    _status_pill("err", na_label),
                                    unsafe_allow_html=True,
                                )
                        else:
                            acc = float(evt.get("accuracy") or 0)
                            full_board[key] = {
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
                            if key in status_boxes:
                                status_boxes[key].markdown(
                                    _status_pill("done", f"Judged · {acc:.0f}%"),
                                    unsafe_allow_html=True,
                                )
                        full_ui["highlight"] = key
                        _paint_full_board()
                        if full_judge_status is not None:
                            full_judge_status.update(
                                label=f"DeepSeek R1 · {done_n}/{tot} scored · pipelined",
                                state="running",
                            )
                    full_progress_slot.progress(
                        min(1.0, done_n / max(1, tot)),
                        text=f"Judge · {done_n}/{tot} (overlap with collect)",
                    )

                full_pipe.on_progress = _on_full_progress
                full_status_ctx = st.status(
                    f"DeepSeek R1 · pipelined with collect · run {run_i}/{n_runs}",
                    expanded=True,
                )
                full_judge_status = full_status_ctx.__enter__()
                full_progress_slot = st.empty()
                full_board_slot = st.empty()
                full_progress_slot.progress(0.0, text="Judge · waiting for first answer…")
                _paint_full_board()

                for evt in iter_collect_live(
                    case_obj,
                    candidates_cfg,
                    blind_map,
                    benchmark_track=benchmark_track,
                    api_key=st.session_state.get("or_key_session"),
                ):
                    if evt.get("type") == "token":
                        key = evt["key"]
                        bufs[key] = bufs.get(key, "") + (evt.get("delta") or "")
                        tok_n[key] = tok_n.get(key, 0) + 1
                        now_paint = time.time()
                        # Body-only remount; throttle ~8 tokens or ~250ms (no overlay remount)
                        if (
                            tok_n[key] == 1
                            or tok_n[key] % 8 == 0
                            or (now_paint - last_paint_at.get(key, 0.0)) >= 0.25
                        ):
                            last_paint_at[key] = now_paint
                            text_boxes[key].markdown(
                                _stream_body_html(
                                    bufs[key], live=True, panel_id=key
                                ),
                                unsafe_allow_html=True,
                            )
                            kpi_boxes[key].markdown(
                                f'<div class="kpi-slot"><p class="kpi-row live">'
                                f'{_kpi_live_line(evt.get("ttft_s"), evt.get("elapsed_s"), evt.get("tps_live"))}'
                                f"</p></div>",
                                unsafe_allow_html=True,
                            )
                        if (now_paint - last_pipe_poll) >= 0.45:
                            last_pipe_poll = now_paint
                            full_pipe.poll()
                            if full_pipe.submitted:
                                phase_slot.markdown(
                                    f'<div class="phase-banner">Run {run_i}/{n_runs} · '
                                    f"collect + judge {full_pipe.done_count}/{full_pipe.total}"
                                    + (
                                        f" · {full_pipe.pending_count} in flight"
                                        if full_pipe.pending_count
                                        else ""
                                    )
                                    + "</div>",
                                    unsafe_allow_html=True,
                                )
                    elif evt.get("type") == "retry":
                        key = evt["key"]
                        bufs[key] = ""
                        tok_n[key] = 0
                        status_boxes[key].markdown(
                            _status_pill(
                                "run",
                                f"Retrying once · {evt.get('reason') or 'transport'}",
                            ),
                            unsafe_allow_html=True,
                        )
                        text_boxes[key].markdown(
                            _stream_body_html("", live=True, panel_id=key),
                            unsafe_allow_html=True,
                        )
                    elif evt.get("type") == "done":
                        cand = evt["candidate"]
                        collected.append(cand)
                        err = bool(cand.meta.error)
                        status_msg = (
                            "Done · judge queued"
                            if not err
                            else f"Error: {str(cand.meta.error)[:60]}"
                        )
                        status_boxes[cand.candidate_key].markdown(
                            _status_pill("err" if err else "done", status_msg),
                            unsafe_allow_html=True,
                        )
                        text = cand.raw_response or bufs.get(cand.candidate_key) or "(empty)"
                        kpi = _kpi_line(cand.meta.model_dump(), text)
                        kpi_boxes[cand.candidate_key].markdown(
                            f'<div class="kpi-slot"><p class="kpi-row">{kpi}</p></div>',
                            unsafe_allow_html=True,
                        )
                        text_boxes[cand.candidate_key].markdown(
                            _stream_body_html(
                                text, live=False, panel_id=cand.candidate_key
                            ),
                            unsafe_allow_html=True,
                        )
                        live_snap[cand.candidate_key] = {
                            "text": text,
                            "status": status_msg,
                            "error": err,
                            "kpi": kpi,
                        }
                        # Submit every fixed candidate. The pipeline terminalizes
                        # collection errors/empty output as explicit N/A without a paid call.
                        if cand.candidate_key:
                            if (
                                not err
                                and (cand.raw_response or "").strip()
                                and t_j0_full is None
                            ):
                                t_j0_full = time.time()
                                _el_this = t_j0_full - t_run_i0
                                _paint_run_timer(
                                    timer_slot,
                                    _run_timer_live(
                                        f"Run {run_i}/{n_runs} · collect∥judge",
                                        n_runs=n_runs,
                                        elapsed_total=t_j0_full - t_run0,
                                        elapsed_this=_el_this,
                                        collect_base=int(
                                            round(collect_s_acc + (t_j0_full - t_collect0))
                                        ),
                                        judge_base=int(round(judge_s_acc)),
                                        bucket="both",
                                    ),
                                    height=230 if n_runs > 1 else 178,
                                    multi=n_runs > 1,
                                )
                            # Collect → judging: append to bottom of live board (FIFO)
                            # before DeepSeek finishes (cloud + local + MedPsy).
                            _ck = cand.candidate_key
                            _clab = (
                                label_live.get(_ck)
                                or cand.display_label
                                or cand.label
                                or _ck
                            )
                            if _ck not in full_started:
                                full_ui["queue_i"] = int(full_ui["queue_i"]) + 1
                                full_board[_ck] = {
                                    "label": _clab,
                                    "status": "judging",
                                    "accuracy": None,
                                    "queue_i": full_ui["queue_i"],
                                }
                                _paint_full_board()
                            full_pipe.submit(cand)
                            full_pipe.poll()

                by_key = {c.candidate_key: c for c in collected}
                collected = [by_key[c["key"]] for c in candidates_cfg if c["key"] in by_key]
                t_collect_end = time.time()
                run_collect_s = t_collect_end - t_collect0
                collect_s_acc += run_collect_s

                phase_slot.markdown(
                    f'<div class="phase-banner">Run {run_i}/{n_runs} · Collect done · '
                    f"finishing judge {full_pipe.done_count}/{full_pipe.total}…</div>",
                    unsafe_allow_html=True,
                )
                if full_pipe.done_count < full_pipe.submitted:
                    _flash_collect_done(n_answers=len(collected))

                _now_tail = time.time()
                _judge_so_far = (
                    (_now_tail - t_j0_full) if t_j0_full is not None else 0.0
                )
                _paint_run_timer(
                    timer_slot,
                    _run_timer_live(
                        f"Run {run_i}/{n_runs} · DeepSeek R1 tail",
                        n_runs=n_runs,
                        elapsed_total=_now_tail - t_run0,
                        elapsed_this=_now_tail - t_run_i0,
                        collect_base=int(round(collect_s_acc)),
                        judge_base=int(round(judge_s_acc + _judge_so_far)),
                        bucket="judge",
                    ),
                    height=230 if n_runs > 1 else 178,
                    multi=n_runs > 1,
                )
                try:
                    full_pipe.set_expected_total(full_pipe.submitted or len(collected))
                    judgments = full_pipe.finalize()
                    # Prefer cfg order for ranking join
                    by_j = {j.candidate_key: j for j in judgments}
                    judgments = [
                        by_j[c["key"]] for c in candidates_cfg if c["key"] in by_j
                    ]
                    if full_judge_status is not None:
                        full_judge_status.update(
                            label=(
                                f"DeepSeek R1 · {len(judgments)}/{len(collected)} done · "
                                f"pipelined · run {run_i}"
                            ),
                            state="complete",
                        )
                finally:
                    try:
                        full_pipe.close(cancel_pending=False)
                    except Exception:
                        pass
                    try:
                        full_status_ctx.__exit__(None, None, None)
                    except Exception:
                        pass
                _t_j_end = time.time()
                if t_j0_full is not None:
                    run_judge_s = _t_j_end - t_j0_full
                else:
                    run_judge_s = max(0.0, _t_j_end - t_collect_end)
                judge_s_acc += run_judge_s
                run_total_s = _t_j_end - t_run_i0
                per_run_timings.append(
                    {
                        "run": run_i,
                        "collect_s": int(round(run_collect_s)),
                        "judge_s": int(round(run_judge_s)),
                        "total_s": int(round(run_total_s)),
                    }
                )

                ranking = build_ranking(judgments)
                for row in ranking:
                    cand = by_key.get(row["key"])
                    if cand:
                        row["label"] = cand.display_label or cand.label
                        row["ttft_s"] = cand.meta.ttft_s
                        row["tps"] = cand.meta.tps
                        row["latency_s"] = cand.meta.latency_s
                        row["cost_usd"] = cand.meta.cost_usd
                        row["model"] = cand.meta.model
                        if cand.meta.ram_mb is not None:
                            row["ram_mb"] = cand.meta.ram_mb
                        if cand.meta.gguf_mb is not None:
                            row["gguf_mb"] = cand.meta.gguf_mb

                from benchmark.costing import cost_breakdown_for_run, run_cost_usd

                _extract_fee = float(
                    getattr(
                        load_confirmed_gold(effective_gold),
                        "extraction_cost_usd",
                        0.0,
                    )
                    or 0.0
                ) if effective_gold else 0.0
                total_cost = run_cost_usd(collected, judgments)
                _run_cost_bd = cost_breakdown_for_run(
                    collected, judgments, extraction_cost_usd=_extract_fee
                )
                abort_multi = n_runs > 1 and systemic_judge_failure(judgments)
                notes = ""
                if abort_multi:
                    notes = (
                        f"Multi aborted after run {run_i}/{n_runs}: systemic judge failure "
                        "(empty JSON / transport / majority zeros). Remaining runs skipped to save credits."
                    )
                    st.warning(notes)
                _full_cohort = build_cohort_id(
                    case_stem=case_stem,
                    gold=load_confirmed_gold(effective_gold),
                    prompt_version="gold-only-v1",
                    model_config={
                        "candidates": candidates_cfg,
                        "judge": prep["cfg"].get("judge") or {},
                    },
                    benchmark_track=benchmark_track,
                    pack_revision=int(_pack_rev_now) or None,
                )
                st.session_state["_active_cohort_id"] = _full_cohort
                artifact = build_run_artifact(
                    config_snapshot=prep["cfg"],
                    judge_temperature=judge_temp,
                    run_id=f"{case_id}-{uuid.uuid4().hex[:10]}",
                    case_id=case_id,
                    started_at=_iteration_started,
                    finished_at=utc_now_iso(),
                    n_index=run_i,
                    batch_id=_batch_id,
                    models_config={
                        "profile": prep["cfg"].get("profile"),
                        "candidates": candidates_cfg,
                        "judge": prep["cfg"].get("judge"),
                        "blind_map": blind_map,
                        "gold_reference": effective_gold.strip() if effective_gold else "",
                        "case_stem": case_stem.strip(),
                        "owner_id": owner_id_for_current_key(
                            st.session_state.get("or_key_session")
                        ),
                        "estimated_breakdown": bd if n_runs == 1 else bd_multi,
                        "pack_revision": int(_pack_rev_now),
                    },
                    candidates=collected,
                    judgments=judgments,
                    ranking=ranking,
                    total_cost_usd=round(total_cost, 6),
                    cost_breakdown=_run_cost_bd,
                    notes=notes,
                    cohort_id=_full_cohort,
                    scoring_version=SCORING_VERSION,
                    pack_revision=int(_pack_rev_now) or None,
                    prompt_version="gold-only-v1",
                    benchmark_track=benchmark_track,
                    run_status=(
                        "cancelled"
                        if any(j.status == "cancelled" for j in judgments)
                        else (
                            "complete"
                            if all(j.status == "valid" for j in judgments)
                            else "partial"
                        )
                    ),
                    reproducibility={
                        "benchmark_track": benchmark_track,
                        "candidate_temperature": (
                            0.2 if uses_controlled_sampling(benchmark_track) else None
                        ),
                        "judge_temperature": judge_temp,
                        "blind_map": blind_map,
                    },
                )
                WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
                art_path = _persist_run_artifact(artifact, WORKSPACE_DIR)
                _account = st.session_state.get("account_session")
                if account_store_configured() and isinstance(_account, AccountSession):
                    try:
                        account_save_artifact(_account, artifact)
                    except Exception as exc:
                        st.warning(f"Encrypted cloud persistence failed: {exc}")
                all_artifacts.append(artifact)
                artifact_paths.append(str(art_path) if art_path else artifact.run_id)
                last_ranking = ranking
                last_judgments = judgments
                last_collected = collected

                if n_runs > 1:
                    snap = snapshot_from_artifact(artifact)
                    completed_snaps.append(snap)
                    st.session_state["multi_progress"] = {
                        "completed": list(completed_snaps),
                        "n_total": n_runs,
                        "batch_done": False,
                        "paths": list(artifact_paths),
                    }
                    _paint_multi_progress(
                        multi_progress_slot,
                        completed_snaps,
                        n_total=n_runs,
                        batch_done=False,
                        toast_html=client_toast_run_done(run_i, n_runs, ranking),
                        height=320,
                    )
                    leader = "—"
                    if ranking:
                        best = min(ranking, key=lambda r: int(r.get("rank") or 99))
                        leader = (
                            f"{short_model(str(best.get('key')))} "
                            f"{float(best.get('accuracy') or 0):.1f}%"
                        )
                    st.toast(
                        f"Run {run_i}/{n_runs} complete · leader {leader}",
                        icon="✅",
                    )

                if abort_multi:
                    break

            t_end = time.time()
            total_s = int(round(t_end - t_run0))
            collect_s = int(round(collect_s_acc))
            judge_s = int(round(judge_s_acc))
            last_this_s = int(per_run_timings[-1]["total_s"]) if per_run_timings else total_s
            # Do NOT force collect+judge == total (pipelined overlap ⇒ sum ≥ wall).
            timings = {
                "collect_s": collect_s,
                "judge_s": judge_s,
                "total_s": total_s,
                "last_run_s": last_this_s,
                "per_run": per_run_timings,
                "mode": "full",
                "n": n_runs,
            }
            st.session_state["last_run_timings"] = timings
            _paint_run_timer(
                timer_slot,
                _run_timer_stop(
                    total_s,
                    this_s=last_this_s,
                    n_runs=n_runs,
                    collect_s=collect_s,
                    judge_s=judge_s,
                    per_run=per_run_timings,
                ),
                height=220,
                multi=n_runs > 1,
                per_run_n=len(per_run_timings),
            )
            st.session_state["live_outputs"] = live_snap
            if last_ranking:
                last_ranking = _current_ranking(last_ranking)
                for _r in last_ranking:
                    _r.setdefault("n_runs", 1)
            st.session_state["last_ranking"] = last_ranking
            if last_judgments:
                _lj0 = last_judgments[0]
                _lj_key = (
                    "candidate_key"
                    if (hasattr(_lj0, "candidate_key") or isinstance(_lj0, dict))
                    else "key"
                )
                last_judgments = filter_current_roster_rows(
                    last_judgments, key_field=_lj_key
                )
            st.session_state["last_judgments"] = last_judgments
            _finish_scope_run()
            st.session_state["benchmark_running"] = False
            st.session_state["last_cost_rows"] = None  # filled below

            _completion_label = (
                f"Aborted early · {len(all_artifacts)}/{n_runs} completed"
                if abort_multi
                else f"Done · N={len(all_artifacts)}"
            )
            from benchmark.costing import batch_total_cost_usd

            _batch_spend = batch_total_cost_usd(all_artifacts)
            phase_slot.markdown(
                f'<div class="phase-banner">{_completion_label} · '
                f"actual spend ≈ ${_batch_spend:.4f} · "
                f"wall {total_s}s</div>",
                unsafe_allow_html=True,
            )

            st.markdown('<div class="sec-label">Results</div>', unsafe_allow_html=True)
            if n_runs > 1 and per_run_timings:
                per_bits = " · ".join(
                    f"run{p['run']} {p['total_s']}s (c{p['collect_s']}+j{p['judge_s']})"
                    for p in per_run_timings
                )
                st.caption(
                    f"**Wall time** · collect {collect_s}s · judge {judge_s}s · "
                    f"**total {total_s}s** · last run {last_this_s}s"
                )
                st.caption(f"Per run · {per_bits}")
            else:
                st.caption(
                    f"**Wall time** · collect {collect_s}s · judge {judge_s}s · "
                    f"**total {total_s}s** (same as sidebar Run clock)"
                )

            cost_rows = []
            for c in filter_current_roster_rows(last_collected, key_field="candidate_key"):
                cost_rows.append(
                    {
                        "Key": c.candidate_key,
                        "Model": c.meta.model,
                        "$": c.meta.cost_usd,
                        "TTFT": c.meta.ttft_s,
                        "TPS": c.meta.tps,
                        "RAM(RSS)": _fmt_ram_mb(c.meta.ram_mb) or "—",
                        "GGUF": _fmt_gguf_mb(c.meta.gguf_mb) or "—",
                    }
                )
            last_judgments = filter_current_roster_rows(
                last_judgments, key_field="candidate_key"
            )
            judge_cost = sum((j.judge_meta.cost_usd or 0) for j in last_judgments)
            cost_rows.append(
                {
                    "Key": "judge",
                    "Model": judge_model,
                    "$": round(judge_cost, 6),
                    "TTFT": None,
                    "TPS": None,
                    "RAM(RSS)": "—",
                    "GGUF": "—",
                }
            )
            st.session_state["last_cost_rows"] = cost_rows
            st.session_state["show_last_run_costs"] = True
            st.session_state["last_multi_n"] = n_runs

            # Always close the progressive strip when Multi ends — including early
            # abort/cancel with a single artifact (len>1 was leaving batch_done=false).
            if n_runs > 1:
                st.session_state["multi_progress"] = finished_multi_progress(
                    completed_snaps,
                    n_total=n_runs,
                    paths=artifact_paths,
                    aborted_early=bool(abort_multi),
                )
                _paint_multi_progress(
                    multi_progress_slot,
                    completed_snaps,
                    n_total=n_runs,
                    batch_done=True,
                    height=160,
                )

            # -------- Multi ×N: protocol mean KPIs; per-run via tabs/popups --------
            if len(all_artifacts) > 1:
                summary, _mean_warn = summarize_multi_batch(all_artifacts)
                if _mean_warn:
                    st.warning(_mean_warn)
                st.session_state["last_multi_paths"] = list(artifact_paths)
                st.session_state["multi_progress"] = finished_multi_progress(
                    completed_snaps,
                    n_total=n_runs,
                    paths=artifact_paths,
                    aborted_early=bool(abort_multi),
                )
                _paint_multi_progress(
                    multi_progress_slot,
                    completed_snaps,
                    n_total=n_runs,
                    batch_done=True,
                    height=160,
                )

                if summary is not None:
                    _persist_summary(summary)
                    st.session_state["last_multi_summary"] = summary.model_dump()

                    st.markdown(
                        f'<div class="sec-label">{t("bench.ranking_mean_label", _ui_lang())}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(reliability_caption(summary))
                    _last_art = all_artifacts[-1] if all_artifacts else None
                    _multi_cohort = (
                        str(getattr(_last_art, "cohort_id", None) or "")
                        if _last_art is not None
                        else ""
                    )
                    _multi_roster_n = len(
                        filter_current_roster_rows(summary.ranking_mean or [])
                    ) or DEFAULT_ROSTER_VERSION
                    st.markdown(
                        honesty_block_html(
                            lang=_ui_lang(),
                            roster_n=_multi_roster_n,
                            scope="same_case",
                            cohort_id=_multi_cohort,
                        ),
                        unsafe_allow_html=True,
                    )
                    _eff_multi = (
                        ((_last_art.reproducibility or {}).get("effective_judge") or "")
                        if _last_art is not None
                        else ""
                    )
                    if _eff_multi:
                        st.caption(
                            f"Effective judge (last run) · `{_eff_multi}`"
                            + (
                                " · verifier may replace primary on systemic failure"
                                if (_last_art.reproducibility or {}).get("verifier_activated")
                                else ""
                            )
                        )
                    st.markdown("##### Ranking table")
                    st.markdown(
                        _reliability_table_html(summary.ranking_mean), unsafe_allow_html=True
                    )
                    st.markdown(
                        screenshot_footer_html(
                            lang=_ui_lang(),
                            scope="same_case",
                            roster_n=_multi_roster_n,
                            cohort_id=_multi_cohort,
                            n_label=f"N={summary.n} runs · successful scores",
                            protocol_id=str(SCORING_VERSION),
                            pack_revision_label=str(
                                st.session_state.get("_ui_pack_revision")
                                or _pack_rev_now
                                or ""
                            )
                            or None,
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown("##### Chart (mean %; whiskers = ±1 std)")
                    st.plotly_chart(
                        fig_judge_mean_accuracy_bars(
                            summary.ranking_mean,
                            title="Mean Clinical Composite Score",
                            height=320,
                        ),
                        use_container_width=True,
                        key="rank_chart_multi_mean",
                    )
                    st.markdown(
                        screenshot_footer_html(
                            lang=_ui_lang(),
                            scope="same_case",
                            roster_n=_multi_roster_n,
                            cohort_id=_multi_cohort,
                            n_label=f"N={summary.n} runs · successful scores",
                            protocol_id=str(SCORING_VERSION),
                            pack_revision_label=str(
                                st.session_state.get("_ui_pack_revision")
                                or _pack_rev_now
                                or ""
                            )
                            or None,
                            extra="mean±std whiskers",
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown("##### Paired sensitivity ranking")
                    if summary.paired_ranking:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "Rank": row.get("rank"),
                                        "Model": short_model(str(row.get("key"))),
                                        "Paired mean %": row.get("accuracy_mean"),
                                        "Coverage %": row.get("coverage_mean"),
                                        "Quality %": row.get("quality_mean"),
                                        "Discipline %": row.get("discipline_mean"),
                                        "Paired N": summary.paired_n,
                                    }
                                    for row in summary.paired_ranking
                                ]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.caption(
                            f"Paired N={summary.paired_n}; at least 5 complete iterations "
                            "are required. Missing scores are never imputed."
                        )
                    if summary.outliers:
                        _exec_notes = [
                            o
                            for o in summary.outliers
                            if "execution_cohort" in o.lower()
                        ]
                        _other_notes = [
                            o for o in summary.outliers if o not in _exec_notes
                        ]
                        if _exec_notes:
                            st.warning(
                                "**Execution cohort varied** across pooled runs — "
                                "primary judge vs verifier/route may differ. "
                                + " · ".join(_exec_notes)
                                + " · mean still pools on cohort_id (requested recipe)."
                            )
                        if _other_notes:
                            st.caption("Notes · " + " · ".join(_other_notes[:4]))
                        _prior_bits = []
                        for art in all_artifacts:
                            for cand in art.candidates or []:
                                pa = list(getattr(cand.meta, "prior_attempts", None) or [])
                                if pa:
                                    _prior_bits.append(
                                        f"{cand.candidate_key}×{len(pa)}"
                                    )
                            for j in art.judgments or []:
                                pa = list(getattr(j, "prior_attempts", None) or [])
                                if pa:
                                    _prior_bits.append(
                                        f"judge:{j.candidate_key}×{len(pa)}"
                                    )
                        if _prior_bits:
                            st.caption(
                                "Retries kept prior attempts · "
                                + ", ".join(sorted(set(_prior_bits))[:8])
                            )

                st.markdown(
                    '<div class="sec-label">Per-run detail · open a tab</div>',
                    unsafe_allow_html=True,
                )
                st.caption("Each finished run keeps its own KPIs — click to open in-page (no popup).")
                tab_cols = st.columns(min(len(artifact_paths), 5) or 1)
                for i, path in enumerate(artifact_paths):
                    snap = completed_snaps[i] if i < len(completed_snaps) else {}
                    ri = snap.get("n_index") or (i + 1)
                    top = "—"
                    ranking = snap.get("ranking") or []
                    if ranking:
                        best = min(ranking, key=lambda r: int(r.get("rank") or 99))
                        top = f"{short_model(str(best.get('key')))} {float(best.get('accuracy') or 0):.0f}%"
                    with tab_cols[i % len(tab_cols)]:
                        if st.button(
                            f"Run {ri}\n{top}",
                            use_container_width=True,
                            key=f"mrun_tab_btn_{ri}_{Path(path).stem[-6:]}",
                        ):
                            _arm_kpi_dialog("multi_run", path=path)
                            st.rerun()

                with st.expander("Last run only (for reference)", expanded=False):
                    if last_ranking:
                        st.plotly_chart(
                            fig_judge_accuracy_bars(
                                last_ranking,
                                height=220,
                                title="Last run · Clinical Composite Score",
                            ),
                            use_container_width=True,
                            key="rank_chart_last_ref",
                        )
                    st.dataframe(pd.DataFrame(cost_rows), use_container_width=True, hide_index=True)

                if summary is not None:
                    # Ranking for persist view = mean order mapped to accuracy_mean
                    st.session_state["last_ranking"] = _mean_rows_to_last_ranking(
                        summary.ranking_mean
                    )
            else:
                # -------- Single run: classic results --------
                if last_judgments:
                    explain = explain_run_scores(case_obj, last_judgments)
                    st.session_state["last_score_explain"] = explain
                    with st.expander("This run — why these scores", expanded=True):
                        b1, b2 = st.columns([3, 1])
                        with b1:
                            st.caption(explain.get("note") or "")
                            st.markdown(
                                "**Weights** · "
                                + " · ".join(
                                    f"`{k}` {v:.0%}"
                                    for k, v in explain["section_weights"].items()
                                )
                            )
                            st.markdown(
                                "**Heaviest** · "
                                + ", ".join(explain.get("heaviest_sections") or [])
                            )
                        with b2:
                            st.markdown(
                                '<label class="guide-open-btn" for="guide_rank">'
                                "Full guide</label>",
                                unsafe_allow_html=True,
                            )
                        rows_ex = []
                        for pm in filter_current_roster_rows(explain.get("per_model") or []):
                            nm, ver = _nv(pm.get("key"))
                            rows_ex.append(
                                {
                                    "Name": nm,
                                    "Version": ver,
                                    "Clinical Composite %": pm["accuracy"],
                                    "Diagnosis": pm.get("diagnosis"),
                                    "Safety": pm.get("safety"),
                                    "Strongest": pm.get("strongest"),
                                    "Weakest": pm.get("weakest"),
                                    "Runs": 1,
                                }
                            )
                        if rows_ex:
                            st.dataframe(
                                pd.DataFrame(rows_ex),
                                use_container_width=True,
                                hide_index=True,
                            )
                        st.caption(
                            "**Quality** = clinical judgment (not style). "
                            "Reference-relative Clinical Composite — not clinical ground truth. "
                            "Exact ties keep the same rank; technical failures are N/A."
                        )

                if last_ranking:
                    last_ranking = _current_ranking(last_ranking)
                    for _r in last_ranking:
                        _r.setdefault("n_runs", 1)
                    st.plotly_chart(
                        fig_judge_accuracy_bars(last_ranking, height=260),
                        use_container_width=True,
                        key="rank_chart_live",
                    )
                tab_l, tab_r = st.columns(2)
                with tab_l:
                    _eff_from_j = sorted(
                        {
                            str(
                                getattr(j, "judge_model", None)
                                or (j.get("judge_model") if isinstance(j, dict) else "")
                                or ""
                            )
                            for j in (last_judgments or [])
                        }
                        - {""}
                    )
                    _eff_label = (
                        _eff_from_j[0]
                        if len(_eff_from_j) == 1
                        else ("mixed" if _eff_from_j else judge_model)
                    )
                    st.caption(
                        "Clinical Composite Score (reference-relative) + KPI · "
                        "uncalibrated LLM-as-judge · not clinical ground truth · "
                        "technical errors remain N/A · "
                        f"effective judge · `{_eff_label}`"
                        + (
                            " (verifier may replace primary on systemic failure)"
                            if _eff_label
                            else ""
                        )
                    )
                    if last_ranking:
                        rows = []
                        _any_partial = False
                        for r in last_ranking:
                            nm, ver = _nv(
                                r.get("key"),
                                label=r.get("label"),
                                model=r.get("model"),
                            )
                            st_raw = str(r.get("status") or "ok").lower()
                            is_na = st_raw in {"n/a", "na", "failed", "error"} or (
                                r.get("accuracy") is None and st_raw != "ok"
                            )
                            is_partial = bool(r.get("partial")) or st_raw == "partial" or is_na
                            _any_partial = _any_partial or is_partial
                            rank = r.get("rank")
                            if is_na:
                                rank_disp = "— · partial"
                                score_disp = "N/A"
                                status_disp = "partial"
                            elif is_partial and rank is not None:
                                rank_disp = f"#{rank} · partial"
                                score_disp = r.get("accuracy")
                                status_disp = "partial"
                            else:
                                rank_disp = rank
                                score_disp = r.get("accuracy")
                                status_disp = "ok"
                            rows.append(
                                {
                                    "#": rank_disp,
                                    "Name": nm,
                                    "Version": ver,
                                    "Clinical Composite %": score_disp,
                                    "Status": status_disp,
                                    "TTFT": r.get("ttft_s"),
                                    "TPS": r.get("tps"),
                                    "RAM(RSS)": _fmt_ram_mb(r.get("ram_mb")) or "—",
                                    "GGUF": _fmt_gguf_mb(r.get("gguf_mb")) or "—",
                                    "$": r.get("cost_usd"),
                                    "Runs": int(r.get("n_runs") or 1),
                                }
                            )
                        if _any_partial:
                            st.markdown(
                                '<div style="margin:0.25rem 0 0.5rem;padding:0.45rem 0.7rem;'
                                "border-radius:8px;border:1px solid #f59e0b;"
                                'background:rgba(251,191,36,0.12);color:#fde68a;font-size:0.85rem">'
                                "<b style='color:#fbbf24'>partial</b> · incomplete coverage "
                                "stays ranked by mean of scored runs when available.</div>",
                                unsafe_allow_html=True,
                            )
                        st.dataframe(
                            pd.DataFrame(rows), use_container_width=True, hide_index=True
                        )
                with tab_r:
                    st.caption("Actual $")
                    st.dataframe(
                        pd.DataFrame(cost_rows), use_container_width=True, hide_index=True
                    )

                if last_judgments:
                    st.markdown(
                        '<div class="sec-label">Scores by clinical dimension</div>',
                        unsafe_allow_html=True,
                    )
                    q_ids = [q.id for q in case_obj.questions]
                    matrix_rows = []
                    for j in last_judgments:
                        nm, ver = _nv(j.candidate_key)
                        row = {"Name": nm, "Version": ver}
                        by_q = {qs.question_id: qs.score for qs in j.question_scores}
                        for qid in q_ids:
                            row[qid] = by_q.get(qid)
                        row["Clinical Composite %"] = j.weighted_accuracy
                        row["Coverage %"] = j.coverage_score
                        row["Quality %"] = j.quality_score
                        row["Discipline %"] = j.discipline_score
                        _nr = next(
                            (
                                int(r.get("n_runs") or r.get("n") or 1)
                                for r in (last_ranking or [])
                                if r.get("key") == j.candidate_key
                            ),
                            1,
                        )
                        row["Runs"] = _nr
                        matrix_rows.append(row)
                    st.dataframe(
                        pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True
                    )
                    st.caption(
                        "Per-question 0–100 from DeepSeek R1 (semantic / synonym-aware). "
                        "The Clinical Composite Score uses case section weights."
                    )

                with st.expander("Judge breakdown", expanded=False):
                    for j in last_judgments:
                        name = short_model(j.candidate_key)
                        st.markdown(f"**{name} · {j.weighted_accuracy}%**")
                        for qs in j.question_scores:
                            st.caption(f"{qs.question_id}: {qs.score}/100 — {qs.rationale}")

                st.session_state.pop("last_multi_summary", None)
                st.session_state.pop("last_multi_paths", None)
                st.session_state.pop("multi_progress", None)

            st.caption(
                f"Saved in your private folder · {short_owner_label()} "
                f"(not visible to other API keys)"
            )
            # No auto dialogs here — Run tabs / History open an in-page panel.
            # Toast avoids the old bug: stale show_run_done reopening on case/gold blur.
            try:
                n_fin = int(st.session_state.get("last_multi_n") or 1)
                if n_fin > 1:
                    st.toast(f"Multi-run ×{n_fin} finished — scroll to Results / Run tabs.", icon="✅")
                else:
                    st.toast("Judge finished — scroll to Results.", icon="✅")
            except Exception:
                pass
            st.session_state.pop("show_run_done", None)
            st.session_state.pop("kpi_dialog_armed", None)
            st.session_state.pop("confirmed_run", None)
            st.session_state.pop("pending_run", None)
            _finish_scope_run()
            st.session_state["benchmark_running"] = False
            st.rerun()


        except Exception as exc:
            try:
                cancel_run(st.session_state["_run_scope"])
                abandon_all_pipelines(st.session_state["_run_scope"])
            except Exception:
                pass
            _finish_scope_run()
            st.session_state["benchmark_running"] = False
            if n_runs > 1:
                # Keep finished run tabs; clear the forever "Waiting for all runs…" strip.
                st.session_state["multi_progress"] = finished_multi_progress(
                    completed_snaps,
                    n_total=n_runs,
                    paths=artifact_paths,
                    aborted_early=True,
                )
                try:
                    _paint_multi_progress(
                        multi_progress_slot,
                        completed_snaps,
                        n_total=n_runs,
                        batch_done=True,
                        height=160,
                    )
                except Exception:
                    pass
            elapsed = int(round(time.time() - t_run0))
            _paint_run_timer(
                timer_slot,
                _run_timer_stop(
                    elapsed,
                    this_s=elapsed,
                    n_runs=n_runs,
                    collect_s=int(round(collect_s_acc)),
                    judge_s=int(round(judge_s_acc)),
                    title="Run clock · failed",
                    phase=f"Failed · {type(exc).__name__}",
                ),
                height=220,
            )
            st.session_state["last_run_timings"] = {
                "collect_s": int(round(collect_s_acc)),
                "judge_s": int(round(judge_s_acc)),
                "total_s": elapsed,
                "mode": "full",
                "n": n_runs,
                "error": f"{type(exc).__name__}: {exc}",
            }
            phase_slot.markdown(
                f'<div class="phase-banner">Failed after {elapsed}s · timer stopped · '
                f"{type(exc).__name__}</div>",
                unsafe_allow_html=True,
            )
            _saved_n = len(artifact_paths) if artifact_paths else 0
            if _saved_n:
                _hint_paths = ", ".join(str(p) for p in artifact_paths[:3])
                if _saved_n > 3:
                    _hint_paths += f", … (+{_saved_n - 3} more)"
                st.warning(
                    f"**{_saved_n}** artifact(s) already saved before the failure — "
                    f"check History / private folder. Paths: `{_hint_paths}`"
                )
            st.error(
                f"Run failed after {elapsed}s — the clock is stopped. "
                f"**{type(exc).__name__}:** {exc}\n\n"
                "Models that already finished may still have used OpenRouter credits."
                + (
                    f"\n\n{_saved_n} run(s) were persisted before abort."
                    if _saved_n
                    else ""
                )
            )
            st.stop()

    # --- Saved run (History / Run tab) — in-page panel, never a popup ---
    _inline_path = st.session_state.get("inline_run_path")
    if _inline_path and not st.session_state.get("benchmark_running"):
        st.markdown(
            '<div class="sec-label">Saved run · in page</div>',
            unsafe_allow_html=True,
        )
        _c_close, _ = st.columns([1, 4])
        with _c_close:
            if st.button("Close panel", key="inline_run_close", use_container_width=True):
                st.session_state.pop("inline_run_path", None)
                st.session_state.pop("inline_run_kind", None)
                st.rerun()
        _render_saved_run_panel(
            str(_inline_path),
            key_prefix=f"inline_{st.session_state.get('inline_run_kind') or 'run'}",
        )

    # Persist ranking view after the run script finishes (next interactions)
    if (
        st.session_state.get("last_ranking")
        and not st.session_state.get("confirmed_run")
        and not st.session_state.get("benchmark_running")
    ):
        _ms = st.session_state.get("last_multi_summary")
        if _ms and int(_ms.get("n") or 0) > 1:
            st.markdown(
                '<div class="sec-label">Last multi-run · mean ranking</div>',
                unsafe_allow_html=True,
            )
            from benchmark.schema import MultiRunSummary as _MRS

            _sum = _MRS.model_validate(_ms)
            st.caption(reliability_caption(_sum))
            st.markdown(
                _reliability_table_html(_sum.ranking_mean), unsafe_allow_html=True
            )
            st.markdown(
                _struct_live_footer_html(
                    n_label=f"this session · N={_sum.n} runs · successful scores",
                    extra="Structured · session mean ranking",
                ),
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                fig_judge_mean_accuracy_bars(
                    _sum.ranking_mean,
                    title="Mean Clinical Composite Score",
                    height=320,
                ),
                use_container_width=True,
                key="rank_chart_saved_multi",
            )
            st.markdown(
                _struct_live_footer_html(
                    n_label=f"this session · N={_sum.n} runs · successful scores",
                    extra="Structured · session mean chart",
                ),
                unsafe_allow_html=True,
            )
            _paths = st.session_state.get("last_multi_paths") or []
            if _paths:
                st.caption("Open a finished run")
                _cols = st.columns(min(len(_paths), 5) or 1)
                for _i, _p in enumerate(_paths):
                    with _cols[_i % len(_cols)]:
                        if st.button(
                            f"Run {_i + 1}",
                            use_container_width=True,
                            key=f"saved_mrun_tab_{_i}",
                        ):
                            _arm_kpi_dialog("multi_run", path=_p)
                            st.rerun()
        else:
            st.markdown('<div class="sec-label">Last ranking</div>', unsafe_allow_html=True)
            _saved_rank = _current_ranking(st.session_state["last_ranking"] or [])
            for _r in _saved_rank:
                _r.setdefault("n_runs", 1)
            st.plotly_chart(
                fig_judge_accuracy_bars(_saved_rank, height=260),
                use_container_width=True,
                key="rank_chart_saved",
            )
            st.markdown(
                _struct_live_footer_html(
                    n_label="this session · ranking",
                    extra="Structured · session ranking",
                ),
                unsafe_allow_html=True,
            )
            rows = []
            _any_partial = False
            for r in _saved_rank:
                nm, ver = _nv(
                    r.get("key"), label=r.get("label"), model=r.get("model")
                )
                st_raw = str(r.get("status") or "ok").lower()
                is_na = st_raw in {"n/a", "na", "failed", "error"} or (
                    r.get("accuracy") is None and st_raw != "ok"
                )
                is_partial = bool(r.get("partial")) or st_raw == "partial" or is_na
                _any_partial = _any_partial or is_partial
                rank = r.get("rank")
                if is_na:
                    rank_disp = "— · partial"
                    score_disp = "N/A"
                    status_disp = "partial"
                elif is_partial and rank is not None:
                    rank_disp = f"#{rank} · partial"
                    score_disp = r.get("accuracy")
                    status_disp = "partial"
                else:
                    rank_disp = rank
                    score_disp = r.get("accuracy")
                    status_disp = "ok"
                rows.append(
                    {
                        "#": rank_disp,
                        "Name": nm,
                        "Version": ver,
                        "Clinical Composite %": score_disp,
                        "Status": status_disp,
                        "TTFT": r.get("ttft_s"),
                        "TPS": r.get("tps"),
                        "RAM(RSS)": _fmt_ram_mb(r.get("ram_mb")) or "—",
                        "GGUF": _fmt_gguf_mb(r.get("gguf_mb")) or "—",
                        "$": r.get("cost_usd"),
                        "Runs": int(r.get("n_runs") or r.get("n") or 1),
                    }
                )
            if _any_partial:
                st.markdown(
                    '<div style="margin:0.25rem 0 0.5rem;padding:0.45rem 0.7rem;'
                    "border-radius:8px;border:1px solid #f59e0b;"
                    'background:rgba(251,191,36,0.12);color:#fde68a;font-size:0.85rem">'
                    "<b style='color:#fbbf24'>partial</b> · incomplete coverage "
                    "stays ranked by mean of scored runs when available.</div>",
                    unsafe_allow_html=True,
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Only show $ after a real Single/Multi this session (not after offline rebuild)
        if st.session_state.get("show_last_run_costs") and st.session_state.get(
            "last_cost_rows"
        ):
            with st.expander("Last live run · API cost", expanded=False):
                st.dataframe(
                    pd.DataFrame(st.session_state["last_cost_rows"]),
                    use_container_width=True,
                    hide_index=True,
                )
        _tm = st.session_state.get("last_run_timings") or {}
        if _tm.get("total_s") is not None:
            n_tm = int(_tm.get("n") or 1)
            line = (
                f"Last wall time · collect {_tm.get('collect_s', '—')}s · "
                f"judge {_tm.get('judge_s', '—')}s · **total {_tm.get('total_s')}s**"
            )
            if n_tm > 1 and _tm.get("last_run_s") is not None:
                line += f" · last run {_tm.get('last_run_s')}s"
            st.caption(line)
            per = _tm.get("per_run") or []
            if n_tm > 1 and per:
                st.caption(
                    "Per run · "
                    + " · ".join(
                        f"run{p['run']} {p['total_s']}s (c{p['collect_s']}+j{p['judge_s']})"
                        for p in per
                    )
                )
        _lj = filter_current_roster_rows(
            st.session_state.get("last_judgments") or [],
            key_field="candidate_key",
        )
        if _lj:
            st.markdown(
                '<div class="sec-label">Scores by clinical dimension</div>',
                unsafe_allow_html=True,
            )
            # Rebuild matrix from saved judgments (current 9 only)
            def _j_key(j):
                return getattr(j, "candidate_key", None) or (
                    j.get("candidate_key") if isinstance(j, dict) else None
                )

            def _j_scores(j):
                if isinstance(j, dict):
                    return j.get("question_scores") or []
                return getattr(j, "question_scores", None) or []

            def _j_weighted(j):
                if isinstance(j, dict):
                    return j.get("weighted_accuracy")
                return getattr(j, "weighted_accuracy", None)

            q_ids = []
            for j in _lj:
                for qs in _j_scores(j):
                    qid = qs.get("question_id") if isinstance(qs, dict) else qs.question_id
                    if qid not in q_ids:
                        q_ids.append(qid)
            q_ids = sorted(q_ids)
            _rank_n = {
                r.get("key"): int(r.get("n_runs") or r.get("n") or 1)
                for r in _current_ranking(st.session_state.get("last_ranking") or [])
            }
            matrix_rows = []
            for j in _lj:
                ck = _j_key(j)
                nm, ver = _nv(ck)
                row = {"Name": nm, "Version": ver}
                by_q = {}
                for qs in _j_scores(j):
                    if isinstance(qs, dict):
                        by_q[qs.get("question_id")] = qs.get("score")
                    else:
                        by_q[qs.question_id] = qs.score
                for qid in q_ids:
                    row[qid] = by_q.get(qid)
                row["Clinical Composite %"] = _j_weighted(j)
                row["Coverage %"] = (
                    j.get("coverage_score")
                    if isinstance(j, dict)
                    else getattr(j, "coverage_score", None)
                )
                row["Quality %"] = (
                    j.get("quality_score")
                    if isinstance(j, dict)
                    else getattr(j, "quality_score", None)
                )
                row["Discipline %"] = (
                    j.get("discipline_score")
                    if isinstance(j, dict)
                    else getattr(j, "discipline_score", None)
                )
                row["Runs"] = _rank_n.get(ck, 1)
                matrix_rows.append(row)
            st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)

with _rebuild_zone:
    # --- Offline: Rebuild mean across homogeneous cohorts ($0 API) ---
    st.markdown(
        f'<div class="sec-label">{t("bench.rebuild_sec_label", _ui_lang())}</div>',
        unsafe_allow_html=True,
    )
    # Rebuild viz follows the default 9-roster + optional/legacy session toggles
    # (History still resolves all CURRENT_ROSTER_KEYS labels when toggled back on).
    _rebuild_optional = [
        k
        for k in OPTIONAL_LEGACY_SLOT_KEYS
        if st.session_state.get(
            {
                "local_gemma": "opt_legacy_local_gemma",
                "local_llama": "opt_legacy_local_llama",
                "qvac_4b_q8": "opt_legacy_qvac_4b_q8",
            }[k]
        )
    ]
    _rebuild_model_ids = rebuild_model_ids(_rebuild_optional)
    _rebuild_roster_n = len(_rebuild_model_ids)
    st.caption(
        "Rebuild stays below Live responses / KPIs (sequential UX). "
        f"Mean chart/table = **{len(_rebuild_model_ids)}-model roster** "
        "(default 9"
        + (
            f" + {len(_rebuild_optional)} optional/legacy"
            if _rebuild_optional
            else "; optional Gemma/Llama/Q8 hidden until toggled"
        )
        + ") · last ≤N **successful** non-zero scored runs per model · "
        "technical N/A and exact Clinical Composite == 0 treated like N/A "
        "(a rare 0 would crush the mean; usually refusal) · "
        "**No API calls** · scored-only mean."
    )

    _hist_for_case = artifacts_for_case(WORKSPACE_DIR, case_id)
    # Scope "selected case only" to the active Case slot's stem cohort history.
    _slot_scoped_arts = filter_artifacts_for_slot(
        [a for _, a in _hist_for_case],
        _active_slot,
    )
    if _active_slot.filled and _slot_scoped_arts:
        _hist_for_case = [
            (p, a)
            for p, a in _hist_for_case
            if a.run_id in {x.run_id for x in _slot_scoped_arts}
        ]
    elif _active_slot.filled and not _slot_scoped_arts:
        # Slot has a stem binding but no runs yet in History for that stem.
        _hist_for_case = []
    _rebuild_cohort_id = st.session_state.get("_restored_cohort_id") or None
    if not _rebuild_cohort_id and _active_slot.cohort_id:
        _rebuild_cohort_id = _active_slot.cohort_id
    if _rebuild_cohort_id and not any(
        a.cohort_id == _rebuild_cohort_id for _, a in _hist_for_case
    ):
        _rebuild_cohort_id = None
    # Resolve active cohort from confirmed gold on artifacts (models/track live there).
    if not _rebuild_cohort_id and effective_gold:
        try:
            _want_gold = load_confirmed_gold(effective_gold).model_dump(
                mode="json", exclude={"confirmed_at"}
            )
        except Exception:
            _want_gold = None
        _gold_cohort_counts: dict = {}
        if _want_gold is not None:
            for _, _art in _hist_for_case:
                if not _art.cohort_id:
                    continue
                _gref = str((_art.models_config or {}).get("gold_reference") or "")
                if not _gref:
                    continue
                try:
                    if load_confirmed_gold(_gref).model_dump(
                        mode="json", exclude={"confirmed_at"}
                    ) != _want_gold:
                        continue
                except Exception:
                    continue
                _gold_cohort_counts[_art.cohort_id] = (
                    int(_gold_cohort_counts.get(_art.cohort_id) or 0) + 1
                )
            if _gold_cohort_counts:
                _rebuild_cohort_id = max(
                    _gold_cohort_counts.items(), key=lambda kv: kv[1]
                )[0]
    if not _rebuild_cohort_id and _hist_for_case and _hist_for_case[0][1].cohort_id:
        _rebuild_cohort_id = _hist_for_case[0][1].cohort_id
    if _rebuild_cohort_id:
        _avail_same = sum(
            1
            for _, a in _hist_for_case
            if a.cohort_id == _rebuild_cohort_id
            and is_mean_poolable_run(a)
        )
    else:
        _avail_same = sum(
            1
            for _, a in _hist_for_case
            if is_mean_poolable_run(a)
        )

    # Portfolio eligibility: same track + v4; roster shapes may differ (per-model N).
    # Filter to active rebuild roster (9 default + opted-in optional/legacy).
    _portfolio_model_ids = list(_rebuild_model_ids)
    # All eligible docs — Rebuild N is per-model obs, not a global last-N slice.
    _portfolio_probe = list_portfolio_runs(
        WORKSPACE_DIR,
        n=None,
        scoring_version=SCORING_VERSION,
        track=str(benchmark_track or "controlled"),
        model_ids=_portfolio_model_ids,
    )
    _avail_portfolio = len(_portfolio_probe)

    # Scope control — highly visible, immediately next to N / Rebuild.
    if "history_rebuild_scope" not in st.session_state:
        st.session_state["history_rebuild_scope"] = "balanced_cases"
    if st.session_state.get("history_rebuild_scope") not in {
        "same_case",
        "portfolio",
        "balanced_cases",
    }:
        st.session_state["history_rebuild_scope"] = "balanced_cases"
    _rebuild_scope = st.radio(
        t("bench.rebuild_scope_label", _ui_lang()),
        options=["same_case", "portfolio", "balanced_cases"],
        format_func=lambda v: (
            f"● {scope_label(v, _ui_lang())} — "
            + (
                t("bench.rebuild_scope_same", _ui_lang())
                if v == "same_case"
                else (
                    t("bench.rebuild_scope_portfolio", _ui_lang())
                    if v == "portfolio"
                    else t("bench.rebuild_scope_balanced", _ui_lang())
                )
            )
        ),
        horizontal=True,
        key="history_rebuild_scope",
        on_change=_on_rebuild_n_pick_change,
    )
    st.markdown(
        honesty_block_html(
            lang=_ui_lang(),
            roster_n=_rebuild_roster_n or DEFAULT_ROSTER_VERSION,
            scope=_rebuild_scope,
            cohort_id=_rebuild_cohort_id,
        ),
        unsafe_allow_html=True,
    )
    st.caption(
        t(
            "disclosure.rebuild_scope_loud",
            _ui_lang(),
            scope=scope_label(_rebuild_scope, _ui_lang()),
        )
    )
    if _rebuild_scope in {"portfolio", "balanced_cases"}:
        st.warning(t("bench.rebuild_customs_warning", _ui_lang()))

    if _rebuild_scope == "portfolio":
        st.caption(t("bench.rebuild_portfolio_intro", _ui_lang()))
        _avail_n = _avail_portfolio
    elif _rebuild_scope == "balanced_cases":
        st.caption(t("bench.rebuild_balanced_intro", _ui_lang()))
        _avail_n = _avail_portfolio
    else:
        st.caption(
            f"**Case {_active_slot_idx}** · **Same-case** mean · one immutable cohort only"
            + (
                f" · cohort `{short_cohort(_rebuild_cohort_id)}`"
                if _rebuild_cohort_id
                else ""
            )
            + ". Cohort hash = normalized case + confirmed gold (excl. timestamp) + "
            "models + track. Other Case slots are excluded. "
            "**New cohort only if case text or locked claims change** "
            "(Confirm on the same content keeps the same set). **No API calls.**"
        )
        _avail_n = _avail_same

    # Other confirm versions in the same case family (never auto-merged).
    _other_family_runs = 0
    if _rebuild_scope == "same_case":
        if _family_cohorts and _rebuild_cohort_id:
            _other_family_runs = sum(
                int(c.get("run_count") or 0)
                for c in _family_cohorts
                if c.get("cohort_id") != _rebuild_cohort_id
            )
        elif _family_cohorts and not effective_gold:
            _other_family_runs = sum(
                int(c.get("run_count") or 0) for c in _family_cohorts
            )

    if _other_family_runs > 0 and _avail_n < 5 and _rebuild_scope == "same_case":
        st.caption(
            t(
                "bench.family_other_cohorts",
                _ui_lang(),
                n=_other_family_runs,
            )
        )

    # Per-model valid N; N=5 remains exploratory.
    _n_options = [5, 10, 20, 30, 50, 70, 100]
    _rb1, _rb2, _rb3 = st.columns([1, 1.6, 0.9])
    with _rb1:
        # Default once — value MUST stay inside options or Streamlit raises TypeError
        # ("bad argument type for built-in operation") after a mid-run reload.
        if "history_rebuild_n_pick" not in st.session_state:
            st.session_state["history_rebuild_n_pick"] = 5
        try:
            _pick = int(st.session_state.get("history_rebuild_n_pick") or 5)
        except (TypeError, ValueError):
            _pick = 5
        if _pick not in _n_options:
            st.session_state["history_rebuild_n_pick"] = 5
        _rebuild_n = st.selectbox(
            t("bench.rebuild_n_label", _ui_lang()),
            options=_n_options,
            format_func=lambda n: (
                f"≤{n} successful / model"
                + (" · exploratory" if n == 5 else "")
                + (" · steadier mean±std" if n == 10 else "")
                + (
                    " · exploratory mean±std"
                    if n in (20, 30, 50, 70)
                    else ""
                )
                + (" · max · exploratory mean±std" if n == 100 else "")
                + (f"  (only {_avail_n} eligible runs)" if _avail_n < n else "")
            ),
            key="history_rebuild_n_pick",
            on_change=_on_rebuild_n_pick_change,
            help=t("bench.rebuild_n_help", _ui_lang()),
        )
    with _rb3:
        _score_help_lang = _ui_lang()
        with st.popover(
            "Come funziona lo score?" if _score_help_lang == "it" else "How does scoring work?",
            use_container_width=True,
        ):
            if _score_help_lang == "it":
                st.markdown(
                    """
    **In parole semplici** — protocollo esplorativo amateur, **non** validazione medica.

    **Composite (voto totale)**  
    Cinque “capitoli” con pesi **non uguali**: diagnosi **30%**, safety **25%**, piano **20%**, esami **15%**, urgenza **10%**. Dentro ogni capitolo: *copertura* (hai detto le cose chiave?) ~50%, *qualità* ~35%, *disciplina* (pochissime aggiunte pericolose) ~15%. Il voto non è “a occhio”: è la media pesata di questi pezzi.

    **Media (mean) — base della classifica**  
    Con N rebuild validi: somma i punteggi e dividi per N. È il risultato “tipico”.  
    Non entrano: fallimenti tecnici (N/A) e score **esattamente 0**.  
    La classifica (#) e le bande CV usano la media (±std). Su N piccoli (5–10) è più stabile da interpretare della mediana.

    **Mediana (◆ sul grafico) — controllo di robustezza**  
    Ordina i N punteggi e prendi quello di mezzo. Su N alti (50–100) è utile se la distribuzione è storta: meno tirata dai picchi. Non sostituisce la classifica mean di default (altrimenti CV e ±std non allineano al ranking).

    **Min–max (solo tabella)**  
    Il peggiore e il migliore tra quei N run: quanto può oscillare il modello.

    **Deviazione standard (±1 std sul grafico)**  
    Quanto i punteggi si sparpagliano intorno alla media. Alta = run molto diversi tra loro.  
    Sul grafico i baffi chiari (con alone scuro) mostrano **media ± 1 std**.

    **Varianza**  
    Stessa idea della dispersione; la std è quella misura riportata in punti di score (più leggibile).

    **CV% (coefficiente di variazione)**  
    Dispersione **in % rispetto alla media** — utile per confrontare modelli con medie diverse.  
    Bande: Stable mean ≤5 · High ≤10 · Medium ≤15 · Low ≤20 · Very Low >20.  
    **Banda CV ≠ qualità clinica** e ≠ validazione ufficiale.

    **Perché i pesi**  
    Le cinque sezioni esistono perché la vignetta chiede cose diverse; i pesi riflettono priorità del protocollo (safety importante). Non è un score da cartella clinica ufficiale.
    """
                )
            else:
                st.markdown(
                    """
    **In plain words** — exploratory amateur protocol, **not** medical validation.

    **Composite (total score)**  
    Five chapters with **unequal** weights: diagnosis **30%**, safety **25%**, plan **20%**, tests **15%**, urgency **10%**. Inside each chapter: *coverage* ~50%, *quality* ~35%, *discipline* ~15%. Weighted average — not a vibe score.

    **Mean — ranking basis**  
    Average of up to N successful non-zero scored runs. Technical N/A and exact-zero skipped. Rank (#) and CV bands follow the mean (±std). At small N (5–10) mean is usually clearer than median.

    **Median (◆) — robustness check**  
    Middle of the N scores. Helpful at large N (50–100) if the distribution is skewed. Default ranking stays on mean so CV / ±std stay aligned.

    **Min–max (table only)**  
    Worst and best among those N runs.

    **Standard deviation (±1 std whiskers)**  
    How spread out scores are around the mean. Black whiskers with white outline on the chart = mean ± 1 std (readable on bars and past bar tips).

    **Variance**  
    Same “spread” idea; std is the readable score-unit version.

    **CV%**  
    Spread as a **% of the mean** — fairer when means differ. Bands: Stable mean ≤5 … Very Low >20. **CV band ≠ clinical quality / ≠ clinical validation.**

    **Why weights**  
    Five sections match five clinical asks in the vignette; weights are protocol priorities, not official chart review.
    """
                )
    _ordered_case_stems = [
        str(s.stem_key)
        for s in sorted(_case_slots, key=lambda x: int(x.index))
        if getattr(s, "stem_key", None)
    ]
    with _rb2:
        if _rebuild_scope in {"portfolio", "balanced_cases"}:
            _n_show = int(_rebuild_n) if _avail_portfolio else 0
            _k_show = (
                len(_ordered_case_stems)
                if _rebuild_scope == "balanced_cases" and _ordered_case_stems
                else (
                    count_distinct_stem_keys(a for _, a in _portfolio_probe)
                    if _avail_portfolio
                    else 0
                )
            )
            st.caption(
                t(
                    (
                        "bench.rebuild_balanced_stats"
                        if _rebuild_scope == "balanced_cases"
                        else "bench.rebuild_portfolio_stats"
                    ),
                    _ui_lang(),
                    n=_n_show,
                    cases=_k_show,
                    avail=_avail_portfolio,
                    track=str(benchmark_track or "controlled"),
                )
            )
        else:
            st.caption(
                f"Saved runs for Case {_active_slot_idx}"
                + (
                    f" · this cohort: **{_avail_n}**"
                    if _rebuild_cohort_id
                    else f": **{_avail_n}**"
                )
                + " · 5 exploratory · ~10 steadier CV · 20+ nicer but costly"
            )
        _can_rebuild = _avail_n >= 1
        _do_rebuild = st.button(
            t(
                "bench.rebuild_btn",
                _ui_lang(),
                n=_rebuild_n,
            ),
            type="primary",
            use_container_width=True,
            disabled=not _can_rebuild,
            key="history_rebuild_btn",
            help=(
                t("bench.rebuild_btn_help_portfolio", _ui_lang())
                if _rebuild_scope == "portfolio"
                else (
                    t("bench.rebuild_btn_help_balanced", _ui_lang())
                    if _rebuild_scope == "balanced_cases"
                    else t("bench.rebuild_btn_help_same", _ui_lang())
                )
            ),
        )

    if _avail_n < 1:
        if _rebuild_scope in {"portfolio", "balanced_cases"}:
            st.info(t("bench.rebuild_need_portfolio", _ui_lang(), n=_avail_n))
        else:
            st.info(
                f"Need at least **1** saved complete run for Case {_active_slot_idx} "
                f"(found {_avail_n}). Run Single once, then rebuild the mean."
            )
    elif _do_rebuild:
        # Pass requested per-model cap; rebuild loads all eligible history and trims.
        _n_use = int(_rebuild_n)
        if _avail_n < int(_rebuild_n):
            st.toast(
                f"Only {_avail_n} eligible runs saved — each model gets ≤{_avail_n} "
                f"obs (requested ≤{_rebuild_n}/model).",
                icon="ℹ️",
            )
        if _rebuild_scope == "portfolio":
            _built = rebuild_portfolio_from_history(
                WORKSPACE_DIR,
                n=_n_use,
                scoring_version=SCORING_VERSION,
                track=str(benchmark_track or "controlled"),
                model_ids=_portfolio_model_ids,
                preloaded=(
                    None
                    if getattr(RUN_STORE, "writes_plaintext", True)
                    else _preloaded_artifacts()
                ),
                pack_revision=int(_pack_rev_now),
                current_pack_revision=int(_pack_rev_now),
            )
        elif _rebuild_scope == "balanced_cases":
            _built = rebuild_balanced_cases_from_history(
                WORKSPACE_DIR,
                n=_n_use,
                scoring_version=SCORING_VERSION,
                track=str(benchmark_track or "controlled"),
                model_ids=_portfolio_model_ids,
                ordered_stem_keys=_ordered_case_stems,
                preloaded=(
                    None
                    if getattr(RUN_STORE, "writes_plaintext", True)
                    else _preloaded_artifacts()
                ),
                pack_revision=int(_pack_rev_now),
                current_pack_revision=int(_pack_rev_now),
            )
        else:
            # Selected-case only: preload slot-scoped artifacts so other Case stems
            # cannot enter the mean even when case_id is shared (caseC).
            _same_preloaded = [a for _, a in _hist_for_case]
            _built = rebuild_multi_from_history(
                WORKSPACE_DIR,
                case_id,
                n=_n_use,
                cohort_id=_rebuild_cohort_id,
                model_ids=_rebuild_model_ids,
                preloaded=_same_preloaded,
                scoring_version=SCORING_VERSION,
                pack_revision=int(_pack_rev_now),
                current_pack_revision=int(_pack_rev_now),
            )
        if not _built.get("ok"):
            st.warning(_built.get("reason") or "Rebuild failed.")
        else:
            _sum_obj = _built["summary"]
            _built["summary"] = (
                _sum_obj.model_dump()
                if hasattr(_sum_obj, "model_dump")
                else _sum_obj
            )
            st.session_state["history_rebuild_result"] = _built
            from benchmark.schema import MultiRunSummary as _MRS

            _sum_persist = _MRS.model_validate(_built["summary"])
            st.session_state["last_multi_summary"] = _sum_persist.model_dump()
            st.session_state["last_multi_paths"] = [
                pr["path"] for pr in (_built.get("per_run") or []) if pr.get("path")
            ]
            st.session_state["last_ranking"] = _mean_rows_to_last_ranking(
                _sum_persist.ranking_mean
            )
            st.session_state["last_multi_n"] = _sum_persist.n
            st.session_state["show_last_run_costs"] = False  # offline rebuild — no live $
            _arm_kpi_dialog("rebuild")
            st.rerun()

    _prev = st.session_state.get("history_rebuild_result") or {}
    _prev_scope = str(_prev.get("scope") or "same_case")
    _prev_ok_for_ui = (
        _prev.get("ok")
        and isinstance(_prev.get("summary"), dict)
        and (
            _prev_scope in {"portfolio", "balanced_cases"}
            or _prev["summary"].get("case_id") == case_id
        )
    )
    if _prev_ok_for_ui:
        _prev_n_docs = _prev.get("n_used")
        if _prev_n_docs is None:
            _prev_n_docs = (
                _prev["summary"].get("n")
                if isinstance(_prev.get("summary"), dict)
                else getattr(_prev.get("summary"), "n", "?")
            )
        if _prev_scope == "portfolio":
            _reopen_label = t(
                "bench.rebuild_reopen_portfolio",
                _ui_lang(),
                n=_prev_n_docs,
                cases=_prev.get("n_cases") or "?",
            )
        elif _prev_scope == "balanced_cases":
            _reopen_label = t(
                "bench.rebuild_reopen_balanced",
                _ui_lang(),
                n=_prev_n_docs,
                cases=_prev.get("n_cases") or "?",
            )
        else:
            _reopen_label = f"Re-open mean popup · N={_prev_n_docs} · $0"
        if st.button(
            _reopen_label,
            use_container_width=False,
            key="history_rebuild_reopen",
        ):
            _arm_kpi_dialog("rebuild")
            st.rerun()
    if _rebuild_scope == "portfolio":
        st.caption(t("bench.rebuild_portfolio_quiet", _ui_lang()))
    elif _rebuild_scope == "balanced_cases":
        st.caption(t("bench.rebuild_balanced_quiet", _ui_lang()))

st.markdown('<div class="sec-label">Run history</div>', unsafe_allow_html=True)
st.caption(
    f"Private to your OpenRouter key ({short_owner_label()}). "
    "Same authenticated account/session = same gold-only History. "
    "Other visitors with a different key cannot see your runs. "
    "Use **Rebuild mean** only across the same immutable cohort and protocol."
)
_hist_all_pairs = RUN_STORE.list_artifacts()
if not has_key:
    st.info(
        "Enter your OpenRouter API key (sidebar / welcome) to unlock **your** History. "
        "Without a key, cloud runs cannot start and History stays empty."
    )
elif not _hist_all_pairs:
    st.info("No saved runs for this API key yet — after a Single/Multi run they appear here.")
else:
    _default_path = st.session_state.get("history_path")
    _labels = []
    _path_by_label = {}
    for pth, a in _hist_all_pairs[:30]:
        try:
            when = (a.finished_at or a.started_at or "")[:19].replace("T", " ")
            top = ""
            if a.ranking:
                top_row = a.ranking[0]
                top = f" · #1 {top_row.get('key')} {top_row.get('accuracy')}%"
            lab = (
                f"{case_display_name(a.case_id)} · {when} · "
                f"${a.total_cost_usd:.3f}{top} · "
                f"{pth.name if pth is not None else a.run_id}"
            )
        except Exception:
            lab = a.run_id if hasattr(a, "run_id") else "run"
        _labels.append(lab)
        _path_by_label[lab] = str(pth) if pth is not None else f"memory:{a.run_id}"

    _idx = 0
    if _default_path:
        for i, lab in enumerate(_labels):
            if _path_by_label[lab] == _default_path:
                _idx = i
                break

    chosen = st.selectbox(
        "Saved run",
        _labels,
        index=_idx,
        key="history_main_pick",
    )
    _sel = _path_by_label[chosen]
    hist = None
    if str(_sel).startswith("memory:"):
        _rid = str(_sel).split(":", 1)[1]
        for _p, _a in RUN_STORE.list_artifacts():
            if _a.run_id == _rid:
                hist = _a
                break
        if hist is None:
            st.error("In-memory run not found.")
    else:
        hist_path = Path(_sel)
        try:
            hist = load_artifact(hist_path)
        except Exception as exc:
            st.error(f"Could not load {hist_path.name}: {exc}")
            hist = None

    if hist is not None:
        h1, h2, h3 = st.columns(3)
        h1.metric("Case", hist.case_id)
        h2.metric("Models", str(len(hist.candidates)))
        h3.metric("Cost $", f"{hist.total_cost_usd:.3f}")
        st.caption("Open a run from the sidebar History dropdown for the ranking chart popup.")

        name_by_key = {
            c.candidate_key: (c.display_label or c.label or c.candidate_key)
            for c in hist.candidates
        }
        score_by_key = {j.candidate_key: j for j in hist.judgments}

        for c in hist.candidates:
            name = name_by_key.get(c.candidate_key, c.candidate_key)
            j = score_by_key.get(c.candidate_key)
            acc = f"{j.weighted_accuracy}%" if j else "—"
            with st.expander(f"{name} · {acc}", expanded=False):
                if c.meta and (c.meta.ttft_s is not None or c.meta.tps is not None):
                    st.caption(
                        f"TTFT {c.meta.ttft_s}s · TPS {c.meta.tps} · "
                        f"${c.meta.cost_usd or 0:.4f}"
                        + (f" · err {c.meta.error}" if c.meta.error else "")
                    )
                if c.raw_response:
                    st.markdown("**Full answer**")
                    _dlg_full_text(c.raw_response)
                elif c.answers:
                    for qid, ans in c.answers.items():
                        st.markdown(f"**{qid}**")
                        _dlg_full_text(ans or "")
                if j:
                    st.markdown("**Judge**")
                    for qs in j.question_scores:
                        st.caption(
                            f"{qs.question_id}: {qs.score}/100 — {qs.rationale}"
                        )

        same_case = []
        for _pth, a in _hist_all_pairs:
            if a.case_id == hist.case_id and a.ranking:
                same_case.append(a)
        if len(same_case) >= 2:
            st.caption(
                f"{len(same_case)} saved runs for {case_display_name(hist.case_id)} — "
                "use **Rebuild mean across N runs** above for the ranking table "
                "(same-cohort protocol, $0 API)."
            )
