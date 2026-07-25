"""Demo studio UI — DeepSeek R1 judge, cost transparency, QVAC SDK check."""

from __future__ import annotations

import html
import json
import os
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from benchmark.config import is_usable_openrouter_key, load_models_config
from benchmark.cases_loader import case_display_name, list_case_ids, load_case
from benchmark.workspace import (
    assert_path_in_workspace,
    maybe_claim_legacy_root_artifacts,
    owner_id_for_current_key,
    scoped_artifacts_dir,
    short_owner_label,
)
from lib.ip_key_store import (
    client_identity,
    identity_caption,
    is_streamlit_cloud,
    load_key_for_client,
    save_key_for_client,
)
from benchmark.judge import (
    build_ranking,
    explain_run_scores,
    judge_candidates_parallel,
    systemic_judge_failure,
)
from benchmark.qvac_bridge import available as qvac_available
from benchmark.qvac_bridge import ensure_sidecar as qvac_ensure_sidecar
from benchmark.qvac_bridge import health as qvac_health
from benchmark.qvac_bridge import reachable as qvac_reachable
from benchmark.qvac_bridge import iter_tokens as qvac_iter_tokens
from benchmark.prompts import candidate_system, candidate_user
from benchmark.report import (
    artifacts_for_case,
    list_run_artifacts,
    load_artifact,
    print_summary_table,
    rebuild_multi_from_history,
    reliability_caption,
    summarize_runs,
    write_artifact,
    write_summary,
)
from benchmark.runner import (
    dry_run_estimate,
    estimate_cost_breakdown,
    iter_collect_live,
    iter_collect_parallel,
    prepare_run,
)
from benchmark.schema import Case, RunArtifact, utc_now_iso
from lib.benchmark_multi_ui import (
    client_toast_run_done,
    progressive_multi_panel_html,
    reliability_badge,
    short_model,
    snapshot_from_artifact,
)
from lib.charts import fig_judge_accuracy_bars, fig_judge_mean_accuracy_bars
import streamlit.components.v1 as components

st.set_page_config(
    page_title="QVAC vs Cloud · Automated Benchmark",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

LIVE_BOX_H = 168

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
# silently spend visitor A's credits. Prefill comes only from the per-IP vault
# (or this browser session after Save). Local .env still helps the developer machine.
_server_env_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
if is_streamlit_cloud():
    os.environ.pop("OPENROUTER_API_KEY", None)
    _server_env_key = ""

# Use st.html (not st.markdown) so blank lines inside <style> cannot leak CSS as page text.
st.html(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* Dark theme — high contrast, do not break Streamlit Material icons */
html, body, .stApp, .stMarkdown, p, label, .stCaption {
  font-family: "IBM Plex Sans", system-ui, sans-serif !important;
}
/* CRITICAL: never force font-family on all spans/[class*=css] — breaks Material icon ligatures */
.stApp {
  background: #0b1220 !important;
  color: #e2e8f0 !important;
}
/* Kill Streamlit top chrome — opaque bar was clipping the hero title */
header[data-testid="stHeader"],
.stApp > header {
  display: none !important;
}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
  display: none !important;
}
/* Content can sit near the top once chrome is hidden */
.block-container {
  padding-top: 1.25rem !important;
  padding-bottom: 0.7rem !important;
  max-width: 1440px;
}
[data-testid="stMainBlockContainer"] {
  padding-top: 0.5rem !important;
}
[data-testid="stAppViewContainer"] > section.main > div {
  padding-top: 0.35rem !important;
}
div[data-testid="stVerticalBlock"] > div { gap: 0.2rem !important; }

/* Custom spend confirm modal (NOT st.dialog — those stick grey + X aborts in-flight API calls) */
.spend-modal-marker { display: none; }
div[data-testid="stVerticalBlock"]:has(.spend-modal-marker) {
  position: fixed !important;
  inset: 0 !important;
  z-index: 100000 !important;
  background: rgba(2, 6, 23, 0.78) !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 1.25rem !important;
  gap: 0.55rem !important;
  margin: 0 !important;
  max-width: none !important;
}
div[data-testid="stVerticalBlock"]:has(.spend-modal-marker) > div {
  width: min(440px, 94vw) !important;
  max-width: 440px !important;
}
.spend-modal-card {
  background: #111827;
  border: 1px solid #475569;
  border-radius: 14px;
  padding: 1.15rem 1.25rem 0.35rem;
  box-shadow: 0 24px 64px rgba(0,0,0,0.55);
}
.spend-modal-card h3 {
  margin: 0 0 0.55rem 0 !important;
  font-size: 1.08rem !important;
  font-weight: 700 !important;
  color: #f8fafc !important;
}
.spend-modal-card p {
  margin: 0 0 0.45rem 0 !important;
  font-size: 0.86rem !important;
  line-height: 1.45 !important;
  color: #cbd5e1 !important;
}
.run-timer-panel {
  position: relative !important;
  z-index: 1 !important;
  background: linear-gradient(165deg, #1c1917 0%, #0f172a 55%, #111827 100%) !important;
  border: 1px solid #f59e0b !important;
  color: #fde68a !important;
  font-family: "IBM Plex Mono", ui-monospace, monospace !important;
  padding: 0.5rem 0.6rem 0.45rem !important;
  border-radius: 10px !important;
  box-shadow: none !important;
  margin: 0 !important;
  width: 100% !important;
  box-sizing: border-box !important;
}
/* Timer sits in normal flow at the END of the sidebar — never sticky/fixed overlay */
.sidebar-timer-dock {
  margin-top: 1.35rem !important;
  padding-top: 0.65rem !important;
  border-top: 1px solid #334155 !important;
  background: #070b14 !important;
  position: relative !important;
  bottom: auto !important;
  z-index: 1 !important;
  clear: both !important;
}
.sidebar-timer-spacer {
  height: 0.75rem !important;
  margin: 0 !important;
  padding: 0 !important;
}
[data-testid="stSidebar"] {
  overflow-x: hidden !important;
}
[data-testid="stSidebar"] iframe {
  max-width: 100% !important;
  width: 100% !important;
  margin: 0 !important;
}
/* Collapsed sidebar must not leave a timer iframe over the main left column */
section[data-testid="stSidebar"][aria-expanded="false"] iframe,
[data-testid="stSidebar"][aria-expanded="false"] iframe {
  display: none !important;
  height: 0 !important;
  visibility: hidden !important;
}
[data-testid="stSidebar"] .element-container:has(.sidebar-timer-dock),
[data-testid="stSidebar"] .element-container:has(.sidebar-timer-spacer) {
  margin-top: 0 !important;
}
.run-timer-panel .t-title {
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #94a3b8 !important;
  margin: 0 0 0.5rem 0 !important;
}
.run-timer-panel .t-big {
  font-size: 1.95rem !important;
  font-weight: 700 !important;
  line-height: 1.1 !important;
  color: #fbbf24 !important;
  margin: 0 0 0.45rem 0 !important;
  white-space: nowrap !important;
}
.run-timer-panel .t-min {
  font-size: 0.52em !important;
  font-weight: 500 !important;
  opacity: 0.75 !important;
  letter-spacing: -0.02em;
  white-space: nowrap !important;
}
.run-timer-panel .t-row {
  display: flex !important;
  justify-content: space-between !important;
  align-items: baseline !important;
  gap: 0.35rem;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  margin: 0.18rem 0 !important;
  color: #fde68a !important;
  flex-wrap: nowrap !important;
}
.run-timer-panel .t-row .lab {
  color: #94a3b8 !important;
  font-weight: 500 !important;
  font-size: 0.82rem !important;
  flex-shrink: 0;
}
.run-timer-panel .t-row .val {
  font-variant-numeric: tabular-nums;
  color: #fef3c7 !important;
  white-space: nowrap !important;
  flex-shrink: 0;
}
.run-timer-panel .t-row .val .t-min {
  font-size: 0.68em !important;
}
.run-timer-panel .t-row.active .val {
  color: #fbbf24 !important;
}
.run-timer-panel .t-sep {
  border: 0;
  border-top: 1px solid #334155;
  margin: 0.45rem 0 !important;
}
.run-timer-panel .phase {
  display: block;
  font-size: 0.74rem !important;
  font-weight: 500 !important;
  color: #cbd5e1 !important;
  margin-top: 0.4rem;
  line-height: 1.3;
}
.run-timer-panel.idle .t-big { color: #64748b !important; }
.run-timer-panel.idle .phase { color: #64748b !important; }
/* legacy class kept for stop markup compatibility */
.run-timer-overlay { /* unused floating mode */ display: none !important; }
.spend-modal-card .muted {

  font-size: 0.72rem !important;
  color: #94a3b8 !important;
}

[data-testid="stSidebar"] {
  background: #070b14 !important;
  min-width: 220px !important;
  max-width: 260px !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 0.45rem !important;
  padding-bottom: 0.5rem !important;
  padding-left: 0.55rem !important;
  padding-right: 0.55rem !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  gap: 0.18rem !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  color: #e2e8f0 !important;
  font-size: 0.72rem !important;
  line-height: 1.25 !important;
  margin: 0 !important;
}
[data-testid="stSidebar"] h2 {
  font-size: 0.85rem !important;
  margin: 0.1rem 0 0.15rem !important;
}
[data-testid="stSidebar"] h3 {
  font-size: 0.78rem !important;
  margin: 0.1rem 0 0.12rem !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  font-size: 0.64rem !important;
  line-height: 1.25 !important;
}
[data-testid="stSidebar"] .stButton button {
  min-height: 1.7rem !important;
  font-size: 0.7rem !important;
  padding: 0.15rem 0.4rem !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="input"] {
  font-size: 0.72rem !important;
  min-height: 1.7rem !important;
}
[data-testid="stSidebar"] hr {
  margin: 0.35rem 0 !important;
}
[data-testid="stSidebar"] div[data-testid="stAlert"] {
  padding: 0.28rem 0.4rem !important;
  font-size: 0.68rem !important;
}
[data-testid="stSidebar"] div[data-testid="stAlert"] p {
  font-size: 0.68rem !important;
  line-height: 1.25 !important;
}
/* Restore Material Symbols used by expanders / widgets */
[data-testid="stIconMaterial"],
span[data-testid="stIconMaterial"],
.material-icons,
.material-symbols-rounded {
  font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
  font-style: normal !important;
  font-weight: 400 !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  speak: never;
}
.demo-hero {
  font-size: 1.12rem !important; font-weight: 750 !important;
  margin: 0 !important; color: #f8fafc !important;
  letter-spacing: -0.02em;
}
.demo-sub {
  color: #94a3b8 !important; font-size: 0.7rem !important;
  margin: 0 0 0.3rem !important;
}
.steps-bar { display: flex; gap: 0.3rem; margin: 0 0 0.35rem; }
.step-pill {
  flex: 1; padding: 0.25rem 0.45rem; border-radius: 8px;
  background: #111827; color: #e2e8f0; font-size: 0.66rem; line-height: 1.25;
  border: 1px solid #334155;
}
.step-pill b { color: #5eead4; margin-right: 0.28rem; font-weight: 700; }
.phase-banner {
  padding: 0.32rem 0.6rem; border-radius: 8px; margin: 0.2rem 0;
  background: #0f766e; color: #ffffff !important; font-size: 0.76rem; font-weight: 650;
}
.sec-label {
  font-size: 0.6rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
  color: #5eead4 !important; margin: 0.08rem 0 0.12rem !important;
}
.model-chip {
  display: block; padding: 0.26rem 0.42rem; border-radius: 8px;
  font-size: 0.66rem; font-weight: 700; border: 1.5px solid; line-height: 1.2;
  background: #111827 !important; color: #f1f5f9 !important;
}
.model-chip span { display: block; font-weight: 500; color: #94a3b8 !important; font-size: 0.56rem; margin-top: 0.06rem; }
.cost-compact {
  font-size: 0.64rem; color: #cbd5e1 !important; margin-top: 0.18rem; line-height: 1.3;
  padding: 0.26rem 0.4rem; border-radius: 7px;
  background: #111827; border: 1px solid #334155;
}
.cost-compact b { color: #f8fafc !important; }
.cost-multi { text-align: center; font-size: 0.7rem; }
.status-pill {
  display: inline-block; font-size: 0.6rem; font-weight: 700;
  padding: 0.08rem 0.38rem; border-radius: 999px; margin-bottom: 0.1rem;
}
.status-pill.ready { background: #164e63; color: #a5f3fc !important; }
.status-pill.wait { background: #7c2d12; color: #fed7aa !important; }
.status-pill.done { background: #14532d; color: #bbf7d0 !important; }
.status-pill.err { background: #7f1d1d; color: #fecaca !important; }
.status-pill.skip { background: #334155; color: #e2e8f0 !important; }
.kpi-row {
  font-family: "IBM Plex Mono", ui-monospace, monospace !important;
  font-size: 0.6rem !important; color: #5eead4 !important; margin: 0.04rem 0 0.1rem !important;
}
.kpi-row.live { color: #fdba74 !important; }
.live-head {
  font-size: 0.76rem !important; font-weight: 700 !important; margin: 0 !important;
  color: #f8fafc !important;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.live-meta { font-size: 0.56rem !important; color: #94a3b8 !important; margin: 0 0 0.1rem !important; }
.panel-frame {
  border: 1px solid #334155; border-radius: 10px; padding: 0.3rem 0.35rem 0.18rem;
  background: #111827; min-height: 100%;
}
.stream-wrap { position: relative; }
.stream-toolbar {
  display: flex; justify-content: flex-end; margin: 0 0 0.15rem 0;
}
.stream-fs-lab {
  cursor: pointer; user-select: none;
  font-size: 0.62rem !important; font-weight: 700 !important;
  color: #5eead4 !important; padding: 0.12rem 0.4rem;
  border: 1px solid #334155; border-radius: 6px; background: #0f172a;
}
.stream-fs-lab:hover { border-color: #2dd4bf; color: #a5f3fc !important; }
.fs-ck { position: absolute !important; opacity: 0 !important; pointer-events: none !important; width: 0 !important; height: 0 !important; }
.fs-overlay {
  display: none; position: fixed; inset: 0; z-index: 100000;
  background: rgba(2, 6, 23, 0.88); padding: 1rem;
  align-items: stretch; justify-content: center;
}
.fs-ck:checked + .fs-overlay { display: flex !important; }
.fs-card {
  flex: 1; max-width: 1100px; margin: 0 auto;
  background: #0f172a; border: 1px solid #334155; border-radius: 12px;
  display: flex; flex-direction: column; min-height: 0; max-height: 100%;
  box-shadow: 0 20px 50px rgba(0,0,0,0.45);
}
.fs-bar {
  display: flex; justify-content: space-between; align-items: center;
  gap: 0.75rem; padding: 0.65rem 0.85rem;
  border-bottom: 1px solid #334155; color: #e2e8f0;
  font-size: 0.85rem; font-weight: 700;
}
.fs-close {
  cursor: pointer; font-size: 1.15rem; line-height: 1;
  color: #f8fafc !important; padding: 0.15rem 0.55rem;
  border-radius: 8px; background: #334155; border: 1px solid #475569;
}
.fs-close:hover { background: #7f1d1d; border-color: #ef4444; }
.fs-pre {
  flex: 1; overflow: auto; margin: 0; padding: 1rem 1.1rem;
  white-space: pre-wrap; word-break: break-word;
  font-family: "IBM Plex Mono", ui-monospace, monospace !important;
  font-size: 0.85rem !important; line-height: 1.45 !important;
  color: #e2e8f0 !important; background: #020617 !important;
}
/* Sidebar / inline openers for client-side guides (no Streamlit rerun) */
label.guide-open-btn {
  display: block !important; width: 100% !important; box-sizing: border-box !important;
  cursor: pointer !important; text-align: center !important;
  margin: 0.2rem 0 !important; padding: 0.45rem 0.6rem !important;
  border-radius: 8px !important; border: 1px solid #334155 !important;
  background: #1e293b !important; color: #f8fafc !important;
  font-size: 0.82rem !important; font-weight: 650 !important;
}
label.guide-open-btn:hover {
  border-color: #2dd4bf !important; color: #a5f3fc !important;
}
.guide-body {
  flex: 1; overflow: auto; margin: 0; padding: 1rem 1.15rem 1.25rem;
  color: #e2e8f0 !important; font-size: 0.88rem !important; line-height: 1.45 !important;
  background: #020617 !important;
}
.guide-body h3 { color: #5eead4 !important; font-size: 1rem !important; margin: 0.9rem 0 0.35rem !important; }
.guide-body h3:first-child { margin-top: 0 !important; }
.guide-body p, .guide-body li { color: #e2e8f0 !important; margin: 0.25rem 0 !important; }
.guide-body ul { margin: 0.2rem 0 0.5rem 1.1rem !important; padding: 0 !important; }
.guide-body code, .guide-body pre {
  background: #111827 !important; color: #fde68a !important;
  border-radius: 6px; padding: 0.15rem 0.35rem;
  font-size: 0.8rem !important;
}
.guide-body pre {
  display: block; padding: 0.65rem 0.75rem; white-space: pre-wrap;
  border: 1px solid #334155; margin: 0.4rem 0 0.7rem;
}
.guide-body table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; font-size: 0.8rem; }
.guide-body th, .guide-body td {
  border: 1px solid #334155; padding: 0.35rem 0.45rem; text-align: left; color: #e2e8f0 !important;
}
.guide-body th { background: #1e293b; color: #5eead4 !important; }
.stream-out {
  font-family: "IBM Plex Mono", ui-monospace, monospace !important;
  font-size: 0.68rem !important; line-height: 1.35 !important; white-space: pre-wrap;
  background: #020617 !important; color: #e2e8f0 !important; border-radius: 8px;
  padding: 0.42rem 0.48rem; min-height: 168px; max-height: 240px; overflow: auto;
  border: 1px solid #1e293b;
}
.stream-out .caret {
  display: inline-block; width: 0.4em; height: 0.9em; vertical-align: text-bottom;
  background: #2dd4bf; animation: caretBlink 0.85s step-end infinite; margin-left: 1px;
}
@keyframes caretBlink { 50% { opacity: 0; } }
/* Buttons: ALWAYS light text — fixes secondary dark-on-dark (contrast~1.0) */
div[data-testid="stButton"] button,
div[data-testid="stButton"] button p,
div[data-testid="stButton"] button span {
  color: #f8fafc !important;
  border-radius: 8px !important;
  min-height: 2.05rem !important;
  opacity: 1 !important;
}
div[data-testid="stButton"] button[kind="primary"] {
  background: #0f766e !important;
  border: 1px solid #14b8a6 !important;
  font-weight: 650 !important;
}
div[data-testid="stButton"] button[kind="secondary"] {
  background: #1e293b !important;
  border: 1px solid #64748b !important;
}
/* Run QVAC only — vivid amber (clearly not Single/Multi teal/slate) */
div.st-key-qvac_only_btn button,
div[class*="st-key-qvac_only_btn"] button,
div[class*="st-key-qvac_only_btn"] button[kind="secondary"],
div[class*="st-key-qvac_only_btn"] button[data-testid="baseButton-secondary"] {
  background: #f59e0b !important;
  background-color: #f59e0b !important;
  border: 2px solid #fbbf24 !important;
  color: #1c1917 !important;
  font-weight: 750 !important;
}
div.st-key-qvac_only_btn button p,
div.st-key-qvac_only_btn button span,
div[class*="st-key-qvac_only_btn"] button p,
div[class*="st-key-qvac_only_btn"] button span {
  color: #1c1917 !important;
}
div.st-key-qvac_only_btn button:hover:not(:disabled),
div[class*="st-key-qvac_only_btn"] button:hover:not(:disabled) {
  background: #fbbf24 !important;
  background-color: #fbbf24 !important;
  border-color: #fde68a !important;
}
div.st-key-qvac_only_btn button:disabled,
div[class*="st-key-qvac_only_btn"] button:disabled {
  background: #78350f !important;
  border-color: #a16207 !important;
  color: #fde68a !important;
}
div.st-key-qvac_only_btn button:disabled p,
div.st-key-qvac_only_btn button:disabled span,
div[class*="st-key-qvac_only_btn"] button:disabled p,
div[class*="st-key-qvac_only_btn"] button:disabled span {
  color: #fde68a !important;
}
div[data-testid="stButton"] button:disabled {
  opacity: 0.55 !important;
}
div[data-testid="stExpander"] details {
  border: 1px solid #334155; border-radius: 8px; background: #111827;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p,
div[data-testid="stExpander"] summary span:not([data-testid="stIconMaterial"]) {
  color: #e2e8f0 !important;
}
textarea, .stTextArea textarea {
  border-radius: 8px !important;
  background: #0f172a !important;
  color: #f1f5f9 !important;
  border: 1px solid #475569 !important;
  font-size: 0.76rem !important;
}
.stCaption, [data-testid="stCaption"] { color: #94a3b8 !important; font-size: 0.66rem !important; }
div[data-testid="stAlert"] { padding: 0.35rem 0.55rem !important; }
/* Select / number / checkbox readable on dark */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {
  background-color: #0f172a !important;
  color: #f1f5f9 !important;
  border-color: #475569 !important;
}
[data-testid="stWidgetLabel"] p { color: #cbd5e1 !important; }
</style>
"""
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
qvac_run_ok = bool(qvac_ok)

# Per-IP remembered key (same PC/network → prefilled; other IP → empty)
_ip_saved_key = load_key_for_client()
if _ip_saved_key and is_usable_openrouter_key(_ip_saved_key):
    if not st.session_state.get("or_key_session"):
        st.session_state["or_key_session"] = _ip_saved_key
elif (
    (not is_streamlit_cloud())
    and client_identity() == "local"
    and is_usable_openrouter_key(_server_env_key)
    and not st.session_state.get("or_key_session")
    and not _ip_saved_key
):
    # Local install convenience: .env prefills only on this machine (identity=local)
    st.session_state["or_key_session"] = _server_env_key

_raw_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
# Drop unusable placeholders left in the process env
if _raw_key and not is_usable_openrouter_key(_raw_key):
    os.environ.pop("OPENROUTER_API_KEY", None)
    _raw_key = ""
has_key = is_usable_openrouter_key(_raw_key)
if st.session_state.get("or_key_session") and is_usable_openrouter_key(
    st.session_state["or_key_session"]
):
    os.environ["OPENROUTER_API_KEY"] = st.session_state["or_key_session"]
    has_key = True
elif st.session_state.get("or_key_session") and not is_usable_openrouter_key(
    st.session_state["or_key_session"]
):
    st.session_state.pop("or_key_session", None)
    os.environ.pop("OPENROUTER_API_KEY", None)
    has_key = False
else:
    # No session key for this visitor — do not keep a leftover process env key
    if not has_key:
        os.environ.pop("OPENROUTER_API_KEY", None)

# Private History: runs live under artifacts/owners/<sha256(key)[:24]>/
# Same OpenRouter key → same history (login). Different key → empty / own runs only.
# Always pull leftover root-level JSON (pre-owners layout) into this workspace —
# that is where the ~6 Custom Case (caseC) runs still lived.
WORKSPACE_DIR = scoped_artifacts_dir()
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
WORKSPACE_DIR = scoped_artifacts_dir()

# --- Startup dialogs: every browser refresh starts a new session → show again ---
# Within one session: API key popup first, then QVAC status (good for screen recordings).
if "boot_welcome_done" not in st.session_state:
    st.session_state.boot_welcome_done = False
if "boot_step" not in st.session_state:
    st.session_state.boot_step = "api"  # api → qvac → done
# Legacy flags kept in sync for any older checks
if "qvac_dialog_shown" not in st.session_state:
    st.session_state.qvac_dialog_shown = False
if "key_dialog_shown" not in st.session_state:
    st.session_state.key_dialog_shown = False

# Clear sticky run flag when idle (a Streamlit dialog X used to abort mid-flight).
if not st.session_state.get("confirmed_run") and not st.session_state.get("pending_run"):
    st.session_state["benchmark_running"] = False


def _mask_api_key(key: str) -> str:
    """Show start + end only (middle hidden) for video / confirmation UI."""
    k = (key or "").strip()
    if not k:
        return "(none)"
    if len(k) <= 16:
        return "•" * min(len(k), 12)
    return f"{k[:10]}…{'•' * 8}…{k[-4:]}"


def _advance_boot(to: str) -> None:
    st.session_state.boot_step = to
    if to == "done":
        st.session_state.boot_welcome_done = True
        st.session_state.key_dialog_shown = True
        st.session_state.qvac_dialog_shown = True
    st.rerun()


def _client_guide_overlay(uid: str, title: str, body_html: str) -> str:
    """Fullscreen guide overlay toggled by <label for=uid> — no Streamlit rerun."""
    u = html.escape(uid)
    t = html.escape(title)
    return f"""
<input type="checkbox" id="{u}" class="fs-ck" />
<div class="fs-overlay">
  <div class="fs-card">
    <div class="fs-bar">
      <span>{t}</span>
      <label for="{u}" class="fs-close" title="Close">✕</label>
    </div>
    <div class="guide-body">{body_html}</div>
  </div>
</div>
"""


def _guides_always_available_html(*, qvac_status_line: str = "") -> str:
    """Inject Setup + Ranking guides once in main DOM (sidebar labels toggle these)."""
    setup_status = html.escape(qvac_status_line or "")
    setup_body = f"""
<h3>What this benchmark uses for MedPsy</h3>
<ul>
  <li><b>QVAC SDK</b> (<code>@qvac/sdk</code>) via local <code>sidecar/</code></li>
  <li><b>MedPsy-4B GGUF</b> under <code>models/</code> (GPU/Metal preferred)</li>
  <li><b>Node.js ≥ 22</b> to run the sidecar</li>
</ul>
<h3>Setup after cloning</h3>
<ol>
  <li>Install Node.js ≥ 22 from nodejs.org</li>
  <li>Place MedPsy GGUF in <code>models/</code> (or set <code>QVAC_MODEL_PATH</code>)</li>
  <li>From repo root, in a second terminal:</li>
</ol>
<pre>./scripts/setup_qvac_sidecar.sh
cd sidecar &amp;&amp; npm start</pre>
<p>Leave that terminal open, then refresh this page.
Check <code>curl -s http://127.0.0.1:8787/health</code>.</p>
<p>When the sidecar is running, MedPsy is included (on-device, $0 API).</p>
<p><b>Status on this machine:</b> {setup_status}</p>
<p style="opacity:.8;font-size:0.8rem">This window is browser-only — opening it does <b>not</b> pause collect/judge.</p>
"""
    rank_body = """
<h3>How ranking works</h3>
<p>Blind DeepSeek R1 · host recomputes scores · literal 100% is not used ·
four models always get distinct accuracies.</p>
<pre>Gold: 100 × (0.50·alignment + 0.30·quality + 0.20·stem)
Rubric: 100 × (0.30·must + 0.20·acceptable + 0.40·quality + 0.10·stem)
→ capped at 96.5
Accuracy = Σ (section_weight × section_score) · run cap ≈ 97%</pre>
<table>
  <tr><th>Param (gold)</th><th>Wt</th><th>Meaning</th></tr>
  <tr><td>alignment</td><td>50%</td><td>Semantic closeness to gold thesis (synonyms / near-equivalents OK)</td></tr>
  <tr><td>quality</td><td>30%</td><td>Clinical judgment 0–1 (not writing style)</td></tr>
  <tr><td>stem</td><td>20%</td><td>Case-specific anchors (anti-generic paste)</td></tr>
</table>
<p>Judge scores <b>clinical meaning</b> — not exact keywords or acronyms.</p>
<h3>Gold vs rubric</h3>
<ul>
  <li><b>GOLD pasted</b> (Custom Case required; Demo 1/2 optional) → 0–100 vs that diagnosis thesis</li>
  <li><b>Gold empty</b> (Demo cases) → teaching rubric in the case JSON wins</li>
</ul>
<p>Strong answers can land ~80–95%. Tie-break: safety → quality → stem → diagnosis.
Multi ×N / Rebuild mean: Mean ± std, CV% reliability (High / Medium / Low).</p>
<p style="opacity:.8;font-size:0.8rem">This window is browser-only — opening it does <b>not</b> pause collect/judge.</p>
"""
    return (
        _client_guide_overlay("guide_setup", "QVAC SDK + MedPsy setup guide", setup_body)
        + _client_guide_overlay("guide_rank", "How ranking is calculated", rank_body)
    )


QVAC_SETUP_GUIDE = """
### What this benchmark uses for MedPsy

- **QVAC SDK** (`@qvac/sdk`) via the local `sidecar/` server on this computer  
- **MedPsy-4B GGUF** under `models/` — inference prefers **GPU/Metal** (`gpu_layers=99`)  
- **Node.js ≥ 22** to run the sidecar  

### Setup (any Mac/PC after cloning this repo)

1. Install Node.js ≥ 22 from https://nodejs.org/  
2. Place the MedPsy GGUF in `models/` (or set `QVAC_MODEL_PATH`)  
3. From the **repository root**, in a second terminal:

```bash
./scripts/setup_qvac_sidecar.sh
cd sidecar && npm start
```

4. Leave that terminal open, then refresh this page.  
   Check `curl -s http://127.0.0.1:8787/health` → should show `"device":"gpu"` (or `cpu` only after a Metal fallback).

When the sidecar is running, MedPsy is included (on-device, $0 API).  
If GPU load fails on some Macs, the sidecar retries on CPU automatically.
"""


@st.dialog("QVAC SDK + MedPsy setup guide")
def qvac_setup_guide_dialog():
    st.markdown(QVAC_SETUP_GUIDE)
    st.markdown(
        f"**Status on this machine right now:** "
        f"{'ready — MedPsy will be included' if qvac_available() else ('sidecar online · MedPsy not loaded' if qvac_reachable() else 'sidecar offline — cloud-only until you start it')}"
    )
    if st.button("Close", type="primary", use_container_width=True):
        st.session_state["show_qvac_guide"] = False
        st.rerun()


@st.dialog("QVAC SDK / MedPsy status")
def qvac_status_dialog(online: bool, loaded: bool):
    if loaded:
        st.success(
            "MedPsy is **active** through the QVAC SDK on this machine. "
            "It will be included in the benchmark (on-device, $0 API)."
        )
        st.caption("Stack: QVAC SDK sidecar · MedPsy-4B GGUF · stock inference settings.")
    elif online:
        st.warning(
            "QVAC sidecar is **online**, but MedPsy is **not loaded** yet. "
            "Cloud-only until the model finishes loading "
            "(or until you fix OpenSSL / restart the sidecar)."
        )
        st.markdown(QVAC_SETUP_GUIDE)
    else:
        st.warning(
            "QVAC sidecar is **offline** on this computer. "
            "This run will use **cloud models only** (ChatGPT / Claude / Gemini via OpenRouter)."
        )
        st.markdown(QVAC_SETUP_GUIDE)
    if st.button("OK · continue", type="primary", use_container_width=True, key="qvac_boot_ok"):
        _advance_boot("done")


def _remember_openrouter_key(key: str) -> None:
    """Activate key for this browser session and remember it for this IP only."""
    key = (key or "").strip()
    if not is_usable_openrouter_key(key):
        return
    os.environ["OPENROUTER_API_KEY"] = key
    st.session_state["or_key_session"] = key
    saved = save_key_for_client(key)
    st.session_state["_ip_key_remembered"] = bool(saved)


@st.dialog("OpenRouter API key")
def key_welcome_dialog():
    """Shown on every fresh page load — prefilled only if this IP saved a key before."""
    # Prefer per-IP vault (survives refresh). Never prefill from a shared Cloud Secret.
    existing = (load_key_for_client() or st.session_state.get("or_key_session") or "").strip()
    if existing and not is_usable_openrouter_key(existing):
        existing = ""

    st.markdown(
        "This app uses **bring-your-own-key**. Confirm or paste your **full** OpenRouter key "
        "for cloud models + DeepSeek R1 judge. "
        "Or continue without a key to rehearse **Run QVAC only · $0** (no ranking)."
    )
    st.caption(identity_caption())
    if existing:
        st.info(f"Key remembered for this IP (hidden): `{_mask_api_key(existing)}`")
    else:
        st.caption(
            "No key on file for this IP — field stays empty so other visitors "
            "cannot use your OpenRouter credits."
        )
    st.caption(
        "Field below hides characters (••••). Keys with `...` placeholders get HTTP 401. "
        "https://openrouter.ai/keys"
    )

    # Prefill password widget once so reload shows dots, not an empty box
    if "dialog_or_key" not in st.session_state:
        st.session_state["dialog_or_key"] = existing
    k = st.text_input(
        "OPENROUTER_API_KEY",
        type="password",
        key="dialog_or_key",
        help="Characters stay hidden. Replace to change the key.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Continue without key", use_container_width=True, key="boot_key_skip"):
            _advance_boot("qvac")
    with c2:
        if st.button(
            "Use saved key",
            use_container_width=True,
            disabled=not bool(existing),
            key="boot_key_keep",
        ):
            _remember_openrouter_key(existing)
            _advance_boot("qvac")
    with c3:
        if st.button("Save / update key", type="primary", use_container_width=True, key="boot_key_save"):
            typed = (k or "").strip()
            if is_usable_openrouter_key(typed):
                _remember_openrouter_key(typed)
                if not st.session_state.get("_ip_key_remembered") and is_streamlit_cloud():
                    st.warning(
                        "Saved for this browser session only — could not bind to an IP "
                        "(proxy). Other visitors still start with an empty field."
                    )
                _advance_boot("qvac")
            elif existing and (not typed or typed == existing):
                _remember_openrouter_key(existing)
                _advance_boot("qvac")
            else:
                st.error(
                    "Enter a complete OpenRouter key starting with sk-or-v1-… "
                    "(no dots/ellipsis in the middle)."
                )


@st.dialog("Saved run results", width="large")
def history_run_dialog(path_str: str):
    """Popup review of a past artifact (from sidebar History)."""
    _hp = Path(path_str)
    if not assert_path_in_workspace(_hp, WORKSPACE_DIR):
        st.error("That run is not in your private history (API key mismatch).")
        if st.button("Close", type="primary", use_container_width=True, key="hist_dlg_close_deny"):
            st.session_state.pop("history_popup_path", None)
            st.rerun()
        return
    try:
        hist = load_artifact(_hp)
    except Exception as exc:
        st.error(f"Could not load run: {exc}")
        if st.button("Close", type="primary", use_container_width=True, key="hist_dlg_close_err"):
            st.session_state.pop("history_popup_path", None)
            st.rerun()
        return

    when = (hist.finished_at or hist.started_at or "")[:19].replace("T", " ")
    st.caption(
        f"{case_display_name(hist.case_id)} · {when} · ${hist.total_cost_usd:.4f} · "
        f"`{Path(path_str).name}`"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Case", case_display_name(hist.case_id))
    m2.metric("Models", str(len(hist.candidates)))
    m3.metric("Cost $", f"{hist.total_cost_usd:.3f}")

    if hist.ranking:
        st.plotly_chart(
            fig_judge_accuracy_bars(
                [
                    {
                        **r,
                        "label": next(
                            (
                                c.display_label or c.label
                                for c in hist.candidates
                                if c.candidate_key == r.get("key")
                            ),
                            r.get("key"),
                        ),
                    }
                    for r in hist.ranking
                ],
                height=220,
            ),
            use_container_width=True,
            key="hist_dlg_rank_chart",
        )
        rows = [
            {
                "#": r.get("rank"),
                "Model": next(
                    (
                        c.display_label or c.label
                        for c in hist.candidates
                        if c.candidate_key == r.get("key")
                    ),
                    r.get("key"),
                ),
                "Acc %": r.get("accuracy"),
            }
            for r in hist.ranking
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
                st.text(c.raw_response[:12000])
            elif c.answers:
                for qid, ans in c.answers.items():
                    st.markdown(f"**{qid}**")
                    st.text((ans or "")[:4000])
            if j:
                st.markdown("**Judge**")
                for qs in j.question_scores:
                    st.caption(f"{qs.question_id}: {qs.score}/100 — {qs.rationale}")

    if st.button("Close", type="primary", use_container_width=True, key="hist_dlg_close"):
        st.session_state.pop("history_popup_path", None)
        st.rerun()


@st.dialog("Run complete")
def run_done_dialog():
    """Notify when judge finishes — Streamlit auto-scroll is unreliable."""
    multi_n = int(st.session_state.get("last_multi_n") or 1)
    if multi_n > 1:
        st.success(f"Multi-run ×{multi_n} finished — mean KPIs and reliability are ready.")
        st.caption(
            "Scroll down to **Results**: official ranking is the **mean across runs** "
            "(±std, CV%). Open each **Run** tab for that iteration’s detail."
        )
    else:
        st.success("Judge finished — ranking and scores are ready.")
        st.caption(
            "Scroll down to **Results** for the chart and tables. "
            "The sidebar Run clock is stopped."
        )
    if st.button("OK", type="primary", use_container_width=True, key="run_done_ok"):
        st.session_state.pop("show_run_done", None)
        st.rerun()


def _reliability_table_html(ranking_mean: list) -> str:
    """Colored High / Medium / Low reliability table for mean-KPI popup."""
    rel_style = {
        "high": ("#14532d", "#86efac", "HIGH"),
        "medium": ("#713f12", "#fde047", "MEDIUM"),
        "low": ("#7f1d1d", "#fca5a5", "LOW"),
    }
    rows_html = []
    for r in ranking_mean or []:
        rel = str(r.get("reliability") or "—").lower()
        bg, fg, lab = rel_style.get(rel, ("#1e293b", "#94a3b8", rel.upper() or "—"))
        badge = (
            f'<span style="display:inline-block;min-width:4.2rem;text-align:center;'
            f'padding:0.2rem 0.45rem;border-radius:999px;background:{bg};color:{fg};'
            f'font-size:0.72rem;font-weight:800;letter-spacing:0.04em;">{lab}</span>'
        )
        rows_html.append(
            "<tr>"
            f"<td style='padding:0.45rem 0.55rem;border-bottom:1px solid #1e293b'>#{r.get('rank')}</td>"
            f"<td style='padding:0.45rem 0.55rem;border-bottom:1px solid #1e293b;font-weight:600'>"
            f"{html.escape(short_model(str(r.get('key'))))}</td>"
            f"<td style='padding:0.45rem 0.55rem;border-bottom:1px solid #1e293b;"
            f"font-weight:700;color:#fbbf24;font-size:1.05rem'>"
            f"{float(r.get('accuracy_mean') or 0):.1f}%</td>"
            f"<td style='padding:0.45rem 0.55rem;border-bottom:1px solid #1e293b;color:#cbd5e1'>"
            f"± {float(r.get('std') or 0):.1f}</td>"
            f"<td style='padding:0.45rem 0.55rem;border-bottom:1px solid #1e293b;color:#cbd5e1'>"
            f"{float(r.get('cv_pct') or 0):.1f}%</td>"
            f"<td style='padding:0.45rem 0.55rem;border-bottom:1px solid #1e293b'>{badge}</td>"
            f"<td style='padding:0.45rem 0.55rem;border-bottom:1px solid #1e293b;color:#94a3b8'>"
            f"{float(r.get('median') or 0):.1f}</td>"
            f"<td style='padding:0.45rem 0.55rem;border-bottom:1px solid #1e293b;color:#64748b;"
            f"font-size:0.85rem'>{float(r.get('min') or 0):.0f}–{float(r.get('max') or 0):.0f}</td>"
            "</tr>"
        )
    return (
        "<div style='overflow-x:auto;margin:0.35rem 0 0.75rem;border:1px solid #334155;"
        "border-radius:12px;background:#0f172a'>"
        "<table style='width:100%;border-collapse:collapse;color:#e2e8f0;font-size:0.9rem'>"
        "<thead><tr style='color:#94a3b8;text-align:left;font-size:0.75rem;"
        "letter-spacing:0.04em;text-transform:uppercase'>"
        "<th style='padding:0.55rem'>#</th><th style='padding:0.55rem'>Model</th>"
        "<th style='padding:0.55rem'>Mean</th><th style='padding:0.55rem'>± Std</th>"
        "<th style='padding:0.55rem'>CV %</th><th style='padding:0.55rem'>Reliability</th>"
        "<th style='padding:0.55rem'>Median</th><th style='padding:0.55rem'>Min–Max</th>"
        "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>"
        "<div style='font-size:0.78rem;color:#94a3b8;margin-bottom:0.5rem'>"
        f"{reliability_badge('high')} CV ≤ 8% &nbsp; "
        f"{reliability_badge('medium')} CV ≤ 18% &nbsp; "
        f"{reliability_badge('low')} CV &gt; 18% &nbsp;·&nbsp; "
        "lower CV = stabler mean</div>"
    )


@st.dialog("Rebuild mean · N runs · 50/30/20 · $0", width="large")
def history_mean_rebuild_dialog():
    """Popup: offline mean KPIs after rescoring saved runs with current formula."""
    from benchmark.schema import MultiRunSummary as _MRS

    payload = st.session_state.get("history_rebuild_result") or {}
    if not payload.get("ok"):
        st.error(payload.get("reason") or "Nothing to show.")
        if st.button("Close", type="primary", use_container_width=True, key="hm_dlg_err"):
            st.session_state.pop("show_history_mean_popup", None)
            st.rerun()
        return

    raw = payload.get("summary") or {}
    try:
        summary = _MRS.model_validate(raw) if isinstance(raw, dict) else raw
    except Exception as exc:
        st.error(f"Summary invalid: {exc}")
        if st.button("Close", type="primary", use_container_width=True, key="hm_dlg_bad"):
            st.session_state.pop("show_history_mean_popup", None)
            st.rerun()
        return

    st.success(
        f"**{case_display_name(summary.case_id)}** · mean over **N={summary.n}** runs · "
        f"rescored **50% alignment / 30% quality / 20% stem** · "
        f"**$0 API** (no OpenRouter / DeepSeek calls)"
    )
    st.caption(reliability_caption(summary))

    st.markdown("##### Ranking table")
    st.markdown(_reliability_table_html(summary.ranking_mean), unsafe_allow_html=True)

    st.markdown("##### Chart (mean %; whiskers = ±1 std)")
    st.plotly_chart(
        fig_judge_mean_accuracy_bars(
            summary.ranking_mean,
            title=f"Mean accuracy · {case_display_name(summary.case_id)} · N={summary.n}",
            height=260,
        ),
        use_container_width=True,
        key="hm_dlg_mean_chart",
    )

    with st.expander("Per-run accuracy (after 50/30/20 rescore)", expanded=False):
        pr_rows = []
        for pr in payload.get("per_run") or []:
            when = (pr.get("finished_at") or "")[:19].replace("T", " ")
            row = {"When": when, "run_id": (pr.get("run_id") or "")[:18]}
            for r in pr.get("ranking") or []:
                row[short_model(str(r.get("key")))] = r.get("accuracy")
            pr_rows.append(row)
        if pr_rows:
            st.dataframe(pd.DataFrame(pr_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Used {payload.get('n_used')} of {payload.get('available')} saved runs · "
        f"{payload.get('formula')}"
    )
    if st.button("Close", type="primary", use_container_width=True, key="hm_dlg_close"):
        st.session_state.pop("show_history_mean_popup", None)
        st.rerun()


@st.dialog("Single run KPIs", width="large")
def multi_run_detail_dialog(path_str: str):
    """Popup KPIs for one finished run inside a Multi ×N batch."""
    _mp = Path(path_str)
    if not assert_path_in_workspace(_mp, WORKSPACE_DIR):
        st.error("That run is not in your private history (API key mismatch).")
        if st.button("Close", type="primary", use_container_width=True, key="mrun_dlg_deny"):
            st.session_state.pop("multi_run_popup_path", None)
            st.rerun()
        return
    try:
        hist = load_artifact(_mp)
    except Exception as exc:
        st.error(f"Could not load run: {exc}")
        if st.button("Close", type="primary", use_container_width=True, key="mrun_dlg_err"):
            st.session_state.pop("multi_run_popup_path", None)
            st.rerun()
        return

    st.caption(
        f"Run {hist.n_index} · {case_display_name(hist.case_id)} · "
        f"${hist.total_cost_usd:.4f} · "
        f"`{Path(path_str).name}`"
    )
    if hist.ranking:
        st.plotly_chart(
            fig_judge_accuracy_bars(
                [
                    {
                        **r,
                        "label": next(
                            (
                                c.display_label or c.label
                                for c in hist.candidates
                                if c.candidate_key == r.get("key")
                            ),
                            r.get("key"),
                        ),
                    }
                    for r in hist.ranking
                ],
                height=220,
                title=f"Run {hist.n_index} · accuracy",
            ),
            use_container_width=True,
            key=f"mrun_dlg_chart_{hist.n_index}",
        )
        rows = [
            {
                "#": r.get("rank"),
                "Model": short_model(str(r.get("key"))),
                "Acc %": r.get("accuracy"),
                "TTFT": r.get("ttft_s"),
                "TPS": r.get("tps"),
                "$": r.get("cost_usd"),
            }
            for r in hist.ranking
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if hist.judgments:
        q_ids = []
        for j in hist.judgments:
            for qs in j.question_scores:
                if qs.question_id not in q_ids:
                    q_ids.append(qs.question_id)
        matrix = []
        for j in hist.judgments:
            row = {"Model": short_model(j.candidate_key)}
            by_q = {qs.question_id: qs.score for qs in j.question_scores}
            for qid in q_ids:
                row[qid] = by_q.get(qid)
            row["weighted %"] = j.weighted_accuracy
            matrix.append(row)
        st.markdown("**Scores by clinical dimension**")
        st.dataframe(pd.DataFrame(matrix), use_container_width=True, hide_index=True)

    if st.button("Close", type="primary", use_container_width=True, key="mrun_dlg_close"):
        st.session_state.pop("multi_run_popup_path", None)
        st.rerun()


@st.dialog("How ranking is calculated", width="large")
def scoring_guide_dialog():
    """Wide, shallow popup: formula + parameters side-by-side (not a deep expander)."""
    st.caption(
        "Blind DeepSeek R1 · host recomputes scores · literal 100% is not used · "
        "four models always get distinct accuracies · "
        "GOLD pasted → score vs gold; empty → teaching rubric"
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Per-section score (semantic)")
        st.code(
            "Gold: 100×(0.50·alignment + 0.30·quality + 0.20·stem)\n"
            "Rubric: 100×(0.30·must + 0.20·acceptable + 0.40·quality + 0.10·stem)\n"
            "→ capped at 96.5 · literal 100% unused",
            language=None,
        )
        st.markdown(
            """
| Param | Wt | Meaning |
|-------|----|---------|
| **alignment** (gold) | 50% | Semantic closeness to gold thesis (synonyms / near-equivalents OK) |
| **quality** | 30% | Clinical judgment — not writing style |
| **stem** | 20% | Case-specific anchors (anti-generic paste) |
| **must/accept** (rubric only) | 30%/20% | Soft checklist by **meaning**, not keywords |
"""
        )
        st.markdown(
            "Judge reads **diagnosis framing, workup intent, advice, next steps** — "
            "not word-for-word matches. Near-equivalent formulations score high."
        )
    with right:
        st.markdown("##### Final ranking %")
        st.code(
            "Accuracy = Σ (section_weight × section_score)\n"
            "run cap ≈ 97% · always unique across the 4 models\n"
            "Multi ×N official rank = mean Acc ± std (CV% = reliability)",
            language=None,
        )
        st.markdown(
            """
| Piece | Role |
|-------|------|
| **Section weights** | Fixed in the case JSON (diagnosis usually heaviest) |
| **Tie-break** | safety → quality → stem → diagnosis |
| **Multi reliability** | CV% = std/mean · High ≤8% · Medium ≤18% · else Low |

**Flow:** same prompt → answers → blind semantic judge → host formula → ranking.
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
# ONE dialog max per script run (Streamlit rule). Never open dialogs mid-benchmark.
# Priority: explicit user KPI click > rebuild mean > run-done toast > guides > boot.
if not _busy_boot:
    if st.session_state.get("multi_run_popup_path"):
        multi_run_detail_dialog(st.session_state["multi_run_popup_path"])
    elif st.session_state.get("history_popup_path"):
        history_run_dialog(st.session_state["history_popup_path"])
    elif st.session_state.get("show_history_mean_popup"):
        history_mean_rebuild_dialog()
    elif st.session_state.get("show_run_done"):
        run_done_dialog()
    elif st.session_state.get("show_scoring_guide"):
        scoring_guide_dialog()
    elif st.session_state.get("show_qvac_guide"):
        qvac_setup_guide_dialog()
    elif not st.session_state.get("boot_welcome_done"):
        # Every new browser session / reload: API key first, then QVAC status
        if st.session_state.get("boot_step", "api") == "api":
            key_welcome_dialog()
        elif st.session_state.get("boot_step") == "qvac":
            qvac_status_dialog(qvac_up, qvac_ok)

if st.session_state.get("or_key_session") and is_usable_openrouter_key(
    st.session_state["or_key_session"]
):
    os.environ["OPENROUTER_API_KEY"] = st.session_state["or_key_session"]
    has_key = True

st.markdown('<p class="demo-hero">QVAC vs Cloud · Automated Benchmark</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="demo-sub">On-device MedPsy via QVAC SDK · BYOK OpenRouter · DeepSeek R1 blind judge · '
    "semantic scoring 50/30/20 · live TTFT / TPS</p>",
    unsafe_allow_html=True,
)
st.markdown(
    """
<div class="steps-bar">
  <div class="step-pill"><b>Step 1</b>Pick clinical case</div>
  <div class="step-pill"><b>Step 2</b>Confirmed diagnosis</div>
  <div class="step-pill"><b>Step 3</b>Run all models → judge → ranking</div>
</div>
""",
    unsafe_allow_html=True,
)
st.caption(
    "**Naming:** Steps 1–3 = workflow · **Custom Case** (main) or **Demo Case 1/2** · "
    "ranking uses real model names (ChatGPT / Claude / Gemini / QVAC)."
)

# Client-side guide overlays in main DOM (sidebar labels toggle via for=… — no run interrupt)
_qvac_guide_status = (
    "ready — MedPsy will be included"
    if qvac_ok
    else (
        "sidecar online · MedPsy not loaded"
        if qvac_up
        else "sidecar offline — cloud-only until you start it"
    )
)
st.markdown(
    _guides_always_available_html(qvac_status_line=_qvac_guide_status),
    unsafe_allow_html=True,
)

# --- Sidebar: API key + QVAC (compact) ---
with st.sidebar:
    st.markdown("**OpenRouter**")
    if has_key:
        st.success("Key OK · cloud + R1")
    else:
        st.warning("No full key · Single/Multi off")
    st.caption(identity_caption())
    key_in = st.text_input(
        "OPENROUTER_API_KEY",
        value="",
        type="password",
        help="Full sk-or-v1-… from openrouter.ai/keys — remembered for this IP only",
        placeholder="sk-or-v1-…",
        label_visibility="collapsed",
    )
    if key_in:
        if is_usable_openrouter_key(key_in):
            _remember_openrouter_key(key_in.strip())
            has_key = True
        else:
            st.error("Key truncated / too short")

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
    # Client-side overlays (no Streamlit rerun) — safe during collect/judge
    st.markdown(
        '<label class="guide-open-btn" for="guide_setup">Setup guide</label>'
        '<label class="guide-open-btn" for="guide_rank">How ranking works</label>'
        '<p style="font-size:0.65rem;color:#94a3b8;margin:0.15rem 0 0.4rem">'
        "Opens without pausing the run · ✕ to close</p>",
        unsafe_allow_html=True,
    )

# --- Steps 1–2 side by side (less scroll) ---
# Internal ids stay caseA/caseB/caseC so saved History (esp. Custom = caseC) keeps working.
CASE_PICKER = {
    "caseC": "Custom Case · your anonymized real case (main)",
    "caseA": "Demo Case 1 · STEMI + sildenafil (teaching)",
    "caseB": "Demo Case 2 · Mania + CKD (teaching)",
}
case_ids = [c for c in ("caseC", "caseA", "caseB") if c in set(list_case_ids())]
st.markdown('<div class="sec-label">Case</div>', unsafe_allow_html=True)


def _clear_all_kpi_popups() -> None:
    """Close every KPI / guide dialog flag (case change must not reopen them)."""
    for k in (
        "history_popup_path",
        "multi_run_popup_path",
        "show_run_done",
        "show_history_mean_popup",
        "show_scoring_guide",
        "show_qvac_guide",
        "history_path",
    ):
        st.session_state.pop(k, None)


def _on_case_change() -> None:
    """Case picker only swaps stem/gold fields — never opens KPI popups."""
    _clear_all_kpi_popups()
    # Reset sidebar History to placeholder so it does not look "selected"
    opts = st.session_state.get("_hist_sidebar_opts") or {}
    placeholder = next((k for k, v in opts.items() if v is None), "— select a run —")
    st.session_state["hist_sidebar_pick"] = placeholder
    # Hide previous case's ranking strip until user runs again or opens History
    st.session_state.pop("last_ranking", None)
    st.session_state.pop("last_multi_summary", None)
    st.session_state.pop("last_multi_paths", None)
    st.session_state.pop("last_multi_n", None)
    st.session_state.pop("multi_progress", None)
    st.session_state.pop("show_last_run_costs", None)


_default_case_idx = case_ids.index("caseC") if "caseC" in case_ids else 0
case_id = st.selectbox(
    "Case",
    case_ids,
    index=_default_case_idx,
    format_func=lambda cid: CASE_PICKER.get(cid, cid),
    label_visibility="collapsed",
    key="case_pick",
    on_change=_on_case_change,
)
st.caption(
    "**Custom Case** (main) = paste symptoms (step 1) + confirmed diagnosis (step 2). "
    "**Demo Case 1 / 2** = recall a teaching vignette + built-in rubric "
    "(gold box can stay empty)."
)
preset = load_case(case_id)
is_custom_real = (preset.mode or "") == "custom_real"

if is_custom_real:
    st.warning(
        "**Custom Case** — paste your anonymized clinical text (step 1) and confirmed "
        "diagnosis / safety traps (step 2). No teaching answer grid. "
        "Your previous Custom Case runs remain in History."
    )

# Sync stem when case changes. Demos: prefill stem, clear gold. Custom: empty both.
if st.session_state.get("_stem_case_id") != case_id:
    st.session_state["demo_case_stem"] = preset.stem if not is_custom_real else ""
    st.session_state["demo_gold_ref"] = ""
    st.session_state["_stem_case_id"] = case_id

col_case, col_gold = st.columns([3, 2], gap="small")
with col_case:
    st.markdown('<div class="sec-label">Step 1 · Clinical case</div>', unsafe_allow_html=True)
    case_stem = st.text_area(
        "case",
        height=96 if not is_custom_real else 110,
        key="demo_case_stem",
        label_visibility="collapsed",
        placeholder="Paste anonymized real case here…" if is_custom_real else "",
    )
with col_gold:
    st.markdown(
        '<div class="sec-label">Step 2 · Confirmed diagnosis'
        + (
            " (required · Custom Case · GOLD wins)"
            if is_custom_real
            else " (optional · Demo · empty = RUBRIC wins)"
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    gold_reference = st.text_area(
        "gold",
        height=96 if not is_custom_real else 110,
        placeholder=(
            "Custom Case · required gold (0–100 vs this thesis):\n"
            "PRIMARY + must-include / traps / plan cues…"
            if is_custom_real
            else "Optional. Leave empty → score vs teaching rubric. "
            "Paste a confirmed diagnosis → 0–100 vs that gold (GOLD wins)."
        ),
        key="demo_gold_ref",
        label_visibility="collapsed",
        disabled=False,
    )
    if is_custom_real:
        st.caption(
            "Custom Case: scores are **0–100 against this gold** "
            "(rubric arrays stay empty on purpose)."
        )
    else:
        st.caption(
            "**Empty** → RUBRIC wins (must/acceptable in the demo case). "
            "**Filled** → GOLD wins (same 0–100 scale against your diagnosis)."
        )

live_case = preset.model_copy(update={"stem": (case_stem or "").strip() or preset.stem})
# Only user-pasted gold counts. Empty demo → rubric; Custom Case requires paste.
gold_reference = (gold_reference or "").strip()
effective_gold = gold_reference

with st.expander("Exact prompt (inference) — identical for all four models", expanded=False):
    st.markdown("**System**")
    st.code(candidate_system())
    st.markdown("**User**")
    st.code(candidate_user(live_case))

# --- Models (compact chips) ---
st.markdown(
    '<div class="sec-label">Models (all four run on the same case)</div>',
    unsafe_allow_html=True,
)
chip_cols = st.columns(4)
for i, c in enumerate(cfg.get("candidates") or []):
    color = c.get("color") or "#64748b"
    with chip_cols[i % 4]:
        st.markdown(
            f'<div class="model-chip" style="background:{color}18;border-color:{color};color:{color}">'
            f'{c.get("label") or c.get("key")}'
            f'<span>{c.get("site")} · {c.get("model")}</span></div>',
            unsafe_allow_html=True,
        )

include_qvac = qvac_ok
opt_a, opt_b = st.columns([2, 1])
with opt_a:
    skip_qvac = st.checkbox("Cloud-only (skip QVAC)", value=False)
with opt_b:
    n_multi = st.number_input("Multi N", min_value=2, max_value=10, value=3)
if skip_qvac:
    include_qvac = False

bd = estimate_cost_breakdown(
    cfg,
    live_case,
    include_qvac=include_qvac and not skip_qvac,
    gold_reference=effective_gold,
    n=1,
)
bd_multi = estimate_cost_breakdown(
    cfg,
    live_case,
    include_qvac=include_qvac and not skip_qvac,
    gold_reference=effective_gold,
    n=int(n_multi),
)


def _fmt_cost_single(breakdown: dict) -> str:
    bits = []
    for m in breakdown.get("per_model") or []:
        bits.append(f"{m.get('key')} ${m.get('estimated_usd', 0):.3f}")
    j = breakdown.get("judge") or {}
    bits.append(f"judge ${j.get('estimated_usd', 0):.3f}")
    total = float(breakdown.get("total_usd", 0) or 0)
    hi = total * 2
    tok = breakdown.get("input_tokens_used_for_estimate", 0)
    chars = breakdown.get("chars_case_plus_gold", 0)
    return (
        '<div class="cost-compact">'
        + " · ".join(bits)
        + f' · <b>${total:.3f}–${hi:.3f}</b>'
        + f'<br/><span style="opacity:.75">{chars} chars · ~{tok} in-tok · upper≈2×</span>'
        + "</div>"
    )


def _fmt_cost_multi(breakdown: dict, n: int) -> str:
    per = float(breakdown.get("total_usd", 0) or 0)
    tot = float(breakdown.get("total_usd_for_n", 0) or 0)
    return (
        f'<div class="cost-compact cost-multi">'
        f"${per:.3f} × {n} → <b>${tot:.3f}–${tot * 2:.3f}</b>"
        f'<br/><span style="opacity:.75">upper≈2×</span></div>'
    )


st.markdown('<div class="sec-label">Step 3 · Run</div>', unsafe_allow_html=True)
col_s, col_m, col_q = st.columns(3)
with col_s:
    single_clicked = st.button(
        "Single run",
        type="secondary",
        use_container_width=True,
        disabled=not has_key,
        help="Quick one-shot. For published-style comparison prefer Multi ×3.",
    )
    st.markdown(_fmt_cost_single(bd), unsafe_allow_html=True)
with col_m:
    multi_clicked = st.button(
        f"Multi run ×{int(n_multi)} · recommended",
        type="primary",
        use_container_width=True,
        disabled=not has_key,
        help="Official demo path: mean/median/std across N runs (default 3).",
    )
    st.markdown(_fmt_cost_multi(bd_multi, int(n_multi)), unsafe_allow_html=True)
with col_q:
    qvac_only_clicked = st.button(
        "Run QVAC only · $0",
        key="qvac_only_btn",
        use_container_width=True,
        disabled=not qvac_run_ok,
        help="Local MedPsy via QVAC SDK sidecar only — no OpenRouter credits. Live token stream for UI rehearsal.",
    )
    st.markdown(
        '<div class="cost-compact cost-multi"><b>$0</b> · MedPsy only · no DeepSeek · no ranking</div>',
        unsafe_allow_html=True,
    )
st.caption(
    "**Recommended:** Multi ×3 for stable ranking. Single run stays available for a fast check. "
    "Judge scores clinical meaning + synonyms on a linear 0–100 coverage scale. "
    "QVAC only = local rehearsal, no judge."
)

# Confirm flow via session state
if single_clicked:
    st.session_state["pending_run"] = {"n": 1, "est": bd["total_usd"], "mode": "full"}
if multi_clicked:
    st.session_state["pending_run"] = {
        "n": int(n_multi),
        "est": bd_multi["total_usd_for_n"],
        "mode": "full",
    }
if qvac_only_clicked:
    # Clear stale failed snapshot (e.g. old 404) so the live panels reset.
    st.session_state.pop("live_outputs", None)
    st.session_state["confirmed_run"] = {"n": 1, "est": 0.0, "mode": "qvac_only"}
    st.rerun()


# Custom modal (looks like a popup, fully unmounts on Yes/Cancel — no stuck grey,
# no Streamlit dialog X that aborts in-flight OpenRouter/QVAC calls).
if st.session_state.get("pending_run") and not st.session_state.get("confirmed_run"):
    pr = st.session_state["pending_run"]
    n = int(pr.get("n") or 1)
    est = float(pr.get("est") or 0)
    est_hi = est * 2
    with st.container():
        st.markdown(
            f"""
<div class="spend-modal-marker"></div>
<div class="spend-modal-card">
  <h3>Confirm OpenRouter spend</h3>
  <p>Estimated range <b>${est:.4f} – ${est_hi:.4f}</b> for <b>{n}</b> run(s)
  (cloud models + DeepSeek R1 judge). QVAC = $0 if included.</p>
  <p class="muted">Lower ≈ token estimate; upper ≈ 2× (conservative — long answers / judge
  tokens often exceed the base estimate). Actual bill is what OpenRouter reports.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if not has_key:
            st.error("No usable OpenRouter key — paste a full sk-or-v1-… key in the sidebar.")
            if st.button("Close", use_container_width=True, key="spend_close_nokey"):
                st.session_state.pop("pending_run", None)
                st.session_state.pop("confirmed_run", None)
                st.rerun()
        else:
            a, b = st.columns(2)
            with a:
                if st.button("Cancel", use_container_width=True, key="spend_cancel"):
                    st.session_state.pop("pending_run", None)
                    st.session_state.pop("confirmed_run", None)
                    st.rerun()
            with b:
                if st.button(
                    "Yes, continue",
                    type="primary",
                    use_container_width=True,
                    key="spend_yes",
                ):
                    pending = st.session_state.pop("pending_run", None)
                    if pending:
                        st.session_state["confirmed_run"] = pending
                    st.rerun()
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


def _stream_html(
    text: str,
    live: bool = False,
    *,
    title: str = "Answer",
    panel_id: str = "ans",
) -> str:
    """Live answer box + client-side fullscreen (no Streamlit rerun → run keeps going)."""
    caret = '<span class="caret"></span>' if live else ""
    body = html.escape(text or "")
    tid = html.escape(title or "Answer")
    uid = "".join(ch if ch.isalnum() else "_" for ch in (panel_id or "ans"))[:32]
    return f"""
<div class="stream-wrap">
  <div class="stream-toolbar">
    <label class="stream-fs-lab" for="fs_{uid}" title="Open full screen (does not pause the run)">⛶ Full screen</label>
  </div>
  <input type="checkbox" id="fs_{uid}" class="fs-ck" />
  <div class="fs-overlay">
    <div class="fs-card">
      <div class="fs-bar">
        <span>{tid}</span>
        <label for="fs_{uid}" class="fs-close" title="Close">✕</label>
      </div>
      <pre class="fs-pre">{body}</pre>
    </div>
  </div>
  <div class="stream-out">{body}{caret}</div>
</div>
"""


def _kpi_live_line(ttft_s, elapsed_s, tps_live) -> str:
    parts = []
    if ttft_s is not None:
        parts.append(f"TTFT {ttft_s}s")
    if tps_live is not None:
        parts.append(f"~{tps_live} TPS")
    if elapsed_s is not None:
        parts.append(f"{elapsed_s}s…")
    return " · ".join(parts) if parts else "streaming…"



def _fmt_s_min(seconds: int | float) -> str:
    """Primary seconds; compact minutes in parentheses (Italian comma), e.g. 150s (2,5m)."""
    s = max(0, int(round(float(seconds))))
    m = s / 60.0
    m_txt = f"{m:.1f}".replace(".", ",")
    return f'{s}s<span class="t-min"> ({m_txt}m)</span>'


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
    return f"""
<div class="run-timer-panel idle">
  <div class="t-title">Run clock · last</div>
  <div class="t-big">{_fmt_s_min(total)}</div>
  {this_row}
  <div class="t-row"><span class="lab">collect</span><span class="val">{_fmt_s_min(last.get("collect_s") or 0)}</span></div>
  <div class="t-row"><span class="lab">judge</span><span class="val">{_fmt_s_min(last.get("judge_s") or 0)}</span></div>
  <hr class="t-sep"/>
  <span class="phase">Done · final scores ready</span>
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
    slot, inner_html: str, *, height: int = 160, live: bool | None = None
) -> None:
    """
    Idle / stopped: plain HTML docked at bottom of left sidebar (no iframe).
    Live ticking: compact iframe only (scripts need it) — never tall enough to
    bleed over the main CASE / STEP 1 column.
    """
    if live is None:
        live = "setInterval" in (inner_html or "")
    # Hard cap — old height=220/280 iframes covered the main left column
    height = min(int(height or 160), 168)
    docked = f'<div class="sidebar-timer-dock">{inner_html}</div>'
    with slot:
        if live:
            doc = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<style>{_TIMER_IFRAME_CSS}html,body{{overflow:hidden;}}</style>"
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
    """Live ticking panel. Baselines from Python survive iframe remounts."""
    phase_js = json.dumps(phase)
    bucket_js = json.dumps(bucket if bucket in ("collect", "judge") else "other")
    multi = int(n_runs) > 1
    this_display = "flex" if multi else "none"
    et = max(0, int(elapsed_total))
    eh = max(0, int(elapsed_this))
    return f"""
<div class="run-timer-panel">
  <div class="t-title">Run clock</div>
  <div class="t-big"><span id="t-total">{_fmt_s_min(et)}</span></div>
  <div class="t-row" style="display:{this_display}">
    <span class="lab">this run</span><span class="val"><span id="t-this">{_fmt_s_min(eh)}</span></span>
  </div>
  <div class="t-row" id="row-c"><span class="lab">collect</span><span class="val"><span id="t-collect">{_fmt_s_min(collect_base)}</span></span></div>
  <div class="t-row" id="row-j"><span class="lab">judge</span><span class="val"><span id="t-judge">{_fmt_s_min(judge_base)}</span></span></div>
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
  if (rowC) rowC.classList.toggle('active', bucket === 'collect');
  if (rowJ) rowJ.classList.toggle('active', bucket === 'judge');
  var paintAt = Date.now();
  var baseTotal = {et};
  var baseThis = {eh};
  var cBase = {int(collect_base)};
  var jBase = {int(judge_base)};
  function fmt(s) {{
    var m = (s / 60).toFixed(1).replace('.', ',');
    return s + 's<span class="t-min"> (' + m + 'm)</span>';
  }}
  function paint() {{
    var add = Math.floor((Date.now() - paintAt) / 1000);
    var totalS = baseTotal + add;
    var thisS = baseThis + add;
    var collectS = cBase + (bucket === 'collect' ? add : 0);
    var judgeS = jBase + (bucket === 'judge' ? add : 0);
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
    return f"""
<div class="run-timer-panel">
  <div class="t-title">{title}</div>
  <div class="t-big">{_fmt_s_min(total_s)}</div>
  {this_row}
  <div class="t-row"><span class="lab">collect</span><span class="val">{_fmt_s_min(collect_s)}</span></div>
  <div class="t-row"><span class="lab">judge</span><span class="val">{_fmt_s_min(judge_s)}</span></div>
  <hr class="t-sep"/>
  <span class="phase">{phase}</span>
</div>
"""


# --- Sidebar: History first, Run clock pinned at the very bottom of the left column ---
with st.sidebar:
    st.markdown("---")
    _hist_paths = list_run_artifacts(WORKSPACE_DIR)[:12]
    st.caption(
        f"Private history · {short_owner_label()}"
        + (" · enter API key to unlock" if not has_key else "")
    )
    if _hist_paths:
        st.markdown("**History**")
        _placeholder = "— select a run —"
        _opts = {_placeholder: None}
        for p in _hist_paths:
            try:
                art = load_artifact(p)
                when = (art.finished_at or art.started_at or "")[5:16].replace("T", " ")
                top = ""
                if art.ranking:
                    top = f" · {art.ranking[0].get('accuracy')}%"
                label = (
                    f"{case_display_name(art.case_id)} · {when} · "
                    f"${art.total_cost_usd:.2f}{top}"
                )
            except Exception:
                label = p.stem
            base = label
            n = 2
            while label in _opts:
                label = f"{base} ·{n}"
                n += 1
            _opts[label] = str(p)
        st.session_state["_hist_sidebar_opts"] = _opts

        def _on_sidebar_hist_change() -> None:
            """Fires only when the user changes the sidebar History dropdown."""
            opts = st.session_state.get("_hist_sidebar_opts") or {}
            chosen = st.session_state.get("hist_sidebar_pick")
            path = opts.get(chosen)
            if path:
                # Explicit user pick → open that run only (close other KPI dialogs)
                st.session_state.pop("multi_run_popup_path", None)
                st.session_state.pop("show_run_done", None)
                st.session_state.pop("show_history_mean_popup", None)
                st.session_state["history_popup_path"] = path
                st.session_state["history_path"] = path
            else:
                st.session_state.pop("history_popup_path", None)

        pick = st.selectbox(
            "Recent runs",
            list(_opts.keys()),
            label_visibility="collapsed",
            key="hist_sidebar_pick",
            on_change=_on_sidebar_hist_change,
        )
        sel_path = _opts.get(pick)
        if st.button(
            "View run results",
            use_container_width=True,
            disabled=not sel_path,
            key="hist_sidebar_view",
            help="Re-open ranking + answers for the selected sidebar run",
        ):
            st.session_state["history_popup_path"] = sel_path
            st.session_state["history_path"] = sel_path
            st.rerun()
    else:
        st.caption("No runs in History yet for this key.")

    # LAST widget in left column = Run clock (space above so it never sits on History)
    st.markdown('<div class="sidebar-timer-spacer"></div>', unsafe_allow_html=True)
    timer_slot = st.empty()
    _pending = st.session_state.get("confirmed_run") or {}
    if _pending:
        _paint_run_timer(
            timer_slot,
            _run_timer_live(
                "Starting…",
                n_runs=int(_pending.get("n") or 1),
                elapsed_total=0,
                elapsed_this=0,
                collect_base=0,
                judge_base=0,
                bucket="collect",
            ),
            height=160,
            live=True,
        )
    else:
        _paint_run_timer(
            timer_slot,
            _run_timer_idle(st.session_state.get("last_run_timings")),
            live=False,
        )

# --- Live response panels ---
roster = list(cfg.get("candidates") or [])
saved_outputs = st.session_state.get("live_outputs") or {}
_run_pending = st.session_state.get("confirmed_run") or {}
running_now = bool(_run_pending)
qvac_only_now = _run_pending.get("mode") == "qvac_only"

st.markdown('<div class="sec-label">Live responses</div>', unsafe_allow_html=True)
st.caption("Click **⛶ Full screen** on a box to read the full answer · ✕ closes · collect/judge keep running.")
card_cols = st.columns(len(roster) or 1)
text_boxes, kpi_boxes, status_boxes = {}, {}, {}
for i, c in enumerate(roster):
    key = c["key"]
    prev = saved_outputs.get(key) or {}
    color = c.get("color") or "#64748b"
    label = c.get("display_label") or c.get("label") or key
    with card_cols[i]:
        st.markdown(
            f'<div class="panel-frame" style="border-top: 3px solid {color}">'
            f'<p class="live-head">{html.escape(str(label))}</p>'
            f'<p class="live-meta">{html.escape(str(c.get("model") or ""))}</p></div>',
            unsafe_allow_html=True,
        )
        status_boxes[key] = st.empty()
        kpi_boxes[key] = st.empty()
        text_boxes[key] = st.empty()

        if running_now:
            if qvac_only_now and key != "qvac":
                status_boxes[key].markdown(
                    _status_pill("skip", "Skipped · $0 rehearsal"),
                    unsafe_allow_html=True,
                )
                text_boxes[key].markdown(
                    _stream_html(
                        "Cloud skipped — QVAC-only rehearsal",
                        live=False,
                        title=str(label),
                        panel_id=key,
                    ),
                    unsafe_allow_html=True,
                )
            else:
                status_boxes[key].markdown(
                    _status_pill(
                        "wait",
                        "Generating…" if (key == "qvac" and qvac_only_now) else "Waiting…",
                    ),
                    unsafe_allow_html=True,
                )
                text_boxes[key].markdown(
                    _stream_html("", live=True, title=str(label), panel_id=key),
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
                    f'<p class="kpi-row">{prev["kpi"]}</p>',
                    unsafe_allow_html=True,
                )
            text_boxes[key].markdown(
                _stream_html(
                    prev.get("text") or "",
                    live=False,
                    title=str(label),
                    panel_id=key,
                ),
                unsafe_allow_html=True,
            )
        else:
            qvac_out = key == "qvac" and (skip_qvac or not include_qvac)
            if qvac_out:
                reason = (
                    "Skipped · cloud-only"
                    if skip_qvac
                    else "Sidecar offline"
                )
                status_boxes[key].markdown(
                    _status_pill("skip", reason),
                    unsafe_allow_html=True,
                )
                text_boxes[key].markdown(
                    _stream_html("", live=False, title=str(label), panel_id=key),
                    unsafe_allow_html=True,
                )
            else:
                status_boxes[key].markdown(
                    _status_pill("ready", "Ready"),
                    unsafe_allow_html=True,
                )
                text_boxes[key].markdown(
                    _stream_html("", live=False, title=str(label), panel_id=key),
                    unsafe_allow_html=True,
                )

def _paint_multi_progress(
    slot,
    completed: list,
    *,
    n_total: int,
    batch_done: bool = False,
    toast_html: str = "",
    height: int = 220,
) -> None:
    """Render multi progress in an iframe so onclick modals/toasts survive sanitizer."""
    body = progressive_multi_panel_html(
        completed, n_total=n_total, batch_done=batch_done
    ) + (toast_html or "")
    # Extra height when toast is present
    h = height + (180 if toast_html else 0)
    slot.empty()
    with slot.container():
        components.html(
            f"""<!doctype html><html><head><meta charset="utf-8"/>
<style>
  body {{ margin:0; background:transparent; font-family: ui-sans-serif, system-ui, sans-serif; }}
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

# --- Execute confirmed run ---
if st.session_state.get("confirmed_run"):
    run_cfg = st.session_state.pop("confirmed_run")
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
    run_mode = run_cfg.get("mode") or "full"
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
    )

    def _abort_run(msg: str, *, phase: str = "Stopped · fix the issue and retry") -> None:
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

    # ---- Free local rehearsal: MedPsy only, live token stream, $0 ----
    if run_mode == "qvac_only":
        if not case_stem.strip():
            _abort_run("Clinical case is empty.")
        if is_custom_real and not gold_reference.strip():
            _abort_run("Custom Case requires confirmed diagnosis in step 2.")
        if not qvac_run_ok:
            _abort_run(
                "QVAC SDK sidecar offline — start it: `cd sidecar && npm start` "
                "(requires OpenSSL 3: `brew install openssl@3`)."
            )

        phase_slot.markdown(
            '<div class="phase-banner">QVAC only · MedPsy streaming on-device · $0</div>',
            unsafe_allow_html=True,
        )
        _paint_run_timer(
            timer_slot,
            _run_timer_live(
                "QVAC only · streaming",
                n_runs=1,
                elapsed_total=time.time() - t_run0,
                elapsed_this=time.time() - t_run0,
                collect_base=0,
                judge_base=0,
                bucket="collect",
            ),
        )
        for c in roster:
            if c["key"] != "qvac":
                status_boxes[c["key"]].markdown(
                    _status_pill("skip", "Skipped"), unsafe_allow_html=True
                )

        status_boxes["qvac"].markdown(
            _status_pill("wait", "Streaming…"), unsafe_allow_html=True
        )
        prompt = candidate_system() + "\n\n" + candidate_user(live_case)
        buf = ""
        done_meta: dict = {}
        err_msg = None
        n_tok = 0
        import time as _time_live

        t0 = _time_live.time()
        ttft_s = None
        for evt in qvac_iter_tokens(prompt):
            et = evt.get("type")
            if et == "token":
                tok = evt.get("token") or ""
                if not tok:
                    continue
                buf += tok
                n_tok += 1
                now = _time_live.time()
                if ttft_s is None:
                    ttft_s = round(now - t0, 2)
                elapsed = round(now - t0, 2)
                gen_elapsed = max(elapsed - (ttft_s or 0), 0.001)
                tps_live = round(n_tok / gen_elapsed, 1) if ttft_s is not None else None
                # Refresh UI every few tokens for screen-recording fluidity
                if n_tok == 1 or n_tok % 3 == 0:
                    text_boxes["qvac"].markdown(
                        _stream_html(
                            buf,
                            live=True,
                            title="QVAC · MedPsy",
                            panel_id="qvac",
                        ),
                        unsafe_allow_html=True,
                    )
                    kpi_boxes["qvac"].markdown(
                        f'<p class="kpi-row live">{_kpi_live_line(ttft_s, elapsed, tps_live)}</p>',
                        unsafe_allow_html=True,
                    )
            elif et == "done":
                done_meta = evt
                if evt.get("content"):
                    buf = str(evt["content"])
            elif et == "error":
                err_msg = str(evt.get("error") or "stream error")
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
                break

        text_boxes["qvac"].markdown(
            _stream_html(
                buf or "(empty)",
                live=False,
                title="QVAC · MedPsy",
                panel_id="qvac",
            ),
            unsafe_allow_html=True,
        )
        if err_msg:
            status_boxes["qvac"].markdown(
                _status_pill("err", err_msg[:80]), unsafe_allow_html=True
            )
        else:
            status_boxes["qvac"].markdown(
                _status_pill("done", "Done · $0"), unsafe_allow_html=True
            )
            kpi = _kpi_line(
                {
                    "ttft_s": done_meta.get("ttft_s") if done_meta.get("ttft_s") is not None else ttft_s,
                    "tps": done_meta.get("tps"),
                    "latency_s": done_meta.get("latency_s"),
                    "cost_usd": 0,
                    "completion_tokens": done_meta.get("completion_tokens") or 0,
                },
                buf,
            )
            device = done_meta.get("device") or qvac_health().get("device") or "?"
            kpi_boxes["qvac"].markdown(
                f'<p class="kpi-row">{kpi} · device {device}</p>',
                unsafe_allow_html=True,
            )

        live_snap = {
            c["key"]: {
                "text": (buf or "") if c["key"] == "qvac" else "",
                "status": (
                    ("Done · $0" if not err_msg else err_msg[:80])
                    if c["key"] == "qvac"
                    else "Skipped"
                ),
                "error": bool(err_msg) if c["key"] == "qvac" else False,
                "kpi": "",
            }
            for c in roster
        }
        if not err_msg:
            live_snap["qvac"]["kpi"] = _kpi_line(
                {
                    "ttft_s": done_meta.get("ttft_s"),
                    "tps": done_meta.get("tps"),
                    "latency_s": done_meta.get("latency_s"),
                    "cost_usd": 0,
                    "completion_tokens": done_meta.get("completion_tokens") or 0,
                },
                buf,
            )
        st.session_state["live_outputs"] = live_snap
        st.session_state.pop("last_ranking", None)
        st.session_state["benchmark_running"] = False
        total_s = int(round(time.time() - t_run0))
        _paint_run_timer(
            timer_slot,
            _run_timer_stop(total_s, n_runs=1, collect_s=total_s, judge_s=0),
            height=220,
        )
        st.session_state["last_run_timings"] = {
            "collect_s": total_s,
            "judge_s": 0,
            "total_s": total_s,
            "mode": "qvac_only",
            "n": 1,
        }
        phase_slot.markdown(
            '<div class="phase-banner">QVAC rehearsal done · $0 · no DeepSeek judge · no ranking</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Wall time · {total_s}s (matches sidebar Run clock)")
        st.info(
            "**QVAC-only** streams MedPsy locally and stops there — "
            "no cloud models, no DeepSeek R1 score, no charts/ranking. "
            "For comparison + judge: paste a **full** OpenRouter key in the sidebar, "
            "then click **Single run**."
        )
        st.stop()

    if not is_usable_openrouter_key(os.environ.get("OPENROUTER_API_KEY")):
        _abort_run(
            "OpenRouter API key missing or invalid (truncated placeholder). "
            "Paste the full key from https://openrouter.ai/keys in the sidebar."
        )
    if not case_stem.strip():
        _abort_run("Clinical case is empty.")
    if is_custom_real and not gold_reference.strip():
        _abort_run(
            "Custom Case requires confirmed diagnosis in step 2 "
            "(no teaching answer grid)."
        )

    try:
        prep = prepare_run(
            case_id,
            skip_qvac=skip_qvac or not include_qvac,
            require_qvac=False,
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

    # Mark slots not in this run (e.g. QVAC offline)
    for c in roster:
        if c["key"] not in active_keys:
            status_boxes[c["key"]].markdown(
                _status_pill("skip", "Skipped"), unsafe_allow_html=True
            )
            text_boxes[c["key"]].text_area(
                "out",
                value="(start QVAC SDK sidecar to include MedPsy)",
                height=LIVE_BOX_H,
                key=f"live_skip_{c['key']}_{uuid.uuid4().hex[:6]}",
                label_visibility="collapsed",
                disabled=True,
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
            )
            collected = []
            bufs = {c["key"]: "" for c in candidates_cfg}
            tok_n = {c["key"]: 0 for c in candidates_cfg}
            label_live = {
                c["key"]: (c.get("display_label") or c.get("label") or c["key"])
                for c in candidates_cfg
            }
            for c in candidates_cfg:
                status_boxes[c["key"]].markdown(
                    _status_pill("wait", "Streaming…"), unsafe_allow_html=True
                )
                text_boxes[c["key"]].markdown(
                    _stream_html(
                        "",
                        live=True,
                        title=str(label_live[c["key"]]),
                        panel_id=c["key"],
                    ),
                    unsafe_allow_html=True,
                )

            for evt in iter_collect_live(case_obj, candidates_cfg, blind_map):
                if evt.get("type") == "token":
                    key = evt["key"]
                    bufs[key] = bufs.get(key, "") + (evt.get("delta") or "")
                    tok_n[key] = tok_n.get(key, 0) + 1
                    # Throttle UI paints — same $ cost, smoother recording
                    if tok_n[key] == 1 or tok_n[key] % 3 == 0:
                        text_boxes[key].markdown(
                            _stream_html(
                                bufs[key],
                                live=True,
                                title=str(label_live.get(key, key)),
                                panel_id=key,
                            ),
                            unsafe_allow_html=True,
                        )
                        kpi_boxes[key].markdown(
                            f'<p class="kpi-row live">{_kpi_live_line(evt.get("ttft_s"), evt.get("elapsed_s"), evt.get("tps_live"))}</p>',
                            unsafe_allow_html=True,
                        )
                elif evt.get("type") == "done":
                    cand = evt["candidate"]
                    collected.append(cand)
                    err = bool(cand.meta.error)
                    status_msg = (
                        "Done" if not err else f"Error: {str(cand.meta.error)[:60]}"
                    )
                    status_boxes[cand.candidate_key].markdown(
                        _status_pill("err" if err else "done", status_msg),
                        unsafe_allow_html=True,
                    )
                    text = cand.raw_response or bufs.get(cand.candidate_key) or "(empty)"
                    kpi = _kpi_line(cand.meta.model_dump(), text)
                    kpi_boxes[cand.candidate_key].markdown(
                        f'<p class="kpi-row">{kpi}</p>',
                        unsafe_allow_html=True,
                    )
                    text_boxes[cand.candidate_key].markdown(
                        _stream_html(
                            text,
                            live=False,
                            title=str(
                                label_live.get(
                                    cand.candidate_key, cand.candidate_key
                                )
                            ),
                            panel_id=cand.candidate_key,
                        ),
                        unsafe_allow_html=True,
                    )
                    live_snap[cand.candidate_key] = {
                        "text": text,
                        "status": status_msg,
                        "error": err,
                        "kpi": kpi,
                    }

            by_key = {c.candidate_key: c for c in collected}
            collected = [by_key[c["key"]] for c in candidates_cfg if c["key"] in by_key]
            run_collect_s = time.time() - t_collect0
            collect_s_acc += run_collect_s

            phase_slot.markdown(
                f'<div class="phase-banner">Run {run_i}/{n_runs} · Collect done · starting judge…</div>',
                unsafe_allow_html=True,
            )
            _flash_collect_done(n_answers=len(collected))

            phase_slot.markdown(
                f'<div class="phase-banner">Run {run_i}/{n_runs} · Judging blind with {judge_label}…</div>',
                unsafe_allow_html=True,
            )
            t_judge0 = time.time()
            _paint_run_timer(
                timer_slot,
                _run_timer_live(
                    f"Run {run_i}/{n_runs} · DeepSeek R1 judging",
                    n_runs=n_runs,
                    elapsed_total=t_judge0 - t_run0,
                    elapsed_this=t_judge0 - t_run_i0,
                    collect_base=int(round(collect_s_acc)),
                    judge_base=int(round(judge_s_acc)),
                    bucket="judge",
                ),
                height=280,
            )
            judgments = []
            with st.status(f"DeepSeek R1 judge · parallel · run {run_i}", expanded=(n_runs == 1)):
                st.write(f"Scoring {len(collected)} answers in parallel…")
                judgments = judge_candidates_parallel(
                    case_obj,
                    collected,
                    judge_model,
                    temperature=judge_temp,
                    gold_reference=effective_gold,
                )
                label_by_key = {
                    c["key"]: (c.get("display_label") or c.get("label") or c["key"])
                    for c in candidates_cfg
                }
                for j in judgments:
                    name = label_by_key.get(j.candidate_key, j.candidate_key)
                    st_note = ""
                    if (j.judge_meta and j.judge_meta.error) or any(
                        "judge_error" in (e or "")
                        for qs in (j.question_scores or [])
                        for e in (qs.errors or [])
                    ):
                        st_note = " · ⚠ judge transport issue"
                    st.write(f"**{name}**: {j.weighted_accuracy}%{st_note}")
            run_judge_s = time.time() - t_judge0
            judge_s_acc += run_judge_s
            run_total_s = time.time() - t_run_i0
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

            total_cost = sum((c.meta.cost_usd or 0) for c in collected) + sum(
                (j.judge_meta.cost_usd or 0) for j in judgments
            )
            abort_multi = n_runs > 1 and systemic_judge_failure(judgments)
            notes = ""
            if abort_multi:
                notes = (
                    f"Multi aborted after run {run_i}/{n_runs}: systemic judge failure "
                    "(empty JSON / transport / majority zeros). Remaining runs skipped to save credits."
                )
                st.warning(notes)
            artifact = RunArtifact(
                run_id=f"{case_id}-{uuid.uuid4().hex[:10]}",
                case_id=case_id,
                started_at=utc_now_iso(),
                finished_at=utc_now_iso(),
                n_index=run_i,
                models_config={
                    "profile": prep["cfg"].get("profile"),
                    "candidates": candidates_cfg,
                    "judge": prep["cfg"].get("judge"),
                    "blind_map": blind_map,
                    "gold_reference": effective_gold.strip() if effective_gold else "",
                    "case_stem": case_stem.strip(),
                    "owner_id": owner_id_for_current_key(),
                    "estimated_breakdown": bd if n_runs == 1 else bd_multi,
                },
                candidates=collected,
                judgments=judgments,
                ranking=ranking,
                total_cost_usd=round(total_cost, 6),
                notes=notes,
            )
            WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
            art_path = write_artifact(artifact, WORKSPACE_DIR)
            all_artifacts.append(artifact)
            artifact_paths.append(str(art_path))
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
        # Keep phases consistent with wall clock (rounding / overhead)
        if collect_s + judge_s != total_s and total_s >= 0:
            overhead = total_s - collect_s - judge_s
            if overhead != 0:
                judge_s = max(0, judge_s + overhead)
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
            ),
            height=220,
        )
        st.session_state["live_outputs"] = live_snap
        st.session_state["last_ranking"] = last_ranking
        st.session_state["last_judgments"] = last_judgments
        st.session_state["benchmark_running"] = False
        st.session_state["last_cost_rows"] = None  # filled below

        phase_slot.markdown(
            f'<div class="phase-banner">Done · N={len(all_artifacts)} · '
            f"actual spend ≈ ${sum(a.total_cost_usd for a in all_artifacts):.4f} · "
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
        for c in last_collected:
            cost_rows.append(
                {
                    "Key": c.candidate_key,
                    "Model": c.meta.model,
                    "$": c.meta.cost_usd,
                    "TTFT": c.meta.ttft_s,
                    "TPS": c.meta.tps,
                }
            )
        judge_cost = sum((j.judge_meta.cost_usd or 0) for j in last_judgments)
        cost_rows.append(
            {
                "Key": "judge",
                "Model": judge_model,
                "$": round(judge_cost, 6),
                "TTFT": None,
                "TPS": None,
            }
        )
        st.session_state["last_cost_rows"] = cost_rows
        st.session_state["show_last_run_costs"] = True
        st.session_state["last_multi_n"] = n_runs

        # -------- Multi ×N: official = mean KPIs; per-run via tabs/popups --------
        if len(all_artifacts) > 1:
            summary = summarize_runs(all_artifacts)
            write_summary(summary, WORKSPACE_DIR)
            st.session_state["last_multi_summary"] = summary.model_dump()
            st.session_state["last_multi_paths"] = list(artifact_paths)
            st.session_state["multi_progress"] = {
                "completed": list(completed_snaps),
                "n_total": n_runs,
                "batch_done": True,
                "paths": list(artifact_paths),
            }
            _paint_multi_progress(
                multi_progress_slot,
                completed_snaps,
                n_total=n_runs,
                batch_done=True,
                height=160,
            )

            st.markdown(
                '<div class="sec-label">Official ranking · mean across runs</div>',
                unsafe_allow_html=True,
            )
            st.caption(reliability_caption(summary))
            st.markdown("##### Ranking table")
            st.markdown(
                _reliability_table_html(summary.ranking_mean), unsafe_allow_html=True
            )
            st.markdown("##### Chart (mean %; whiskers = ±1 std)")
            st.plotly_chart(
                fig_judge_mean_accuracy_bars(
                    summary.ranking_mean,
                    title=f"Mean accuracy · N={summary.n}",
                    height=280,
                ),
                use_container_width=True,
                key="rank_chart_multi_mean",
            )
            if summary.outliers:
                st.caption("Notes · " + " · ".join(summary.outliers[:4]))

            st.markdown(
                '<div class="sec-label">Per-run detail · open a tab</div>',
                unsafe_allow_html=True,
            )
            st.caption("Each finished run keeps its own KPIs — click to open the popup.")
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
                        st.session_state.pop("history_popup_path", None)
                        st.session_state.pop("show_run_done", None)
                        st.session_state.pop("show_history_mean_popup", None)
                        st.session_state["multi_run_popup_path"] = path
                        st.rerun()

            with st.expander("Last run only (for reference)", expanded=False):
                if last_ranking:
                    st.plotly_chart(
                        fig_judge_accuracy_bars(
                            last_ranking, height=220, title="Last run · accuracy"
                        ),
                        use_container_width=True,
                        key="rank_chart_last_ref",
                    )
                st.dataframe(pd.DataFrame(cost_rows), use_container_width=True, hide_index=True)

            # Ranking for persist view = mean order mapped to accuracy_mean
            st.session_state["last_ranking"] = [
                {
                    "key": r["key"],
                    "rank": r["rank"],
                    "accuracy": r["accuracy_mean"],
                    "label": short_model(str(r["key"])),
                    "status": "ok",
                    "std": r.get("std"),
                    "cv_pct": r.get("cv_pct"),
                }
                for r in summary.ranking_mean
            ]
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
                    label_by_key = {
                        c["key"]: (c.get("display_label") or c.get("label") or c["key"])
                        for c in candidates_cfg
                    }
                    for pm in explain.get("per_model") or []:
                        rows_ex.append(
                            {
                                "Model": label_by_key.get(pm["key"], pm["key"]),
                                "Acc %": pm["accuracy"],
                                "Diagnosis": pm.get("diagnosis"),
                                "Safety": pm.get("safety"),
                                "Strongest": pm.get("strongest"),
                                "Weakest": pm.get("weakest"),
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
                        "Tie-break: safety → quality → stem → diagnosis."
                    )

            if last_ranking:
                st.plotly_chart(
                    fig_judge_accuracy_bars(last_ranking, height=260),
                    use_container_width=True,
                    key="rank_chart_live",
                )
            tab_l, tab_r = st.columns(2)
            with tab_l:
                st.caption("Accuracy + KPI · status=error = not a fair clinical grade")
                if last_ranking:
                    rows = [
                        {
                            "#": r["rank"],
                            "Model": r.get("label") or r["key"],
                            "Acc %": r["accuracy"],
                            "Status": (
                                "ok"
                                if r.get("status", "ok") == "ok"
                                else f"error · {r.get('status_note') or 'failed'}"
                            ),
                            "TTFT": r.get("ttft_s"),
                            "TPS": r.get("tps"),
                            "$": r.get("cost_usd"),
                        }
                        for r in last_ranking
                    ]
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
                label_by_key = {
                    c["key"]: (c.get("display_label") or c.get("label") or c["key"])
                    for c in candidates_cfg
                }
                q_ids = [q.id for q in case_obj.questions]
                matrix_rows = []
                for j in last_judgments:
                    row = {"Model": label_by_key.get(j.candidate_key, j.candidate_key)}
                    by_q = {qs.question_id: qs.score for qs in j.question_scores}
                    for qid in q_ids:
                        row[qid] = by_q.get(qid)
                    row["weighted %"] = j.weighted_accuracy
                    matrix_rows.append(row)
                st.dataframe(
                    pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True
                )
                st.caption(
                    "Per-question 0–100 from DeepSeek R1 (semantic / synonym-aware). "
                    "Weighted % uses case weights."
                )

            with st.expander("Judge breakdown", expanded=False):
                label_by_key = {
                    c["key"]: (c.get("display_label") or c.get("label") or c["key"])
                    for c in candidates_cfg
                }
                for j in last_judgments:
                    name = label_by_key.get(j.candidate_key, j.candidate_key)
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
        # Flag only — never call a @st.dialog here (would collide with another dialog
        # already opened earlier in the same script run → StreamlitAPIException).
        st.session_state["show_run_done"] = True
        st.session_state.pop("confirmed_run", None)
        st.session_state.pop("pending_run", None)
        st.session_state["benchmark_running"] = False
        st.rerun()


    except Exception as exc:
        st.session_state["benchmark_running"] = False
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
        st.error(
            f"Run failed after {elapsed}s — the clock is stopped. "
            f"**{type(exc).__name__}:** {exc}\n\n"
            "Models that already finished may still have used OpenRouter credits."
        )
        st.stop()

# Persist ranking view after the run script finishes (next interactions)
elif st.session_state.get("last_ranking"):
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
        st.plotly_chart(
            fig_judge_mean_accuracy_bars(
                _sum.ranking_mean,
                title=f"Mean accuracy · N={_sum.n}",
                height=260,
            ),
            use_container_width=True,
            key="rank_chart_saved_multi",
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
                        st.session_state.pop("history_popup_path", None)
                        st.session_state.pop("show_run_done", None)
                        st.session_state.pop("show_history_mean_popup", None)
                        st.session_state["multi_run_popup_path"] = _p
                        st.rerun()
    else:
        st.markdown('<div class="sec-label">Last ranking</div>', unsafe_allow_html=True)
        st.plotly_chart(
            fig_judge_accuracy_bars(st.session_state["last_ranking"], height=260),
            use_container_width=True,
            key="rank_chart_saved",
        )
        rows = [
            {
                "#": r["rank"],
                "Model": r.get("label") or r["key"],
                "Acc %": r["accuracy"],
                "Status": (
                    "ok"
                    if r.get("status", "ok") == "ok"
                    else f"error · {r.get('status_note') or 'failed'}"
                ),
                "TTFT": r.get("ttft_s"),
                "TPS": r.get("tps"),
                "$": r.get("cost_usd"),
            }
            for r in st.session_state["last_ranking"]
        ]
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
    _lj = st.session_state.get("last_judgments") or []
    if _lj:
        st.markdown(
            '<div class="sec-label">Scores by clinical dimension</div>',
            unsafe_allow_html=True,
        )
        # Rebuild matrix from saved judgments
        q_ids = sorted(
            {qs.question_id for j in _lj for qs in j.question_scores}
        )
        matrix_rows = []
        for j in _lj:
            row = {"Model": j.candidate_key}
            by_q = {qs.question_id: qs.score for qs in j.question_scores}
            for qid in q_ids:
                row[qid] = by_q.get(qid)
            row["weighted %"] = j.weighted_accuracy
            matrix_rows.append(row)
        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)
# --- Offline: Rebuild mean across N runs (formula 50/30/20, $0 API) ---
st.markdown(
    '<div class="sec-label">Rebuild mean across N runs · $0 API</div>',
    unsafe_allow_html=True,
)
st.caption(
    f"**{case_display_name(case_id)}** · take the N newest saved runs, **rescore every one** with "
    "gold formula **50% alignment / 30% quality / 20% stem** (including older logs), "
    "then open a **popup** with table + chart. **No API calls** — local CPU only."
)
_hist_for_case = artifacts_for_case(WORKSPACE_DIR, case_id)
_avail_n = len(_hist_for_case)

# Always offer 3 / 5 / 10; clamp at rebuild time if fewer runs exist
_n_options = [3, 5, 10]
_rb1, _rb2 = st.columns([1, 2])
with _rb1:
    _default_idx = 1 if _avail_n >= 5 else 0  # prefer 5 when possible
    _rebuild_n = st.selectbox(
        "Average over N runs",
        options=_n_options,
        index=_default_idx,
        format_func=lambda n: (
            f"{n} runs"
            + (" · recommended" if n == 5 else "")
            + (" · best stability" if n == 10 else "")
            + (f"  (only {_avail_n} saved)" if _avail_n < n else "")
        ),
        key="history_rebuild_n_pick",
        help="Tiers: 3 · 5 · 10. If fewer runs are saved, rebuild uses all available.",
    )
with _rb2:
    st.caption(f"Saved runs for {case_display_name(case_id)}: **{_avail_n}**")
    _can_rebuild = _avail_n >= 2
    _do_rebuild = st.button(
        f"Rebuild mean · {_rebuild_n} runs · open KPI popup · $0",
        type="primary",
        use_container_width=True,
        disabled=not _can_rebuild,
        key="history_rebuild_btn",
        help="Offline rescore 50/30/20 + mean. Zero API cost.",
    )

if _avail_n < 2:
    st.info(
        f"Need at least **2** saved runs for {case_display_name(case_id)} "
        f"(found {_avail_n}). Run Single a few times, then rebuild the mean."
    )
elif _do_rebuild:
    _n_use = min(int(_rebuild_n), _avail_n)
    if _n_use < int(_rebuild_n):
        st.toast(
            f"Only {_avail_n} runs saved — averaging {_n_use} (requested {_rebuild_n}).",
            icon="ℹ️",
        )
    _built = rebuild_multi_from_history(
        WORKSPACE_DIR, case_id, n=_n_use
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
        st.session_state["last_ranking"] = [
            {
                "key": r["key"],
                "rank": r["rank"],
                "accuracy": r["accuracy_mean"],
                "label": short_model(str(r["key"])),
                "status": "ok",
                "std": r.get("std"),
                "cv_pct": r.get("cv_pct"),
            }
            for r in _sum_persist.ranking_mean
        ]
        st.session_state["last_multi_n"] = _sum_persist.n
        st.session_state["show_last_run_costs"] = False  # offline rebuild — no live $
        st.session_state["show_history_mean_popup"] = True
        st.rerun()

_prev = st.session_state.get("history_rebuild_result") or {}
if (
    _prev.get("ok")
    and isinstance(_prev.get("summary"), dict)
    and _prev["summary"].get("case_id") == case_id
):
    if st.button(
        f"Re-open mean popup · N={_prev['summary'].get('n')} · $0",
        use_container_width=False,
        key="history_rebuild_reopen",
    ):
        st.session_state["show_history_mean_popup"] = True
        st.rerun()

st.markdown('<div class="sec-label">Run history</div>', unsafe_allow_html=True)
st.caption(
    f"Private to your OpenRouter key ({short_owner_label()}). "
    "Same key on this app = same History for Custom Case + Demo 1/2. "
    "Other visitors with a different key cannot see your runs. "
    "Use **Rebuild mean across N runs** above for offline mean KPIs (formula 50/30/20)."
)
_hist_all = list_run_artifacts(WORKSPACE_DIR)
if not has_key:
    st.info(
        "Enter your OpenRouter API key (sidebar / welcome) to unlock **your** History. "
        "Without a key, cloud runs cannot start and History stays empty."
    )
elif not _hist_all:
    st.info("No saved runs for this API key yet — after a Single/Multi run they appear here.")
else:
    _default_path = st.session_state.get("history_path")
    _labels = []
    _path_by_label = {}
    for pth in _hist_all[:30]:
        try:
            a = load_artifact(pth)
            when = (a.finished_at or a.started_at or "")[:19].replace("T", " ")
            top = ""
            if a.ranking:
                top_row = a.ranking[0]
                top = f" · #1 {top_row.get('key')} {top_row.get('accuracy')}%"
            lab = (
                f"{case_display_name(a.case_id)} · {when} · "
                f"${a.total_cost_usd:.3f}{top} · {pth.name}"
            )
        except Exception:
            lab = pth.name
        _labels.append(lab)
        _path_by_label[lab] = str(pth)

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
    hist_path = Path(_path_by_label[chosen])
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
                    st.text(c.raw_response[:12000])
                elif c.answers:
                    for qid, ans in c.answers.items():
                        st.markdown(f"**{qid}**")
                        st.text((ans or "")[:4000])
                if j:
                    st.markdown("**Judge**")
                    for qs in j.question_scores:
                        st.caption(
                            f"{qs.question_id}: {qs.score}/100 — {qs.rationale}"
                        )

        same_case = []
        for pth in _hist_all:
            try:
                a = load_artifact(pth)
            except Exception:
                continue
            if a.case_id == hist.case_id and a.ranking:
                same_case.append(a)
        if len(same_case) >= 2:
            st.caption(
                f"{len(same_case)} saved runs for {case_display_name(hist.case_id)} — "
                "use **Rebuild mean across N runs** above for the chart "
                "(current formula, $0 API)."
            )
