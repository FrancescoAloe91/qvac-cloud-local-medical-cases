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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from benchmark.config import is_usable_openrouter_key, load_models_config
from benchmark.cases_loader import list_case_ids, load_case
from benchmark.config import ARTIFACTS_DIR
from benchmark.judge import build_ranking, explain_run_scores, judge_candidates_parallel
from benchmark.scoring import scoring_guide_markdown
from benchmark.qvac_bridge import available as qvac_available
from benchmark.qvac_bridge import ensure_sidecar as qvac_ensure_sidecar
from benchmark.qvac_bridge import health as qvac_health
from benchmark.qvac_bridge import reachable as qvac_reachable
from benchmark.qvac_bridge import iter_tokens as qvac_iter_tokens
from benchmark.prompts import candidate_system, candidate_user
from benchmark.report import (
    list_run_artifacts,
    load_artifact,
    print_summary_table,
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
from lib.charts import fig_judge_accuracy_bars
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Automated Benchmark · Demo Studio",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
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
  position: sticky !important;
  bottom: 10px !important;
  z-index: 40 !important;
  background: linear-gradient(165deg, #1c1917 0%, #0f172a 55%, #111827 100%) !important;
  border: 1px solid #f59e0b !important;
  color: #fde68a !important;
  font-family: "IBM Plex Mono", ui-monospace, monospace !important;
  padding: 0.7rem 0.75rem 0.65rem !important;
  border-radius: 12px !important;
  box-shadow: 0 10px 28px rgba(0,0,0,0.35) !important;
  margin: 0.85rem 0 0.35rem 0 !important;
}
[data-testid="stSidebar"] .run-timer-panel {
  position: relative !important;
  bottom: auto !important;
  margin-top: 0.75rem !important;
}
.run-timer-panel .t-title {
  font-size: 0.62rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #94a3b8 !important;
  margin: 0 0 0.45rem 0 !important;
}
.run-timer-panel .t-big {
  font-size: 1.55rem !important;
  font-weight: 700 !important;
  line-height: 1.1 !important;
  color: #fbbf24 !important;
  margin: 0 0 0.35rem 0 !important;
}
.run-timer-panel .t-row {
  display: flex !important;
  justify-content: space-between !important;
  align-items: baseline !important;
  gap: 0.4rem;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  margin: 0.12rem 0 !important;
  color: #fde68a !important;
}
.run-timer-panel .t-row .lab {
  color: #94a3b8 !important;
  font-weight: 500 !important;
  font-size: 0.68rem !important;
}
.run-timer-panel .t-row .val {
  font-variant-numeric: tabular-nums;
  color: #fef3c7 !important;
}
.run-timer-panel .t-row.active .val {
  color: #fbbf24 !important;
}
.run-timer-panel .t-sep {
  border: 0;
  border-top: 1px solid #334155;
  margin: 0.4rem 0 !important;
}
.run-timer-panel .phase {
  display: block;
  font-size: 0.62rem !important;
  font-weight: 500 !important;
  color: #cbd5e1 !important;
  margin-top: 0.35rem;
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

# --- Startup dialogs (once per session) ---
if "qvac_dialog_shown" not in st.session_state:
    st.session_state.qvac_dialog_shown = False
if "key_dialog_shown" not in st.session_state:
    st.session_state.key_dialog_shown = False

# Clear sticky run flag when idle (a Streamlit dialog X used to abort mid-flight).
if not st.session_state.get("confirmed_run") and not st.session_state.get("pending_run"):
    st.session_state["benchmark_running"] = False


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
            "MedPsy is available through the QVAC SDK on this machine. "
            "It will be included in the benchmark (on-device, $0 API)."
        )
        st.caption("Stack: QVAC SDK sidecar · MedPsy-4B GGUF · stock inference settings.")
    elif online:
        st.warning(
            "The QVAC SDK sidecar is online, but MedPsy is not loaded yet. "
            "This run will use **cloud models only** until the model finishes loading "
            "(or until you fix OpenSSL / restart the sidecar)."
        )
        st.markdown(QVAC_SETUP_GUIDE)
    else:
        st.warning(
            "The QVAC SDK sidecar is offline on this computer. "
            "This run will use **cloud models only** (ChatGPT / Claude / Gemini via OpenRouter)."
        )
        st.markdown(QVAC_SETUP_GUIDE)
    if st.button("OK", type="primary", use_container_width=True):
        st.session_state.qvac_dialog_shown = True
        st.rerun()


@st.dialog("OpenRouter API key required")
def key_required_dialog():
    st.markdown(
        "This app uses **bring-your-own-key**. Paste your **full** OpenRouter key "
        "to run cloud + DeepSeek R1 judge. "
        "Or continue without a key to rehearse **Run QVAC only · $0** (no ranking)."
    )
    st.caption(
        "Keys that contain `...` are placeholders and will get HTTP 401. "
        "Copy the complete key from https://openrouter.ai/keys"
    )
    k = st.text_input("OPENROUTER_API_KEY", type="password", key="dialog_or_key")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Continue without key", use_container_width=True):
            st.session_state.key_dialog_shown = True
            st.rerun()
    with c2:
        if st.button("Save key for this session", type="primary", use_container_width=True):
            if is_usable_openrouter_key(k):
                os.environ["OPENROUTER_API_KEY"] = k.strip()
                st.session_state["or_key_session"] = k.strip()
                st.session_state.key_dialog_shown = True
                st.rerun()
            else:
                st.error(
                    "Enter a complete OpenRouter key starting with sk-or-v1-… "
                    "(no dots/ellipsis in the middle)."
                )


@st.dialog("Saved run results", width="large")
def history_run_dialog(path_str: str):
    """Popup review of a past artifact (from sidebar History)."""
    try:
        hist = load_artifact(Path(path_str))
    except Exception as exc:
        st.error(f"Could not load run: {exc}")
        if st.button("Close", type="primary", use_container_width=True, key="hist_dlg_close_err"):
            st.session_state.pop("history_popup_path", None)
            st.rerun()
        return

    when = (hist.finished_at or hist.started_at or "")[:19].replace("T", " ")
    st.caption(f"{hist.case_id} · {when} · ${hist.total_cost_usd:.4f} · `{Path(path_str).name}`")

    m1, m2, m3 = st.columns(3)
    m1.metric("Case", hist.case_id)
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


# Prefer the setup guide when requested from the sidebar (so online users
# can always re-open the install steps they only see when the sidecar is offline).
if st.session_state.get("show_qvac_guide"):
    qvac_setup_guide_dialog()
elif st.session_state.get("history_popup_path"):
    history_run_dialog(st.session_state["history_popup_path"])
elif not st.session_state.qvac_dialog_shown:
    qvac_status_dialog(qvac_up, qvac_ok)

if not has_key and not st.session_state.get("or_key_session") and not st.session_state.key_dialog_shown:
    # Defer until after qvac dialog is dismissed; show on next paint
    if st.session_state.qvac_dialog_shown:
        key_required_dialog()

if st.session_state.get("or_key_session") and is_usable_openrouter_key(
    st.session_state["or_key_session"]
):
    os.environ["OPENROUTER_API_KEY"] = st.session_state["or_key_session"]
    has_key = True

st.markdown('<p class="demo-hero">QVAC vs Cloud · Health Benchmark</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="demo-sub">On-device MedPsy via QVAC SDK · BYOK OpenRouter · DeepSeek R1 blind judge · '
    "live TTFT / TPS on every model</p>",
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
    "**Naming:** Steps 1–3 = workflow · **Case A/B/C** = clinical scenario · "
    "ranking uses real model names (ChatGPT / Claude / Gemini / QVAC)."
)

# --- Sidebar: API key + QVAC (compact) ---
with st.sidebar:
    st.markdown("**OpenRouter**")
    if has_key:
        st.success("Key OK · cloud + R1")
    else:
        st.warning("No full key · Single/Multi off")
    key_in = st.text_input(
        "OPENROUTER_API_KEY",
        value="",
        type="password",
        help="Full sk-or-v1-… from openrouter.ai/keys",
        placeholder="sk-or-v1-…",
        label_visibility="collapsed",
    )
    if key_in:
        if is_usable_openrouter_key(key_in):
            os.environ["OPENROUTER_API_KEY"] = key_in.strip()
            st.session_state["or_key_session"] = key_in.strip()
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
    if st.button("Setup guide", use_container_width=True):
        st.session_state["show_qvac_guide"] = True
        st.rerun()

    st.caption(f"Judge · {(judge_cfg.get('display_label') or judge_cfg.get('model') or 'R1')[:42]}")

    # Compact recent-run picker → popup ONLY when user changes this dropdown (or View)
    _hist_paths = list_run_artifacts(ARTIFACTS_DIR)[:12]
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
                label = f"{art.case_id} · {when} · ${art.total_cost_usd:.2f}{top}"
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

# --- Steps 1–2 side by side (less scroll) ---
CASE_PICKER = {
    "caseA": "Case A · Teaching — STEMI + sildenafil",
    "caseB": "Case B · Teaching — Mania + CKD",
    "caseC": "Case C · CUSTOM — your real anonymized case",
}
case_ids = [c for c in ("caseA", "caseB", "caseC") if c in set(list_case_ids())]
st.markdown('<div class="sec-label">Case (A / B / C)</div>', unsafe_allow_html=True)


def _on_case_change() -> None:
    """Case A/B/C is for editing + RUN — never open a saved-run popup."""
    st.session_state.pop("history_popup_path", None)
    # Reset sidebar History to placeholder so it does not look "selected"
    opts = st.session_state.get("_hist_sidebar_opts") or {}
    placeholder = next((k for k, v in opts.items() if v is None), "— select a run —")
    st.session_state["hist_sidebar_pick"] = placeholder


case_id = st.selectbox(
    "Case",
    case_ids,
    index=0,
    format_func=lambda cid: CASE_PICKER.get(cid, cid),
    label_visibility="collapsed",
    key="case_pick",
    on_change=_on_case_change,
)
st.caption(
    "**Case A/B** = teaching vignette + built-in rubric (gold box can stay empty — "
    "`gold_summary` is sent to the judge automatically). "
    "**Case C** = you paste symptoms (step 1) + checklist gold (step 2); nothing auto-filled."
)
preset = load_case(case_id)
is_custom_real = (preset.mode or "") == "custom_real"

if is_custom_real:
    st.warning(
        "**Case C · CUSTOM** — paste your anonymized clinical text (step 1) and confirmed "
        "diagnosis / safety traps (step 2). No teaching answer grid."
    )

# Sync stem when case changes. A/B: prefill stem, clear gold. C: empty both.
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
        + (" (required · Case C)" if is_custom_real else " (leave empty for A/B)")
        + "</div>",
        unsafe_allow_html=True,
    )
    gold_reference = st.text_area(
        "gold",
        height=96 if not is_custom_real else 110,
        placeholder=(
            "Case C checklist (required):\n"
            "1) Primary diagnosis:\n"
            "2) Key differentials to reject:\n"
            "3) Safety traps:\n"
            "4) Urgency:\n"
            "5) Must-do plan steps:"
            if is_custom_real
            else "Leave empty for Cases A/B — teaching gold_summary is used automatically"
        ),
        key="demo_gold_ref",
        label_visibility="collapsed",
        disabled=not is_custom_real,
    )

live_case = preset.model_copy(update={"stem": (case_stem or "").strip() or preset.stem})
gold_reference = (gold_reference or "").strip()
# Teaching A/B: auto-pass case.gold_summary when the gold box is empty
if not gold_reference and not is_custom_real and (preset.gold_summary or "").strip():
    gold_reference = preset.gold_summary.strip()
effective_gold = gold_reference

exp_l, exp_r = st.columns(2)
with exp_l:
    with st.expander("Exact prompt (inference)", expanded=False):
        st.caption("Identical system + user for all four models.")
        st.markdown("**System**")
        st.code(candidate_system())
        st.markdown("**User**")
        st.code(candidate_user(live_case))
with exp_r:
    with st.expander("How ranking is calculated (must · acceptable · quality · stem)", expanded=True):
        st.markdown(scoring_guide_markdown())
        st.markdown("**This case — section weights**")
        for q in live_case.questions:
            st.markdown(f"- **{q.id}** ({q.weight:.0%}) · {q.kind}")

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
    total = breakdown.get("total_usd", 0)
    tok = breakdown.get("input_tokens_used_for_estimate", 0)
    chars = breakdown.get("chars_case_plus_gold", 0)
    return (
        '<div class="cost-compact">'
        + " · ".join(bits)
        + f' · <b>${total:.3f}</b>'
        + f'<br/><span style="opacity:.75">{chars} chars · ~{tok} in-tok</span>'
        + "</div>"
    )


def _fmt_cost_multi(breakdown: dict, n: int) -> str:
    per = breakdown.get("total_usd", 0)
    tot = breakdown.get("total_usd_for_n", 0)
    return (
        f'<div class="cost-compact cost-multi">'
        f"${per:.3f} × {n} → <b>${tot:.3f}</b></div>"
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
    with st.container():
        st.markdown(
            f"""
<div class="spend-modal-marker"></div>
<div class="spend-modal-card">
  <h3>Confirm OpenRouter spend</h3>
  <p>Estimated <b>${est:.4f}</b> for <b>{n}</b> run(s)
  (cloud models + DeepSeek R1 judge). QVAC = $0 if included.</p>
  <p class="muted">This spends your OpenRouter credits and produces ranking + charts.</p>
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


def _stream_html(text: str, live: bool = False) -> str:
    caret = '<span class="caret"></span>' if live else ""
    return f'<div class="stream-out">{html.escape(text or "")}{caret}</div>'


def _kpi_live_line(ttft_s, elapsed_s, tps_live) -> str:
    parts = []
    if ttft_s is not None:
        parts.append(f"TTFT {ttft_s}s")
    if tps_live is not None:
        parts.append(f"~{tps_live} TPS")
    if elapsed_s is not None:
        parts.append(f"{elapsed_s}s…")
    return " · ".join(parts) if parts else "streaming…"



def _run_timer_idle(last: dict | None = None) -> str:
    """Always-visible sidebar clock (idle or last finished timings) — static HTML."""
    last = last or {}
    total = last.get("total_s")
    if total is None:
        return """
<div class="run-timer-panel idle">
  <div class="t-title">Run clock</div>
  <div class="t-big">0s</div>
  <div class="t-row"><span class="lab">collect</span><span class="val">0s</span></div>
  <div class="t-row"><span class="lab">judge</span><span class="val">0s</span></div>
  <hr class="t-sep"/>
  <span class="phase">Ready · starts when you click RUN</span>
</div>
"""
    n = int(last.get("n") or 1)
    this_row = ""
    if n > 1 and last.get("last_run_s") is not None:
        this_row = (
            f'<div class="t-row"><span class="lab">last run</span>'
            f'<span class="val">{int(last["last_run_s"])}s</span></div>'
        )
    return f"""
<div class="run-timer-panel idle">
  <div class="t-title">Run clock · last</div>
  <div class="t-big">{int(total)}s</div>
  {this_row}
  <div class="t-row"><span class="lab">collect</span><span class="val">{int(last.get("collect_s") or 0)}s</span></div>
  <div class="t-row"><span class="lab">judge</span><span class="val">{int(last.get("judge_s") or 0)}s</span></div>
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
  font-size:0.62rem;font-weight:600;letter-spacing:0.08em;
  text-transform:uppercase;color:#94a3b8;margin:0 0 0.45rem 0;
}
.run-timer-panel .t-big{
  font-size:1.55rem;font-weight:700;line-height:1.1;color:#fbbf24;margin:0 0 0.35rem 0;
}
.run-timer-panel .t-row{
  display:flex;justify-content:space-between;align-items:baseline;gap:0.4rem;
  font-size:0.78rem;font-weight:600;margin:0.12rem 0;color:#fde68a;
}
.run-timer-panel .t-row .lab{color:#94a3b8;font-weight:500;font-size:0.68rem;}
.run-timer-panel .t-row .val{font-variant-numeric:tabular-nums;color:#fef3c7;}
.run-timer-panel .t-row.active .val{color:#fbbf24;}
.run-timer-panel .t-sep{border:0;border-top:1px solid #334155;margin:0.4rem 0;}
.run-timer-panel .phase{
  display:block;font-size:0.62rem;font-weight:500;color:#cbd5e1;margin-top:0.35rem;line-height:1.3;
}
.run-timer-panel.idle .t-big{color:#64748b;}
.run-timer-panel.idle .phase{color:#64748b;}
"""


def _paint_run_timer(slot, inner_html: str, *, height: int = 248) -> None:
    """Render timer in an iframe so <script> ticks (st.html DOMPurify strips scripts)."""
    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{_TIMER_IFRAME_CSS}</style></head><body>"
        f"{inner_html}</body></html>"
    )
    with slot:
        components.html(doc, height=height, scrolling=False)


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
  <div class="t-big"><span id="t-total">{et}</span>s</div>
  <div class="t-row" style="display:{this_display}">
    <span class="lab">this run</span><span class="val"><span id="t-this">{eh}</span>s</span>
  </div>
  <div class="t-row" id="row-c"><span class="lab">collect</span><span class="val"><span id="t-collect">{int(collect_base)}</span>s</span></div>
  <div class="t-row" id="row-j"><span class="lab">judge</span><span class="val"><span id="t-judge">{int(judge_base)}</span>s</span></div>
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
  function paint() {{
    var add = Math.floor((Date.now() - paintAt) / 1000);
    var totalS = baseTotal + add;
    var thisS = baseThis + add;
    var collectS = cBase + (bucket === 'collect' ? add : 0);
    var judgeS = jBase + (bucket === 'judge' ? add : 0);
    if (totEl) totEl.textContent = String(totalS);
    if (thisEl) thisEl.textContent = String(thisS);
    if (colEl) colEl.textContent = String(collectS);
    if (judEl) judEl.textContent = String(judgeS);
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
) -> str:
    dual = int(n_runs) > 1
    this_show = int(this_s) if this_s is not None else int(total_s)
    this_row = ""
    if dual:
        this_row = (
            f'<div class="t-row"><span class="lab">last run</span>'
            f'<span class="val">{this_show}s</span></div>'
        )
    return f"""
<div class="run-timer-panel">
  <div class="t-title">Run clock · done</div>
  <div class="t-big">{int(total_s)}s</div>
  {this_row}
  <div class="t-row"><span class="lab">collect</span><span class="val">{int(collect_s)}s</span></div>
  <div class="t-row"><span class="lab">judge</span><span class="val">{int(judge_s)}s</span></div>
  <hr class="t-sep"/>
  <span class="phase">Done · final scores ready</span>
</div>
"""


# --- Always-visible Run clock in sidebar (bottom) ---
with st.sidebar:
    st.markdown("---")
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
        )
    else:
        _paint_run_timer(
            timer_slot,
            _run_timer_idle(st.session_state.get("last_run_timings")),
            height=220,
        )

# --- Live response panels ---
roster = list(cfg.get("candidates") or [])
saved_outputs = st.session_state.get("live_outputs") or {}
_run_pending = st.session_state.get("confirmed_run") or {}
running_now = bool(_run_pending)
qvac_only_now = _run_pending.get("mode") == "qvac_only"

st.markdown('<div class="sec-label">Live responses</div>', unsafe_allow_html=True)
card_cols = st.columns(len(roster) or 1)
text_boxes, kpi_boxes, status_boxes = {}, {}, {}
for i, c in enumerate(roster):
    key = c["key"]
    prev = saved_outputs.get(key) or {}
    color = c.get("color") or "#64748b"
    with card_cols[i]:
        st.markdown(
            f'<div class="panel-frame" style="border-top: 3px solid {color}">'
            f'<p class="live-head">{c.get("label") or c.get("key")}</p>'
            f'<p class="live-meta">{c.get("model")}</p></div>',
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
                    _stream_html("Cloud skipped — QVAC-only rehearsal", live=False),
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
                text_boxes[key].markdown(_stream_html("", live=True), unsafe_allow_html=True)
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
                _stream_html(prev.get("text") or "", live=False),
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
                    _stream_html("", live=False),
                    unsafe_allow_html=True,
                )
            else:
                status_boxes[key].markdown(
                    _status_pill("ready", "Ready"),
                    unsafe_allow_html=True,
                )
                text_boxes[key].markdown(
                    _stream_html("", live=False),
                    unsafe_allow_html=True,
                )

# --- Execute confirmed run ---
if st.session_state.get("confirmed_run"):
    run_cfg = st.session_state.pop("confirmed_run")
    st.session_state["benchmark_running"] = True
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

    # ---- Free local rehearsal: MedPsy only, live token stream, $0 ----
    if run_mode == "qvac_only":
        if not case_stem.strip():
            st.error("Clinical case is empty.")
            st.stop()
        if is_custom_real and not gold_reference.strip():
            st.error("Case C · CUSTOM requires confirmed diagnosis in step 2.")
            st.stop()
        if not qvac_run_ok:
            st.error(
                "QVAC SDK sidecar offline — start it: `cd sidecar && npm start` "
                "(requires OpenSSL 3: `brew install openssl@3`)."
            )
            st.stop()

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
                        f'<div class="stream-out">{html.escape(buf)}'
                        f'<span class="caret"></span></div>',
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
            f'<div class="stream-out">{html.escape(buf or "(empty)")}</div>',
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
        st.error(
            "OpenRouter API key missing or invalid (truncated placeholder). "
            "Paste the full key from https://openrouter.ai/keys in the sidebar."
        )
        st.stop()
    if not case_stem.strip():
        st.error("Clinical case is empty.")
        st.stop()
    if is_custom_real and not gold_reference.strip():
        st.error(
            "Case C · CUSTOM requires confirmed diagnosis in step 2 "
            "(no teaching answer grid)."
        )
        st.stop()

    try:
        prep = prepare_run(
            case_id,
            skip_qvac=skip_qvac or not include_qvac,
            require_qvac=False,
        )
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

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
        for c in candidates_cfg:
            status_boxes[c["key"]].markdown(
                _status_pill("wait", "Streaming…"), unsafe_allow_html=True
            )
            text_boxes[c["key"]].markdown(_stream_html("", live=True), unsafe_allow_html=True)

        for evt in iter_collect_live(case_obj, candidates_cfg, blind_map):
            if evt.get("type") == "token":
                key = evt["key"]
                bufs[key] = bufs.get(key, "") + (evt.get("delta") or "")
                tok_n[key] = tok_n.get(key, 0) + 1
                # Throttle UI paints — same $ cost, smoother recording
                if tok_n[key] == 1 or tok_n[key] % 3 == 0:
                    text_boxes[key].markdown(
                        _stream_html(bufs[key], live=True), unsafe_allow_html=True
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
                    _stream_html(text, live=False), unsafe_allow_html=True
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
                st.write(f"**{name}**: {j.weighted_accuracy}%")
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
                "estimated_breakdown": bd if n_runs == 1 else bd_multi,
            },
            candidates=collected,
            judgments=judgments,
            ranking=ranking,
            total_cost_usd=round(total_cost, 6),
            notes="",
        )
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        write_artifact(artifact, ARTIFACTS_DIR)
        all_artifacts.append(artifact)
        last_ranking = ranking
        last_judgments = judgments
        last_collected = collected

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
            f"**total {total_s}s** (timer) · last run {last_this_s}s"
        )
        st.caption(f"Per run · {per_bits}")
    else:
        st.caption(
            f"**Wall time** · collect {collect_s}s · judge {judge_s}s · "
            f"**total {total_s}s** (same as sidebar Run clock)"
        )

    if last_judgments:
        explain = explain_run_scores(case_obj, last_judgments)
        st.session_state["last_score_explain"] = explain
        with st.expander("Why these scores (formula · weights · discriminators)", expanded=True):
            st.markdown(scoring_guide_markdown())
            st.markdown("---")
            st.markdown(f"**This run · formula** · `{explain['formula']}`")
            st.caption(explain.get("note") or "")
            st.markdown(
                "**Section weights** · "
                + " · ".join(f"`{k}` {v:.0%}" for k, v in explain["section_weights"].items())
            )
            st.markdown(
                "**Heaviest / discriminating** · "
                + ", ".join(explain.get("heaviest_sections") or [])
                + (
                    f" · ids: {', '.join(explain.get('main_discriminators') or [])}"
                    if explain.get("main_discriminators")
                    else ""
                )
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
                st.dataframe(pd.DataFrame(rows_ex), use_container_width=True, hide_index=True)
            st.caption(
                "**Quality** = clinical judgment on that section (correct call, coherent DDx/plan, "
                "case-specific, safe) — not writing style. "
                "Ties broken by safety → quality → stem specificity → diagnosis."
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

    res_l, res_r = st.columns([1.2, 1])
    with res_l:
        if last_ranking:
            st.plotly_chart(
                fig_judge_accuracy_bars(last_ranking, height=230),
                use_container_width=True,
                key="rank_chart_live",
            )
    with res_r:
        st.caption("Table · accuracy + KPI")
        if last_ranking:
            rows = [
                {
                    "#": r["rank"],
                    "Model": r.get("label") or r["key"],
                    "Acc %": r["accuracy"],
                    "TTFT": r.get("ttft_s"),
                    "TPS": r.get("tps"),
                    "$": r.get("cost_usd"),
                }
                for r in last_ranking
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Actual $")
        st.dataframe(pd.DataFrame(cost_rows), use_container_width=True, hide_index=True)

    # Per-question clinical scores (meaning of the comparison)
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
        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Per-question 0–100 from DeepSeek R1 (semantic / synonym-aware). "
            "Weighted % uses case weights (e.g. diagnosis heavy, safety trap on Case A)."
        )

    with st.expander("Judge breakdown (last run)", expanded=False):
        label_by_key = {
            c["key"]: (c.get("display_label") or c.get("label") or c["key"])
            for c in candidates_cfg
        }
        for j in last_judgments:
            name = label_by_key.get(j.candidate_key, j.candidate_key)
            st.markdown(f"**{name} · {j.weighted_accuracy}%**")
            for qs in j.question_scores:
                st.caption(f"{qs.question_id}: {qs.score}/100 — {qs.rationale}")

    if len(all_artifacts) > 1:
        summary = summarize_runs(all_artifacts)
        write_summary(summary, ARTIFACTS_DIR)
        with st.expander("Multi-run stats", expanded=True):
            st.code(print_summary_table(summary))

    st.caption(f"Saved under `{ARTIFACTS_DIR}`")


# Persist ranking view after the run script finishes (next interactions)
elif st.session_state.get("last_ranking"):
    st.markdown('<div class="sec-label">Last ranking</div>', unsafe_allow_html=True)
    res_l, res_r = st.columns([1.2, 1])
    with res_l:
        st.plotly_chart(
            fig_judge_accuracy_bars(st.session_state["last_ranking"], height=230),
            use_container_width=True,
            key="rank_chart_saved",
        )
    with res_r:
        rows = [
            {
                "#": r["rank"],
                "Model": r.get("label") or r["key"],
                "Acc %": r["accuracy"],
                "TTFT": r.get("ttft_s"),
                "TPS": r.get("tps"),
                "$": r.get("cost_usd"),
            }
            for r in st.session_state["last_ranking"]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if st.session_state.get("last_cost_rows"):
            st.caption("Actual $")
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
st.markdown('<div class="sec-label">Run history</div>', unsafe_allow_html=True)
st.caption(
    "Every Single/Multi run saves full answers + judge JSON under `artifacts/`. "
    "**Multi run ×N** computes mean/median/std for that batch. "
    "Pick a past run below to re-read each model’s answer."
)
_hist_all = list_run_artifacts(ARTIFACTS_DIR)
if not _hist_all:
    st.info("No saved runs yet — after a Single/Multi run they appear here.")
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
            lab = f"{a.case_id} · {when} · ${a.total_cost_usd:.3f}{top} · {pth.name}"
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
                    height=200,
                ),
                use_container_width=True,
                key="hist_rank_chart",
            )

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
            with st.expander(
                f"Average across {len(same_case)} saved runs · {hist.case_id}",
                expanded=False,
            ):
                summary = summarize_runs(same_case)
                st.code(print_summary_table(summary))
