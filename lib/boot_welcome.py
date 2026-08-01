"""Shared BYOK + QVAC SDK boot dialogs (Comprehension home and Structured page).

API key: every fresh Streamlit session re-prompts (BYOK).
QVAC SDK ack: remembered in ``.ui_prefs.json`` (never store API keys there).
"""

from __future__ import annotations

import streamlit as st

from benchmark.config import is_usable_openrouter_key
from lib.ui_prefs import load_qvac_sdk_ack, save_qvac_sdk_ack

QVAC_SETUP_GUIDE = """
**QVAC SDK (on-device MedPsy)**  
Local GGUF inference runs through a sidecar on this machine — not through OpenRouter.
If the sidecar is offline, cloud slots can still run with your OpenRouter key; MedPsy
slots stay unavailable until the sidecar is up.

Setup: start the sidecar (`sidecar/`), ensure MedPsy GGUFs are present under `models/`.
"""


def init_boot_state() -> None:
    if "qvac_sdk_ack" not in st.session_state:
        st.session_state.qvac_sdk_ack = load_qvac_sdk_ack()
    if "boot_welcome_done" not in st.session_state:
        st.session_state.boot_welcome_done = False
    if "boot_step" not in st.session_state:
        st.session_state.boot_step = "api"


def _mask_api_key(key: str) -> str:
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
    st.rerun()


def _remember_openrouter_key(key: str) -> None:
    key = (key or "").strip()
    if not is_usable_openrouter_key(key):
        return
    st.session_state["or_key_session"] = key


def _advance_boot_after_key() -> None:
    if st.session_state.get("qvac_sdk_ack"):
        _advance_boot("done")
    else:
        _advance_boot("qvac")


def _acknowledge_qvac_boot() -> None:
    st.session_state.qvac_sdk_ack = True
    save_qvac_sdk_ack(True)
    _advance_boot("done")


@st.dialog("OpenRouter API key", width="small")
def key_welcome_dialog() -> None:
    existing = (st.session_state.get("or_key_session") or "").strip()
    if existing and not is_usable_openrouter_key(existing):
        existing = ""

    st.markdown(
        "This app uses **bring-your-own-key**. Confirm or paste your **full** OpenRouter key "
        "for cloud models + DeepSeek R1 judge. "
        "Or continue without a key to rehearse **QVAC-only** collect (no paid ranking)."
    )
    st.caption("Keys stay in this Streamlit session only · never written to `.ui_prefs.json`.")
    if existing:
        st.info(f"Key available for this session (hidden): `{_mask_api_key(existing)}`")
    if "dialog_or_key" not in st.session_state:
        st.session_state["dialog_or_key"] = existing
    k = st.text_input(
        "OPENROUTER_API_KEY",
        type="password",
        key="dialog_or_key",
        help="Full sk-or-v1-… key · no ellipsis placeholders",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Continue without key", use_container_width=True, key="boot_key_skip"):
            _advance_boot_after_key()
    with c2:
        if st.button(
            "Confirm current key",
            use_container_width=True,
            disabled=not bool(existing),
            key="boot_key_keep",
        ):
            _remember_openrouter_key(existing)
            _advance_boot_after_key()
    with c3:
        if st.button(
            "Use / update key", type="primary", use_container_width=True, key="boot_key_save"
        ):
            typed = (k or "").strip()
            if is_usable_openrouter_key(typed):
                _remember_openrouter_key(typed)
                _advance_boot_after_key()
            elif existing and (not typed or typed == existing):
                _remember_openrouter_key(existing)
                _advance_boot_after_key()
            else:
                st.error(
                    "Enter a complete OpenRouter key starting with sk-or-v1-… "
                    "(no dots/ellipsis in the middle)."
                )


@st.dialog("QVAC SDK / MedPsy status", width="small")
def qvac_status_dialog(*, online: bool, loaded: bool) -> None:
    if loaded:
        st.success(
            "MedPsy is **active** through the QVAC SDK on this machine. "
            "On-device slots can run ($0 API for those candidates)."
        )
        st.caption("Stack: QVAC SDK sidecar · MedPsy GGUF · stock inference settings.")
    elif online:
        st.warning(
            "QVAC sidecar is **online**, but MedPsy is **not loaded** yet. "
            "On-device paused until the model finishes loading."
        )
        st.markdown(QVAC_SETUP_GUIDE)
    else:
        st.warning(
            "QVAC sidecar is **offline** on this computer. "
            "Cloud models still work with OpenRouter; MedPsy slots stay unavailable."
        )
        st.markdown(QVAC_SETUP_GUIDE)
    if st.button("OK · continue", type="primary", use_container_width=True, key="qvac_boot_ok"):
        _acknowledge_qvac_boot()


def run_boot_dialogs(
    *,
    qvac_online: bool,
    qvac_loaded: bool,
    pending_spend: bool = False,
    running: bool = False,
) -> None:
    """Show API key → QVAC dialogs when idle. No-op mid-run / mid-spend confirm."""
    init_boot_state()
    if pending_spend or running:
        return
    if st.session_state.get("boot_welcome_done"):
        return
    step = st.session_state.get("boot_step", "api")
    if step == "api":
        key_welcome_dialog()
    elif step == "qvac":
        qvac_status_dialog(online=qvac_online, loaded=qvac_loaded)
