"""Shared BYOK + QVAC SDK boot dialogs (Comprehension home and Structured page).

API key: every fresh Streamlit session re-prompts (BYOK).
QVAC SDK ack: remembered in ``.ui_prefs.json`` (never store API keys there).

Structured may set ``st.session_state["boot_show_account"] = True`` before calling
``run_boot_dialogs`` to show optional Supabase account UI in the key dialog.
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
    # Legacy flags kept in sync for older Structured checks.
    if "qvac_dialog_shown" not in st.session_state:
        st.session_state.qvac_dialog_shown = False
    if "key_dialog_shown" not in st.session_state:
        st.session_state.key_dialog_shown = False


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
        st.session_state.key_dialog_shown = True
        st.session_state.qvac_dialog_shown = True
    st.rerun()


def _remember_openrouter_key(key: str) -> None:
    key = (key or "").strip()
    if not is_usable_openrouter_key(key):
        return
    st.session_state["or_key_session"] = key
    # Optional Structured account vault (no-op if store not configured).
    try:
        from lib.deployment import is_streamlit_cloud
        from lib.secure_account_store import AccountSession
        from lib.secure_account_store import configured as account_store_configured
        from lib.secure_account_store import save_openrouter_key as account_save_key

        account = st.session_state.get("account_session")
        if account_store_configured() and isinstance(account, AccountSession):
            account_save_key(account, key)
            st.session_state["_account_key_remembered"] = True
        else:
            st.session_state["_account_key_remembered"] = False
        _ = is_streamlit_cloud  # silence unused when account path unused
    except Exception:
        st.session_state["_account_key_remembered"] = False


def _advance_boot_after_key() -> None:
    """Always show QVAC SDK status after the API-key step (every fresh session)."""
    _advance_boot("qvac")


def _acknowledge_qvac_boot() -> None:
    st.session_state.qvac_sdk_ack = True
    save_qvac_sdk_ack(True)
    _advance_boot("done")


def _render_account_block() -> None:
    """Optional Supabase account UI (Structured). Fail soft if store unavailable."""
    try:
        from lib.deployment import is_streamlit_cloud
        from lib.secure_account_store import AccountSession
        from lib.secure_account_store import configured as account_store_configured
        from lib.secure_account_store import sign_in as account_sign_in
        from lib.secure_account_store import sign_up as account_sign_up
    except Exception:
        return

    account = st.session_state.get("account_session")
    if account_store_configured() and not isinstance(account, AccountSession):
        st.markdown("**Private account storage**")
        email = st.text_input("Email", key="account_email")
        password = st.text_input("Password", type="password", key="account_password")
        auth_in, auth_up = st.columns(2)
        with auth_in:
            if st.button("Sign in", use_container_width=True, key="account_sign_in"):
                try:
                    account = account_sign_in(email, password)
                    st.session_state["account_session"] = account
                    st.session_state.pop("_account_key_loaded", None)
                    st.session_state.pop("_account_artifacts_synced", None)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Sign-in failed: {exc}")
        with auth_up:
            if st.button("Create account", use_container_width=True, key="account_sign_up"):
                try:
                    account = account_sign_up(email, password)
                    if account is None:
                        st.success("Check your email, confirm the account, then sign in.")
                    else:
                        st.session_state["account_session"] = account
                        st.session_state.pop("_account_key_loaded", None)
                        st.session_state.pop("_account_artifacts_synced", None)
                        st.rerun()
                except Exception as exc:
                    st.error(f"Account creation failed: {exc}")
        st.caption(
            "API key and benchmark artifacts are encrypted before storage; "
            "Supabase RLS restricts every row to the authenticated user."
        )
    elif isinstance(account, AccountSession):
        st.success(f"Signed in · {account.email}")
    else:
        try:
            from lib.deployment import is_hosted_byok_required

            hosted = is_streamlit_cloud() or is_hosted_byok_required()
        except Exception:
            hosted = is_streamlit_cloud()
        if hosted:
            st.caption(
                "Hosted without Supabase · key and history stay session-only (not durable)."
            )


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
    if st.session_state.get("boot_show_account"):
        _render_account_block()
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
    other_dialog_open: bool = False,
    show_account: bool = False,
) -> None:
    """Show API key → QVAC dialogs when idle. No-op mid-run / mid-spend confirm.

    Call once per script run from the active page (Comprehension home or Structured).
    Fresh Streamlit sessions always re-ask for the API key; QVAC SDK ack is local.
    """
    init_boot_state()
    if show_account:
        st.session_state["boot_show_account"] = True
    if pending_spend or running or other_dialog_open:
        return
    if st.session_state.get("boot_welcome_done"):
        return
    step = st.session_state.get("boot_step", "api")
    if step == "api":
        key_welcome_dialog()
    elif step == "qvac":
        qvac_status_dialog(online=qvac_online, loaded=qvac_loaded)
    elif step != "done":
        # Recover unknown step (e.g. partial migration) → re-ask key.
        st.session_state.boot_step = "api"
        key_welcome_dialog()
