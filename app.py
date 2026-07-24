"""QVAC vs Cloud LLMs - Health Test — Medical benchmark dashboard."""

import os
import time

import pandas as pd
import streamlit as st

from lib import charts, diagnosis_compare, medpsy, metrics, reset, session_store, ui
from lib.browser import cloud_url, copy_to_clipboard, open_all_cloud_tabs
from lib.cases import CASE_IDS, build_prompt, case_meta, case_text_for, default_case_text
from lib.i18n import DEFAULT_LANG, t
from lib.lang_switch import apply_language_switch
from lib.metrics import TABLE_MODEL_SHORT, _L
from lib.tiers import MODEL_CONFIG, build_qvac_prompt
from lib.runtime_env import is_streamlit_cloud

_CLOUD_TIERS_IMPORT_ERROR = None
try:
    from lib.cloud_tiers import (
        effective_tier_labels,
        load_tier_labels,
        normalize_tier_labels_dict,
        save_tier_labels,
        tier_label,
    )
except Exception as _cloud_tiers_exc:  # pragma: no cover - cloud diagnostics
    import traceback as _tb

    _CLOUD_TIERS_IMPORT_ERROR = (
        f"{type(_cloud_tiers_exc).__name__}: {_cloud_tiers_exc}\n"
        f"{_tb.format_exc()}"
    )

    def load_tier_labels():
        return {
            "chatgpt": "5.5 Instant",
            "claude": "Sonnet 5 · Extra",
            "gemini": "3.5 Flash · extended thinking",
            "qvac": "MedPsy-4B (cloud demo)",
        }

    def normalize_tier_labels_dict(labels):
        return dict(labels or load_tier_labels())

    def effective_tier_labels(labels=None):
        return normalize_tier_labels_dict(labels)

    def save_tier_labels(labels):
        return None

    def tier_label(model_key, labels=None):
        return (labels or load_tier_labels()).get(model_key, "")

st.set_page_config(
    page_title="QVAC vs Cloud LLMs - Health Test",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject_css()

STREAMLIT_CLOUD = is_streamlit_cloud()

if _CLOUD_TIERS_IMPORT_ERROR:
    st.error("cloud_tiers failed to load — using fallback labels. Details:")
    st.code(_CLOUD_TIERS_IMPORT_ERROR)

ALL_KEYS = ("chatgpt", "claude", "gemini", "qvac")


def widget_key(model_key: str) -> str:
    return f"out_{model_key}"


def set_output_widget(model_key: str, value: str) -> None:
    st.session_state[widget_key(model_key)] = value


if "lang" not in st.session_state:
    st.session_state.lang = DEFAULT_LANG
for k, v in [
    ("benchmark_results", {}),
    ("user_outputs", {}),
    ("case_text", default_case_text(DEFAULT_LANG)),
    ("case_id", "case1"),
    ("case_slot", "case1"),
    ("case_text_version", 0),
    ("browser_info", None),
    ("use_gold_standard", False),
    ("gold_standard_text", ""),
    ("lang_switch_notice", None),
    ("saved_slots", {}),
    ("view_slot", None),
    ("last_snapshot_ready", None),
    ("sell_step", None),
    ("qvac_thinking", ""),
]:
    if k not in st.session_state:
        st.session_state[k] = v

if "cloud_tier_labels" not in st.session_state:
    st.session_state.cloud_tier_labels = load_tier_labels()
st.session_state.cloud_tier_labels = normalize_tier_labels_dict(st.session_state.cloud_tier_labels)

if not st.session_state.saved_slots:
    st.session_state.saved_slots = session_store.load_slots()

lang = st.session_state.lang


def _tier_kw() -> dict:
    labels = effective_tier_labels(st.session_state.cloud_tier_labels)
    st.session_state.cloud_tier_labels["qvac"] = labels["qvac"]
    return {"tier_labels": labels}


def _tier_labels() -> dict:
    return effective_tier_labels(st.session_state.cloud_tier_labels)


def _case_label(case_id=None) -> str:
    cid = case_id or st.session_state.case_id
    return t(f"cases.{cid}", lang) if cid else t("case.specialty.custom", lang)


def _save_slot_id():
    """Slot sidebar (case1–case5) per salvataggio — il caso 5 resta case5 anche se editi il testo."""
    cid = st.session_state.case_id
    if cid in CASE_IDS:
        return cid
    slot = st.session_state.get("case_slot")
    return slot if slot in CASE_IDS else None


def effective_outputs() -> dict:
    out = {}
    results = st.session_state.benchmark_results
    for key in ALL_KEYS:
        wk = widget_key(key)
        edited = st.session_state.get(wk, st.session_state.user_outputs.get(key, "")).strip()
        if edited:
            out[key] = edited
        elif key in results:
            out[key] = results[key].get("content", "")
        else:
            out[key] = ""
    return out


def _load_case(case_id: str) -> None:
    st.session_state.case_id = case_id
    st.session_state.case_slot = case_id
    st.session_state.case_text = case_text_for(case_id, lang)
    st.session_state.case_text_version += 1
    st.session_state.benchmark_results = {}
    st.session_state.user_outputs = {}
    st.session_state.view_slot = None
    for key in ALL_KEYS:
        if widget_key(key) in st.session_state:
            del st.session_state[widget_key(key)]


def _recall_slot(case_id: str) -> None:
    """Richiama risultato salvato e sincronizza editor caso (+ gold se presente)."""
    st.session_state.view_slot = case_id
    st.session_state.case_id = case_id
    st.session_state.case_slot = case_id
    st.session_state.case_text = case_text_for(case_id, lang)
    st.session_state.case_text_version += 1
    snap = st.session_state.saved_slots.get(case_id) or {}
    if snap.get("use_gold"):
        st.session_state.use_gold_standard = True
        st.session_state.gold_standard_text = snap.get("gold_standard_text") or ""
    else:
        st.session_state.use_gold_standard = False
        st.session_state.gold_standard_text = ""
    if "gold_standard_input" in st.session_state:
        del st.session_state["gold_standard_input"]


def _case5_save_ui(slots: dict) -> dict:
    """Labels for case 5 multi-save (rolling window of CASE5_MAX_RUNS)."""
    count = session_store.slot_run_count(slots, "case5")
    max_r = session_store.CASE5_MAX_RUNS
    at_cap = count >= max_r
    return {
        "count": count,
        "max": max_r,
        "next_n": min(count + 1, max_r),
        "at_cap": at_cap,
    }


def _save_current_snapshot() -> None:
    ready = st.session_state.get("last_snapshot_ready")
    slot_id = _save_slot_id()
    if not ready or not slot_id or ready.get("case_id") != slot_id:
        return
    if slot_id == "case5":
        before = session_store.slot_run_count(st.session_state.saved_slots, "case5")
        st.session_state.saved_slots = session_store.save_slot(
            st.session_state.saved_slots, slot_id, ready
        )
        after = session_store.slot_run_count(st.session_state.saved_slots, "case5")
        max_r = session_store.CASE5_MAX_RUNS
        if before >= max_r:
            st.toast(
                t("sidebar.save_case5_run_toast_rolling", lang, max=max_r),
                icon="💾",
            )
        else:
            st.toast(
                t("sidebar.save_case5_run_toast", lang, n=after, max=max_r),
                icon="💾",
            )
        st.rerun()
        return
    st.session_state.saved_slots = session_store.save_slot(
        st.session_state.saved_slots, slot_id, ready
    )
    st.session_state.view_slot = slot_id
    st.toast(t("sidebar.save_slot_toast", lang, case=_case_label(slot_id)), icon="💾")
    st.rerun()


def _render_sidebar_slot(index: int, case_id: str) -> None:
    """Slot = una riga: info a sinistra, Recall/Load a destra."""
    filled = session_store.slot_is_filled(st.session_state.saved_slots, case_id)
    viewing = st.session_state.view_slot == case_id
    short = t(f"case.short.{case_id}", lang)
    snap = st.session_state.saved_slots.get(case_id) or {}
    saved_at = session_store.slot_latest_saved_at(snap)
    n_runs = (
        session_store.slot_run_count(st.session_state.saved_slots, case_id)
        if case_id == "case5"
        else 0
    )
    runs_tag = f"{n_runs}/{session_store.CASE5_MAX_RUNS}" if case_id == "case5" and filled else ""
    state = "viewing" if viewing else ("filled" if filled else "empty")

    if viewing:
        btn_label = t("sidebar.slot_viewing", lang)
    elif filled:
        btn_label = t("sidebar.slot_recall", lang)
    else:
        btn_label = t("sidebar.slot_load", lang)

    with st.container(border=True):
        col_info, col_btn = st.columns([6.5, 2.5], gap="small")
        with col_info:
            st.markdown(
                ui.case_slot_header_html(index, short, state, saved_at, lang, runs_tag=runs_tag),
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button(
                btn_label,
                key=f"slot_btn_{case_id}",
                use_container_width=True,
                type="primary" if (viewing or st.session_state.case_id == case_id) else "secondary",
                disabled=viewing,
            ):
                if filled:
                    _recall_slot(case_id)
                else:
                    _load_case(case_id)
                st.rerun()


def _render_sidebar_case_slots() -> None:
    """Slot 1–5 in colonna singola — compatti ma leggibili."""
    n_filled = len([c for c in CASE_IDS if session_store.slot_is_filled(st.session_state.saved_slots, c)])
    st.markdown(
        ui.sidebar_section_title_html(t("sidebar.saved_cases", lang), f"{n_filled}/{len(CASE_IDS)}"),
        unsafe_allow_html=True,
    )
    for i, cid in enumerate(CASE_IDS, 1):
        _render_sidebar_slot(i, cid)


def _render_sidebar_finale_actions() -> None:
    with st.container(border=True):
        st.markdown('<span class="sidebar-finale-marker"></span>', unsafe_allow_html=True)
        if st.button(
            "🎬 " + t("sidebar.finale_btn", lang),
            key="btn_finale_ranking",
            use_container_width=True,
        ):
            finale_ranking_dialog()


def _render_sidebar_danger_zone() -> None:
    """Destructive actions — separated from ranking / guide controls."""
    st.divider()
    with st.container(border=True):
        st.markdown('<span class="sidebar-danger-zone-marker"></span>', unsafe_allow_html=True)
        if st.button(
            "🗑️ " + t("sidebar.clear_saved", lang),
            key="btn_clear_saved_cases",
            use_container_width=True,
            help=t("sidebar.clear_saved_help", lang),
        ):
            clear_saved_dialog()


def _render_main_save_bar(can_save: bool, slot_id: str) -> None:
    """Salva solo nell'area risultati (main dashboard)."""
    if not can_save or not slot_id:
        return
    with st.container(border=True):
        sc1, sc2 = st.columns([1, 2.2])
        with sc1:
            if slot_id == "case5":
                c5 = _case5_save_ui(st.session_state.saved_slots)
                lbl = (
                    t("sidebar.save_case5_run_rolling", lang, max=c5["max"])
                    if c5["at_cap"]
                    else t(
                        "sidebar.save_case5_run",
                        lang,
                        n=c5["next_n"],
                        max=c5["max"],
                    )
                )
            else:
                lbl = "💾 " + t("sidebar.save_slot", lang)
            if st.button(lbl, key="btn_save_main", type="primary", use_container_width=True):
                _save_current_snapshot()
        with sc2:
            if slot_id == "case5":
                c5 = _case5_save_ui(st.session_state.saved_slots)
                if c5["at_cap"]:
                    st.markdown(f"**{t('session.save_case5_ready_rolling', lang, max=c5['max'])}**")
                    st.caption(t("session.save_case5_ready_short_rolling", lang, max=c5["max"]))
                else:
                    st.markdown(
                        f"**{t('session.save_case5_ready', lang, n=c5['next_n'], max=c5['max'])}**"
                    )
                    st.caption(
                        t(
                            "session.save_case5_ready_short",
                            lang,
                            n=c5["next_n"],
                            max=c5["max"],
                        )
                    )
            else:
                st.markdown(f"**{t('session.save_ready', lang, case=_case_label(slot_id))}**")
                st.caption(t("session.save_ready_short", lang, case=_case_label(slot_id)))


# =====================================================================
# Dialogs (defined after `lang` so titles are translated on every rerun)
# =====================================================================


@st.dialog(t("onboarding.title", lang), width="large")
def onboarding_dialog():
    steps = [
        ("onboarding.step1_title", "onboarding.step1_body"),
        ("onboarding.step2_title", "onboarding.step2_body"),
        ("onboarding.step3_title", "onboarding.step3_body"),
        ("onboarding.step4_title", "onboarding.step4_body"),
    ]
    for title_key, body_key in steps:
        st.markdown(f"**{t(title_key, lang)}**")
        st.caption(t(body_key, lang))
    st.divider()
    if st.button(t("onboarding.cta", lang), type="primary", use_container_width=True):
        st.rerun()


@st.dialog(t("reset.confirm_title", lang))
def reset_confirm_dialog():
    st.warning(t("reset.confirm_body", lang))
    c1, c2 = st.columns(2)
    with c1:
        if st.button(t("reset.confirm_cancel", lang), use_container_width=True):
            st.rerun()
    with c2:
        if st.button(t("reset.confirm_yes", lang), type="primary", use_container_width=True):
            reset.reset_session()
            st.rerun()


@st.dialog(t("sidebar.finale_title", lang), width="large")
def finale_ranking_dialog():
    dialog_L = _L(lang)
    slots = st.session_state.saved_slots
    history = session_store.slots_as_history(slots)
    avg_df = metrics.build_final_consensus_average(history, lang)
    case5_runs = session_store.slot_runs(slots.get("case5"))
    gold_entries = [
        r["entry"] for r in case5_runs if r.get("entry", {}).get("mode") == "gold_standard"
    ]
    if gold_entries:
        gold_df, gold_case = metrics.build_final_gold_ranking_average(gold_entries, lang)
    else:
        gold_df, gold_case = metrics.build_final_gold_ranking(history, lang)
    st.caption(t("session.finale_caption", lang))

    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown(f"##### {t('session.finale_avg', lang)}")
        n_std = len([
            c for c in CASE_IDS[:4]
            if session_store.slot_is_filled(slots, c)
            and (session_store.slot_latest_snapshot(slots.get(c)) or {}).get("entry", {}).get("mode") != "gold_standard"
        ])
        st.caption(t("session.finale_avg_help", lang, n=max(n_std, 0)))
        if not avg_df.empty:
            st.plotly_chart(charts.fig_consensus_ranking_bars(avg_df, lang, height=220, **_tier_kw()), use_container_width=True)
            ui.show_model_table(
                avg_df,
                lang,
                _tier_labels(),
                columns=[dialog_L["model"], dialog_L["score_cons_rescaled"], dialog_L["rank_consensus"]],
            )
        else:
            st.info(t("session.finale_avg_empty", lang))
    with fc2:
        st.markdown(f"##### {t('session.finale_gold', lang)}")
        if not gold_df.empty:
            st.caption(t("session.finale_gold_help", lang, case=gold_case))
            if len(gold_entries) > 1:
                st.caption(
                    t(
                        "session.finale_gold_avg_help",
                        lang,
                        n=len(gold_entries),
                        max=session_store.CASE5_MAX_RUNS,
                    )
                )
            st.plotly_chart(charts.fig_clinical_ranking_bars(gold_df, lang, height=220, **_tier_kw()), use_container_width=True)
            ui.show_model_table(
                gold_df,
                lang,
                _tier_labels(),
                columns=[dialog_L["model"], dialog_L["score_clin_short"], dialog_L["rank_clinical"]],
            )
        else:
            st.info(t("session.finale_gold_empty", lang))
    if st.button(t("decision.close", lang), type="primary", use_container_width=True):
        st.rerun()


@st.dialog(t("sidebar.clear_saved_title", lang))
def clear_saved_dialog():
    st.warning(t("sidebar.clear_saved_body", lang))
    c1, c2 = st.columns(2)
    with c1:
        if st.button(t("reset.confirm_cancel", lang), use_container_width=True):
            st.rerun()
    with c2:
        st.markdown('<span class="dialog-danger-confirm-marker"></span>', unsafe_allow_html=True)
        if st.button(t("sidebar.clear_saved_confirm", lang), use_container_width=True, key="btn_clear_saved_confirm"):
            session_store.clear_slots()
            st.session_state.saved_slots = {}
            st.session_state.view_slot = None
            st.session_state.last_snapshot_ready = None
            st.rerun()


@st.dialog(t("card.full_dialog_title", lang), width="large")
def full_response_dialog(model_key: str):
    cfg = MODEL_CONFIG[model_key]
    text = st.session_state.get(widget_key(model_key), "").strip()
    st.markdown(f"### {cfg['icon']} {cfg['name']}")
    if text:
        st.markdown(text)
        if model_key == "qvac" and st.session_state.get("qvac_thinking"):
            with st.expander("🧠 " + t("card.reasoning_expander", lang)):
                st.caption(t("card.reasoning_caption", lang))
                st.markdown(
                    f'<div class="qvac-live-stream" style="max-height:320px;">'
                    f'{st.session_state.qvac_thinking}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info(t("card.status_empty", lang))


@st.dialog(t("explain.dialog_title", lang), width="large")
def explain_scores_dialog(explanations: list, use_gold: bool):
    st.caption(t("explain.dialog_intro", lang))
    if not explanations:
        st.info(t("warn.no_output", lang))
        return

    model_tabs = st.tabs([f"{e['icon']} {e['name']}" for e in explanations])
    for tab, e in zip(model_tabs, explanations):
        with tab:
            cons = e["consensus"]
            st.markdown(
                f"#### {e['icon']} {e['name']} — "
                f"{t('cols.final_score', lang)}: **{e['final_score']}%**"
            )

            st.markdown(f"**{t('explain.consensus_label', lang)}**")
            st.caption(t("explain.consensus_desc", lang))

            if cons["semantic_available"]:
                d1, d2, d3, d4 = st.columns(4)
                dims = cons["dimensions"]
                d1.metric(t("cols.diagnosis_match_short", lang), f"{dims['diagnosis']}%" if dims["diagnosis"] is not None else "—")
                d2.metric(t("cols.plan_next_steps_short", lang), f"{dims['management']}%" if dims["management"] is not None else "—")
                d3.metric(t("cols.clinical_summary_short", lang), f"{dims['summary']}%" if dims["summary"] is not None else "—")
                d4.metric(t("cols.urgency_agreement_short", lang), f"{dims['urgency']}%" if dims["urgency"] is not None else "—")

                for dim_key, dim_label_key in (
                    ("diagnosis", "cols.diagnosis_match_short"),
                    ("management", "cols.plan_next_steps_short"),
                    ("summary", "cols.clinical_summary_short"),
                ):
                    pairs = cons["pairs"].get(dim_key) or []
                    if pairs:
                        st.caption(t("explain.dimension_pairs", lang, dim=t(dim_label_key, lang)))
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        t("explain.select_model", lang): t("explain.vs", lang, name=p["other"]),
                                        t("cols.semantic_short", lang): f"{p['value']}%",
                                    }
                                    for p in pairs
                                ]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
            else:
                st.caption(t("explain.semantic_unavailable_note", lang))

            st.markdown(f"**{t('explain.formula_label', lang)}**")
            st.code(
                t(
                    "explain.formula_consensus_dim",
                    lang,
                    formula=cons["formula"],
                    raw=cons["raw"],
                    rescaled=cons["rescaled"],
                ),
                language=None,
            )
            st.caption(t("explain.consensus_rescale_note", lang))

            if use_gold and e.get("gold"):
                g = e["gold"]
                st.divider()
                st.markdown(f"**{t('explain.gold_label', lang)}**")
                st.caption(t("explain.gold_desc", lang))
                gd = g["dimensions"]
                gcol1, gcol2, gcol3, gcol4 = st.columns(4)
                gcol1.metric(t("cols.primary_accuracy", lang), f"{gd['diagnosis']}%" if gd["diagnosis"] is not None else "—")
                gcol2.metric(t("cols.ddx_coverage", lang), f"{gd['management']}%" if gd["management"] is not None else "—")
                gcol3.metric(t("cols.semantic_gold_short", lang), f"{gd['urgency']}%" if gd["urgency"] is not None else "—")
                gcol4.metric(t("cols.clinical_summary_short", lang), f"{gd['summary']}%" if gd["summary"] is not None else "—")
                st.metric(f"🎓 {t('explain.grade_label', lang)}", f"{g['grade_10']} / 10")
                st.caption(t("explain.grade_rubric", lang))
                st.code(
                    t(
                        "explain.formula_gold_dim",
                        lang,
                        formula=g["formula"],
                        score=g["score"],
                    ),
                    language=None,
                )

            u = e.get("urgency", {})
            st.divider()
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                st.caption(t("explain.urgency_label", lang))
                st.write(u.get("label") or "—")
            with fcol2:
                st.caption(t("explain.privacy_label", lang))
                st.write(f"{e['privacy']}%")


@st.dialog(t("decision.cloud_dialog_title", lang))
def cloud_decision_dialog():
    st.error(t("decision.cloud_dialog_body", lang))
    if st.button(t("decision.close", lang), use_container_width=True):
        st.rerun()


@st.dialog(t("decision.qvac_dialog_title", lang))
def qvac_decision_dialog():
    st.success(t("decision.qvac_dialog_body", lang))
    if st.button(t("decision.close", lang), use_container_width=True):
        st.rerun()


# --- Sidebar ---
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">🩺 <b>QVAC Health Test</b></div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Prefer the reproducible runner? Open **Automated Benchmark** "
        "(sidebar page) — OpenRouter + LLM judge, no paste."
    )
    st.divider()
    _render_sidebar_case_slots()
    st.divider()
    _render_sidebar_finale_actions()

    if st.button(t("sidebar.guide_btn", lang), use_container_width=True, type="primary"):
        onboarding_dialog()

    if st.button(
        f"🔄 {t('sidebar.reset', lang)}",
        use_container_width=True,
        help=t("sidebar.reset_help", lang),
    ):
        reset_confirm_dialog()

    with st.expander("☁️ " + t("sidebar.cloud_tiers", lang), expanded=False):
        st.caption(t("sidebar.cloud_tiers_help", lang))
        labels = dict(st.session_state.cloud_tier_labels)
        labels["chatgpt"] = st.text_input("ChatGPT", value=labels.get("chatgpt", ""), key="tier_chatgpt")
        labels["claude"] = st.text_input("Claude", value=labels.get("claude", ""), key="tier_claude")
        labels["gemini"] = st.text_input("Gemini", value=labels.get("gemini", ""), key="tier_gemini")
        qvac_label = medpsy.runtime_tier_label()
        st.text_input("QVAC MedPsy", value=qvac_label, disabled=True, key="tier_qvac_display")
        st.caption(t("sidebar.cloud_tiers_qvac_auto", lang))
        if st.button(t("sidebar.cloud_tiers_save", lang), use_container_width=True):
            labels["qvac"] = qvac_label
            st.session_state.cloud_tier_labels = labels
            save_tier_labels(labels)
            st.toast(t("sidebar.cloud_tiers_saved", lang), icon="✅")
        st.caption(t("sidebar.cloud_tiers_howto", lang))

    st.markdown(
        f'<p class="sidebar-footer-note">{t("sidebar.privacy_note", lang)}</p>',
        unsafe_allow_html=True,
    )
    _render_sidebar_danger_zone()

# --- Header ---
hdr_l, hdr_r = st.columns([5, 1.3])
with hdr_l:
    st.markdown(f'<p class="app-title fade-in">🩺 {t("title", lang)}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="app-subtitle">{t("subtitle", lang)}</p>', unsafe_allow_html=True)
    st.markdown(
        ui.live_chip_html(medpsy.runtime_header_chip()),
        unsafe_allow_html=True,
    )
with hdr_r:
    lc1, lc2 = st.columns(2)
    with lc1:
        if st.button("🇬🇧 EN", use_container_width=True, type="primary" if lang == "en" else "secondary"):
            if lang != "en":
                apply_language_switch(lang, "en")
                st.session_state.lang = "en"
                st.rerun()
    with lc2:
        if st.button("🇮🇹 IT", use_container_width=True, type="primary" if lang == "it" else "secondary"):
            if lang != "it":
                apply_language_switch(lang, "it")
                st.session_state.lang = "it"
                st.rerun()

if st.session_state.lang_switch_notice:
    st.info(st.session_state.lang_switch_notice)

if STREAMLIT_CLOUD:
    st.info(t("cloud.demo_banner", lang))
    st.caption(t("cloud.demo_link", lang))

# --- Compact progress stepper (replaces bulky numbered section headers) ---
_run_done = bool(st.session_state.benchmark_results)
_any_pasted = any(
    (st.session_state.get(widget_key(k), "") or st.session_state.user_outputs.get(k, "")).strip()
    for k in ALL_KEYS
)
st.markdown(
    ui.stepper_html(
        [
            (t("stepper.case", lang), "done"),
            (t("stepper.run", lang), "done" if _run_done else "current"),
            (t("stepper.results", lang), "done" if _any_pasted else ("current" if _run_done else "todo")),
        ]
    ),
    unsafe_allow_html=True,
)

# =====================================================================
# 1) Caso attivo — selezione solo dalla colonna sinistra (slot)
# =====================================================================
st.markdown(ui.eyebrow_html("🗂️", t("eyebrow.case", lang)), unsafe_allow_html=True)

if st.session_state.view_slot:
    st.info(t("sidebar.viewing_saved", lang, case=_case_label(st.session_state.view_slot)))
else:
    st.caption(t("case.sidebar_nav_hint", lang))

if st.session_state.case_id:
    _am = case_meta(st.session_state.case_id)
    st.markdown(
        ui.case_info_bar_html(
            _am["icon"],
            _am["color"],
            t(_am["specialty_key"], lang),
            t(f"cases.{st.session_state.case_id}", lang),
            t("case.focus_label", lang),
            t(_am["focus_key"], lang),
        ),
        unsafe_allow_html=True,
    )
elif st.session_state.get("case_slot") in CASE_IDS:
    _am = case_meta(st.session_state.case_slot)
    st.markdown(
        ui.case_info_bar_html(
            _am["icon"],
            _am["color"],
            t(_am["specialty_key"], lang),
            t(f"case.short.{st.session_state.case_slot}", lang),
            t("case.focus_label", lang),
            t(_am["focus_key"], lang),
        ),
        unsafe_allow_html=True,
    )
else:
    st.caption("✏️ " + t("case.custom_hint", lang))

with st.expander("📝 " + t("case.clinical", lang), expanded=(st.session_state.case_id is None)):
    case_text = st.text_area(
        t("case.clinical", lang),
        value=st.session_state.case_text,
        height=100,
        key=f"case_editor_v{st.session_state.case_text_version}",
        label_visibility="collapsed",
    )
    if case_text != st.session_state.case_text:
        st.session_state.case_text = case_text
        # Caso 5 è un template da compilare: resta sullo slot case5 per poter salvare.
        if st.session_state.case_id != "case5":
            st.session_state.case_id = None
        else:
            st.session_state.case_slot = "case5"

full_prompt = build_prompt(st.session_state.case_text, lang)

# =====================================================================
# 2) Run — QVAC local inference + cloud prompt
# =====================================================================
st.markdown(ui.eyebrow_html("⚙️", t("eyebrow.run", lang)), unsafe_allow_html=True)

btn_run, btn_cloud = st.columns([2, 1])
with btn_run:
    run_benchmark = st.button("🚀 " + t("benchmark.run", lang), type="primary", use_container_width=True)
with btn_cloud:
    open_cloud = st.button(
        "🌐 " + t("browser.open_btn", lang),
        use_container_width=True,
        help=t("browser.help", lang),
    )

if open_cloud:
    st.session_state.browser_info = open_all_cloud_tabs(lang)
    st.rerun()

st.caption(t("run.caption", lang))

qvac_prompt = build_qvac_prompt(full_prompt, lang)

with st.expander(t("prompt.expander", lang), expanded=False):
    st.code(full_prompt, language="text")
    if st.button("📋 " + t("prompt.copy", lang)):
        if copy_to_clipboard(full_prompt):
            st.success(t("prompt.copied", lang))

st.caption(t("caption.cloud", lang))

if run_benchmark:
    if _save_slot_id() == "case5" or st.session_state.get("case_slot") == "case5":
        st.session_state.view_slot = None
    qvac = None
    if not medpsy.ollama_available():
        st.error(t("error.ollama_offline", lang))
    elif not medpsy.model_ready():
        st.error(t("error.model_missing", lang))
    else:
        live_metrics_bar = st.empty()
        with st.status(t("status.step_loading", lang), expanded=True) as status:
            status.update(label=t("status.step_generating", lang))
            stream_box = st.empty()
            partial_text = ""
            result = None
            t0 = time.time()
            ttft_s = None
            token_count = 0
            for event in medpsy.stream_inference(qvac_prompt):
                if event.get("delta"):
                    piece = event["delta"]
                    if ttft_s is None:
                        ttft_s = round(time.time() - t0, 2)
                    partial_text += piece
                    token_count += max(1, len(piece.split()))
                    elapsed = time.time() - t0
                    gen_elapsed = max(elapsed - (ttft_s or 0), 0.001)
                    live_tps = round(token_count / gen_elapsed, 1) if ttft_s else 0.0
                    preview = partial_text[-1800:]
                    stream_box.markdown(
                        f'<div class="qvac-live-stream">{preview}▌</div>',
                        unsafe_allow_html=True,
                    )
                    live_metrics_bar.markdown(
                        f'<div class="qvac-live-metrics-bar">'
                        f'⏱ <b>{elapsed:.1f}s</b> · '
                        f'📝 <b>{token_count}</b> {t("status.live_words", lang)} · '
                        f'⚡ TTFT <b>{ttft_s:.1f}s</b> · '
                        f'📈 <b>{live_tps:.1f}</b> tok/s'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                if event.get("done"):
                    result = event
            stream_box.empty()
            if result and not result.get("error"):
                stats = result.get("stats", {})
                live_metrics_bar.markdown(
                    f'<div class="qvac-live-metrics-bar qvac-live-metrics-done">'
                    f'✅ {t("status.step_done", lang)} · '
                    f'⏱ <b>{stats.get("latency_s", 0):.1f}s</b> · '
                    f'📝 <b>{len(result.get("content", "").split())}</b> {t("status.live_words", lang)} · '
                    f'⚡ TTFT <b>{stats.get("ttft_s", 0):.1f}s</b> · '
                    f'📈 <b>{stats.get("tps", 0):.1f}</b> tok/s'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                live_metrics_bar.empty()

            if not result or result.get("error"):
                status.update(label=t("status.step_error", lang), state="error")
                st.error(t("error.ollama_runtime", lang, error=(result or {}).get("error", "unknown")))
            else:
                status.update(label=t("status.step_done", lang), state="complete")
                qvac = result

    if qvac:
        qvac.update(model_key="qvac", name=MODEL_CONFIG["qvac"]["name"], tier="local")
        content = qvac.get("content", "")
        st.session_state.benchmark_results = {"qvac": qvac}
        set_output_widget("qvac", content)
        st.session_state.user_outputs["qvac"] = content
        st.session_state.qvac_thinking = qvac.get("thinking", "")
        # Deliberately do NOT touch chatgpt/claude/gemini here: running (or
        # re-running) QVAC must never erase what the user already pasted from
        # the cloud sites for this same clinical case.

if st.session_state.browser_info:
    info = st.session_state.browser_info
    st.success(f"✅ {t('browser.opened', lang)}: {', '.join(info['opened'])}")
    st.caption(info.get("note", ""))

# --- Gold standard (compact) ---
with st.expander("🎯 " + t("gold.section", lang), expanded=st.session_state.use_gold_standard):
    use_gold = st.checkbox(
        t("gold.use", lang),
        value=st.session_state.use_gold_standard,
        help=t("gold.help", lang),
    )
    st.session_state.use_gold_standard = use_gold
    if use_gold:
        st.caption(t("gold.caption", lang))
        st.caption(t("gold.qvac_variance", lang, max=session_store.CASE5_MAX_RUNS))
        gold_text = st.text_area(
            t("gold.input", lang),
            value=st.session_state.gold_standard_text,
            height=70,
            placeholder=t("gold.placeholder", lang),
            key="gold_standard_input",
            label_visibility="collapsed",
        )
        st.session_state.gold_standard_text = gold_text
    else:
        st.session_state.gold_standard_text = ""

# =====================================================================
# 4) Model output cards
# =====================================================================
st.markdown(ui.eyebrow_html("💬", t("eyebrow.output", lang)), unsafe_allow_html=True)
results = st.session_state.benchmark_results
c1, c2 = st.columns(2)
c3, c4 = st.columns(2)

for col, key in zip([c1, c2, c3, c4], ALL_KEYS):
    cfg = MODEL_CONFIG[key]

    with col:
        wk = widget_key(key)
        if wk not in st.session_state:
            initial = st.session_state.user_outputs.get(key, "")
            if not initial and key in results:
                initial = results[key].get("content", "")
            st.session_state[wk] = initial

        current_text = st.session_state.get(wk, "").strip()

        badge_html = (
            f'<span class="tier-local">{t("badge.local", lang)}</span>'
            if not cfg["cloud"] and current_text
            else ""
        )

        word_count = len(current_text.split()) if current_text else 0
        status_html = (
            f'<span class="status-pill status-filled">{t("card.status_filled", lang, n=word_count)}</span>'
            if current_text
            else f'<span class="status-pill status-empty">{t("card.status_empty", lang)}</span>'
        )

        link_html = (
            f'<a class="card-link-pill" href="{cloud_url(key, lang)}" target="_blank" rel="noopener">'
            f'{t("output.link", lang)}</a>'
            if cfg["cloud"] and cfg.get("url")
            else ""
        )
        vendor_line = cfg["vendor"]
        tier_txt = tier_label(key, effective_tier_labels(st.session_state.cloud_tier_labels))
        if tier_txt:
            vendor_line = f'{vendor_line} · <span class="cloud-tier-tag">{tier_txt}</span>'
        st.markdown(
            f'<div class="model-card fade-in" style="--model-color:{cfg["color"]};">'
            f'<div class="model-card-head">'
            f'<span class="model-card-name"><span class="m-icon">{cfg["icon"]}</span>{cfg["name"]} '
            f'{badge_html}{link_html}</span>{status_html}</div>'
            f'<div class="model-vendor">{vendor_line}</div>'
            f'<div class="model-instructions">'
            f'{t("card.instructions_cloud", lang, name=cfg["name"]) if cfg["cloud"] else t("card.instructions_local", lang)}'
            f"</div></div>",
            unsafe_allow_html=True,
        )

        placeholder = (
            t("output.placeholder_cloud", lang)
            if cfg["cloud"]
            else t("output.placeholder_qvac", lang)
        )
        st.text_area(
            t("output.diagnosis", lang),
            height=160,
            placeholder=placeholder,
            key=wk,
            label_visibility="collapsed",
        )
        st.session_state.user_outputs[key] = st.session_state[wk]

        fbtn1, fbtn2 = st.columns([2, 1])
        with fbtn1:
            if key == "qvac" and key in results:
                stats = results[key].get("stats", {})
                if stats:
                    st.caption(
                        t(
                            "output.stats",
                            lang,
                            ttft=stats.get("ttft_s") or "—",
                            tps=stats.get("tps") or "—",
                            latency=stats.get("latency_s") or "—",
                            ram=stats.get("ram_gb") or "—",
                        )
                    )
        with fbtn2:
            if st.button(t("card.view_full", lang), key=f"view_{key}", use_container_width=True, disabled=not current_text):
                full_response_dialog(key)

st.button("🔄 " + t("recalc", lang), key="recalc_btn", use_container_width=True)

outputs = effective_outputs()
has_any_output = any(v.strip() for v in outputs.values())
L = _L(lang)

view_slot = st.session_state.view_slot
saved_slots = st.session_state.saved_slots
viewing_saved = bool(view_slot and view_slot in saved_slots)

compare = None
ranking_df = pd.DataFrame()
model_keys: list = []
use_gold_cmp = False
display_results = results

if viewing_saved:
    snap = saved_slots[view_slot]
    runs = session_store.slot_runs(snap)
    latest = session_store.slot_latest_snapshot(snap) or snap
    if len(runs) > 1:
        ranking_df = metrics.build_averaged_ranking_from_snapshots(runs, lang)
    elif compare and model_keys:
        ranking_df = metrics.build_unified_ranking(compare, model_keys, lang)
    else:
        ranking_df = session_store.ranking_df_from_snapshot(latest, lang)
    compare = latest.get("compare") or snap.get("compare")
    model_keys = latest.get("model_keys") or list(compare.get("diagnoses", {}).keys())
    use_gold_cmp = bool(latest.get("use_gold") or compare.get("mode") == "gold_standard")
    display_results = latest.get("results") or snap.get("results") or {}
    vs_label = latest.get("case_label") or snap.get("case_label") or _case_label(view_slot)
    saved_at = session_store.slot_latest_saved_at(snap)
    time_label = (
        t("sidebar.slot_saved_at", lang, time=session_store.format_saved_at(saved_at, lang))
        if saved_at
        else ""
    )
    run_note = (
        t("sidebar.viewing_case5_avg", lang, n=len(runs))
        if view_slot == "case5" and len(runs) > 1
        else ""
    )
    bc1, bc2 = st.columns([4, 1])
    with bc1:
        msg = t("sidebar.viewing_saved", lang, case=vs_label)
        if time_label:
            msg += f" · {time_label}"
        if run_note:
            msg += f" · {run_note}"
        st.info(msg)
    with bc2:
        if st.button(t("sidebar.return_live", lang), use_container_width=True, key="btn_return_live"):
            st.session_state.view_slot = None
            st.rerun()
elif not has_any_output:
    st.warning(t("warn.no_output", lang))
else:
    compare_input = {
        k: {"output": outputs[k], "name": MODEL_CONFIG[k]["name"]}
        for k in ALL_KEYS
        if outputs[k].strip()
    }
    if compare_input:
        gold_ref = (
            st.session_state.gold_standard_text.strip()
            if st.session_state.use_gold_standard
            else None
        )
        compare = diagnosis_compare.compare_all(compare_input, gold_standard_text=gold_ref)
        model_keys = list(compare_input.keys())
        use_gold_cmp = compare.get("mode") == "gold_standard"
        ranking_df = metrics.build_unified_ranking(compare, model_keys, lang)
        slot_id = _save_slot_id() or st.session_state.case_id
        entry = metrics.build_session_entry(
            _case_label(slot_id), slot_id, ranking_df, compare, lang
        )
        if entry and slot_id in CASE_IDS:
            gold_txt = (
                st.session_state.gold_standard_text.strip()
                if st.session_state.use_gold_standard
                else ""
            )
            st.session_state.last_snapshot_ready = session_store.make_snapshot(
                slot_id,
                _case_label(slot_id),
                compare,
                ranking_df,
                model_keys,
                lang,
                entry,
                results,
                gold_standard_text=gold_txt,
            )
        else:
            st.session_state.last_snapshot_ready = None

if compare is not None:
        # KPI command center — always visible, side by side
        # -------------------------------------------------------------
        st.markdown(ui.eyebrow_html("📌", t("eyebrow.results", lang)), unsafe_allow_html=True)

        qvac_latency = display_results.get("qvac", {}).get("stats", {}).get("latency_s")
        time_saved = metrics.time_saved_seconds(qvac_latency) if qvac_latency else None

        def _leader_row(rank_col, score_col):
            if ranking_df.empty:
                return None, "—"
            row = ranking_df[ranking_df[rank_col] == 1].iloc[0]
            return row, row[score_col]

        k1, k2, k3 = st.columns(3)
        if use_gold_cmp:
            best_clin_row, best_clin_score = _leader_row(L["rank_clinical"], L["score_clin_short"])
            best_cons_row, best_cons_score = _leader_row(L["rank_consensus"], L["score_cons_rescaled"])
            with k1:
                best_html = (
                    f'{MODEL_CONFIG[best_clin_row["key"]]["icon"]} {best_clin_row[L["model"]]}<br>'
                    f'<span style="font-size:1.1rem;font-weight:700;color:#f5c518">{best_clin_score}%</span>'
                    if best_clin_row is not None
                    else "—"
                )
                st.markdown(
                    ui.kpi_tile_html(t("kpi.best_clinical", lang), best_html, accent=True, delay=1),
                    unsafe_allow_html=True,
                )
            with k2:
                best_html = (
                    f'{MODEL_CONFIG[best_cons_row["key"]]["icon"]} {best_cons_row[L["model"]]}<br>'
                    f'<span style="font-size:1.1rem;font-weight:700;color:#6ee7b7">{best_cons_score}%</span>'
                    if best_cons_row is not None
                    else "—"
                )
                st.markdown(
                    ui.kpi_tile_html(t("kpi.best_consensus", lang), best_html, delay=2),
                    unsafe_allow_html=True,
                )
            with k3:
                tri_agree = compare.get("urgency_agreement", 0.0)
                st.markdown(
                    ui.kpi_tile_html(
                        t("kpi.triage_agreement", lang), f"{tri_agree:.0f}%",
                        sub=t("kpi.triage_agreement_help", lang), delay=3,
                    ),
                    unsafe_allow_html=True,
                )
        else:
            best_cons_row, best_cons_score = _leader_row(L["rank_consensus"], L["score_cons_rescaled"])
            with k1:
                best_html = (
                    f'{MODEL_CONFIG[best_cons_row["key"]]["icon"]} {best_cons_row[L["model"]]}<br>'
                    f'<span style="font-size:1.1rem;font-weight:700;color:#6ee7b7">{best_cons_score}%</span>'
                    if best_cons_row is not None
                    else "—"
                )
                st.markdown(
                    ui.kpi_tile_html(t("kpi.best_model", lang), best_html, accent=True, delay=1),
                    unsafe_allow_html=True,
                )
            with k2:
                tri_agree = compare.get("urgency_agreement", 0.0)
                st.markdown(
                    ui.kpi_tile_html(
                        t("kpi.triage_agreement", lang), f"{tri_agree:.0f}%",
                        sub=t("kpi.triage_agreement_help", lang), delay=2,
                    ),
                    unsafe_allow_html=True,
                )
            with k3:
                ts_val = f"{time_saved:.0f}s" if time_saved is not None else "—"
                st.markdown(
                    ui.kpi_tile_html(
                        t("kpi.time_saved", lang),
                        ts_val,
                        sub=t("kpi.time_saved_help", lang, overhead=metrics.CLOUD_MANUAL_OVERHEAD_S),
                        delay=3,
                    ),
                    unsafe_allow_html=True,
                )

        # -------------------------------------------------------------
        # Ranking hero — order, chart and score immediately visible,
        # right under the KPI tiles, with a written "why" underneath.
        # Since TPS/TTFT can't be measured on cloud sites without an API,
        # this is deliberately centered on real diagnostic content instead.
        # -------------------------------------------------------------
        st.markdown(ui.eyebrow_html("🏆", t("ranking.section", lang)), unsafe_allow_html=True)

        slot_id = _save_slot_id()
        can_save_main = bool(
            not viewing_saved
            and st.session_state.get("last_snapshot_ready")
            and slot_id
            and st.session_state.last_snapshot_ready.get("case_id") == slot_id
        )
        _render_main_save_bar(can_save_main, slot_id)

        if not ranking_df.empty:
            if use_gold_cmp:
                best_clin = ranking_df[ranking_df[L["rank_clinical"]] == 1].iloc[0]
                best_cons = ranking_df[ranking_df[L["rank_consensus"]] == 1].iloc[0]
                st.info(
                    t(
                        "ranking.dual_summary",
                        lang,
                        clin_name=TABLE_MODEL_SHORT.get(
                            best_clin["key"], best_clin[L["model"]]
                        ),
                        clin_score=best_clin[L["score_clin_short"]],
                        cons_name=best_cons[L["model"]],
                        cons_score=best_cons[L["score_cons_rescaled"]],
                    )
                )
                st.caption(t("ranking.dual_explanation", lang))
                st.caption(t("ranking.clinical_chart_note", lang))

            if use_gold_cmp:
                rk_col1, rk_col2 = st.columns(2)
                with rk_col1:
                    st.plotly_chart(
                        charts.fig_consensus_ranking_bars(ranking_df, lang, height=280, **_tier_kw()),
                        use_container_width=True,
                        key="chart_ranking_consensus",
                    )
                with rk_col2:
                    st.plotly_chart(
                        charts.fig_clinical_ranking_bars(ranking_df, lang, height=280, **_tier_kw()),
                        use_container_width=True,
                        key="chart_ranking_clinical",
                    )
            else:
                st.plotly_chart(
                    charts.fig_consensus_ranking_bars(ranking_df, lang, height=280, **_tier_kw()),
                    use_container_width=True,
                    key="chart_ranking_hero",
                )
            st.markdown(ui.score_story_html(use_gold_cmp, lang), unsafe_allow_html=True)
            st.markdown(ui.privacy_strip_html(ranking_df, lang), unsafe_allow_html=True)

            consensus_df = metrics.build_consensus_table(
                compare, model_keys, lang, tier_labels=_tier_labels()
            )
            ui.show_model_table(
                consensus_df,
                lang,
                _tier_labels(),
            )
            st.caption(t("ranking.legend", lang))
            if not compare.get("semantic_available"):
                st.caption(t("kpi.semantic_unavailable", lang))

            narrative_items = metrics.build_ranking_narrative(ranking_df, compare, model_keys, lang, use_gold_cmp)
            if narrative_items:
                st.markdown(ui.eyebrow_html("📝", t("narrative.section", lang)), unsafe_allow_html=True)
                st.caption(t("narrative.section_caption", lang))
                for item in narrative_items:
                    st.markdown(
                        ui.narrative_card_html(
                            item["rank"],
                            item["icon"],
                            item["color"],
                            item["name"],
                            item["score"],
                            item["bullets"],
                            item["tone"],
                        ),
                        unsafe_allow_html=True,
                    )
        else:
            st.info(t("warn.no_output", lang))

        # -------------------------------------------------------------
        # Full KPI breakdown — always visible, no tab clicks: same data as
        # the table above, plus the mode explanation, gold-standard table
        # (if any), supporting metrics and the full "why" dialog trigger.
        # -------------------------------------------------------------
        st.markdown(ui.eyebrow_html("📊", t("kpi.glance_title", lang)), unsafe_allow_html=True)
        st.caption(t("kpi.glance_caption", lang))
        if use_gold_cmp:
            st.info(t("kpi.mode_note_gold", lang))
        else:
            st.info(t("kpi.mode_note_consensus", lang))

        if use_gold_cmp:
            st.markdown("#### " + t("kpi.gold_section", lang))
            st.caption(t("kpi.gold_caption", lang))
            gold_df = metrics.build_gold_table(compare, model_keys, lang)
            ui.show_model_table(gold_df, lang, _tier_labels())
            if compare.get("gold", {}).get("gold_diagnoses"):
                with st.expander(t("kpi.gold_diagnoses", lang)):
                    for i, d in enumerate(compare["gold"]["gold_diagnoses"], 1):
                        st.markdown(f"{i}. {d}")

        gc1, gc2, gc3 = st.columns(3)
        gc1.metric(t("kpi.primary_agreement", lang), f"{compare['primary_agreement']}%")
        gc2.metric(t("kpi.qvac_cloud", lang), f"{compare.get('qvac_cloud_concordance', 0)}%")
        gc3.metric(t("kpi.models_compared", lang), compare["active_count"])
        if compare.get("consensus_keywords"):
            st.caption(
                t("kpi.consensus_keywords", lang) + ": "
                + ", ".join(compare["consensus_keywords"][:12])
            )
        if compare["matrix"]:
            with st.expander(t("kpi.matrix_expander", lang)):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                t("matrix.pair", lang): k,
                                t("matrix.concordance", lang): v,
                            }
                            for k, v in compare["matrix"].items()
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        if st.button(t("kpi.explain_button", lang), use_container_width=True, key="btn_explain_scores"):
            explanations = metrics.build_score_explanations(compare, model_keys, lang, use_gold_cmp)
            explain_scores_dialog(explanations, use_gold_cmp)

        tab_perf, tab_triage, tab_charts, tab_session = st.tabs(
            [
                "⚡ " + t("kpi.performance", lang),
                "🚦 " + t("kpi.triage", lang),
                "📊 " + t("charts.section", lang),
                "📈 " + t("kpi.session", lang),
            ]
        )

        with tab_perf:
            st.caption(t("kpi.performance_caption", lang))
            perf_df = metrics.build_performance_table(display_results, lang)
            ui.show_model_table(perf_df, lang, _tier_labels())

        with tab_triage:
            st.caption(t("kpi.triage_caption", lang))
            urgency_df = metrics.build_urgency_table(compare, model_keys, lang)
            ui.show_model_table(urgency_df, lang, _tier_labels())
            st.metric(t("kpi.triage_agreement", lang), f"{compare.get('urgency_agreement', 0.0)}%")
            st.plotly_chart(
                charts.fig_urgency_compare(compare.get("urgency", {}), lang, tier_labels=_tier_labels()),
                use_container_width=True,
                key="chart_urgency_tab",
            )

        with tab_charts:
            if not ranking_df.empty:
                st.caption(t("charts.dims_privacy_note", lang))
                (
                    chart_tab_radar,
                    chart_tab_bars,
                    chart_tab_heatmap,
                    chart_tab_keywords,
                    chart_tab_urgency,
                ) = st.tabs(
                    [
                        t("charts.tab_radar", lang),
                        t("charts.tab_bars", lang),
                        t("charts.tab_heatmap", lang),
                        t("charts.tab_keywords", lang),
                        t("charts.tab_urgency", lang),
                    ]
                )
                with chart_tab_radar:
                    st.plotly_chart(
                        charts.fig_radar(
                            ranking_df,
                            use_gold_cmp,
                            lang,
                            sem_available=compare.get("semantic_available", False),
                            tier_labels=_tier_labels(),
                        ),
                        use_container_width=True,
                        key="chart_radar",
                    )
                with chart_tab_bars:
                    if use_gold_cmp:
                        st.plotly_chart(
                            charts.fig_consensus_ranking_bars(ranking_df, lang, **_tier_kw()),
                            use_container_width=True,
                            key="chart_bars_consensus",
                        )
                        st.plotly_chart(
                            charts.fig_clinical_ranking_bars(ranking_df, lang, **_tier_kw()),
                            use_container_width=True,
                            key="chart_bars_clinical",
                        )
                    else:
                        st.plotly_chart(
                            charts.fig_consensus_ranking_bars(ranking_df, lang, **_tier_kw()),
                            use_container_width=True,
                            key="chart_bars",
                        )
                with chart_tab_heatmap:
                    if len(model_keys) >= 2:
                        st.plotly_chart(
                            charts.fig_concordance_heatmap(
                                compare.get("matrix_keyed", {}), model_keys, lang, tier_labels=_tier_labels()
                            ),
                            use_container_width=True,
                            key="chart_heatmap",
                        )
                    else:
                        st.info(t("warn.no_output", lang))
                with chart_tab_keywords:
                    kw_counts = compare.get("consensus_keyword_counts", {})
                    if kw_counts:
                        st.plotly_chart(
                            charts.fig_keyword_bar(kw_counts, len(model_keys), lang),
                            use_container_width=True,
                            key="chart_keywords",
                        )
                    else:
                        st.info(t("warn.no_output", lang))
                with chart_tab_urgency:
                    st.plotly_chart(
                        charts.fig_urgency_compare(
                            compare.get("urgency", {}), lang, tier_labels=_tier_labels()
                        ),
                        use_container_width=True,
                        key="chart_urgency_sub",
                    )
            else:
                st.info(t("warn.no_output", lang))

        with tab_session:
            slots = st.session_state.saved_slots
            history = session_store.slots_as_history(slots)
            st.caption(t("session.caption", lang))
            n_filled = len([c for c in CASE_IDS if session_store.slot_is_filled(slots, c)])
            st.caption(t("sidebar.saved_count", lang, n=n_filled, total=len(CASE_IDS)))

            if not history:
                st.info(t("session.slots_empty", lang))
            else:
                st.markdown("#### " + t("session.saved_cases", lang))
                hist_df = metrics.build_session_history_table(history, lang)
                st.dataframe(
                    hist_df,
                    use_container_width=True,
                    hide_index=True,
                    height=ui.table_height(len(hist_df)),
                )

                avg_df = metrics.build_final_consensus_average(history, lang)
                gold_df, gold_case = metrics.build_final_gold_ranking(history, lang)

                if not avg_df.empty or not gold_df.empty:
                    st.markdown(ui.eyebrow_html("🎬", t("session.finale_title", lang)), unsafe_allow_html=True)
                    st.caption(t("session.finale_caption", lang))

                fin_col1, fin_col2 = st.columns(2)
                with fin_col1:
                    if not avg_df.empty:
                        st.markdown("##### " + t("session.finale_avg", lang))
                        st.caption(
                            t(
                                "session.finale_avg_help",
                                lang,
                                n=len([e for e in history if e.get("mode") != "gold_standard"]),
                            )
                        )
                        st.plotly_chart(
                            charts.fig_consensus_ranking_bars(avg_df, lang, height=240, **_tier_kw()),
                            use_container_width=True,
                            key="chart_finale_avg",
                        )
                        ui.show_model_table(
                            avg_df,
                            lang,
                            _tier_labels(),
                            columns=[L["model"], L["score_cons_rescaled"], L["rank_consensus"]],
                        )
                    else:
                        st.info(t("session.finale_avg_empty", lang))

                with fin_col2:
                    if not gold_df.empty:
                        st.markdown("##### " + t("session.finale_gold", lang))
                        st.caption(t("session.finale_gold_help", lang, case=gold_case))
                        st.plotly_chart(
                            charts.fig_clinical_ranking_bars(gold_df, lang, height=240, **_tier_kw()),
                            use_container_width=True,
                            key="chart_finale_gold",
                        )
                        ui.show_model_table(
                            gold_df,
                            lang,
                            _tier_labels(),
                            columns=[L["model"], L["score_clin_short"], L["rank_clinical"]],
                        )
                    else:
                        st.info(t("session.finale_gold_empty", lang))

                if history:
                    lb_df = metrics.build_leaderboard_df(history, lang)
                    st.markdown("#### " + t("leaderboard.title", lang))
                    st.plotly_chart(
                        charts.fig_leaderboard(lb_df, lang, tier_labels=_tier_labels()),
                        use_container_width=True,
                        key="chart_leaderboard",
                    )
                    ui.show_model_table(lb_df, lang, _tier_labels())

        st.markdown(ui.eyebrow_html("🪙", t("decision.section", lang)), unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                f'<p class="decision-lead fade-in">{t("decision.lead", lang)}</p>',
                unsafe_allow_html=True,
            )
            b1, b2 = st.columns(2)
            with b1:
                if st.button("🔴 " + t("decision.cloud", lang), use_container_width=True):
                    cloud_decision_dialog()
                st.markdown(
                    f'<p class="decision-caption">{t("decision.cloud_caption", lang)}</p>',
                    unsafe_allow_html=True,
                )
            with b2:
                if st.button("🟢 " + t("decision.qvac", lang), use_container_width=True):
                    qvac_decision_dialog()
                st.markdown(
                    f'<p class="decision-caption">{t("decision.qvac_caption", lang)}</p>',
                    unsafe_allow_html=True,
                )

st.markdown(
    '<div class="footer-note">🩺 QVAC vs Cloud LLMs · Demonstration benchmark only — not a substitute for medical advice.</div>',
    unsafe_allow_html=True,
)
