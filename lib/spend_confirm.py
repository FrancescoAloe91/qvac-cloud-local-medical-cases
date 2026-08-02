"""Shared pre-run cost estimate + explicit Yes/Cancel gate.

Inline card (never ``st.dialog``) so ✕ cannot abort an in-flight collect.
Used by Comprehension home and Structured.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import streamlit as st


def fmt_cost_single(breakdown: Dict[str, Any]) -> str:
    bits = []
    for m in breakdown.get("per_model") or []:
        if float(m.get("estimated_usd", 0) or 0) <= 0:
            continue
        bits.append(f"{m.get('key')} ${m.get('estimated_usd', 0):.3f}")
    ex = breakdown.get("extractor") or {}
    ex_usd = float(ex.get("estimated_usd", 0) or 0)
    if ex_usd > 0:
        bits.append(f"extract ${ex_usd:.3f}")
    j = breakdown.get("judge") or {}
    bits.append(f"judge ${j.get('estimated_usd', 0):.3f}")
    total = float(breakdown.get("total_usd", 0) or 0)
    hi = float(breakdown.get("total_usd_upper", 0) or 0) or total
    cal_n = int(breakdown.get("calibration_n") or 0)
    src = "History-calibrated" if breakdown.get("calibrated") else "formula"
    j_out = j.get("completion_tokens_per_call", "?")
    return (
        '<div class="cost-compact run-cost-cell">'
        + " · ".join(bits)
        + f' · <b>${total:.3f}–${hi:.3f}</b>'
        + f'<br/><span style="opacity:.75">rough estimate · often over · {src}'
        + (f" n={cal_n}" if cal_n else "")
        + f" · judge ~{j_out} tok (not 16k cap)</span>"
        + "</div>"
    )


def fmt_cost_multi(breakdown: Dict[str, Any], n: int) -> str:
    tot = float(breakdown.get("total_usd_for_n", 0) or 0)
    hi = float(breakdown.get("total_usd_upper_for_n", 0) or 0) or tot
    extract = float((breakdown.get("extractor") or {}).get("estimated_usd", 0) or 0)
    cal_n = int(breakdown.get("calibration_n") or 0)
    src = "History-calibrated" if breakdown.get("calibrated") else "formula"
    extract_bit = (
        f" · +extract ${extract:.3f} once" if extract > 0 else " · extract already paid"
    )
    return (
        f'<div class="cost-compact cost-multi run-cost-cell">'
        f"<b>${tot:.3f}–${hi:.3f}</b> · ×{n}{extract_bit}"
        f'<br/><span style="opacity:.75">rough estimate · often over · {src}'
        + (f" n={cal_n}" if cal_n else "")
        + "</span></div>"
    )


def render_spend_confirm_card(
    *,
    pending_key: str = "pending_run",
    confirmed_key: str = "confirmed_run",
    has_key: bool = False,
    track_label: str = "benchmark",
    on_confirm: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_also_clear: Optional[list] = None,
) -> None:
    """Inline confirm card. Caller should ``st.stop()`` after this when pending."""
    pr = st.session_state.get(pending_key) or {}
    n = int(pr.get("n") or 1)
    rounds = int(pr.get("rounds") or n)
    est = float(pr.get("est") or 0)
    est_hi = float(pr.get("est_hi") or 0) or est
    mode = str(pr.get("mode") or "full")
    multi_case = bool(pr.get("multi_case"))
    show_fc = bool(st.session_state.get("show_cost_forecast", True))

    if not show_fc:
        spend_body = (
            f"Start <b>{rounds}</b> round(s) on <b>{track_label}</b> "
            f"({'on-device / judge path' if mode != 'full' else 'cloud + judge'}). "
            f"Cost forecast is hidden — billed truth = OpenRouter usage."
        )
    elif mode in {"local_only", "qvac_only"}:
        spend_body = (
            f"Rough OpenRouter estimate <b>${est:.4f} – ${est_hi:.4f}</b> "
            f"(often over) for <b>{rounds}</b> {track_label} round(s) · "
            f"<b>judge path</b> (collect on-device ≈ $0). "
            f"Billed truth = OpenRouter usage."
        )
    elif multi_case:
        spend_body = (
            f"Rough OpenRouter estimate <b>${est:.4f} – ${est_hi:.4f}</b> "
            f"(often over) for <b>Multi×all</b> · <b>{rounds}</b> rounds "
            f"({track_label}). Cloud + DeepSeek R1 judge; on-device = $0 if included. "
            f"Billed truth = OpenRouter usage."
        )
    else:
        spend_body = (
            f"Rough OpenRouter estimate <b>${est:.4f} – ${est_hi:.4f}</b> "
            f"(often over) for <b>{rounds}</b> {track_label} round(s) "
            f"(cloud models + DeepSeek R1 judge). "
            f"On-device = $0 if included. Billed truth = OpenRouter usage."
        )

    st.markdown(
        f"""
<div class="spend-confirm-card">
  <p class="spend-title">Confirm before starting</p>
  <p class="spend-body">
    {spend_body}
  </p>
  <p class="spend-note">Cancel goes back. Yes starts the run immediately — no overlay ✕.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Cloud+judge needs a key; pure on-device rehearsal may continue without.
    needs_key = mode == "full" or float(est) > 0 or float(est_hi) > 0
    if needs_key and not has_key:
        st.error(
            "No usable OpenRouter key — paste a full sk-or-v1-… key "
            "(boot dialog or sidebar)."
        )
        if st.button("Close", use_container_width=True, key=f"{pending_key}_close_nokey"):
            st.session_state.pop(pending_key, None)
            st.session_state.pop(confirmed_key, None)
            for k in cancel_also_clear or []:
                st.session_state.pop(k, None)
            st.rerun()
        return

    a, b = st.columns(2)
    with a:
        if st.button("Cancel", use_container_width=True, key=f"{pending_key}_cancel"):
            st.session_state.pop(pending_key, None)
            st.session_state.pop(confirmed_key, None)
            for k in cancel_also_clear or []:
                st.session_state.pop(k, None)
            st.rerun()
    with b:
        if st.button(
            "Yes · start run",
            type="primary",
            use_container_width=True,
            key=f"{pending_key}_yes",
        ):
            pending = st.session_state.pop(pending_key, None)
            if pending:
                if on_confirm is not None:
                    on_confirm(dict(pending))
                else:
                    st.session_state[confirmed_key] = pending
            st.session_state.pop(pending_key, None)
            st.rerun()
