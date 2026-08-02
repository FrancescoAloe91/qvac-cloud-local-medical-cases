"""Client-side Setup / Ranking guide overlays (no Streamlit rerun).

Shared by Comprehension home and Structured so left-rail guide labels
toggle the same main-DOM overlays on both tracks.
"""

from __future__ import annotations

import html


def client_guide_overlay(uid: str, title: str, body_html: str) -> str:
    """Fullscreen guide overlay toggled by ``<label for=uid>`` — no rerun."""
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


def guides_always_available_html(*, qvac_status_line: str = "") -> str:
    """Inject Setup + Ranking guides once in main DOM (sidebar labels toggle these)."""
    setup_status = html.escape(qvac_status_line or "")
    setup_body = f"""
<h3>What you need for on-device MedPsy</h3>
<ul>
  <li><b>QVAC software</b> running locally (the sidecar folder)</li>
  <li><b>MedPsy model file</b> in the <code>models/</code> folder</li>
  <li><b>Node.js 22+</b> from nodejs.org</li>
</ul>
<h3>Setup after cloning</h3>
<ol>
  <li>Install Node.js 22+ from nodejs.org</li>
  <li>Put the MedPsy model file in <code>models/</code></li>
  <li>From the project folder, in a second terminal:</li>
</ol>
<pre>./scripts/setup_qvac_sidecar.sh
cd sidecar &amp;&amp; npm start</pre>
<p>Leave that terminal open, then refresh this page.</p>
<p>When the sidecar is running, MedPsy is included (on your machine, $0 API).</p>
<p><b>Status on this machine:</b> {setup_status}</p>
<p style="opacity:.8;font-size:0.8rem">This window is browser-only — opening it does <b>not</b> pause a run.</p>
"""
    rank_body = """
<h3>How ranking works</h3>
<p>A blind AI judge (DeepSeek R1) scores each answer against your locked reference.
Technical failures count as N/A (skipped), not as clinical zeros. Exact ties stay ties.</p>
<p><b>In plain words, each section score mixes:</b></p>
<ul>
  <li><b>~50% coverage</b> — how much of the reference checklist the answer covers</li>
  <li><b>~35% clinical quality</b> — coherence, priorities, usefulness, caution</li>
  <li><b>~15% discipline</b> — penalties only when the judge can verify unsupported,
  contradictory, or dangerous additions</li>
</ul>
<p>The overall <b>Clinical Composite</b> weights diagnosis, safety, plan, tests, and urgency.
Synonyms and faithful paraphrases count. The judge is not human-calibrated unless
calibration fixtures have been checked.</p>
<p>Models with at least one scored run enter the average ranking. Incomplete runs keep
their rank and show a <b>partial</b> badge. Missing scores are never invented.</p>
<p>N=5 is exploratory — it measures repeatability on this reference, not general
medical validity. Screenshots should keep at least one honesty note
(API ≠ browser apps · scores vs your reference · N=5).</p>
<p><strong>Comprehension</strong> and <strong>Structured</strong> averages are never mixed.</p>
<p style="opacity:.8;font-size:0.8rem">This window is browser-only — opening it does <b>not</b> pause a run.</p>
"""
    return (
        client_guide_overlay("guide_setup", "QVAC + MedPsy setup", setup_body)
        + client_guide_overlay("guide_rank", "How ranking works", rank_body)
    )


def sidebar_guides_block_html() -> str:
    """Labels that toggle the main-DOM overlays without pausing a run."""
    return (
        '<div class="sidebar-guides-block">'
        '<label class="guide-open-btn" for="guide_setup">Setup guide</label>'
        '<label class="guide-open-btn" for="guide_rank">How ranking works</label>'
        '<span class="guides-hint">Opens without pausing the run · ✕ to close</span>'
        "</div>"
    )
