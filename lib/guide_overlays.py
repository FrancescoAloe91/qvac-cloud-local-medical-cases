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
<p>Blind DeepSeek R1 · strict evidence validation · whole-run independent verifier on anomalies ·
technical failures are N/A and exact ties remain ties.</p>
<pre>Section score = 50% graded coverage + 35% clinical quality + 15% discipline
coverage = continuous 0..1 for every frozen reference claim
helpful / neutral additions = no penalty
verified unsupported / contradictory / dangerous = proportional discipline
unverifiable harmful additions = dropped (audit marker; not fail-closed)
Clinical Composite Score = 30% diagnosis + 25% safety + 20% plan + 15% tests + 10% urgency</pre>
<table>
  <tr><th>Signal</th><th>Role</th><th>Meaning</th></tr>
  <tr><td>Graded coverage</td><td>50%</td><td>Partial and complete semantic coverage on a 0..1 continuum</td></tr>
  <tr><td>Clinical quality</td><td>35%</td><td>Coherence, prioritization, usefulness and caution</td></tr>
  <tr><td>Discipline</td><td>15%</td><td>Verified unsupported / contradictory / dangerous additions only; unverifiable harm is dropped, not auto-penalized</td></tr>
  <tr><td>Failure status</td><td>N/A</td><td>Transport, timeout, malformed evidence or cancellation</td></tr>
</table>
<p>Synonyms and faithful paraphrases count. Every match/contradiction must cite candidate evidence.
Judge is an uncalibrated LLM-as-judge unless human calibration fixtures have been checked.
Quality is independent of coverage by design (v4). Verifier = systemic re-judge, not human calibration.
UI shows <b>Clinical Composite</b>; artifact JSON may still use the field name <code>accuracy</code> for compatibility.</p>
<p>Each model with ≥1 scored run enters the mean ranking (sorted by mean of scored runs),
independently of other models' N/A results. Incomplete coverage (Failed% &gt; 0 or scored &lt; requested)
keeps the rank and shows a <b>partial</b> badge — technical N/A are never clinical zeros.
Every mean keeps its own N; missing scores are never imputed.
N=5 remains exploratory (not bit-identical reruns) and measures repeatability on this reference, not general clinical validity.
Local format-repair (same parser as cloud) only re-asks A# markers — it does not invent clinical content.
Screenshots should keep at least one honesty caption (API≠web · reference-relative · N=5).
<strong>Comprehension</strong> and <strong>Structured A1–A5</strong> History / Rebuild means are never pooled.</p>
<p style="opacity:.8;font-size:0.8rem">This window is browser-only — opening it does <b>not</b> pause collect/judge.</p>
"""
    return (
        client_guide_overlay("guide_setup", "QVAC SDK + MedPsy setup guide", setup_body)
        + client_guide_overlay("guide_rank", "How ranking is calculated", rank_body)
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
