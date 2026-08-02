"""Client-side Setup / Ranking guide overlays (no Streamlit rerun).

Shared by Comprehension home and Structured so left-rail guide labels
toggle the same main-DOM overlays on both tracks.
"""

from __future__ import annotations

import html
from typing import Optional

from lib.i18n import t


def client_guide_overlay(uid: str, title: str, body_html: str) -> str:
    """Fullscreen guide overlay toggled by ``<label for=uid>`` — no rerun."""
    u = html.escape(uid)
    ttl = html.escape(title)
    close_lab = "Close"
    return f"""
<input type="checkbox" id="{u}" class="fs-ck" autocomplete="off" />
<div class="fs-overlay" hidden style="display:none !important;visibility:hidden !important">
  <div class="fs-card">
    <div class="fs-bar">
      <span>{ttl}</span>
      <button type="button" class="fs-close" data-fs="{u}" title="{close_lab}" aria-label="{close_lab}">✕</button>
    </div>
    <div class="guide-body">{body_html}</div>
  </div>
</div>
"""


def _setup_body(*, lang: Optional[str], qvac_status_line: str) -> str:
    setup_status = html.escape(qvac_status_line or "")
    if (lang or "en").startswith("it"):
        return f"""
<h3>Cosa serve per MedPsy on-device</h3>
<ul>
  <li><b>Software QVAC</b> in locale (cartella sidecar)</li>
  <li><b>File modello MedPsy</b> nella cartella <code>models/</code></li>
  <li><b>Node.js 22+</b> da nodejs.org</li>
</ul>
<h3>Setup dopo il clone</h3>
<ol>
  <li>Installa Node.js 22+ da nodejs.org</li>
  <li>Metti il file MedPsy in <code>models/</code></li>
  <li>Dalla cartella del progetto, in un secondo terminale:</li>
</ol>
<pre>./scripts/setup_qvac_sidecar.sh
cd sidecar &amp;&amp; npm start</pre>
<p>Lascia quel terminale aperto, poi ricarica questa pagina.</p>
<p>Con il sidecar avviato, MedPsy è incluso (sul tuo PC, $0 API).</p>
<p><b>Stato su questa macchina:</b> {setup_status}</p>
<p style="opacity:.8;font-size:0.8rem">Finestra solo browser — aprirla <b>non</b> mette in pausa una run.</p>
"""
    return f"""
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


def _rank_body(*, lang: Optional[str]) -> str:
    if (lang or "en").startswith("it"):
        return """
<h3>Come funziona la classifica</h3>
<p>Un giudice AI cieco (DeepSeek R1) valuta ogni risposta rispetto al tuo
riferimento bloccato. I fallimenti tecnici contano come N/A (saltati), non come
zeri clinici. I pareggi restano pareggi.</p>
<p><b>In parole semplici, ogni sezione mescola:</b></p>
<ul>
  <li><b>~50% copertura</b> — quanto della checklist di riferimento è coperto</li>
  <li><b>~35% qualità clinica</b> — coerenza, priorità, utilità, cautela</li>
  <li><b>~15% disciplina</b> — penalità solo se il giudice può verificare
  aggiunte non supportate, contraddittorie o pericolose</li>
</ul>
<p>Il <b>Clinical Composite</b> pesa diagnosi, safety, piano, test e urgenza.
Sinonimi e parafrasi fedeli contano. Il giudice non è calibrato da umani salvo
fixture di calibrazione.</p>
<h3>Live Multi (tab run)</h3>
<p>I modelli con almeno una run scored entrano in classifica. Le run incomplete
possono mostrare un badge <b>partial</b> nella tabella live. I punteggi mancanti
non vengono inventati.</p>
<h3>Media Rebuild (History)</h3>
<p>La media Rebuild usa <b>solo score riusciti</b> — niente badge partial nella
classifica media. Zeri esatti e N/A tecnici sono <b>esclusi</b> dalla media e
compaiono nella tabella Failures/N/A sotto.</p>
<p>N piccolo è esplorativo — misura ripetibilità su questo riferimento, non
validità medica generale. Gli screenshot devono tenere almeno una nota di
onestà (API ≠ app browser · score vs tuo riferimento · N).</p>
<p><strong>Comprehension</strong> e <strong>Structured</strong> non si mescolano mai.</p>
<p style="opacity:.8;font-size:0.8rem">Finestra solo browser — aprirla <b>non</b> mette in pausa una run.</p>
"""
    return """
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
<h3>Live Multi (run tabs)</h3>
<p>Models with at least one scored run enter the live ranking. Incomplete runs may
keep a <b>partial</b> badge on the live Multi table. Missing scores are never invented.</p>
<h3>Rebuild mean (History)</h3>
<p>Rebuild averages use <b>successful scores only</b> — no partial badge on the
mean ranking. Exact zeros and technical N/A are <b>excluded</b> from the average
and shown in the Failures/N/A table below.</p>
<p>Small N is exploratory — it measures repeatability on this reference, not general
medical validity. Screenshots should keep at least one honesty note
(API ≠ browser apps · scores vs your reference · N).</p>
<p><strong>Comprehension</strong> and <strong>Structured</strong> averages are never mixed.</p>
<p style="opacity:.8;font-size:0.8rem">This window is browser-only — opening it does <b>not</b> pause a run.</p>
"""


def guides_always_available_html(
    *,
    qvac_status_line: str = "",
    lang: Optional[str] = None,
) -> str:
    """Inject Setup + Ranking guides once in main DOM (sidebar labels toggle these)."""
    setup_title = t("guide.setup_title", lang)
    rank_title = t("guide.rank_title", lang)
    return (
        client_guide_overlay(
            "guide_setup",
            setup_title,
            _setup_body(lang=lang, qvac_status_line=qvac_status_line),
        )
        + client_guide_overlay(
            "guide_rank",
            rank_title,
            _rank_body(lang=lang),
        )
    )


def sidebar_guides_block_html(*, lang: Optional[str] = None) -> str:
    """Labels that toggle the main-DOM overlays without pausing a run."""
    return (
        '<div class="sidebar-guides-block">'
        f'<label class="guide-open-btn" for="guide_setup">'
        f'{html.escape(t("guide.setup_btn", lang))}</label>'
        f'<label class="guide-open-btn" for="guide_rank">'
        f'{html.escape(t("guide.rank_btn", lang))}</label>'
        f'<span class="guides-hint">{html.escape(t("guide.hint", lang))}</span>'
        "</div>"
    )
