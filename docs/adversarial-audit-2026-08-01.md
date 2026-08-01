# Adversarial X audit — 2026-08-01

Ruthless external critic for public posts. Scoring formulas are **not** changed
to hide issues. Mitigations are disclosure and product labelling only.

**Framing:** this repository is a transparent amateur comparison exercise
(author-supplied gold, OpenRouter API routes, uncalibrated LLM-as-judge). It is
**not** an official MedPsy blog evaluation and **not** a powered clinical study.
Repo docs, comments, and posts must stay in that neutral register — never frame
case selection or protocol choices as optimizing for a favored local model.

**Live note:** Streamlit / live Multi workers were not interrupted for this audit.

Counts: **5 Critical · 6 High · 6 Medium · 5 Low**.

## CI note (main Quality gate)

| | SHA | What |
|---|---|---|
| Failed | `19c21b67c869d0452144314f680c6ef7529e7751` | ruff F401 — unused `CURRENT_ROSTER_KEYS` import in `app.py` (Static checks) |
| Green | `587c2bf92591d5239212070e92296ac0fd04783c` | unused import removed; Quality gate success |

Agents must poll Quality gate to **success** on HEAD after every push.

## Product state under audit

| Area | State |
|---|---|
| Default live roster | 9 slots: cloud GPT-5.5 / Claude Sonnet 5 / Gemini 3.5 Flash + Phi-3.5 + MedGemma 1.5 4B IT + Med42 8B + UltraMedical 8B + MedPsy 1.7B Q4 + MedPsy 4B Q4 |
| Legacy opt-in | Gemma 2 2B, Llama 3.2 3B, MedPsy 4B Q8 (History labels retained; OFF by default) |
| Cases | Base Case 1–5 + growable Case 6+; default pack revision 3 seeds slots **2–7** (anaphylaxis, STEMI, DKA, stroke, suicidal OD, psychosis vs delirium). **Case 1 is not in the current pack** |
| Medical peer cutover | MedGemma-1.5-4B-IT Q4 |
| Rebuild mean | Per-model **scored-N** fill; **successful-only** pool; exact Clinical Composite == 0 treated like N/A (MedGemma ~2% zeros on caseC empirically; MedPsy/main peers 0) |
| Prepare | Structured Q1–A5 gold prepares **locally**; free-form still uses OpenRouter extract |
| Cost forecast | Typical completions + History priors — not `max_tokens` as billable; billed truth = OpenRouter `usage.cost` |
| Judge | Single primary DeepSeek R1 (blind); optional whole-run verifier only on systemic failure |

## Critical

### C1 — Gold-relative ≠ clinical truth or official MedPsy blog
- **Attack:** Ranks posted as clinical superiority, or equated to official MedPsy marketing/blog numbers.
- **Fair?** Fair. Metric is agreement with author-supplied frozen gold on this amateur bench.
- **Mitigation:** Always say reference-relative · not clinical truth · not official MedPsy blog. Prefer Clinical Composite wording. Do not retune weights.

### C2 — OpenRouter API ≠ ChatGPT / Claude / Gemini consumer web
- **Attack:** “We beat ChatGPT/Claude” while slots are pinned OpenRouter API routes.
- **Fair?** Fair. Consumer UIs differ (tools, prompts, browsing, memory, often model).
- **Mitigation:** Lead with “OpenRouter API routes, not consumer web.” Align GitHub/OG copy.

### C3 — Single uncalibrated DeepSeek R1 judge
- **Attack:** One competitor LLM owns the podium without human-reviewed calibration.
- **Fair?** Fair. Blind help ≠ ground truth; verifier is systemic-failure only.
- **Mitigation:** Disclose uncalibrated single LLM-as-judge; reviewed fixtures before science posters.

### C4 — Amateur bench sold as product/science win
- **Attack:** Exploratory Multi/Rebuild framed as proof a local model replaces cloud clinical systems.
- **Fair?** Fair. Docs already say research/demo/exploratory; X crops captions.
- **Mitigation:** Keep amateur framing on every post; show mean±std, N per model, cohort hash.

### C5 — Case selection and mid-campaign stem swaps
- **Attack:** Pack revision replaced Case 2, force-seeded Case 6–7 psych/tox emergencies, left Case 1 out of pack — looks like moving goalposts or specialty stacking.
- **Fair?** Fair as selection-bias critique. Cross-era stems are not one case. Disclose pack revision and titles; do not pool pre/post-swap series.
- **Mitigation:** Caption pack revision + stem titles. Frame Case 6–7 as emergency psych/tox breadth in an amateur suite — never as domain hunting.

## High

### H1 — Rebuild scored-only / successful-only hides failure theater
- **Attack:** Clean leaderboard after Rebuild; Failed%/partial gone → “you buried N/A.”
- **Fair?** Mostly fair. Intentional scored-only comparison; reliability optics need captions.
- **Mitigation:** Disclose that exact Clinical Composite == 0 is treated like N/A in Rebuild mean (excluded from scored-N). Empirically MedGemma ~2/105 (~2%) zeros on caseC; MedPsy/main peers 0. Caption: “Scored-only · N = successful non-zero scores.” Offer a separate reliability view when posting ops honesty.

### H2 — Default 9 vs legacy ≤12 / History cross-roster pooling
- **Attack:** Means mix eras with Gemma/Llama/Q8 on vs default-9 runs.
- **Fair?** Fair.
- **Mitigation:** Roster version on charts; warn on Portfolio across roster shapes.

### H3 — Dual MedPsy + MedGemma 1.5 peer optics
- **Attack:** Two MedPsy quants look like independent wins; MedGemma 1.5 moved the peer baseline mid-campaign.
- **Fair?** Mostly fair.
- **Mitigation:** Group MedPsy family vs peers; tag MedGemma 1.5 cutover; no silent cross-era peer pooling.

### H4 — Re-Prepare / Confirm opens a new cohort
- **Attack:** “Same case” after re-Confirm; Rebuild won’t merge / means jump.
- **Fair?** Fair (`cohort_id` hashes gold contract).
- **Mitigation:** Banner “New Confirm = new cohort”; short cohort hash on screenshots.

### H5 — Portfolio mean ≠ same-case mean
- **Attack:** “Last N runs lead” when Portfolio is ≤N per-model across cases.
- **Fair?** Fair.
- **Mitigation:** Label Same-case vs Portfolio; show n_scored per model.

### H6 — Local recovery capped harder than cloud
- **Attack:** Cloud repair/fill vs local ≤1 recovery call.
- **Fair?** Fair if undisclosed as identical weapons.
- **Mitigation:** Disclose caps; optional strict parity = new track/cohort.

## Medium

### M1 — Prepare path asymmetry (local Q1–A5 vs API free-form)
- Seeded/Q1–A5 Prepare without extractor variance; free-form depends on extract quality.
- **Mitigation:** Disclose Prepare path on artifacts.

### M2 — Cost forecast vs billed usage
- Forecast is rough (typical + History priors); invoice is OpenRouter usage.
- **Mitigation:** Label forecast “rough · often over”; show billed totals after runs.

### M3 — Medical peer swaps mid-campaign
- Pre–UltraMedical / pre–MedGemma 1.5 eras are not comparable without tags.
- **Mitigation:** Roster version on artifacts; Portfolio warning.

### M4 — Credits mid-Multi
- HTTP 402 → per-model technical N/A; History kept; abort only on systemic judge failure.
- **Mitigation:** Distinct credits/4xx surfacing; prior-iterations banner.

### M5 — Field name `accuracy`
- JSON alias quote-tweeted as clinical accuracy.
- **Mitigation:** Prefer `clinical_composite_score` in next schema; keep alias.

### M6 — High-variance means without error bars
- Mean bars alone look stable.
- **Mitigation:** Default ±std + exploratory caption; no score massage.

## Low

### L1 — YAML `site: chatgpt.com` branding (UI rename; cohort-safe).
### L2 — Controlled `allow_fallbacks` vs “pinned provider” (`strict_controlled` + `routed_provider`).
### L3 — GitHub/OG marketing copy lag vs in-app honesty captions.
### L4 — Simulated USDT wallet optics next to medical claims.
### L5 — TTFT/TPS cropped as quality proxies (ops metrics only).

## Explicit non-goals

No weight tweaks, forced tie-breaks, dropping high-variance runs, or silent
rescore to flatten rankings. No repository language that frames protocol or case
choices as optimizing for a favored local model.

## Mitigation status (disclosure package)

Critical and High items below are treated as **known limitations** mitigated by
loud product disclosure — not as infinite scoring todos. Residual risk remains
if screenshots crop captions or posts omit the template.

| ID | Status | Where disclosure lands |
|---|---|---|
| C1 Gold ≠ clinical truth / ≠ MedPsy blog | Disclosure-mitigated | Dashboard honesty block · README/PRESENTATION box · X template |
| C2 OpenRouter API ≠ consumer web | Disclosure-mitigated | Honesty block · GitHub-facing README lead · X template |
| C3 Single uncalibrated DeepSeek R1 | Disclosure-mitigated | Honesty block · docs box · X template |
| C4 Amateur / exploratory framing | Disclosure-mitigated | Honesty block · mean footer · X template |
| C5 Case selection / pack revision | Disclosure-mitigated (process) | Pack revision captions; still disclose stem titles when posting |
| H1 Rebuild scored-only | Disclosure-mitigated | Honesty block · Rebuild captions · screenshot footer |
| H2 Roster 9 / cross-roster pooling | Disclosure-mitigated | Roster version on honesty + footer |
| H3 Dual MedPsy + MedGemma 1.5 | Disclosure-mitigated | Honesty roster line · peer cutover tags in product state |
| H4 New Confirm = new cohort | Disclosure-mitigated | Confirm caption · honesty block · short cohort hash |
| H5 Portfolio ≠ Same-case | Disclosure-mitigated | Loud scope label · honesty · footer |
| H6 Local recovery capped | Disclosure-mitigated | Honesty bullet · docs box |

Medium/Low remain documented optics; no formula changes. X copy-paste:
[docs/x-post-template.md](x-post-template.md).

Canvas: `adversarial-x-audit-2026-08-01.canvas.tsx` (Cursor canvases directory).
