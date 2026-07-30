# Adversarial dissing audit — 2026-07-30

Ruthless external critic for X/posts. Scoring formulas **not** changed to hide issues.
Mitigations = product/comms only.

**Live note:** User re-Prepare + Confirm then launched Multi 10×12 — that Confirm opens a **new cohort**. Do not pool with prior Confirm versions as “same case.”

Counts: **4 Critical · 5 High · 5 Medium · 5 Low**.

## OpenRouter credits mid-Multi (code-confirmed)

Credits exhaustion does **not** wipe History. `benchmark/openrouter.py` stores HTTP errors on `meta.error` (e.g. HTTP 402); 402 is **not** retryable. Per-model failures become technical N/A. Iteration `run_status` is `partial` unless every judgment is `valid` (or `cancelled`). `write_artifact` / persist runs **after each** Multi iteration — already-completed iterations remain saved. Multi aborts remaining iterations only on `systemic_judge_failure` (zero valid + judge infrastructure failure). Candidate-only credit death does not clear prior artifacts or History.

## Critical

### C1 — Gold-relative self-reference theater
- **Attack:** “MedPsy beats GPT on clinical accuracy” while scores are agreement with the user’s pasted reference.
- **Fair?** Fair. Field still named `accuracy`; Clinical Composite is author-supplied-gold relative.
- **Mitigation:** Never say accuracy without “reference-relative”; prefer Clinical Composite in UI/exports. Do not retune weights.

### C2 — OpenRouter API ≠ ChatGPT/Claude/Gemini consumer UI
- **Attack:** “We beat ChatGPT/Claude” while slots are OpenRouter API routes (GPT-5.5 / Sonnet 5 / Gemini 3.5 Flash).
- **Fair?** Fair. Consumer UI ≠ API; YAML `site:` still points at brand domains; GitHub description still consumer-branded.
- **Mitigation:** Every public claim: “OpenRouter API routes, not consumer web.” UI chip rename; leave YAML `site` alone if cohort-sensitive.

### C3 — Uncalibrated competitor LLM-as-judge
- **Attack:** DeepSeek R1 ranks the field; no human-reviewed calibration for public claims.
- **Fair?** Fair. Blind help ≠ clinical ground truth (`SCORING.md` Calibration).
- **Mitigation:** Always disclose uncalibrated LLM-as-judge; ship reviewed fixtures before scientific posters.

### C4 — Small-N / one-case / Multi volatility as a winner story
- **Attack:** One Addison (or any) Multi with MedPsy rank flips posted as a product win/loss.
- **Fair?** Fair. N=5 exploratory; high CV is computed for a reason; fresh Confirm Multi ≠ prior cohort replication.
- **Mitigation:** Show mean±std, Failed%, cohort hash, N per model; never crop honesty captions.

## High

### H1 — Re-Prepare / Confirm → new cohort
- **Attack:** User thinks “same case”; Rebuild won’t merge / means jump.
- **Fair?** Fair. `cohort_id` hashes confirmed gold contract (ex `confirmed_at`) + models/track.
- **Mitigation:** Banner “New Confirm = new cohort”; put short cohort hash on screenshots.

### H2 — Portfolio ≠ same-case (per-model N)
- **Attack:** “Last 10 runs MedPsy leads” when Portfolio is ≤N **per model** across cases.
- **Fair?** Fair. Not a global last-N document slice; roster shapes may differ.
- **Mitigation:** Label Same-case vs Portfolio; show n_valid/n_failed per model.

### H3 — “Comparable means” with high variance
- **Attack:** Mean bars without error bars look like a stable ranking.
- **Fair?** Mostly fair. Math includes std/CV; screenshots can omit them.
- **Mitigation:** Default ±std + Failed% + exploratory; no score massage.

### H4 — MedPsy brand pooling vs medical peers
- **Attack:** Three MedPsy quants look like three independent wins; peers swapped mid-campaign.
- **Fair?** Mostly fair. Distinct GGUFs, but family disclosure required; BioMistral/OpenBioLLM→UltraMedical eras differ.
- **Mitigation:** Group MedPsy family; tag peer roster version; no cross-era silent pooling.

### H5 — Local recovery capped vs cloud
- **Attack:** Cloud format-repair + targeted fill; local ≤1 recovery call.
- **Fair?** Fair asymmetry if marketed as identical weapons (`runner.py`, `SCORING.md`).
- **Mitigation:** Disclose caps; optional strict track (new cohort) — not silent rescore.

## Medium

### M1 — Credits mid-run honesty
- See section above. Attack “cloud scored 0 / you wiped runs” is partly unfair if History kept and N/A labelled technical.
- **Mitigation:** Distinct “credits/HTTP 4xx” surfacing; preflight estimate for Multi 10×12.

### M2 — Failed% / partial pooling history
- Partials now pool so Failed% is honest (517b117). Era labelling needed so critics don’t cry goalpost-move.
- **Mitigation:** Caption eligibility rule version; don’t silently rewrite old rankings as vindication.

### M3 — BioMistral / OpenBioLLM mid-campaign swaps
- Fair: peer leaderboard changed with roster engineering, not only MedPsy skill.
- **Mitigation:** Roster version on artifacts; warn on cross-roster Portfolio.

### M4 — Streamlit OOM / restart
- Hosted session memory can look empty after restart; local owner disk artifacts remain.
- **Mitigation:** Disclose durable path; flash artifact count after each iteration.

### M5 — `accuracy` field name
- JSON `accuracy` quote-tweeted as clinical accuracy.
- **Mitigation:** Schema flag / rename in next artifact version; keep alias.

## Low

### L1 — YAML `site: chatgpt.com` branding (UI rename; cohort-safe).
### L2 — Controlled `allow_fallbacks` vs “pinned provider” (use `strict_controlled` + show `routed_provider`).
### L3 — GitHub/OG marketing copy lag vs in-app honesty captions.
### L4 — Simulated USDT wallet optics next to medical claims.
### L5 — TTFT/TPS cropped as quality proxies (ops metrics only).

## Explicit non-goals

No weight tweaks, forced tie-breaks, dropping high-variance runs, or silent rescore to flatten MedPsy variance.

Streamlit / live Multi workers were **not** interrupted for this audit.
