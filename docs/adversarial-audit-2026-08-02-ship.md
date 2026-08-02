# Adversarial audit — 2026-08-02 (post UX parity · ship check)

Ruthless external critic before public means (~70 scored runs / suite narrative).
Scoring formulas **not** changed in this pass. Focus: honesty gaps, dual-track
optics, pack provenance, residual “beta”, ops panels, boot/spend parity.

## Product framing under audit

| Area | State |
|------|--------|
| Default | Comprehension home · wire `comprehension-v1` |
| Optional | Structured A1–A5 · separate History |
| Pack | rev 3 · Case 1–10 · Cursor/internet-assembled · ED-biased |
| Boot | API key every session → QVAC status every session |
| Spend | Yes/Cancel on both tracks (incl. MedPsy-only) |
| Ops | Failures/N/A **table** only (ops chart removed) |

---

## Critical (still true — disclose, do not overclaim)

### C1 — Gold-relative ≠ clinical truth / medical validity
**Attack:** Caption suite means as “MedPsy beats GPT clinically.”  
**Fair?** Fair if caption omits exercise framing.  
**Status:** Disclosed in README / PRESENTATION / honesty block / x-post.  
**Mitigate:** Lead every post with exercise + no medical validity.  
**Resolve (no formula change):** Keep loud UI honesty; never remove provenance.

### C2 — OpenRouter API ≠ consumer ChatGPT / Claude / Gemini web
**Status:** Disclosed.  
**Mitigate:** Screenshot footer always says API routes.  
**Resolve:** Same — caption discipline only.

### C3 — Single uncalibrated DeepSeek R1 judge
**Status:** Disclosed.  
**Mitigate:** Say “LLM-as-judge” on every chart.  
**Resolve later (optional):** human calibration set on 1–2 held-out cases — out of scope for formula freeze.

### C4 — Amateur / exploratory N — even with ~70 runs
**Attack:** “N≈70 proves superiority.”  
**Fair?** Fair if language implies powered study / p-values.  
**Status:** Disclosed exploratory; larger N only stabilizes descriptive mean±std.  
**Mitigate:** Ban significance language; report mean±std · N · pack_rev · scope.  
**Resolve:** Process only — post checklist + x-post template (done).

### C5 — Dual-track confusion
**Attack:** Pool or swap Comprehension vs Structured captions.  
**Status:** UI labels + never-pool captions + separate protocols.  
**Mitigate:** Always print track name on charts.  
**Resolve:** Keep Structured labeled optional; never auto-merge Histories.

### C6 — Pack provenance / ED monoculture
**Attack:** “10 real ED cases validate the model.”  
**Fair?** Fair if provenance hidden.  
**Status:** README + PRESENTATION + pack `disclosure` now state Cursor/internet fixtures.  
**Mitigate:** Mention provenance whenever quoting the 10-case suite.  
**Resolve:** Keep disclosure; diversify pack only as future content (not this ship).

### C7 — Comprehension gold = QnA contract, not undivided prose
**Attack:** UI once implied prose was scored.  
**Status:** Freeze copy + honesty bullets fixed (gold_raw + narrative twin).  
**Mitigate:** Keep claim count / fingerprint after Freeze.  
**Resolve:** True prose-claim gold would change semantics — **out of scope** (would snaturare).

### C8 — Photocopy sections
**Attack:** Five dimensions look independent; free-form photocopies.  
**Status:** Honesty disclose.  
**Mitigate:** Do not market per-section rankings as independent Structured answers.  
**Resolve:** Soft-section parse rate metric = backlog (disclosure-only for now).

---

## High

| ID | Issue | Disclosure already? | Mitigate now | Full resolve (no formula change) |
|----|--------|---------------------|--------------|----------------------------------|
| H1 | Custom / auto scaffold weak | Yes (captions) | Multi×all = pack only | Block custom from Multi×all UI if reintroduced |
| H2 | Same `case_id` family across slots | Dual-read + balanced default | Default Rebuild = balanced after Multi×all | Keep balanced as suite headline |
| H3 | Judge N/A / repair optics | Ops table + retry caption | Always show Failures/N/A beside mean | Keep scored-only mean |
| H4 | Cost forecast ≠ invoice | Spend card “often over” | Show billed total after run | Keep estimate labels |
| H5 | Legacy module names `beta_*` | Wire ids renamed; modules internal | Do not surface in UI | Optional code rename later |
| H6 | Hosted demo often cloud-only | README / PRESENTATION | Caption local vs hosted | Same |

---

## Medium

| ID | Issue | Notes |
|----|--------|------|
| M1 | `accuracy` JSON field alias | UI says Clinical Composite; JSON compat remains |
| M2 | Streamlit page `comprehension_redirect` | Prefer Home; redirect when idle |
| M3 | i18n partial on Structured | English aligned; not a honesty gap |
| M4 | Debug logs removed from app | Good — avoid residual agent noise |

---

## Verification checklist (adversary pass)

- [x] Boot: key then QVAC every fresh session  
- [x] Spend Yes/Cancel both tracks  
- [x] Honesty not duplicated under Rebuild  
- [x] Ops chart removed; table remains  
- [x] Protocol chip shows `comprehension-v1` (not beta)  
- [x] Pack / README / PRESENTATION / x-post state exercise + provenance  
- [x] Tracks never pool (protocol isolation)  
- [x] ~70-run means: descriptive solidity OK; still not clinical validity  

## Bottom line for a ~70-run Comprehension suite mean

Descriptively **much stronger** than N=5 anecdote: report mean±std, N scored,
Balanced scope, pack_rev 3, protocol `comprehension-v1`, roster version, and the
full honesty stack. An adversary can still win on **medical-validity framing**,
**API≠web**, **judge uncalibrated**, and **fixture provenance** if those lines
are cropped from screenshots — keep them in the frame.
