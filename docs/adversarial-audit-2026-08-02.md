# Adversarial audit — 2026-08-02 (Comprehension home)

Ruthless external critic after making **Comprehension** the Streamlit home and
moving rigid A1–A5 to **Structured**. Scoring formulas unchanged.

**Live note:** In-flight Multi workers were not killed for this audit.

## Product framing under audit

| Area | State |
|---|---|
| Home | `app.py` = **Comprehension** (discursive free-form) |
| Secondary | `pages/structured_graded.py` = **Structured · A1–A5** |
| Legacy URL | `pages/beta_comprehension.py` kept so mid-flight Multi sessions survive |
| Protocol isolation | Comprehension `beta-comprehension-v1` **never pools** with graded History |
| Boot | BYOK key every session · QVAC SDK ack in `.ui_prefs.json` (no keys) |

## Critical (still true)

### C1 — Gold-relative ≠ clinical truth / official MedPsy blog
Ranks measure agreement with author-supplied frozen gold on this amateur bench.

### C2 — OpenRouter API ≠ consumer ChatGPT / Claude / Gemini web

### C3 — Single uncalibrated DeepSeek R1 judge

### C4 — Amateur / exploratory N — not a powered study

### C5 — Dual-track confusion (new optics risk)
- **Attack:** Screenshot Comprehension wins, caption as if Structured A1–A5 (or
  vice versa); or pool means across tracks after the rename from “Beta”.
- **Fair?** Fair.
- **Mitigation:** Home honesty block; README two-track table; screenshot footers
  with protocol / scope; UI labels **Comprehension** vs **Structured · A1–A5**;
  internal ids keep `beta_*` for History continuity — disclose that rename is UI-only.

## High

### H1 — Rebuild scored-only / ops reliability split
Unchanged: clinical mean excludes technical N/A and exact-zero composites;
ops chart shows failure theater separately.

### H2 — Free-form judge fragility (MedPsy thinking / JSON repair)
Think-strip + repair hygiene reduce N/A; residual N/A ≠ “model mute”. Disclose
corrective retry cost on DeepSeek.

### H3 — Home rename mid-campaign
Moving home to Comprehension can look like goalpost moves if Structured results
were previously the public face.
- **Mitigation:** State clearly that Structured remains available; Comprehension
  was always a separate protocol; publish track name on every chart.

## Medium

### M1 — Legacy `beta_comprehension` page URL
Kept for in-flight runs; adversaries may claim “two apps”. Caption: prefer Home.

### M2 — Boot dialogs vs mid-run
Boot must not arm during `beta_running` / Structured `benchmark_running`.

## Disclosure checklist for posts

1. Track name: **Comprehension** or **Structured A1–A5** (never ambiguous “the bench”).
2. Reference-relative · not clinical truth · not official MedPsy blog.
3. OpenRouter API routes ≠ consumer web apps.
4. Uncalibrated DeepSeek R1 judge.
5. mean±std · N per model · cohort/protocol · roster version.
6. Do not pool cross-track History.

## CI

Poll Quality gate to **success** on HEAD after push.
