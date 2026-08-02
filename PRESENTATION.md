# QVAC vs Cloud LLMs — Health Test
### Pitch one-pager (demo / investors / workshop)

---

## Framing (read first)

This is a **transparent amateur exercise**: same frozen gold, same roster recipe,
blind LLM-as-judge. It is **not** a medical device, **not** clinical advice,
**not** a powered clinical study, and **not** an official MedPsy blog evaluation.
Artifact scores are **reference-relative Clinical Composite** numbers — agreement
with author-supplied gold on this bench, not real-world diagnostic truth.

**Case pack honesty:** Comprehension Cases 1–10 were **assembled with Cursor
and/or adapted from public internet clinical teaching material**. They are
exercise fixtures, not validated hospital charts.

---

## The problem

Clinical data should not leave the device without DPA, consent, and an audit trail — yet clinicians still want LLM assistance. Cloud APIs are capable but **send the case off-device**; on-device GGUFs keep data local but need an honest, comparable score.

## The proposal

A **gold-only clinical LLM benchmark** with **two isolated tracks**:

| Track | Role | What you are really measuring |
|-------|------|-------------------------------|
| **Comprehension** | **Home / default** | Free-form narrative capability — how models write diagnosis, tests, urgency, safety, plan in natural language vs curated `gold_raw` |
| **Structured A1–A5** | **Optional secondary** | Rigid slot / format compliance — Prepare→Confirm claim gold and `A1:`…`A5:` answers |

KPIs **never pool** across tracks. Same C/Q/D math underneath; different collect contracts.

| | Cloud (OpenRouter BYOK) | QVAC / local GGUFs |
|---|---|---|
| Cost per run | OpenRouter usage (candidates + optional extract + judge) | $0 collect; judge still billed if used |
| Clinical data | Leaves via API (BYOK) | Collect stays on-device; cloud judge/extractor still leave if used |
| TTFT / TPS | Measured from API stream | Measured from local sidecar |
| Models | Pinned API routes (GPT / Claude / Gemini) | MedPsy + open peer GGUFs |

N=5 is exploratory. Showing **larger descriptive N** (e.g. ~70 scored runs in a suite mean) improves stability of mean±std — it still does **not** create clinical validation or significance claims. **Controlled** track uses temp 0.2 (separate cohort from `native_defaults`). TTFT/TPS are ops metrics, not hardware-normalized. Keep the honesty block + mean footer visible in every screenshot.

### Known limitations / How to read results

- **Reference-relative** Clinical Composite — not clinical truth / not medical validity.
- Cloud slots = **OpenRouter API routes** ≠ consumer ChatGPT / Claude / Gemini web.
- **Uncalibrated single LLM-as-judge** (DeepSeek R1).
- **Comprehension** scores curated `gold_raw` Q1–A5; prose is the narrative twin.
  Photocopy caveat: unmarked free-form notes may repeat into all five sections.
- Pack = acute ED-biased Case 1–10 · Cursor/internet-assembled fixtures.
- **Structured** = optional format-stress — do not headline it as clinical IQ.
- **Rebuild mean = scored-only.** Exact Clinical Composite == 0 treated like N/A
  (refusal / empty clinical content). Ops honesty = Failures/N/A **table** only.
- Label **Same-case** vs **Portfolio** vs **Balanced cases**; a **new cohort**
  starts when case text or locked reference claims change — Confirm alone on
  the same content keeps the same set id.
- Roster version **default 9** · MedPsy family · medical peers where relevant.
- Local recovery is **capped**.

Post template: [docs/x-post-template.md](docs/x-post-template.md).

## How it works (60 seconds)

### Comprehension (recommended demo path)

1. Open **Home** · pick Case 1–10 (or New case) · confirm Freeze (gold_raw contract)
2. Cost estimate → **Yes · start run** (Single / Multi ×N / Multi×all pack)
3. Free-form collect → blind DeepSeek R1 → session KPIs under Live responses
4. Rebuild mean offline · prefer **Balanced cases** after Multi×all · show mean±std · N

### Structured (optional)

1. Paste case → Prepare → edit claims → Confirm (new cohort if claims/case change)
2. Same roster · rigid A1–A5 answers · separate History / Rebuild

### Scoring (shared C/Q/D math · different wire stamps)

**Per section:** 50% graded reference coverage · 35% clinical quality · 15% evidence discipline  
**Section weights:** Diagnosis 30% · Safety 25% · Plan 20% · Tests 15% · Urgency 10%  

Same formula weights on both tracks; wire `scoring_version` stamps differ (**`comprehension-v1`** on Home vs **`graded-clinical-v4`** on Structured) so History/Rebuild never pool across tracks.

Exact ties remain ties. Technical N/A is not a synthetic zero.

## What it proves (and what it does not)

**Yes**

- Prompt parity and a transparent, reference-relative comparison
- Privacy path for **local QVAC collect** (extract/judge may still use OpenRouter)
- Blind LLM-as-judge with evidence checks and bounded repair
- Exploratory multi-run repeatability on fixed gold cohorts (descriptive mean±std)

**No**

- Not a medical device and not clinical advice
- Does not validate that stems or gold are clinically correct
- Larger N (including ~70-run suite means) ≠ powered superiority study
- Cloud slots are **API routes**, not consumer free web tiers
- Hosted Streamlit demo is often **cloud roster only** (no QVAC sidecar)
- Salvage/section-repair never invents clinical content
- Cost UI = rough estimate · billed truth = OpenRouter usage

## Demo

| Mode | Link / command |
|---|---|
| **Public** | [Live demo](https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app) — hosted path usually has no QVAC sidecar (cloud roster only) |
| **Full (live QVAC)** | `git clone` → `./install.sh` → sidecar + `streamlit run app.py` → `http://localhost:8501` |

## Stack

Streamlit · Python · OpenRouter (BYOK) · DeepSeek R1 judge · Qwen whole-run verifier (optional) · QVAC SDK sidecar · MedPsy / peer GGUFs · Plotly

## Closing line

> *“Same frozen gold — free-form Comprehension on Home for real narrative capability; optional Structured if you care about slot contracts. Amateur exercise, loud disclaimers, no medical validity.”*

**Repo:** https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test
