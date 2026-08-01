# QVAC vs Cloud LLMs — Health Test
### Pitch one-pager (demo / investors / workshop)

---

## The problem

Clinical data should not leave the device without DPA, consent, and an audit trail — yet clinicians still want LLM assistance. Cloud APIs are capable but **send the case off-device**; on-device GGUFs keep data local but need an honest, comparable score.

## The proposal

A **gold-only clinical LLM benchmark**: the same anonymized case and user-supplied reference, judged relative to frozen source quotes — not invented scores, not free-tier copy-paste, not clinical validation.

| | Cloud (OpenRouter BYOK) | QVAC / local GGUFs |
|---|---|---|
| Cost per run | OpenRouter usage (candidates + extractor + judge) | $0 collect; judge still billed if used |
| Clinical data | Leaves via API (BYOK) | Collect stays on-device; cloud judge/extractor still leave if used |
| TTFT / TPS | Measured from API stream | Measured from local sidecar |
| Models | Pinned API routes (GPT / Claude / Gemini) | MedPsy + open peer GGUFs |

This is a **research/demo tool**. Artifact “accuracy” is a **Clinical Composite Score relative to the user reference**, not external clinical truth. The judge is an **uncalibrated LLM-as-judge** until human-reviewed fixtures exist. N=5 is exploratory. **Controlled** track uses temp 0.2 (separate cohort from `native_defaults`). TTFT/TPS are ops metrics, not hardware-normalized. Keep the honesty block + mean footer visible in every screenshot.

### Known limitations / How to read results

Transparent **amateur** comparison — not a medical device, not a powered study,
not an official MedPsy blog evaluation. Do not lead with “beat ChatGPT.”

- **Reference-relative** Clinical Composite — agreement with author-supplied frozen gold, not clinical truth.
- Cloud slots = **OpenRouter API routes** ≠ consumer ChatGPT / Claude / Gemini web.
- **Uncalibrated single LLM-as-judge** (DeepSeek R1).
- **Exploratory** Multi default N=5 — show mean±std and N per model.
- **Rebuild mean = scored-only.** Technical failures and exact Clinical Composite == 0
  are treated like N/A (excluded from N). Rationale: a rare exact 0 would crush the
  mean, so it is equated to a non-score. Exact 0 usually = candidate **refusal** or
  no usable clinical content vs gold (valid judge score, not transport crash). This
  History: MedGemma ~2/109 (~2%) refusal zeros on caseC; rates vary by model — not a
  family ranking claim. Main ranking table = **n scored** (no Failed% column);
  zeros + technical N/A live in a **separate ops reliability** chart (counts + %).
- Label **Same-case** vs **Portfolio**; **New Confirm = new cohort** (short hash).
- Roster version **default 9** · MedPsy family · medical peers where relevant.
- Local recovery is **capped** (not identical cloud repair/fill weapons).

Post template: [docs/x-post-template.md](docs/x-post-template.md).

## How it works (60 seconds)

1. Paste **one** anonymized case (`caseC`) + a free-form reference (diagnosis, tests, urgency, safety, plan)
2. **Prepare → review/edit → Confirm** freezes gold; exact user `source_quote` is the scored claim
3. Same five questions to cloud (OpenRouter) and local (QVAC sidecar) under the same prompt
4. Blind **DeepSeek R1** judge → reference-relative Clinical Composite Score; optional whole-run verifier only if systemic judge failure
5. Dashboard ranking for that cohort; **Multi ×5** for exploratory means (sample SD / median / IQR). History resume = restore the exact prior confirmed gold (same claim splits) — never auto-merge different Confirm versions under one rebuild. Optional History **Portfolio** scope = ≤N observations **per model** across cases (same track/scoring_version; not a global last-N run slice) — exploratory, not clinical validation.

### Scoring (current protocol · graded-clinical-v4)

**Per section:** 50% graded reference coverage · 35% clinical quality · 15% evidence discipline  
**Section weights:** Diagnosis 30% · Safety 25% · Plan 20% · Tests 15% · Urgency 10%  

Coverage is claim-level vs frozen quotes; quality is independent of coverage; unsupported / contradictory / dangerous additions hit discipline only when the quote is verified in the answer. Exact ties remain ties. Technical N/A is not a synthetic zero.

## What it proves (and what it does not)

**Yes**

- Prompt parity and a transparent, reference-relative comparison
- Privacy path for **local QVAC collect** (extract/judge may still use OpenRouter)
- Blind LLM-as-judge with evidence checks and bounded repair
- Exploratory multi-run repeatability on one fixed case/reference cohort

**No**

- Not a medical device and not clinical advice
- Does not validate that the user’s case or reference is clinically correct
- N=5 is exploratory — not a general superiority claim about any model family
- Cloud slots are **API routes**, not claims about consumer free web tiers
- Hosted Streamlit demo is often **cloud roster only** (no QVAC sidecar); full on-device path needs a local install
- Salvage/section-repair never invents clinical content; missing sections stay N/A
- OpenRouter prefer-order is pinned; **fallbacks remain on** (not bit-reproducible backends)
- Local format-repair re-asks `A#` markers only — same parser; no invented medicine
- Schema `critical` flag ignored · equal claim weights; Verify ≠ human calibration
- Cost UI = length-aware estimate · billed truth = OpenRouter usage
- GGUF SHA not pinned by default — set `MEDPSY_GGUF_SHA256` to pin

## Demo

| Mode | Link / command |
|---|---|
| **Public** | [Live demo](https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app) — hosted path usually has no QVAC sidecar (cloud roster only) |
| **Full (live QVAC)** | `git clone` → `./install.sh` → sidecar + `streamlit run app.py` → `http://localhost:8501` |

## Stack

Streamlit · Python · OpenRouter (BYOK) · DeepSeek R1 judge · Qwen whole-run verifier (optional) · QVAC SDK sidecar · MedPsy / peer GGUFs · Plotly

## Closing line

> *“Same case, same frozen quotes — cloud APIs vs local MedPsy generation. You see the composite score and decide.”*

**Repo:** https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test
