# Cloud & local medical LLMs — Medical Cases

Experimental, open-source comparison of pinned **OpenRouter API** cloud models
and **on-device / local medical LLMs** (QVAC SDK + peer GGUFs) on teaching
medical cases. Blind DeepSeek R1 judge, evidence-linked scoring, frozen
author-supplied reference.

**This is a research / demo exercise — not a medical device, not clinical
advice, not a powered clinical study, and not an official MedPsy blog
evaluation.** Scores measure agreement with a frozen reference on this bench,
not real-world diagnostic accuracy.

Live app: https://qvac-cloud-local-medical-cases.streamlit.app  

Repository: https://github.com/FrancescoAloe91/qvac-cloud-local-medical-cases

## Two tracks (KPIs never pool)

| Track | Role | Entry | What it stresses | Protocol / History |
|-------|------|--------|------------------|--------------------|
| **Comprehension** | **Default home** | `app.py` (`streamlit_app.py` on Community Cloud) | Free-form clinical narrative vs curated Q1–A5 `gold_raw` (prose is the narrative twin) · acute ED pack Case 1–10 | `comprehension-v1` · `case_id=comprehension` (legacy `beta-*` History still pools via dual-read) |
| **Structured A1–A5** | **Optional secondary** | Sidebar → Structured | Rigid `A1:`…`A5:` slots vs Prepare→Confirm claim gold | Graded / caseC History · A1–A5 Rebuild |

Same Clinical Composite math (Coverage / Quality / Discipline → section → weighted mean).
**Never** mix Comprehension and Structured means, Rebuild windows, or screenshot claims.

### Comprehension (home) — narrative meaning

Comprehension is the **main** surface: models write like clinicians (or peers)
in free prose. That better matches how real assistants are used — storytelling,
prioritization, safety language — instead of forcing a quiz format.
The judge still scores five clinical dimensions (diagnosis / tests / urgency /
safety / plan) against a **curated `gold_raw` Q1–A5 contract**; the undivided
`reference_prose` is the human-readable twin, not the claim list itself.
Unmarked free-form answers may be **photocopied** into all five sections —
dimensions are not fully independent Structured answers. Prefer **Balanced
cases** Rebuild after Multi×all across the 10-case pack.

### Structured A1–A5 (optional) — format / contract stress

Structured remains available when you care about **slot compliance** and
claim-linked Prepare→Confirm gold (the original graded workflow). It is easy
to over-read as “clinical IQ”; it mostly stresses schema discipline and
editable claim contracts. Use it as a secondary check, not as the headline
track for free-form model capability.

Boot (every fresh session): OpenRouter BYOK key dialog → QVAC SDK status OK
(QVAC ack may be remembered in local `.ui_prefs.json`; keys are never stored there).

## Case pack provenance (Comprehension Case 1–10)

The default Comprehension pack (`benchmark/default_cases/comprehension.json`,
revision 3) is an **acute ED-biased** suite (AKI/hyperK, anaphylaxis, STEMI,
DKA, stroke, overdose, psychosis/delirium, septic shock, PE, UGIB).

**Provenance honesty:** these vignettes were **assembled with Cursor and/or
adapted from public internet clinical teaching material** for this amateur
bench. They are **not** de-identified real patient charts from a hospital EHR,
**not** prospectively validated cases, and **not** a claim of clinical
correctness. Treat stems + gold as **author-supplied exercise fixtures**.

## Known limitations / How to read results

Transparent **amateur** comparison. Do not lead posts with “beat ChatGPT /
Claude / Gemini” (those names mean **OpenRouter API** routes here, not
consumer web apps).

- **Reference-relative** Clinical Composite — agreement with the frozen gold on
  this bench, not clinical truth and not medical validity.
- Cloud slots = **OpenRouter API routes** ≠ consumer ChatGPT / Claude / Gemini web.
- **Uncalibrated single LLM-as-judge** (DeepSeek R1).
- **Exploratory** Multi — show mean±std and N per model. Larger N (e.g. tens of
  scored runs per model across the suite) improves descriptive stability; it
  still does **not** create a powered superiority study.
- **Comprehension** = main free-form / meaning track. **Structured** = optional
  format-stress track. Never pool.
- **Rebuild mean = scored-only.** Technical failures and exact Clinical Composite
  == 0 are treated like N/A (excluded from N). Exact 0 usually means refusal or
  no usable clinical content vs gold. Main ranking shows **n scored**; zeros +
  technical N/A appear in a **Failures/N/A table** (ops honesty), not as a
  fourth competing clinical chart.
- Label **Same-case** vs **Portfolio** vs **Balanced cases**; a **new cohort**
  (short hash) starts when case text or locked reference claims change —
  Confirm alone on the same content keeps the same set id.
- Roster version **default 9** · MedPsy family · medical peers where relevant.
- Local recovery is **capped** (not identical cloud repair/fill weapons).
- Cost UI **estimate ≠ OpenRouter invoice** — billed truth is usage.
- Comprehension free-form may be **photocopied** into five sections — not
  independent Structured A1–A5 slots.
- Pack cases are **teaching-style fixtures**, not EHR / medical validity.
- Export JSON may still include the key `accuracy` for back-compat; prefer
  `clinical_composite` / `clinical_composite_mean` when reading scores.

**Accepted residuals (not product defects):** gold-relative ≠ clinical truth;
OpenRouter API ≠ consumer ChatGPT/Claude/Gemini web; uncalibrated LLM-as-judge;
teaching pack ≠ real charts; git history may still contain old internal audit
blobs even though the current tree does not. Do not re-commit adversarial audit
notes into this repo.

Screenshot rule: track name · protocol · pack_rev · mean±std · N · scope ·
roster · cohort. Template: [docs/x-post-template.md](docs/x-post-template.md).

## How the system works (end-to-end)

### 1. Gold-only reference

1. Paste one anonymized real/custom clinical case.
2. Paste a free-form reference covering diagnosis, tests, urgency, safety, and
   initial plan.
3. **Prepare reference** runs a pinned cloud extractor once and shows editable
   five-section claims. **Confirm reference** freezes `confirmed_at` and the
   gold JSON; candidates start only after confirm. Changing the raw reference
   clears prepared/confirmed state. CLI still requires a pre-confirmed gold JSON.
4. The exact user `source_quote` is the canonical scored claim. Extractor
   summaries, rewrites, or weights cannot change claim meaning or score weight.
   The schema `critical` flag is forced off / ignored — all claims share equal
   weight. Prepare may invent-then-repair segmentation; Confirm locks the
   verbatim quotes you approve.
5. Run a track: **`controlled`** (default best-effort: temp 0.2 + preferred
   provider, OpenRouter fallbacks **on**), **`strict_controlled`** (opt-in: no
   fallback, `require_parameters`, route miss → technical N/A), or
   **`native_defaults`**. Tracks never pool. Multi/Rebuild same-case means pool
   only on matching `cohort_id` (requested recipe + scoring gold contract
   **without** display-only section summaries). `execution_cohort_id` (actual
   routed models/providers + GGUF digests + QVAC runtime) is audit metadata and
   does **not** split batch means. Default install is cloud-only until GGUFs +
   sidecar are present; if the QVAC sidecar is down, Compare is cloud-only.
   Pasting the same raw case/reference alone is not enough: a new Prepare that
   changes claim splits starts a new cohort. Re-Confirm without editing claims
   keeps the same cohort id. The UI can **restore** an exact prior confirmed
   gold from History when the pasted case+raw reference match a case family.
   Confirm supports add/split/delete/move of claims; summaries are display-only.
   Active protocol is gold-only; strings under `OLD/` are archived (not the live
   judge Clinical Composite path).

Demo cases and rubric scoring are not part of the active protocol.

### 2. Candidates collect answers

Nine default roster slots answer the same five questions under the same prompt
(up to **12** when Optional / legacy slots are re-enabled):

- **Cloud** (toggle): OpenRouter API routes in [benchmark/models.yaml](benchmark/models.yaml)
  — `openai/gpt-5.5`, `anthropic/claude-sonnet-5`, `google/gemini-3.5-flash`
- **QVAC MedPsy** (toggle): 1.7B Q4 + 4B Q4 via QVAC sidecar (4B Q8 is optional/legacy)
- **Generic local LLMs** (toggle): Phi-3.5 Mini Instruct Q4 by default —
  Gemma 2 2B and Llama 3.2 3B are optional/legacy Band B slots
- **Medical local LLMs** (toggle): MedGemma 1.5 4B IT Q4, Med42 8B Q4,
  UltraMedical 8B Q4 — medical-specialized peers (not MedPsy)

Dashboard presets (English): **Medical on-device only** (5 = dual MedPsy + medical),
**All on-device** (≤6), **Full roster** (≤9; + optional legacy ≤12), **Cloud only** (3).
Re-enable Gemma / Llama / MedPsy Q8 under **Optional / legacy slots**. Download
medical weights with `./scripts/download_medical_peers.sh` (or
`./scripts/download_all_ggufs.sh`).

Candidate identities are blinded before judging. Every candidate keeps a hard
**3000-token output budget** (cloud and local). That budget was **not** raised
when the judge output budget was increased; only the judge may emit up to 16k
tokens on the primary call.

Collection retries at most once for a retryable transport fault, a local
sidecar/GGUF fault, explicit truncation, or sections the model left unwritten.
Transport retries the call; truncation and unwritten sections request only the
affected questions in one targeted call (local: format-repair **or** one
multi-gap fill — never N sequential) and retain everything already answered.

### 3. Reading answers (cloud vs local)

Section recovery is tolerant about presentation and strict about substance.
Small on-device models seldom reproduce the requested `A#:` layout exactly, and
a different layout is not a clinical failure, so the parser accepts:

- `A1:`, `Q1 [diagnosis]:`, `## A1 [DIAGNOSIS]`, `Answer 1 —`, and an `A1:`
  marker restated mid-sentence after prose;
- Markdown headings, bold or italic emphasis, ordinals, list bullets, missing
  colons, and arbitrary casing;
- synonymous or localized section labels such as impression, workup,
  investigations, triage, contraindications, or management;
- sections written out of order, and a question restated before its answer
  (both fragments are attributed to that one section).

Reasoning wrappers such as `<think>…</think>` are removed before scoring.
The parser never generates, completes, or copies clinical content, so a section
the model genuinely never produced still ends as N/A. Cloud answers usually
follow the template more closely; the same meaning-first parser is used for both
tracks so local formatting quirks are not mistaken for missing medicine. When a
local GGUF returns substantial prose with almost no `A#:` markers, one
**format-repair** pass may re-ask for markers only — it does not invent clinical
content and uses the same parser afterward.

### 4. Blind judge flow

The primary judge is `deepseek/deepseek-r1`. For each blinded candidate:

1. **Primary call** — up to **16384** completion tokens (JSON object). This is
   the only token-budget change relative to earlier runs; candidate caps stay at
   3000.
2. **Local salvage first** — deterministic normalization repairs harmless
   wrappers, aliases, extra fields, lists, and numeric strings. Presentation-
   tolerant evidence matching accepts quotes that are textually present after
   Markdown/whitespace/case/punctuation normalization. Clinically meaningful
   numeric punctuation (decimals, ranges like `10–20`, `±`, slash ratios) is
   preserved. Matching is token-sequence / word-boundary safe. Combined quotes
   are accepted only when every substantial sentence is present; partial spans
   do not count. No fuzzy or semantic matching. Unverifiable coverage quotes
   zero that claim locally; unverifiable dangerous/contradictory additions are
   dropped with an audit marker (no invented quote, no automatic penalty).
   Clinical quality is independent of coverage (`graded-clinical-v4`).
3. **Section repair (not a doomed full redo)** — if some sections remain
   invalid after salvage, the host requests **only those sections**. When the
   primary hit the length cap (common on long Claude API / OpenAI API answers)
   or more than one section failed, repair runs **one section at a time** with a
   4096-token budget each, retaining already-accepted sections. There is no full
   five-section corrective retry that would truncate again.
4. **Whole-run verifier only when systemic** — if bounded primary recovery
   still leaves systemic judge failure (≥2 technical failures and ≥30% of the
   fixed cohort), and an eligible independent verifier is configured
   (`qwen/qwen3.5-397b-a17b`, outside the candidate roster and extractor
   family), that verifier re-judges the **entire** fixed candidate set and
   becomes the sole effective judge for that ranking. One anomaly never
   activates it. Primary and verifier scores are never mixed. There is no
   Claude API (or other candidate) verifier path.

### 5. Clinical Composite Score

Each section score is:

```text
50% graded reference coverage + 35% clinical quality + 15% evidence discipline
```

Clinical quality is scored independently of coverage (v4). The final score is
the weighted mean of sections (Diagnosis 30%, Safety 25%, Plan 20%, Tests 15%,
Urgency 10%). Helpful and neutral additions are unpenalized; unsupported,
contradictory, and dangerous content has a proportional discipline effect when
the judge quote is present in the candidate answer. Exact ties remain ties.
Candidate answers are never photocopied into missing sections — genuinely
absent sections stay missing / N/A.

Artifact JSON may still label the field `accuracy` for compatibility; that
value is the Clinical Composite Score relative to the user reference. It is
**not** clinical accuracy, correctness against an external truth set, or
clinical validation. Fallback scores, artificial score caps, and forced
tie-breaks are not used.

Full scoring contract: [benchmark/SCORING.md](benchmark/SCORING.md).

### 6. N/A semantics

Transport, timeout, malformed evidence that cannot be salvaged, cancellation,
empty output, provider blocks, and genuinely unusable judge JSON are technical
N/A—not synthetic zero scores. They are excluded from means and reported with
reason counts. Genuinely absent clinical content is never scored, imputed, or
credited.

### 7. Multi-run completion

Multi runs continue when any valid judged results remain and abort early only
when judge infrastructure is globally unavailable. Every fixed candidate reaches
exactly one terminal row before an iteration is written and the next starts.
Candidate-specific N/A never aborts a batch. STOP persists a cancelled/partial
artifact. Charts omit N/A rather than drawing 0%. Terminal N/A rows stay at
completion (100%) and do not regress into a stuck “corrective retry” (75%) UI
state.

### 8. Offline rescore / rebuild

History can rebuild last-N means with **zero API cost**:

- Recompute section scores from stored claim assessments with the current host
  formula.
- Re-validate stored judge JSON with current local salvage; presentation/schema
  salvageable N/A rows may recover offline.
- **Same case** means require five valid observations from one immutable cohort.
  Controlled and native-default runs never mix. Rebuild succeeds once any model
  reaches N≥5 even if the History button appears earlier.
- **Portfolio** (optional scope next to Rebuild) averages ≤N **per-model**
  observations (newest first) across cases for the same track and
  `scoring_version` — not a global last-N run-document slice. Roster shapes
  may differ; cloud models with older history still appear when recent runs
  were medical-only. Exploratory cross-case mean, not clinical validation;
  incompatible scoring versions are never pooled.
- Prior rankings are preserved under `reproducibility.offline_rescore`.

A local utility script `scripts/_offline_rescore_all.py` can batch-rescore an
owner artifact tree the same way (no paid calls).

## Models and claims

Cloud candidates are exact API routes pinned in
[benchmark/models.yaml](benchmark/models.yaml). These are OpenRouter API
comparisons. They are not claims about the consumer ChatGPT, Claude, or Gemini
free web tiers.

Requested and routed model/provider metadata are stored when OpenRouter supplies
them. The sidecar reports a stop reason so a cut-off on-device answer is visible
to the host, and its default context window (`QVAC_CTX_SIZE`, 8192) holds the
prompt plus the same 3000-token output budget cloud candidates receive. Every
roster slot reaches exactly one terminal row: a GGUF that fails to load or to
stream is reported as a technical N/A instead of disappearing from the cohort.

Latency and throughput are descriptive operational metrics; they are not
hardware-normalized comparisons between cloud and local execution.

## Statistics and reproducibility

Each artifact uses schema v2 and includes:

- immutable cohort hash;
- case/reference and model configuration;
- scoring and prompt versions;
- requested/routed model metadata;
- blind map and protocol track;
- git/config/prompt/scoring hashes where available;
- per-candidate validity status and failure reason;
- requested, valid, and failed observation counts;
- batch and iteration IDs, effective judge, retry/failure metadata, sampling,
  token cap, timing, and available local hardware measurements.

Each model with at least one scored observation enters the aggregate ranking,
sorted by the mean of its scored runs — even when another model has fewer valid
results or technical N/A failures. Incomplete coverage (Failed% &gt; 0 or
scored &lt; requested) keeps the rank and shows a **partial** badge; N/A is never
treated as a clinical zero. Every mean retains its own N; missing scores are
never imputed and never discard valid data from other models. N=5 is explicitly
exploratory (reruns are not bit-identical). The dashboard reports sample SD,
median, and IQR as repeatability signals for that exact case/reference, not
general clinical validity.

Optional SHA digests and provider-routed model metadata may vary across
OpenRouter responses; treat them as reproducibility aids, not guarantees of
identical provider backends. Empty `EXPECTED_SHA` / unset `MEDPSY_GGUF_SHA256`
and `allow_fallbacks=True` mean a rerun can land on a different routed backend
or GGUF digest — set `MEDPSY_GGUF_SHA256` to pin local weights.

The primary view keeps every valid per-model observation. A secondary paired
complete-case sensitivity ranking uses only iterations where every model has a
valid score; it never imputes missing scores.

## Privacy and hosted persistence

The app no longer stores raw keys in an IP-address vault and never copies a
visitor key into process-global environment state. **Comprehension** and
**Structured** differ on Cloud — read both bullets.

- **Local Comprehension** (`app.py` on your machine): key-scoped workspace under
  `artifacts/owners/<key-fingerprint>/` (or `_local_no_key/` for QVAC-only
  rehearsal). **Same OpenRouter key = shared History/customs/means vault** for
  that workspace on the machine. Saved run JSON and custom drafts may contain
  case text, reference quotes, and model answers — treat the folder as sensitive
  PHI/clinical **plaintext**; do not commit or share it. Anonymize cases before
  paste.
- **Comprehension on Streamlit Cloud** (home track): session-memory only for run
  artifacts and custom drafts — **no** plaintext run JSON and **no**
  `comprehension_custom_drafts.json` on the host FS. Requires visitor BYOK.
  Anonymous (no key) visitors get a per-browser ephemeral owner id
  (`_cloud_ephemeral_<hash16>`), never the shared `_local_no_key` bucket.
  History/Rebuild for that visit lives in the browser session and is lost on
  refresh.
- **Structured on Streamlit Cloud** (`pages/structured_graded.py`): same
  session-memory rule — **no** plaintext run JSON / case drafts on the host FS
  (even without Supabase). Anonymous visitors use per-browser
  `_cloud_ephemeral_<hash16>`, not `_local_no_key`.
- **Structured + Supabase hosted** (Auth + Fernet `APP_ENCRYPTION_KEY`):
  configure Supabase Auth and apply
  `supabase/migrations/202607270001_secure_benchmark.sql`. Workspace directories
  use the Supabase user id. After decrypt, artifacts stay in session memory —
  plaintext is **not** written to disk on the host. Rows are encrypted at rest
  with RLS on `auth.uid()`.
- The public Streamlit demo typically has **no QVAC sidecar**; on-device MedPsy
  requires a local install (`./install.sh` + sidecar).
- Files under `artifacts/`, `.env`, secrets, and GGUF weights are gitignored.

Scores are **reference-relative** (Clinical Composite vs the user-confirmed
gold), not external clinical accuracy. **Mean policy:** exact Clinical
Composite **0** is excluded from averages (treated like N/A for pooling);
technical N/A are separate; low **non-zero** scores still count. Public
screenshots / official ranking claim = **pack Case 1–10** (not custom-only
means). Keep at least one honesty caption visible (OpenRouter API ≠ web ·
author-supplied gold · uncalibrated judge · N=5 exploratory). Legacy Ollama
i18n/`lib/medpsy` helpers are unused on the gold Automated Benchmark path
(live path = QVAC sidecar).

Hosted BYOK: both Comprehension (`app.py`) and Structured strip any process-wide
`OPENROUTER_API_KEY` on Streamlit Cloud so a host secret cannot silently pay for
visitors. Process git history may retain older scrubbed internal notes — treat
that as an accepted residual; the working tree stays audit-doc clean.

See [.env.example](.env.example) for deployment variables. Keep the encryption
key in the hosting secret manager, never in git.

## Install and run

Requirements: Python 3.9+ (hashed locks and CI use 3.9; `./install.sh` prefers
3.10+), Node.js 22.17+, and OpenSSL 3 on macOS for the local
sidecar.

```bash
git clone https://github.com/FrancescoAloe91/qvac-cloud-local-medical-cases.git
cd qvac-cloud-local-medical-cases

chmod +x install.sh
./install.sh                 # MedPsy 4B Q4
# ./install.sh --full-models # all local GGUFs

# Terminal A
cd sidecar && npm start

# Terminal B
source .venv/bin/activate
streamlit run app.py
```

Application dependencies are fully pinned with hashes in `requirements.txt`.
Tests use `requirements-dev.txt`. Edit `requirements.in` /
`requirements-dev.in`, then regenerate locks with `pip-compile`; do not hand-edit
the lock files.

The CLI is also gold-only:

```bash
python -m benchmark list-cases
python -m benchmark dry-run --case caseC --stem-file case.txt --gold-file gold.json
python -m benchmark run --case caseC --stem-file case.txt --gold-file gold.json --n 5
```

`gold.json` must be the confirmed five-section contract emitted by the UI or an
equivalent schema-valid file.

`dry-run` / UI cost estimates are length-aware OpenRouter projections: baseline
covers cloud candidates, the gold extractor, and the primary judge at its 16k
completion cap; the upper bound also includes possible section repair (up to
5×4k per candidate) and an optional whole-run verifier. Actual billed spend
comes from OpenRouter usage.

**Actual cost model** (`benchmark/costing.py`): each artifact stores
`run_cost_usd` (candidates + judge/repair/verifier for that iteration only).
Gold extraction is `batch_shared_cost_usd` and is charged **once per distinct
`batch_id` / Prepare** (same-batch Multi → ×1; portfolio / multi-batch history
sums once per batch). Multi-run totals = sum(run costs) + shared extraction(s).
Estimates remain separate from billed usage.

Windows: `launch_dashboard.bat` / `stop_dashboard.bat` start and stop the
**QVAC SDK sidecar** (not Ollama). Legacy Ollama setup lives under `OLD/legacy/`
(including archived `setup_medpsy.sh`).

## Live judging UI

- Left: FIFO collect order. Rows never reorder.
- Each row shows deterministic pipeline-stage progress, the current stage, and
  elapsed seconds. Stage percentages are not presented as an ETA: queued 10%,
  request sent 25%, response validation 70%, corrective retry 75–88%,
  independent verification 92%, and completion 100%.
- Right: dynamic provisional Clinical Composite Score histogram.
- Completed technical failures display `N/A · technical` and stay terminal.
- Repeated text logs were removed to avoid duplicating the queue state.
- Responsive rules stack the board and review cards on narrow screens and force
  long labels/claims to wrap instead of overlap.

## Verification

Offline checks never call a paid API:

```bash
python -m compileall -q app.py benchmark lib tests
pytest -q
node --check sidecar/qvac_server.mjs
```

GitHub Actions runs these checks, installs hashed dependencies, uses `npm ci`,
runs `npm audit --omit=dev --audit-level=high`, and verifies that private/runtime
files are not tracked.

## Project layout

```text
streamlit_app.py               Community Cloud entrypoint (→ app.py)
app.py                         Streamlit app (local / alternate Cloud main)
benchmark/gold.py              extraction validation + cohort identity
benchmark/judge.py             evidence validation, N/A semantics, verifier
benchmark/scoring.py           Clinical Composite Score (50/35/15)
benchmark/report.py            artifacts, offline rescore, homogeneous stats
benchmark/cases/caseC.json     only active case template
lib/secure_account_store.py    Supabase Auth/RLS encrypted persistence
supabase/migrations/           hosted storage schema and policies
sidecar/qvac_server.mjs        QVAC SDK bridge and track sampling
scripts/_offline_rescore_all.py  batch offline rescore utility (no API)
tests/                         offline methodology/runtime tests
```
