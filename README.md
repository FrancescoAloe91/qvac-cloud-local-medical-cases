# QVAC vs Cloud LLMs — Gold-only Health Benchmark

Experimental, open-source comparison of pinned OpenRouter API models and
on-device GGUFs through the QVAC SDK. It uses frozen, user-supplied source
quotes as canonical claims, a blind DeepSeek R1 judge, strict evidence
validation, and an independent whole-run verifier when primary judging fails.

This is a research/demo tool, not a medical device. It does not validate the
clinical truth of user input. If the case or reference is wrong or incomplete,
the result is not clinically meaningful.

Live app: https://francescoaloe91-qvac-vs-cloud-llms-health-test-app-wihxyd.streamlit.app
Repository: https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test

## Active protocol

1. Paste one anonymized real/custom clinical case.
2. Paste a free-form reference covering diagnosis, tests, urgency, safety, and
   initial plan.
3. When Run is clicked, the reference is frozen and a pinned cloud extractor
   automatically reorganizes only source-supported claims before any candidate
   is called. There is no separate extraction/review step.
4. Run either the `controlled` or `native_defaults` track. They are separate
   cohorts and are never pooled.
5. Candidate identities are blinded for judging. Evidence matching ignores only
   presentation differences (Markdown, whitespace, case, and punctuation) and
   accepts combined quotes only when every substantial sentence is textually
   present in the candidate answer. It never uses fuzzy or semantic matching.
6. The **Clinical Composite Score** computes each section from 50% graded
   reference coverage, 35%
   clinical quality, and 15% evidence discipline. Helpful and neutral additions
   are unpenalized; unsupported, contradictory, and dangerous content has a
   proportional effect.
7. Transport, timeout, malformed evidence, cancellation, and empty output are
   technical N/A—not synthetic zero scores.

Demo cases, rubric scoring, fallback scores, score caps, and forced tie-breaks
have been removed from the active benchmark. Exact ties remain ties.

Full scoring contract: [benchmark/SCORING.md](benchmark/SCORING.md).

## Models and claims

Cloud candidates are exact API routes pinned in
[benchmark/models.yaml](benchmark/models.yaml):

- `openai/gpt-5.5`
- `anthropic/claude-sonnet-5`
- `google/gemini-3.5-flash`

These are OpenRouter API comparisons. They are not claims about the consumer
ChatGPT, Claude, or Gemini free web tiers.

The primary judge is `deepseek/deepseek-r1`. The independent verifier is
`qwen/qwen3.5-397b-a17b`, outside the candidate roster and extractor family.
Requested and routed model/provider metadata are
stored when OpenRouter supplies them. Retryable transport failures and rejected
schema/evidence receive bounded automatic retries. Before another paid judge
call, deterministic local repair normalizes harmless wrappers, aliases, extra
fields, lists, and numeric strings without changing evidence text or clinical
words. A corrective retry requests only invalid sections when valid sections
can be retained from the same judge. Candidate collection retries once only for
retryable transport or explicit truncation. Transport retries the call;
truncation requests only missing/cut sections and retains parsed sections.
Markdown, punctuation, spacing, Unicode styling, and case are normalized
locally before declaring a section missing. Genuinely absent clinical content
remains N/A. The independent verifier is activated only after
systemic residual failure (at least two affected candidates and at least the
30% cohort threshold). If activated, it re-judges
the complete fixed candidate set; one ranking never mixes judge cohorts. Empty or partial candidate
answers, invented evidence, and genuinely unusable JSON remain technical N/A.
Clinical meaning is the judge's first coverage/quality criterion. The host
accepts evidence assembled from textually present sentences, but never uses
unconstrained fuzzy matching that could accept changed clinical facts.

Local candidates are loaded from gitignored GGUF files through the QVAC SDK
sidecar:

- Gemma 2 2B IT Q4
- Llama 3.2 3B Instruct Q4
- Phi-3.5 Mini Instruct Q4
- MedPsy 1.7B Q4, 4B Q4, and 4B Q8

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
- requested, valid, and failed observation counts.
- batch and iteration IDs, effective judge, retry/failure metadata, sampling,
  token cap, timing, and available local hardware measurements.

Each model enters the aggregate ranking after five valid observations in one
cohort, even when another model has fewer valid results or N/A failures. Every
mean retains its own N; missing scores are never imputed and never discard valid
data from other models. N=5 is explicitly exploratory. The dashboard reports
sample SD, median, and IQR as repeatability signals for that exact case/reference,
not general clinical validity.

The primary view keeps every valid per-model observation. A secondary paired
complete-case sensitivity ranking uses only iterations where every model has a
valid score; it never imputes missing scores. All results measure
agreement/usefulness relative to one user reference, not clinical accuracy or
clinical validation.

## Privacy and hosted persistence

The app no longer stores raw keys in an IP-address vault and never copies a
visitor key into process-global environment state.

- Local mode: key and data are scoped to the current session/local workspace.
- Hosted mode: configure Supabase Auth, apply
  `supabase/migrations/202607270001_secure_benchmark.sql`, and provide a Fernet
  `APP_ENCRYPTION_KEY`.
- Supabase Row Level Security restricts rows to `auth.uid()`.
- API keys and full artifacts (case, reference, answers, judgments) are encrypted
  before storage.
- Files under `artifacts/`, `.env`, secrets, and GGUF weights are gitignored.

See [.env.example](.env.example) for deployment variables. Keep the encryption
key in the hosting secret manager, never in git.

## Install and run

Requirements: Python 3.9+, Node.js 22.17+, and OpenSSL 3 on macOS for the local
sidecar.

```bash
git clone https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test.git
cd qvac-vs-cloud-llms-health-test

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

## Live judging UI

- Left: FIFO collect order. Rows never reorder.
- Each row shows deterministic pipeline-stage progress, the current stage, and
  elapsed seconds. Stage percentages are not presented as an ETA: queued 10%,
  request sent 25%, response validation 70%, corrective retry 75–88%,
  independent verification 92%, and completion 100%.
- Right: dynamic provisional Clinical Composite Score histogram.
- Completed technical failures display `N/A · technical`.
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
app.py                         Streamlit app
benchmark/gold.py              extraction validation + cohort identity
benchmark/judge.py             evidence validation, N/A semantics, verifier
benchmark/scoring.py           deterministic balanced claim scoring
benchmark/report.py            atomic artifacts + homogeneous statistics
benchmark/cases/caseC.json     only active case template
lib/secure_account_store.py    Supabase Auth/RLS encrypted persistence
supabase/migrations/           hosted storage schema and policies
sidecar/qvac_server.mjs        QVAC SDK bridge and track sampling
tests/                         offline methodology/runtime tests
```
