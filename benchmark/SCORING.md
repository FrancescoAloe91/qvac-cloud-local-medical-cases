# Gold-only scoring contract

Every active run requires an anonymized real/custom case and a user-supplied
reference covering diagnosis, tests, urgency, safety, and plan. Demo/rubric
scoring is not part of the active benchmark.

The cloud extractor may segment free-form text into sections, but the exact
user `source_quote` is the canonical scored claim. Extractor summaries,
rewrites, duplicate quotes, and extractor-selected critical weights cannot
change claim meaning or score weight.

## Clinical Composite Score

The blind judge grades coverage of every frozen reference claim continuously
from 0 to 1 and classifies additional content as helpful, neutral, unsupported,
contradictory, or dangerous. Every nonzero coverage decision and every added
claim needs verbatim candidate evidence.

Host evidence contract (not the same as technical N/A):

- Unverifiable **coverage** quotes → that claim’s coverage is set to **0** locally
  (section stays scorable; no paid retry for presentation alone).
- Unverifiable **helpful / neutral / unsupported** additions → dropped (no
  inventing text).
- Unverifiable **contradictory / dangerous** additions → **dropped** with an
  audit marker (`judge_unverified_harm_dropped:*`); no invented quote and no
  automatic discipline penalty. This is **not fail-closed**: unverifiable harm
  does not invent a penalty. Verified harmful quotes still apply proportional
  discipline.
- Clinical **quality is independent of coverage** (`graded-clinical-v4`). The
  judge remains responsible for low quality on clinically bad answers; the host
  does not clamp quality to coverage.
- Candidate parsing never photocopies an unstructured full response into all
  five sections. Missing sections stay missing / N/A.
- Technical N/A remains reserved for empty/partial candidates, transport,
  unusable schema after salvage/repair, timeout, and cancellation.

Evidence validation is presentation-tolerant but text-strict. It normalizes
Markdown/HTML, whitespace, letter case, Unicode presentation, list markers, and
styling punctuation, while preserving clinically meaningful numeric punctuation
(decimals, ranges like `10–20`, `±`, slash ratios). Matching is token-sequence /
word-boundary safe so short tokens like `renal` do not match inside `adrenal`.
A judge quote assembled from multiple non-contiguous sentences is accepted only
when every substantial sentence is textually present. Partial spans (for example
“half of a long quote”) do **not** count. No fuzzy or semantic quote matching is
used, so changed or invented clinical words remain invalid.

For each section:

```text
graded coverage = unweighted mean of per-claim coverage values (0..1)
discipline starts at 1.0
helpful or neutral addition penalty = 0
unsupported penalty = 0.25 × severity
contradictory penalty = 0.75 × severity
dangerous penalty = 1.00 × severity
discipline = max(0, 1 - min(1, total fixed penalty))
section Clinical Composite Score = 50% coverage + 35% clinical quality + 15% discipline
```

The final Clinical Composite Score is the predeclared weighted mean:

- Diagnosis: 30%
- Safety: 25%
- Plan: 20%
- Tests: 15%
- Urgency: 10%

Coverage remains the largest component. Clinical quality rewards coherence,
prioritization, usefulness, and appropriate caution independently of coverage
(v4). Discipline discourages speculation and harm without penalizing reasonable
additions, and only when the cited candidate quote is present.

Artifact fields may still be named `accuracy` for schema compatibility. That
number is the Clinical Composite Score relative to one user reference. It is
not clinical accuracy, correctness against an external truth set, or clinical
validation. There are no fallback scores, artificial score caps, or forced
tie-breaks on the gold `graded-clinical-v4` path. Legacy helpers
`linear_item_score` / `semantic_item_score` still reference `ITEM_SCORE_CAP`
(96.5) for archived rubric/semantic modes only — they are not applied to
active gold Clinical Composite scores. Exact ties keep the same rank.

## Judge pipeline and token budgets

Candidate answers (all nine cloud + local models) keep a hard **3000-token**
output budget. Raising the judge budget did **not** raise candidate limits.

Primary judge (`deepseek/deepseek-r1`) flow per candidate:

1. **Primary call** — up to **16384** completion tokens as a JSON object.
2. **Local salvage first** — deterministic normalization of wrappers, aliases,
   singleton/list forms, extra fields, and numeric strings. Presentation salvage
   for evidence quotes (Markdown/whitespace/case/punctuation). Unverifiable
   quote rows are dropped or zeroed locally. Salvage never invents evidence and
   never changes clinical words.
3. **Section repair** — paid retry only for sections still invalid after salvage.
   Already-accepted sections are retained. If the primary hit the length cap
   (common on long Claude/OpenAI answers) or more than one section failed,
   repair runs **one section at a time** with a **4096-token** budget each.
   There is no full five-section corrective redo that would truncate again.
4. **Whole-run verifier** — only after systemic residual failure: at least two
   technical failures and at least 30% of the fixed cohort, with an eligible
   independent verifier configured (`qwen/qwen3.5-397b-a17b`, outside the
   candidate roster and extractor family). The verifier re-judges the entire
   fixed set and becomes the sole effective judge for that ranking. A single
   anomaly never activates it. Primary and verifier cohorts are never mixed.
   There is no Claude (or other candidate-model) verifier path.

These recovery paths may repair formatting or judge-output defects; they do not
relax candidate completeness, evidence integrity, or the scoring formula.

## Failure semantics and ranking

Collection errors, empty/partial output, timeout, judge transport failure,
unusable schema after salvage/repair, and cancellation are technical N/A
observations. Presentation-only evidence problems are handled locally (coverage
zero; unverifiable harmful additions are dropped with an audit marker, never
invented as fail-closed penalties), not as N/A. Technical N/A rows are excluded from means
and reported with reason counts. There is no synthetic zero.

### Recovering the candidate's sections

A candidate answer is read by clinical meaning, not by one exact template.
Deterministic parsing accepts numbered markers (`A1:`, `Q1 [diagnosis]:`,
`## A1 [DIAGNOSIS]`, `Answer 1 —`, and a marker restated mid-sentence),
Markdown headings and emphasis, ordinals, list bullets, absent colons, any
casing, Unicode presentation, sections in any order, and synonymous or
localized labels (impression, workup, investigations, triage,
contraindications, management, and similar). A restated question followed by
its answer contributes both fragments to the same section, and a heading always
ends the previous section so content is never attributed to the wrong question.
Reasoning wrappers (`<think>…</think>` and equivalents) are removed before
scoring, so a private monologue is neither graded nor allowed to displace a real
answer.

This tolerance is presentational only. The parser never writes, completes, or
duplicates clinical content, so recovery is possible exactly when the model
itself produced the content. Cloud answers usually follow the template more
closely; local GGUFs often need the tolerant path. The same parser is used for
both so formatting quirks are not treated as missing medicine.

A hit output-length cap is not by itself a failure. A truncated response whose
required sections all carry content is judged normally; it becomes N/A only when
a required section has no content at all.

Candidate collection spends at most one retry, on a retryable transport fault, a
local sidecar/GGUF fault, explicit truncation, or sections the model left
unwritten. Truncation and unwritten sections regenerate only the affected
questions and retain already parsed sections verbatim.

### What remains N/A

After tolerant parsing and the bounded retry, an observation is still technical
N/A when the candidate returned nothing, when a required clinical section has no
recoverable content (including a degenerate loop that never reaches the later
questions), when collection or the local runtime failed outright, when the
provider blocked the content, or when the judge produced unusable JSON or
evidence that is not verbatim in the answer after local salvage and section
repair. Genuinely absent clinical content is never scored, imputed, or credited.

## Repeated runs

Each model enters the aggregate ranking when it has at least five valid
observations from one immutable cohort. Models may therefore have different N:
an N/A for one model neither removes nor delays valid data from another. Every
mean exposes its own valid and failed counts, and no missing score is imputed. A
cohort hash includes the case, confirmed reference, model configuration,
prompt/scoring versions, and protocol track. Controlled and native-default runs
never mix.

Multi runs continue when any valid judged results remain and abort early only
when judge infrastructure is globally unavailable. The judge pipeline has a
wall-clock budget; unfinished rows become explicit timeout N/A. STOP persists a
cancelled/partial artifact and reason. Charts omit N/A rather than drawing 0%.
Every fixed candidate reaches exactly one terminal row before an iteration is
written and the next iteration starts. Candidate-specific N/A never aborts a
batch. Terminal N/A rows remain at completion progress and do not regress into a
stuck corrective-retry UI stage.

N=5 is labelled exploratory. Reruns are not bit-identical. Sample SD, median,
and IQR describe repeatability for that exact case/reference; they do not
establish general clinical validity.

The primary analysis reports all valid observations with per-model valid N,
failed N, and failure rate. A secondary paired complete-case sensitivity
ranking includes only iterations in the same batch where every model has a
valid score. It is labelled sensitivity analysis and never replaces or imputes
retained observations.

## Offline rescore and rebuild

Saved artifacts can be rescored without API calls:

- Recompute section Clinical Composite Scores from stored claim assessments with
  the current host formula (`graded-clinical-v4`) when the artifact already
  carries independent (unclamped) quality.
- Older `graded-clinical-v3` artifacts keep their **stored ranking** when the
  original unclamped quality cannot be recovered; offline metadata records the
  formula used and never silently stamps v3 runs as v4.
- Re-validate stored judge JSON with current local salvage; N/A rows that were
  only presentation/schema failures may recover offline.
- History “rebuild last N” builds exploratory or official means from the newest
  same-cohort runs (official means still need ≥5 valid cohort observations).
- Prior rankings are stamped under `reproducibility.offline_rescore.stored_ranking`
  so old vs new comparisons remain possible.

`scripts/_offline_rescore_all.py` batch-applies the same offline path over an
owner artifact directory. New rescores label `graded-clinical-v4`; it does not
rewrite old artifacts as v4 silently. It does not call OpenRouter.

## Calibration

The LLM judge is **uncalibrated for public claims** until human-reviewed fixtures
under `fixtures/calibration/` have been checked and the offline comparison
helper passes. Treat scores as LLM-as-judge estimates until then. The whole-run
verifier is an independent re-judge for systemic primary failure — it is **not**
calibration.
