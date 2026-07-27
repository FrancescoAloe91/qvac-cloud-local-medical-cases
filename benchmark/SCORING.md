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
claim needs verbatim candidate evidence. The host rejects invalid evidence
rather than converting it to a low score.

Evidence validation is presentation-tolerant but text-strict. It normalizes
Markdown, whitespace, letter case, Unicode presentation, and punctuation.
It also accepts a judge quote assembled from multiple non-contiguous sentences
only when every substantial sentence is textually present in the candidate
answer. No fuzzy or semantic quote matching is used, so changed or invented
clinical words remain invalid.

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
prioritization, usefulness, and appropriate caution. Discipline discourages
speculation and harm without penalizing reasonable additions. Nothing is
converted to a binary pass/fail merely for convenience.

## Failure semantics and ranking

Collection errors, empty/partial output, timeout, judge transport failure,
invalid schema, invalid evidence, and cancellation are technical N/A
observations. They are excluded from means and reported with reason counts.
There is no synthetic zero, fallback score, or forced tie-break.
Exact ties keep the same rank.

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
itself produced the content.

A hit output-length cap is not by itself a failure. A truncated response whose
required sections all carry content is judged normally; it becomes N/A only when
a required section has no content at all.

Before a paid corrective call, deterministic local normalization repairs only
unambiguous wrappers, aliases, singleton/list forms, extra fields, and numeric
strings. It never changes evidence text or clinical words. A schema/evidence
rejection retries only invalid sections when accepted sections from the same
judge can be retained. Candidate collection spends at most one retry, on a
retryable transport fault, a local sidecar/GGUF fault, explicit truncation, or
sections the model left unwritten. Truncation and unwritten sections regenerate
only the affected questions and retain already parsed sections verbatim.

### What remains N/A

After tolerant parsing and the bounded retry, an observation is still technical
N/A when the candidate returned nothing, when a required clinical section has no
recoverable content (including a degenerate loop that never reaches the later
questions), when collection or the local runtime failed outright, when the
provider blocked the content, or when the judge produced unusable JSON or
evidence that is not verbatim in the answer. Genuinely absent clinical content
is never scored, imputed, or credited.

If bounded primary recovery leaves systemic judge failure—at least two
technical failures and at least 30% of the fixed cohort—and an eligible
independent verifier is configured, the verifier re-judges the entire fixed
candidate set and becomes the effective judge for that ranking. A single
anomaly never activates whole-run verification. Scores from
primary and verifier judges are never mixed. Without an eligible verifier, the
failed observations remain N/A. These recovery paths may repair formatting or judge-output defects;
they do not relax candidate completeness, evidence integrity, or the scoring
formula.

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
batch.

N=5 is labelled exploratory. Sample SD, median, and IQR describe repeatability
for that exact case/reference; they do not establish general clinical validity.

The primary analysis reports all valid observations with per-model valid N,
failed N, and failure rate. A secondary paired complete-case sensitivity
ranking includes only iterations in the same batch where every model has a
valid score. It is labelled sensitivity analysis and never replaces or imputes
retained observations.

The Clinical Composite Score measures agreement and usefulness relative to one
user-supplied reference. It is not clinical accuracy, correctness against an
external truth set, or clinical validation.
