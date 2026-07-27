# Gold-only scoring contract

Every active run requires an anonymized real/custom case and a user-supplied
reference covering diagnosis, tests, urgency, safety, and plan. Demo/rubric
scoring is not part of the active benchmark.

The cloud extractor may reorganize free-form text, but each atomic claim needs a
verbatim source quote. The reference is frozen and organized automatically
before model collection starts.

## Primary correctness

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
graded coverage = critical-weighted mean of per-claim coverage values (0..1)
discipline starts at 1.0
helpful or neutral addition penalty = 0
unsupported penalty = 0.25 × severity
contradictory penalty = 0.75 × severity
dangerous penalty = 1.00 × severity
discipline = max(0, 1 - total penalty / number of reference claims)
section correctness = 50% coverage + 35% clinical quality + 15% discipline
```

The final primary correctness is the predeclared weighted mean:

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
There is no synthetic zero, fallback score, score cap, or forced tie-break.
Exact ties keep the same rank.

Retryable transport errors receive bounded retries. A schema/evidence rejection
receives a corrective retry, followed—when configured—by one independent
verifier. These recovery paths may repair formatting or judge-output defects;
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

N=5 is labelled exploratory. Sample SD, median, and IQR describe repeatability
for that exact case/reference; they do not establish general clinical validity.
