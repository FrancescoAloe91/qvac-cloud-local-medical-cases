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

## Repeated runs

An aggregate ranking is withheld until every compared model has the same minimum
of five valid observations from one immutable cohort. A cohort hash includes the
case, confirmed reference, model configuration, prompt/scoring versions, and
protocol track. Controlled and native-default runs never mix.

N=5 is labelled exploratory. Sample SD, median, and IQR describe repeatability
for that exact case/reference; they do not establish general clinical validity.
