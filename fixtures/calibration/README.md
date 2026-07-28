# Judge calibration fixtures (offline)

The LLM-as-judge is **calibrated** only when:

1. Human reviewers fill expected score ranges in fixtures under this directory, and
2. `scripts/compare_judge_calibration.py` (or `python -m benchmark.calibration`)
   reports all fixtures within tolerance.

The whole-run **verifier** is an independent re-judge for systemic primary
failure. It is **not** calibration.

## Fixture schema (example JSON)

```json
{
  "fixture_id": "caseC-migraine-partial",
  "case_id": "caseC",
  "reviewed_by": "clinician_initials",
  "reviewed_at": "2026-07-28",
  "notes": "Partial coverage answer; no dangerous additions.",
  "candidate_answers": {
    "diagnosis": "Probable migraine without aura.",
    "tests": "Consider MRI if red flags.",
    "urgency": "Routine outpatient.",
    "safety": "Avoid triptans if vascular risk.",
    "plan": "Analgesia and follow-up."
  },
  "expected": {
    "diagnosis": {
      "coverage_min": 0.4,
      "coverage_max": 0.8,
      "quality_min": 0.5,
      "quality_max": 1.0,
      "discipline_min": 0.8,
      "discipline_max": 1.0,
      "score_min": 55,
      "score_max": 85
    }
  }
}
```

Store one fixture per file: `fixtures/calibration/<fixture_id>.json`.
Do not commit live OpenRouter responses with secrets; keep fixtures offline.
