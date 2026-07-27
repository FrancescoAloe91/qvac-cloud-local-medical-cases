from __future__ import annotations

import pytest

from benchmark.report import summarize_runs
from benchmark.schema import RunArtifact


def _artifact(index: int, *, cohort: str = "cohort-a", failure: bool = False):
    ranking = [
        {
            "key": "chatgpt",
            "accuracy": None if failure else 80.0 + index,
            "status": "n/a" if failure else "ok",
            "status_note": "timed_out" if failure else "",
        },
        {
            "key": "claude",
            "accuracy": 82.0 + index,
            "status": "ok",
            "status_note": "",
        },
    ]
    return RunArtifact(
        run_id=f"run-{index}",
        case_id="caseC",
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:01:00Z",
        n_index=index + 1,
        ranking=ranking,
        cohort_id=cohort,
    )


def test_mixed_cohorts_are_rejected():
    with pytest.raises(ValueError, match="mixed cohorts"):
        summarize_runs([_artifact(0), _artifact(1, cohort="cohort-b")])


def test_failure_excludes_only_under_threshold_model():
    artifacts = [_artifact(i) for i in range(5)]
    artifacts[2] = _artifact(2, failure=True)
    summary = summarize_runs(artifacts)
    assert [row["key"] for row in summary.ranking_mean] == ["claude"]
    assert summary.candidate_stats["chatgpt"]["n_valid"] == 4
    assert summary.candidate_stats["chatgpt"]["n_failed"] == 1
    assert summary.candidate_stats["claude"]["n_valid"] == 5


def test_equal_five_valid_runs_enable_exploratory_ranking():
    summary = summarize_runs([_artifact(i) for i in range(5)])
    assert summary.n == 5
    assert len(summary.ranking_mean) == 2
    assert all(row["exploratory"] for row in summary.ranking_mean)
    assert summary.candidate_stats["chatgpt"]["std"] is not None


def test_unequal_valid_n_keeps_all_eligible_models():
    artifacts = [_artifact(i) for i in range(6)]
    artifacts[1] = _artifact(1, failure=True)
    summary = summarize_runs(artifacts)
    assert {row["key"] for row in summary.ranking_mean} == {"chatgpt", "claude"}
    assert summary.candidate_stats["chatgpt"]["n_valid"] == 5
    assert summary.candidate_stats["claude"]["n_valid"] == 6
    assert summary.n == 5

