from __future__ import annotations

import pytest

from benchmark.report import (
    planned_on_device_model_contract,
    summarize_multi_batch,
    summarize_runs,
)
from benchmark.schema import RunArtifact


def _artifact(index: int, *, cohort: str = "cohort-a", failure: bool = False):
    ranking = [
        {
            "key": "chatgpt",
            "accuracy": None if failure else 80.0 + index,
            "status": "n/a" if failure else "ok",
            "status_note": "timed_out" if failure else "",
            "coverage": None if failure else 75.0 + index,
            "quality": None if failure else 85.0 + index,
            "discipline": None if failure else 95.0,
        },
        {
            "key": "claude",
            "accuracy": 82.0 + index,
            "status": "ok",
            "status_note": "",
            "coverage": 80.0 + index,
            "quality": 88.0 + index,
            "discipline": 96.0,
        },
    ]
    return RunArtifact(
        run_id=f"run-{index}",
        case_id="caseC",
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:01:00Z",
        n_index=index + 1,
        batch_id="batch-a",
        ranking=ranking,
        cohort_id=cohort,
    )


def test_mixed_cohorts_are_rejected():
    with pytest.raises(ValueError, match="mixed cohorts"):
        summarize_runs([_artifact(0), _artifact(1, cohort="cohort-b")])


def test_mixed_execution_cohort_same_cohort_id_still_summarizes():
    """Best-effort routing / per-run N/A may diverge execution_cohort_id; mean pools on cohort_id."""
    artifacts = []
    for i in range(5):
        art = _artifact(i)
        art = art.model_copy(
            update={
                "execution_cohort_id": "exec-gemini-na" if i == 2 else "exec-ok",
            }
        )
        artifacts.append(art)
    summary = summarize_runs(artifacts)
    assert summary.n == 5
    assert {row["key"] for row in summary.ranking_mean} == {"chatgpt", "claude"}
    assert any("execution_cohort_id varied" in note for note in summary.outliers)
    assert any("Same-case Multi mean" in note for note in summary.outliers)
    assert any("Portfolio cross-case" in note for note in summary.outliers)


def test_failure_keeps_partial_model_ranked_by_mean():
    """Models with technical N/A stay listed and ranked by mean of scored runs."""
    artifacts = [_artifact(i) for i in range(5)]
    artifacts[2] = _artifact(2, failure=True)
    summary = summarize_runs(artifacts)
    by_key = {row["key"]: row for row in summary.ranking_mean}
    assert set(by_key) == {"chatgpt", "claude"}
    assert by_key["claude"]["eligible"] is True
    assert by_key["claude"]["partial"] is False
    assert by_key["claude"]["n_failed"] == 0
    assert by_key["claude"]["failure_rate"] == 0.0
    # chatgpt: 4/5 scored → still ranked by mean, marked partial
    assert by_key["chatgpt"]["eligible"] is True
    assert by_key["chatgpt"]["partial"] is True
    assert by_key["chatgpt"]["rank"] is not None
    assert by_key["chatgpt"]["n_runs"] == 4
    assert by_key["chatgpt"]["n_requested"] == 5
    assert by_key["chatgpt"]["n_failed"] == 1
    assert by_key["chatgpt"]["failure_rate"] == 0.2
    assert by_key["chatgpt"]["accuracy_mean"] is not None
    assert summary.candidate_stats["chatgpt"]["n_valid"] == 4
    assert summary.candidate_stats["chatgpt"]["n_failed"] == 1
    assert summary.candidate_stats["claude"]["n_valid"] == 5
    assert summary.candidate_stats["chatgpt"]["failure_rate"] == 0.2
    assert any("Partial (ranked by mean" in note for note in summary.outliers)
    # Mean order: both ranked; chatgpt mean of [80,81,83,84]=82 vs claude higher
    assert by_key["claude"]["rank"] == 1
    assert by_key["chatgpt"]["rank"] == 2


def test_ranking_mean_failed_column_for_eligible_partial_failures():
    """Eligible models (enough valid runs) still expose non-zero Failed %."""
    artifacts = [_artifact(i) for i in range(6)]
    artifacts[1] = _artifact(1, failure=True)
    summary = summarize_runs(artifacts, min_valid_for_ranking=5)
    chatgpt = next(row for row in summary.ranking_mean if row["key"] == "chatgpt")
    assert chatgpt["eligible"] is True
    assert chatgpt["partial"] is True
    assert chatgpt["rank"] is not None
    assert chatgpt["n_runs"] == 5
    assert chatgpt["n_requested"] == 6
    assert chatgpt["n_failed"] == 1
    assert chatgpt["failure_rate"] == pytest.approx(1 / 6, rel=1e-3)


def test_all_na_model_listed_but_unranked():
    """Zero scored observations → stay in table with Failed %, no rank."""
    artifacts = [_artifact(i, failure=True) for i in range(5)]
    for i, art in enumerate(artifacts):
        # claude always scores so ranking is non-empty
        art.ranking[1]["accuracy"] = 82.0 + i
        art.ranking[1]["status"] = "ok"
        art.ranking[1]["status_note"] = ""
    summary = summarize_runs(artifacts, min_valid_for_ranking=5)
    by_key = {row["key"]: row for row in summary.ranking_mean}
    assert by_key["chatgpt"]["eligible"] is False
    assert by_key["chatgpt"]["rank"] is None
    assert by_key["chatgpt"]["partial"] is False
    assert by_key["chatgpt"]["n_failed"] == 5
    assert by_key["chatgpt"]["accuracy_mean"] is None
    assert by_key["claude"]["rank"] == 1
    assert any("unranked (no scored observations)" in note for note in summary.outliers)


def test_equal_five_valid_runs_enable_exploratory_ranking():
    summary = summarize_runs([_artifact(i) for i in range(5)])
    assert summary.n == 5
    assert summary.paired_n == 5
    assert len(summary.paired_ranking) == 2
    assert summary.ranking_mean[0]["coverage_mean"] is not None


def test_paired_sensitivity_uses_only_complete_iterations():
    artifacts = [_artifact(i) for i in range(6)]
    artifacts[0] = _artifact(0, failure=True)

    summary = summarize_runs(artifacts)

    assert summary.paired_n == 5
    assert {row["key"] for row in summary.paired_ranking} == {"chatgpt", "claude"}
    assert all(row["n_runs"] == 5 for row in summary.paired_ranking)
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


def test_empty_cohort_id_is_rejected():
    with pytest.raises(ValueError, match="empty cohort_id"):
        summarize_runs([_artifact(0, cohort=""), _artifact(1, cohort="")])


def test_summarize_multi_batch_falls_back_to_majority_cohort():
    """Only-local Multi used to crash when sidecar GGUF labels split cohorts."""
    artifacts = [_artifact(i) for i in range(5)]
    artifacts[2] = _artifact(2, cohort="cohort-b")
    artifacts[4] = _artifact(4, cohort="cohort-c")

    with pytest.raises(ValueError, match="mixed cohorts"):
        summarize_runs(artifacts)

    summary, warning = summarize_multi_batch(artifacts)
    assert summary is not None
    assert summary.n >= 1
    assert len(summary.ranking_mean) == 2
    assert warning and "3/5" in warning
    assert any("matching cohort_id" in note for note in summary.outliers)


def test_summarize_multi_batch_none_when_no_usable_cohort():
    summary, warning = summarize_multi_batch(
        [_artifact(0, cohort=""), _artifact(1, cohort="")]
    )
    assert summary is None
    assert warning and "Mean ranking unavailable" in warning


def test_planned_on_device_model_contract_ignores_runtime_label_noise():
    slots = [
        {"key": "qvac_1_7b", "model": "medpsy-1.7b-q4", "provider": "qvac"},
        {"key": "local_medgemma", "model": "medgemma-1.5-4b-it-q4", "provider": "qvac"},
    ]
    planned = planned_on_device_model_contract(slots)
    assert [r["key"] for r in planned] == ["local_medgemma", "qvac_1_7b"]
    # Same planned slots → identical contract even if a caller would have used
    # oscillating sidecar names (Q4_K_M vs q4 / imat suffix).
    again = planned_on_device_model_contract(slots)
    assert planned == again


def test_mixed_batch_ids_disable_paired_sensitivity():
    artifacts = [_artifact(i) for i in range(5)]
    artifacts[2] = artifacts[2].model_copy(update={"batch_id": "batch-b"})
    summary = summarize_runs(artifacts)
    assert summary.paired_n == 0
    assert summary.paired_ranking == []
    assert len(summary.ranking_mean) == 2


def test_paired_sensitivity_requires_same_batch_id_equality():
    artifacts = [_artifact(i) for i in range(5)]
    for art in artifacts:
        assert art.batch_id == "batch-a"
    summary = summarize_runs(artifacts)
    assert summary.paired_n == 5

