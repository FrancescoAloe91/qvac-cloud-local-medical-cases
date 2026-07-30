"""Portfolio rebuild: last-N across cases; exclude incompatible scoring versions."""

from __future__ import annotations

from pathlib import Path

from lib.model_labels import CURRENT_ROSTER_KEYS
from benchmark.report import (
    list_portfolio_runs,
    rebuild_multi_from_history,
    rebuild_portfolio_from_history,
    summarize_runs,
    write_artifact,
)
from benchmark.schema import RunArtifact


ROSTER = list(CURRENT_ROSTER_KEYS)


def _ranking(acc: float = 80.0):
    return [
        {
            "key": key,
            "accuracy": acc + i,
            "status": "ok",
            "rank": i + 1,
            "coverage": 70.0,
            "quality": 80.0,
            "discipline": 90.0,
        }
        for i, key in enumerate(ROSTER)
    ]


def _write(
    out_dir: Path,
    *,
    run_id: str,
    case_id: str,
    finished_at: str,
    scoring_version: str = "graded-clinical-v4",
    track: str = "controlled",
    roster: list | None = None,
    run_status: str = "complete",
    cohort_id: str | None = None,
    acc: float = 80.0,
) -> None:
    keys = roster if roster is not None else ROSTER
    art = RunArtifact(
        run_id=run_id,
        case_id=case_id,
        started_at=finished_at,
        finished_at=finished_at,
        n_index=1,
        batch_id=f"batch-{case_id}",
        models_config={
            "candidates": [{"key": k, "model": f"test/{k}"} for k in keys],
            "judge": {"model": "deepseek/deepseek-r1"},
            "gold_reference": "{}",
            "case_stem": f"stem-{case_id}",
        },
        ranking=_ranking(acc) if keys == ROSTER else [
            {
                "key": k,
                "accuracy": acc,
                "status": "ok",
                "rank": 1,
                "coverage": 70.0,
                "quality": 80.0,
                "discipline": 90.0,
            }
            for k in keys
        ],
        judgments=[],
        cohort_id=cohort_id or f"cohort-{case_id}-{scoring_version}-{track}",
        scoring_version=scoring_version,
        prompt_version="gold-only-v1",
        benchmark_track=track,  # type: ignore[arg-type]
        run_status=run_status,  # type: ignore[arg-type]
    )
    write_artifact(art, out_dir)


def test_list_portfolio_pulls_across_two_case_stems(tmp_path: Path):
    for i in range(3):
        _write(
            tmp_path,
            run_id=f"a{i}",
            case_id="caseA",
            finished_at=f"2026-01-01T1{i}:00:00Z",
            cohort_id="cohort-a",
        )
    for i in range(3):
        _write(
            tmp_path,
            run_id=f"b{i}",
            case_id="caseB",
            finished_at=f"2026-01-02T1{i}:00:00Z",
            cohort_id="cohort-b",
        )
    pairs = list_portfolio_runs(
        tmp_path, n=5, scoring_version="graded-clinical-v4", track="controlled"
    )
    assert len(pairs) == 5
    case_ids = {a.case_id for _, a in pairs}
    assert case_ids == {"caseA", "caseB"}
    # Newest first chronologically
    assert pairs[0][1].run_id == "b2"


def test_list_portfolio_excludes_different_scoring_version(tmp_path: Path):
    for i in range(5):
        _write(
            tmp_path,
            run_id=f"v4-{i}",
            case_id="caseA",
            finished_at=f"2026-01-01T1{i}:00:00Z",
            scoring_version="graded-clinical-v4",
        )
    for i in range(3):
        _write(
            tmp_path,
            run_id=f"v3-{i}",
            case_id="caseB",
            finished_at=f"2026-01-03T1{i}:00:00Z",
            scoring_version="graded-clinical-v3",
            cohort_id=f"cohort-v3-{i}",
        )
    pairs = list_portfolio_runs(
        tmp_path, n=10, scoring_version="graded-clinical-v4", track="controlled"
    )
    assert len(pairs) == 5
    assert all(a.scoring_version == "graded-clinical-v4" for _, a in pairs)
    assert all(a.case_id == "caseA" for _, a in pairs)


def test_rebuild_portfolio_ok_across_cases(tmp_path: Path):
    for i in range(3):
        _write(
            tmp_path,
            run_id=f"pa{i}",
            case_id="caseA",
            finished_at=f"2026-01-01T1{i}:00:00Z",
            cohort_id="cohort-pa",
            acc=70.0 + i,
        )
    for i in range(3):
        _write(
            tmp_path,
            run_id=f"pb{i}",
            case_id="caseB",
            finished_at=f"2026-01-02T1{i}:00:00Z",
            cohort_id="cohort-pb",
            acc=75.0 + i,
        )
    built = rebuild_portfolio_from_history(
        tmp_path, n=5, scoring_version="graded-clinical-v4", track="controlled"
    )
    assert built["ok"] is True
    assert built["scope"] == "portfolio"
    assert built["n_used"] == 5
    assert built["n_cases"] == 2
    assert built["official"] is False
    assert built["api_cost_usd"] == 0.0
    summary = built["summary"]
    assert summary.case_id == "portfolio"
    assert summary.ranking_mean
    assert built["mean_rank"]


def test_same_case_rebuild_path_unchanged(tmp_path: Path):
    for i in range(5):
        _write(
            tmp_path,
            run_id=f"sc{i}",
            case_id="caseC",
            finished_at=f"2026-01-01T1{i}:00:00Z",
            cohort_id="cohort-same",
        )
    # Extra different-case run must not enter same-case rebuild.
    _write(
        tmp_path,
        run_id="other",
        case_id="caseA",
        finished_at="2026-01-05T10:00:00Z",
        cohort_id="cohort-other",
    )
    built = rebuild_multi_from_history(
        tmp_path, "caseC", n=5, cohort_id="cohort-same"
    )
    assert built["ok"] is True
    assert built["scope"] == "same_case"
    assert built["n_used"] == 5
    assert built["cohort_id"] == "cohort-same"
    assert all(pr.get("case_id") == "caseC" for pr in built["per_run"])


def test_same_case_rebuild_excludes_cancelled_even_with_cohort_id(tmp_path: Path):
    """Abort stamps may share cohort_id; only complete runs enter official means."""
    for i in range(5):
        _write(
            tmp_path,
            run_id=f"ok{i}",
            case_id="caseC",
            finished_at=f"2026-01-01T1{i}:00:00Z",
            cohort_id="cohort-same",
            run_status="complete",
        )
    _write(
        tmp_path,
        run_id="abort-stale",
        case_id="caseC",
        finished_at="2026-01-01T19:00:00Z",
        cohort_id="cohort-same",
        run_status="cancelled",
    )
    built = rebuild_multi_from_history(
        tmp_path, "caseC", n=5, cohort_id="cohort-same"
    )
    assert built["ok"] is True
    assert built["n_used"] == 5
    assert all(pr.get("run_id") != "abort-stale" for pr in built["per_run"])
    # Four complete + one cancelled with same cohort → mean from the 4 completes.
    for i in range(4):
        _write(
            tmp_path,
            run_id=f"only4-{i}",
            case_id="caseD",
            finished_at=f"2026-02-01T1{i}:00:00Z",
            cohort_id="cohort-d",
            run_status="complete",
        )
    _write(
        tmp_path,
        run_id="only4-cancel",
        case_id="caseD",
        finished_at="2026-02-01T19:00:00Z",
        cohort_id="cohort-d",
        run_status="cancelled",
    )
    thin = rebuild_multi_from_history(
        tmp_path, "caseD", n=5, cohort_id="cohort-d"
    )
    assert thin["ok"] is True
    assert thin["available"] == 4
    assert thin["n_used"] == 4


def test_summarize_mixed_cohorts_opt_in(tmp_path: Path):
    arts = []
    for i in range(5):
        _write(
            tmp_path,
            run_id=f"m{i}",
            case_id="caseA" if i < 3 else "caseB",
            finished_at=f"2026-01-01T1{i}:00:00Z",
            cohort_id=f"c{i}",
        )
        from benchmark.report import load_artifact, list_run_artifacts

        arts = [load_artifact(p) for p in list_run_artifacts(tmp_path)]
    import pytest

    with pytest.raises(ValueError, match="mixed cohorts"):
        summarize_runs(arts)
    summary = summarize_runs(arts, allow_mixed_cohorts=True)
    assert summary.case_id == "portfolio"
    assert summary.n >= 5


def test_portfolio_pools_heterogeneous_roster_shapes(tmp_path: Path):
    """6-slot medical-style runs + full-roster runs both enter Portfolio."""
    medical_six = [
        "qvac_1_7b",
        "qvac",
        "qvac_4b_q8",
        "local_medgemma",
        "local_biomistral",
        "local_openbiollm",
    ]
    # Prefer keys that exist in CURRENT_ROSTER_KEYS; fall back to subset of ROSTER.
    medical_six = [k for k in medical_six if k in ROSTER] or ROSTER[:6]
    for i in range(3):
        _write(
            tmp_path,
            run_id=f"med{i}",
            case_id="caseMed",
            finished_at=f"2026-03-01T1{i}:00:00Z",
            roster=medical_six,
            cohort_id=f"cohort-med-{i}",
            acc=72.0 + i,
        )
    for i in range(3):
        _write(
            tmp_path,
            run_id=f"full{i}",
            case_id="caseFull",
            finished_at=f"2026-03-02T1{i}:00:00Z",
            roster=ROSTER,
            cohort_id=f"cohort-full-{i}",
            acc=78.0 + i,
        )
    pairs = list_portfolio_runs(
        tmp_path, n=10, scoring_version="graded-clinical-v4", track="controlled"
    )
    assert len(pairs) == 6
    ids = {a.run_id for _, a in pairs}
    assert any(r.startswith("med") for r in ids)
    assert any(r.startswith("full") for r in ids)

    built = rebuild_portfolio_from_history(
        tmp_path, n=5, scoring_version="graded-clinical-v4", track="controlled"
    )
    assert built["ok"] is True
    assert built["n_used"] == 5
    summary = built["summary"]
    # Shared keys appear in ranking; cloud-only keys need ≥5 obs from full runs only.
    shared = {row["key"] for row in summary.ranking_mean}
    assert medical_six[0] in shared or any(
        summary.candidate_stats.get(k, {}).get("n_valid", 0) >= 1 for k in medical_six
    )
    # At least one medical key has observations from both shapes (≥3 from med + some full).
    med_key = medical_six[1]
    n_valid = int(summary.candidate_stats.get(med_key, {}).get("n_valid") or 0)
    assert n_valid >= 3
