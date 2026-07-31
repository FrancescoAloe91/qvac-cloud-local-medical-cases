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
    """Abort stamps may share cohort_id; only complete/partial runs enter means."""
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


def test_partial_technical_na_counts_in_failed_pct_same_case_and_portfolio(
    tmp_path: Path,
):
    """partial runs (per-model N/A) enter means; Failed % > 0; model stays listed.

    Regression: run_status=partial used to be excluded like cancelled, so every
    mean table showed Failed 0% while History still showed N/A rows.
    """
    cohort = "cohort-partial-na"
    keys = ["chatgpt", "claude"]

    def _write_mixed(*, run_id: str, finished_at: str, fail_chatgpt: bool):
        ranking = []
        for i, key in enumerate(keys):
            if fail_chatgpt and key == "chatgpt":
                ranking.append(
                    {
                        "key": key,
                        "accuracy": None,
                        "status": "n/a",
                        "status_note": "candidate_partial",
                        "rank": None,
                        "coverage": None,
                        "quality": None,
                        "discipline": None,
                    }
                )
            else:
                ranking.append(
                    {
                        "key": key,
                        "accuracy": 80.0 + i,
                        "status": "ok",
                        "rank": i + 1,
                        "coverage": 70.0,
                        "quality": 80.0,
                        "discipline": 90.0,
                    }
                )
        art = RunArtifact(
            run_id=run_id,
            case_id="caseC",
            started_at=finished_at,
            finished_at=finished_at,
            n_index=1,
            batch_id="batch-partial",
            models_config={
                "candidates": [{"key": k, "model": f"test/{k}"} for k in keys],
                "judge": {"model": "deepseek/deepseek-r1"},
                "gold_reference": "{}",
                "case_stem": "stem-caseC",
            },
            ranking=ranking,
            judgments=[],
            cohort_id=cohort,
            scoring_version="graded-clinical-v4",
            prompt_version="gold-only-v1",
            benchmark_track="controlled",
            run_status="partial" if fail_chatgpt else "complete",
        )
        write_artifact(art, tmp_path)

    # 4 complete + 1 partial with chatgpt N/A
    for i in range(4):
        _write_mixed(
            run_id=f"ok{i}",
            finished_at=f"2026-04-01T1{i}:00:00Z",
            fail_chatgpt=False,
        )
    _write_mixed(
        run_id="na-partial",
        finished_at="2026-04-01T19:00:00Z",
        fail_chatgpt=True,
    )

    same = rebuild_multi_from_history(
        tmp_path, "caseC", n=5, cohort_id=cohort
    )
    assert same["ok"] is True
    assert same["n_used"] == 5
    by_key = {r["key"]: r for r in same["summary"].ranking_mean}
    assert "chatgpt" in by_key
    assert by_key["chatgpt"]["n_failed"] == 1
    assert by_key["chatgpt"]["n_requested"] == 5
    assert by_key["chatgpt"]["n_runs"] == 4
    assert by_key["chatgpt"]["failure_rate"] == 0.2
    assert by_key["chatgpt"]["partial"] is True
    assert by_key["chatgpt"]["eligible"] is True
    assert by_key["chatgpt"]["rank"] is not None
    assert by_key["chatgpt"]["accuracy_mean"] is not None
    assert by_key["claude"]["n_failed"] == 0
    assert by_key["claude"]["failure_rate"] == 0.0
    assert by_key["claude"]["partial"] is False

    port = rebuild_portfolio_from_history(
        tmp_path, n=5, scoring_version="graded-clinical-v4", track="controlled"
    )
    assert port["ok"] is True
    assert port["n_used"] == 5
    p_by = {r["key"]: r for r in port["summary"].ranking_mean}
    assert p_by["chatgpt"]["n_failed"] >= 1
    assert float(p_by["chatgpt"]["failure_rate"] or 0) > 0.0
    assert p_by["chatgpt"]["partial"] is True
    assert p_by["chatgpt"]["rank"] is not None
    # Cancelled still excluded from portfolio
    _write(
        tmp_path,
        run_id="cancel-noise",
        case_id="caseC",
        finished_at="2026-04-02T10:00:00Z",
        cohort_id=cohort,
        run_status="cancelled",
        roster=keys,
    )
    pairs = list_portfolio_runs(
        tmp_path, n=10, scoring_version="graded-clinical-v4", track="controlled"
    )
    assert all(a.run_id != "cancel-noise" for _, a in pairs)
    assert any(a.run_id == "na-partial" for _, a in pairs)

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
        "local_med42",
        "local_ultramedical",
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


def test_portfolio_per_model_n_includes_older_cloud_history(tmp_path: Path):
    """Recent medical-only runs must not hide cloud models with older history.

    Regression: Portfolio used to slice the last N *global* run documents, so
    medical-only recency dropped chatgpt/claude/gemini even when History still
    had full-roster runs. N is now a per-model observation cap.
    """
    medical_six = [
        k
        for k in (
            "qvac_1_7b",
            "qvac",
            "qvac_4b_q8",
            "local_medgemma",
            "local_med42",
            "local_ultramedical",
        )
        if k in ROSTER
    ] or ROSTER[:6]
    cloud = [k for k in ("chatgpt", "claude", "gemini") if k in ROSTER]
    assert cloud, "roster must include at least one cloud key"

    # Older: 10 full-roster runs with cloud.
    for i in range(10):
        _write(
            tmp_path,
            run_id=f"full-old-{i}",
            case_id="caseFull",
            finished_at=f"2026-01-01T{10 + i:02d}:00:00Z",
            roster=ROSTER,
            cohort_id=f"cohort-full-{i}",
            acc=60.0 + i,
        )
    # Newer: 5 medical-only runs (would fill a global last-5 / last-20 slice).
    for i in range(5):
        _write(
            tmp_path,
            run_id=f"med-new-{i}",
            case_id="caseMed",
            finished_at=f"2026-06-01T1{i}:00:00Z",
            roster=medical_six,
            cohort_id=f"cohort-med-{i}",
            acc=70.0 + i,
        )

    # Global last-5 slice would be medical-only — cloud absent.
    global5 = list_portfolio_runs(
        tmp_path, n=5, scoring_version="graded-clinical-v4", track="controlled"
    )
    assert len(global5) == 5
    assert all(a.run_id.startswith("med-new") for _, a in global5)

    built = rebuild_portfolio_from_history(
        tmp_path, n=20, scoring_version="graded-clinical-v4", track="controlled"
    )
    assert built["ok"] is True
    assert built["n_per_model_cap"] == 20
    # Contributing docs: 5 medical + 10 full (cloud needs the older ones).
    assert built["n_used"] == 15
    by_key = {r["key"]: r for r in built["summary"].ranking_mean}
    for ck in cloud:
        assert ck in by_key, f"{ck} missing from portfolio mean"
        assert int(by_key[ck]["n_requested"]) == 10
        assert int(by_key[ck]["n_runs"]) == 10
        assert float(by_key[ck]["failure_rate"] or 0) == 0.0
    # Medical keys see newest ≤20: 5 med + 10 full = 15 obs.
    med_key = medical_six[0]
    assert med_key in by_key
    assert int(by_key[med_key]["n_requested"]) == 15


def test_trim_n_skips_technical_na_and_uses_older_scored(tmp_path: Path):
    """N = scored quota: interleaved N/A must not shrink a model's scored mean.

    Newest runs fail for chatgpt; older History still has scored rows. Rebuild
    with N=5 must pull five scored chatgpt observations (and keep the N/A seen
    while scanning for Failed %), not stop after five raw ranking rows.
    """
    cohort = "cohort-na-interleave"
    keys = ["chatgpt", "claude"]

    def _write_row(*, run_id: str, finished_at: str, chatgpt_ok: bool, acc: float):
        ranking = []
        for i, key in enumerate(keys):
            if key == "chatgpt" and not chatgpt_ok:
                ranking.append(
                    {
                        "key": key,
                        "accuracy": None,
                        "status": "n/a",
                        "status_note": "candidate_partial",
                        "rank": None,
                        "coverage": None,
                        "quality": None,
                        "discipline": None,
                    }
                )
            else:
                ranking.append(
                    {
                        "key": key,
                        "accuracy": acc + i,
                        "status": "ok",
                        "rank": i + 1,
                        "coverage": 70.0,
                        "quality": 80.0,
                        "discipline": 90.0,
                    }
                )
        art = RunArtifact(
            run_id=run_id,
            case_id="caseC",
            started_at=finished_at,
            finished_at=finished_at,
            n_index=1,
            batch_id="batch-na-interleave",
            models_config={
                "candidates": [{"key": k, "model": f"test/{k}"} for k in keys],
                "judge": {"model": "deepseek/deepseek-r1"},
                "gold_reference": "{}",
                "case_stem": "stem-caseC",
            },
            ranking=ranking,
            judgments=[],
            cohort_id=cohort,
            scoring_version="graded-clinical-v4",
            prompt_version="gold-only-v1",
            benchmark_track="controlled",
            run_status="partial" if not chatgpt_ok else "complete",
        )
        write_artifact(art, tmp_path)

    # Older: 5 fully scored (chatgpt accuracy 60..64).
    for i in range(5):
        _write_row(
            run_id=f"old-ok-{i}",
            finished_at=f"2026-05-01T1{i}:00:00Z",
            chatgpt_ok=True,
            acc=60.0 + i,
        )
    # Newer: 3 technical N/A for chatgpt (would fill a raw last-5 window).
    for i in range(3):
        _write_row(
            run_id=f"new-na-{i}",
            finished_at=f"2026-06-01T1{i}:00:00Z",
            chatgpt_ok=False,
            acc=90.0 + i,
        )

    same = rebuild_multi_from_history(
        tmp_path, "caseC", n=5, cohort_id=cohort
    )
    assert same["ok"] is True
    by_key = {r["key"]: r for r in same["summary"].ranking_mean}
    assert by_key["chatgpt"]["n_runs"] == 5
    assert by_key["chatgpt"]["n_failed"] == 3
    assert by_key["chatgpt"]["n_requested"] == 8
    # Mean of the five older scored rows (60..64), not a 2-scored truncated window.
    assert by_key["chatgpt"]["accuracy_mean"] == 62.0
    assert by_key["chatgpt"]["partial"] is True
    assert by_key["chatgpt"]["rank"] is not None
    # Claude always scored; newest ≤5 scored, no N/A in its window once filled.
    assert by_key["claude"]["n_runs"] == 5
    assert by_key["claude"]["n_failed"] == 0

    port = rebuild_portfolio_from_history(
        tmp_path, n=5, scoring_version="graded-clinical-v4", track="controlled"
    )
    assert port["ok"] is True
    p_by = {r["key"]: r for r in port["summary"].ranking_mean}
    assert int(p_by["chatgpt"]["n_runs"]) == 5
    assert int(p_by["chatgpt"]["n_failed"]) == 3
    assert float(p_by["chatgpt"]["accuracy_mean"]) == 62.0


def test_rebuild_accepts_per_model_n_40_and_50(tmp_path: Path):
    """UI selectbox offers 40/50; backend must not clamp those caps to 30."""
    for i in range(3):
        _write(
            tmp_path,
            run_id=f"p{i}",
            case_id="caseA",
            finished_at=f"2026-01-01T1{i}:00:00Z",
            cohort_id=f"cohort-p-{i}",
        )
    for n in (40, 50):
        port = rebuild_portfolio_from_history(
            tmp_path, n=n, scoring_version="graded-clinical-v4", track="controlled"
        )
        assert port["ok"] is True
        assert port["n_per_model_cap"] == n
        same = rebuild_multi_from_history(
            tmp_path, "caseA", n=n, cohort_id="cohort-p-2"
        )
        assert same["ok"] is True
        assert same["n_per_model_cap"] == n
    # Still hard-capped above the UI max.
    over = rebuild_portfolio_from_history(
        tmp_path, n=99, scoring_version="graded-clinical-v4", track="controlled"
    )
    assert over["ok"] is True
    assert over["n_per_model_cap"] == 50
