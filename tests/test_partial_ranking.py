"""Partial mean ranking: keep models with technical N/A ranked by scored mean."""

from __future__ import annotations

from benchmark.report import print_summary_table, summarize_runs
from benchmark.schema import RunArtifact
from lib.benchmark_multi_ui import _ranking_table_html
from lib.model_labels import rerank_rows


def _run(index: int, *, chatgpt_fail: bool = False) -> RunArtifact:
    ranking = [
        {
            "key": "chatgpt",
            "accuracy": None if chatgpt_fail else 70.0 + index,
            "status": "n/a" if chatgpt_fail else "ok",
            "status_note": "candidate_partial" if chatgpt_fail else "",
            "coverage": None if chatgpt_fail else 65.0,
            "quality": None if chatgpt_fail else 75.0,
            "discipline": None if chatgpt_fail else 90.0,
        },
        {
            "key": "qvac_1_7b",
            "accuracy": 60.0 + index,
            "status": "ok",
            "status_note": "",
            "coverage": 55.0,
            "quality": 70.0,
            "discipline": 88.0,
        },
        {
            "key": "claude",
            "accuracy": 85.0 + index,
            "status": "ok",
            "status_note": "",
            "coverage": 80.0,
            "quality": 90.0,
            "discipline": 95.0,
        },
    ]
    return RunArtifact(
        run_id=f"partial-{index}",
        case_id="caseC",
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:01:00Z",
        n_index=index + 1,
        batch_id="batch-partial-rank",
        ranking=ranking,
        cohort_id="cohort-partial-rank",
        run_status="partial" if chatgpt_fail else "complete",
    )


def test_partial_model_still_listed_and_ranked_by_mean():
    # 5 runs; chatgpt N/A on one → 4 scored observations, still ranked.
    artifacts = [_run(i, chatgpt_fail=(i == 1)) for i in range(5)]
    summary = summarize_runs(artifacts, min_valid_for_ranking=5)
    by_key = {row["key"]: row for row in summary.ranking_mean}

    assert by_key["chatgpt"]["eligible"] is True
    assert by_key["chatgpt"]["partial"] is True
    assert by_key["chatgpt"]["rank"] is not None
    assert by_key["chatgpt"]["n_runs"] == 4
    assert by_key["chatgpt"]["n_failed"] == 1
    assert by_key["chatgpt"]["accuracy_mean"] is not None

    assert by_key["qvac_1_7b"]["partial"] is False
    assert by_key["claude"]["partial"] is False
    # claude highest mean → rank 1; chatgpt / qvac follow by mean
    assert by_key["claude"]["rank"] == 1
    assert by_key["chatgpt"]["rank"] < by_key["qvac_1_7b"]["rank"]

    text = print_summary_table(summary)
    assert "·p" in text or "chatgpt" in text


def test_rerank_preserves_partial_scored_rows():
    rows = rerank_rows(
        [
            {
                "key": "qvac_1_7b",
                "accuracy": 61.0,
                "status": "partial",
                "partial": True,
                "eligible": True,
            },
            {
                "key": "claude",
                "accuracy": 90.0,
                "status": "ok",
                "partial": False,
                "eligible": True,
            },
            {
                "key": "gemini",
                "accuracy": None,
                "status": "n/a",
                "eligible": False,
            },
        ]
    )
    by_key = {r["key"]: r for r in rows}
    assert by_key["claude"]["rank"] == 1
    assert by_key["qvac_1_7b"]["rank"] == 2
    assert by_key["qvac_1_7b"]["partial"] is True
    assert by_key["gemini"]["rank"] is None


def test_single_run_ranking_table_shows_partial_banner_for_na():
    html = _ranking_table_html(
        [
            {"key": "claude", "rank": 1, "accuracy": 88.0, "status": "ok"},
            {
                "key": "qvac_1_7b",
                "rank": None,
                "accuracy": None,
                "status": "n/a",
                "status_note": "candidate_partial",
            },
        ]
    )
    assert "partial" in html.lower()
    assert "Partial run" in html
    assert "N/A" in html
    assert "#1" in html
