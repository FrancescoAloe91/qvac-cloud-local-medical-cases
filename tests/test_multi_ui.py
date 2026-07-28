from lib.benchmark_multi_ui import (
    _ranking_table_html,
    live_judging_board_html,
    snapshot_from_artifact,
)
from lib.charts import fig_judge_accuracy_bars
from benchmark.schema import (
    CandidateAnswer,
    JudgeResult,
    ModelCallMeta,
    QuestionScore,
    RunArtifact,
)


def test_live_judging_board_shows_stage_progress_and_elapsed_time():
    rendered = live_judging_board_html(
        {
            "gemini": {
                "label": "Gemini",
                "status": "judging",
                "queue_i": 2,
                "progress_pct": 70,
                "progress_label": "validating response",
                "elapsed_s": 18.7,
            }
        }
    )

    assert "70%" in rendered
    assert "validating response" in rendered
    assert "18s elapsed" in rendered
    assert "completed pipeline stages, not ETA" in rendered


def test_scored_row_shows_completion_and_judge_score_separately():
    rendered = live_judging_board_html(
        {
            "claude": {
                "label": "Claude",
                "status": "scored",
                "queue_i": 1,
                "accuracy": 82.4,
                "coverage": 80.0,
                "quality": 90.0,
                "discipline": 95.0,
                "progress_pct": 100,
                "elapsed_s": 31.2,
            }
        }
    )

    assert "82.4" in rendered
    assert "100% complete · 31s" in rendered
    assert "C 80 · Q 90 · D 95" in rendered
    assert "Clinical Composite Score" in rendered


def test_accuracy_chart_does_not_render_na_as_zero_bar():
    figure = fig_judge_accuracy_bars(
        [
            {
                "key": "gemini",
                "status": "n/a",
                "accuracy": None,
                "status_note": "judge_timeout",
            }
        ]
    )

    assert not figure.data
    assert "technical failures are N/A" in figure.layout.annotations[0].text


def test_secondary_ranking_table_shows_na_not_zero_percent():
    html = _ranking_table_html(
        [
            {"key": "claude", "accuracy": 88.0, "status": "ok", "rank": 1},
            {
                "key": "gemini",
                "accuracy": None,
                "status": "n/a",
                "status_note": "judge_schema_invalid",
            },
        ]
    )
    assert "N/A" in html
    assert "technical" in html
    assert "0.0%" not in html


def test_snapshot_from_artifact_preserves_na_status():
    art = RunArtifact(
        run_id="run-na",
        case_id="caseC",
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:01:00Z",
        ranking=[
            {"key": "claude", "accuracy": 80.0, "status": "ok", "rank": 1},
            {"key": "gemini", "accuracy": None, "status": "n/a", "rank": None},
        ],
        judgments=[
            JudgeResult(
                blind_id="c1",
                candidate_key="gemini",
                question_scores=[QuestionScore(question_id="diagnosis", score=0.0)],
                weighted_accuracy=0.0,
                judge_model="judge",
                judge_meta=ModelCallMeta(model="judge", provider="openrouter"),
                status="judge_schema_invalid",
                failure_reason="evidence",
            )
        ],
        candidates=[
            CandidateAnswer(
                candidate_key="gemini",
                label="Gemini",
                blind_id="c1",
                answers={},
                meta=ModelCallMeta(model="gemini", provider="openrouter"),
            )
        ],
        cohort_id="cohort-a",
    )
    snap = snapshot_from_artifact(art)
    gem = next(r for r in snap["ranking"] if r["key"] == "gemini")
    assert gem["accuracy"] is None
    assert gem["status"] == "n/a"
    dim = next(d for d in snap["dimensions"] if d["key"] == "gemini")
    assert dim["weighted"] is None
