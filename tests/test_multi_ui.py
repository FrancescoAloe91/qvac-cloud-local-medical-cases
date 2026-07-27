from lib.benchmark_multi_ui import live_judging_board_html
from lib.charts import fig_judge_accuracy_bars


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
