from lib.benchmark_multi_ui import (
    RELIABILITY_BAND_COLORS,
    LiveJudgingBoard,
    _ranking_table_html,
    cv_reliability_cells_html,
    finished_multi_progress,
    live_judging_board_html,
    na_failure_label,
    progressive_multi_panel_html,
    reliability_band_from_cv,
    snapshot_from_artifact,
)
from benchmark.report import (
    CV_HIGH_MAX,
    CV_LOW_MAX,
    CV_MEDIUM_MAX,
    CV_SUPER_HIGH_MAX,
)
from lib.charts import fig_judge_accuracy_bars
from benchmark.schema import (
    CandidateAnswer,
    JudgeResult,
    ModelCallMeta,
    QuestionScore,
    RunArtifact,
)


def test_progressive_panel_uses_tab_label_for_beta_rounds():
    html = progressive_multi_panel_html(
        [
            {
                "n_index": 8,
                "run_id": "beta-abc",
                "tab_label": "R8 · Case 1",
                "modal_title": "R8 · Case 1 · AKI · table + histogram",
                "total_cost_usd": 0.01,
                "ranking": [
                    {"key": "chatgpt", "accuracy": 80.0, "status": "ok", "rank": 1}
                ],
            }
        ],
        n_total=14,
        batch_done=False,
        footer_html='<div class="screenshot-footer">mean±std · test footer</div>',
    )
    assert "R8 · Case 1" in html
    assert "R8 · Case 1 · AKI · table + histogram" in html
    assert "Waiting for all runs" in html
    assert "Completed <b style=\"color:#fbbf24\">1</b> / 14" in html or "1</b> / 14" in html
    assert "screenshot-footer" in html
    assert "test footer" in html


def test_live_judging_board_paint_appends_footer():
    class _Slot:
        def __init__(self) -> None:
            self.last = ""

        def markdown(self, body, **_kwargs):
            self.last = body

    board = LiveJudgingBoard(
        title="Live judging · Comprehension",
        label_by_key={"chatgpt": "ChatGPT"},
        footer_html='<div class="screenshot-footer">live footer</div>',
    )
    slot = _Slot()
    board.bind(board_slot=slot)
    board.ensure_queued("chatgpt")
    assert "screenshot-footer" in slot.last
    assert "live footer" in slot.last
    assert "Live judging · Comprehension" in slot.last


def test_live_judging_session_updates_provisional_scores():
    board = LiveJudgingBoard(
        title="Live judging · Comprehension",
        label_by_key={"chatgpt": "ChatGPT", "claude": "Claude"},
    )
    board.ensure_queued("chatgpt")
    board.on_progress(
        {
            "phase": "progress",
            "key": "chatgpt",
            "done": 0,
            "total": 2,
            "percent": 40,
            "stage": "scoring sections",
            "elapsed_s": 5,
        }
    )
    assert board.board["chatgpt"]["progress_pct"] == 40
    board.on_progress(
        {
            "phase": "done",
            "key": "chatgpt",
            "done": 1,
            "total": 2,
            "accuracy": 77.0,
            "coverage": 70,
            "quality": 80,
            "discipline": 90,
            "failed": False,
            "elapsed_s": 11,
        }
    )
    assert board.board["chatgpt"]["status"] == "scored"
    assert board.highlight == "chatgpt"
    assert na_failure_label("collect_failed", "candidate error") == "N/A · collect error"


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


def test_failed_board_row_stays_terminal_without_accuracy():
    """App paints failed N/A with accuracy=None — must not look like stuck 75%."""
    from lib.benchmark_multi_ui import accuracy_histogram_html

    rendered = live_judging_board_html(
        {
            "claude": {
                "label": "Claude",
                "status": "failed",
                "accuracy": None,
                "queue_i": 1,
                "progress_pct": 100,
                "progress_label": "complete",
                "elapsed_s": 42.0,
            },
            "gemini": {
                "label": "Gemini",
                "status": "judging",
                "accuracy": None,
                "queue_i": 2,
                "progress_pct": 75,
                "progress_label": "corrective retry",
                "elapsed_s": 12.0,
            },
        }
    )
    assert "N/A · technical" in rendered
    assert "100% complete · 42s" in rendered
    # Failed row must not keep the in-flight progress chrome.
    assert rendered.count("rank-live-progress-track") == 1
    assert "corrective retry" in rendered

    hist = accuracy_histogram_html(
        [
            {
                "key": "claude",
                "status": "failed",
                "accuracy": None,
                "label": "Claude",
            },
            {
                "key": "chatgpt",
                "status": "scored",
                "accuracy": 88.0,
                "label": "ChatGPT",
            },
        ],
        include_pending=True,
    )
    assert "N/A" in hist
    assert "88.0%" in hist
    assert 'hist-num">0.0%' not in hist
    assert 'hist-num">N/A<' in hist


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
    assert "partial" in html.lower()
    assert "Partial run" in html


def test_finished_multi_progress_marks_batch_done_after_early_abort():
    """Full Multi used to set batch_done only when len(artifacts)>1."""
    snap = {"n_index": 1, "run_id": "r1", "ranking": [], "total_cost_usd": 0.1}
    state = finished_multi_progress(
        [snap], n_total=5, paths=["/tmp/r1.json"], aborted_early=True
    )
    assert state["batch_done"] is True
    assert state["aborted_early"] is True
    assert state["completed_runs"] == 1
    assert state["requested_runs"] == 5
    html = progressive_multi_panel_html(
        state["completed"], n_total=state["n_total"], batch_done=state["batch_done"]
    )
    assert "Waiting for all runs" not in html
    assert "still running below" not in html


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


def test_cv_reliability_bands_match_report_thresholds():
    assert reliability_band_from_cv(CV_SUPER_HIGH_MAX) == "super_high"
    assert reliability_band_from_cv(CV_SUPER_HIGH_MAX + 0.1) == "high"
    assert reliability_band_from_cv(CV_HIGH_MAX) == "high"
    assert reliability_band_from_cv(CV_HIGH_MAX + 0.1) == "medium"
    assert reliability_band_from_cv(CV_MEDIUM_MAX) == "medium"
    assert reliability_band_from_cv(CV_MEDIUM_MAX + 0.1) == "low"
    assert reliability_band_from_cv(CV_LOW_MAX) == "low"
    assert reliability_band_from_cv(CV_LOW_MAX + 0.1) == "very_low"
    assert reliability_band_from_cv(None) == ""


def test_cv_reliability_cells_use_legend_colors():
    samples = [4.0, 8.0, 12.0, 18.0, 25.0]
    expected = ["super_high", "high", "medium", "low", "very_low"]
    for cv, band in zip(samples, expected):
        cv_td, badge, got = cv_reliability_cells_html(cv)
        assert got == band
        bg, fg, _ = RELIABILITY_BAND_COLORS[band]
        assert bg in cv_td and fg in cv_td and f"{cv:.1f}%" in cv_td
        assert bg in badge and fg in badge


def test_ops_reliability_chart_stacks_percentages():
    from lib.charts import fig_rebuild_ops_reliability_bars

    fig = fig_rebuild_ops_reliability_bars(
        [
            {
                "key": "chatgpt",
                "n_scored": 3,
                "n_zero": 1,
                "n_technical_na": 2,
                "n_seen": 6,
                "pct_scored": 50.0,
                "pct_zero": 16.7,
                "pct_technical_na": 33.3,
            }
        ]
    )
    assert len(fig.data) == 3
    assert fig.layout.barmode == "stack"
    assert "not clinical mean" in (fig.layout.title.text or "").lower()


def test_mean_chart_whiskers_are_outlined_above_bars():
    """±1 std whiskers: white outline + black core, drawn above bars."""
    from lib.charts import fig_judge_mean_accuracy_bars

    fig = fig_judge_mean_accuracy_bars(
        [
            {
                "key": "chatgpt",
                "eligible": True,
                "rank": 1,
                "accuracy_mean": 80.0,
                "std": 12.0,
                "cv_pct": 15.0,
                "n_runs": 5,
                "median": 81.0,
            }
        ]
    )
    assert fig.data[0].type == "bar"
    # Whiskers must not live only under the bar (tips past bar top vanish).
    outline = fig.data[1]
    core = fig.data[2]
    assert outline.type == "scatter" and core.type == "scatter"
    assert "255,255,255" in str(outline.error_x.color)
    assert float(outline.error_x.thickness) > float(core.error_x.thickness)
    assert "15,23,42" in str(core.error_x.color)
    assert list(outline.error_x.array) == [12.0]


def test_reliability_table_html_rebuild_mean_has_bars_and_cv_bands():
    """Shared graded/Beta Rebuild mean table: score bars + CV tint + n scored."""
    from lib.benchmark_multi_ui import reliability_table_html

    html = reliability_table_html(
        [
            {
                "key": "chatgpt",
                "rank": 1,
                "accuracy_mean": 82.5,
                "coverage_mean": 80,
                "quality_mean": 85,
                "discipline_mean": 78,
                "std": 3.2,
                "cv_pct": 3.9,
                "median": 83.0,
                "min": 78.0,
                "max": 86.0,
                "n_runs": 5,
                "eligible": True,
            },
            {
                "key": "claude",
                "rank": 2,
                "accuracy_mean": 70.0,
                "coverage_mean": 72,
                "quality_mean": 68,
                "discipline_mean": 71,
                "std": 12.0,
                "cv_pct": 17.1,
                "median": 69.0,
                "min": 55.0,
                "max": 88.0,
                "n_runs": 5,
                "eligible": True,
            },
        ],
        successful_only=True,
    )
    assert "Clin. Composite" in html
    assert "C/Q/D" in html
    assert "Reliability" in html
    assert "n scored" in html
    # Rebuild scored-only: no Failed <th> column (footer may mention Failed%)
    assert ">Failed</th>" not in html
    assert "linear-gradient" in html  # mean + median conditional bars
    assert "82.5%" in html
    assert "83.0%" in html  # median bar label
    assert "80/85/78" in html
    assert "3.9%" in html
    assert "Stable mean" in html
    assert "Super High" not in html
    assert "≠ clinical validation" in html or "clinical validation" in html
    assert "17.1%" in html
    # CV cell tint uses band background
    assert "background:#064e3b" in html or "background:#9a3412" in html


def test_reliability_table_html_live_mean_keeps_failed_and_partial():
    from lib.benchmark_multi_ui import reliability_table_html

    html = reliability_table_html(
        [
            {
                "key": "chatgpt",
                "rank": 1,
                "accuracy_mean": 90.0,
                "std": 1.0,
                "cv_pct": 1.1,
                "median": 90.0,
                "min": 89.0,
                "max": 91.0,
                "n_runs": 2,
                "n_requested": 3,
                "n_failed": 1,
                "failure_rate": 1 / 3,
                "partial": True,
                "eligible": True,
            }
        ],
        successful_only=False,
    )
    assert ">Failed</th>" in html
    assert ">Runs</th>" in html
    assert "partial" in html.lower()
    assert "1 (33%)" in html


def test_ops_reliability_table_html_shows_relative_percentages():
    from lib.benchmark_multi_ui import ops_reliability_table_html

    html = ops_reliability_table_html(
        [
            {
                "key": "chatgpt",
                "n_scored": 3,
                "n_zero": 1,
                "n_technical_na": 2,
                "n_excluded": 3,
                "n_seen": 6,
                "pct_scored": 50.0,
                "pct_zero": 16.7,
                "pct_technical_na": 33.3,
                "pct_excluded": 50.0,
            }
        ]
    )
    assert "Technical N/A" in html
    assert "Failed / excluded" in html
    assert "2 (33%)" in html
    assert "3 (50%)" in html
    assert "1 (17%)" in html or "1 (16%)" in html
    assert "Chart below" not in html
    assert "Table columns" in html


def test_paint_rebuild_ops_reliability_panels_order_and_skip_empty():
    """Shared helper paints Failures/N/A table only (no ops chart KPI)."""
    from lib.benchmark_multi_ui import (
        ops_reliability_has_scan_data,
        paint_rebuild_ops_reliability_panels,
    )

    class _St:
        def __init__(self):
            self.calls = []

        def markdown(self, *a, **k):
            self.calls.append(("markdown", a, k))

        def caption(self, *a, **k):
            self.calls.append(("caption", a, k))

        def plotly_chart(self, *a, **k):
            self.calls.append(("plotly_chart", a, k))

    empty = _St()
    assert not ops_reliability_has_scan_data([])
    assert paint_rebuild_ops_reliability_panels(empty, []) is False
    assert empty.calls == []

    rows = [
        {
            "key": "chatgpt",
            "n_scored": 3,
            "n_zero": 1,
            "n_technical_na": 2,
            "n_excluded": 3,
            "n_seen": 6,
            "pct_scored": 50.0,
            "pct_zero": 16.7,
            "pct_technical_na": 33.3,
            "pct_excluded": 50.0,
        }
    ]
    st_mod = _St()
    assert ops_reliability_has_scan_data(rows)
    assert (
        paint_rebuild_ops_reliability_panels(
            st_mod,
            rows,
            n_per_model_cap=10,
            chart_key="beta_rebuild_ops_table",
            table_footer_html="<div>table-foot</div>",
        )
        is True
    )
    kinds = [c[0] for c in st_mod.calls]
    assert "plotly_chart" not in kinds
    md_texts = [str(c[1][0]) for c in st_mod.calls if c[0] == "markdown"]
    assert any("Failures / N/A" in t for t in md_texts)
    assert not any("Ops reliability chart" in t for t in md_texts)
    assert not any("Chart below" in t for t in md_texts)
    assert any("table-foot" in t for t in md_texts)
