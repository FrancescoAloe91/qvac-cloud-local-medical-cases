"""Fail-closed calibration constraints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.calibration import (
    check_section,
    compare_fixture,
    load_fixture,
    summarize_directory,
)


def test_missing_observed_metric_fails():
    fails = check_section(
        {"coverage": 0.5},
        {"coverage_min": 0.4, "coverage_max": 0.6, "quality_min": 0.2, "quality_max": 0.8},
    )
    assert any("missing observed metric: quality" in f for f in fails)


def test_full_domain_range_rejected():
    fails = check_section(
        {"coverage": 0.5, "quality": 0.5, "discipline": 0.5, "score": 50},
        {
            "coverage_min": 0.0,
            "coverage_max": 1.0,
            "quality_min": 0.0,
            "quality_max": 1.0,
            "discipline_min": 0.0,
            "discipline_max": 1.0,
            "score_min": 0,
            "score_max": 100,
        },
    )
    assert any("full-domain" in f for f in fails)


def test_malformed_fixture_raises(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        load_fixture(bad)


def test_unreviewed_never_calibrated():
    fixture = {
        "fixture_id": "x",
        "reviewed_by": "",
        "reviewed_at": "",
        "expected": {
            "diagnosis": {
                "coverage_min": 0.4,
                "coverage_max": 0.6,
                "quality_min": 0.4,
                "quality_max": 0.6,
                "discipline_min": 0.4,
                "discipline_max": 0.6,
                "score_min": 40,
                "score_max": 60,
            }
        },
    }
    obs = {
        "diagnosis": {
            "coverage": 0.5,
            "quality": 0.5,
            "discipline": 0.5,
            "score": 50,
        }
    }
    row = compare_fixture(fixture, obs)
    assert row["ok"] is True
    assert row["reviewed"] is False
    assert row["calibrated"] is False


def test_reviewed_nontrivial_can_calibrate():
    fixture = {
        "fixture_id": "y",
        "reviewed_by": "owner",
        "reviewed_at": "2026-01-01T00:00:00Z",
        "expected": {
            "diagnosis": {
                "coverage_min": 0.4,
                "coverage_max": 0.6,
                "quality_min": 0.4,
                "quality_max": 0.6,
                "discipline_min": 0.4,
                "discipline_max": 0.6,
                "score_min": 40,
                "score_max": 60,
            }
        },
    }
    obs = {
        "diagnosis": {
            "coverage": 0.5,
            "quality": 0.5,
            "discipline": 0.5,
            "score": 50,
        }
    }
    row = compare_fixture(fixture, obs)
    assert row["calibrated"] is True
    summary = summarize_directory({"y": obs}, directory=None)
    # directory None with no fixtures → empty; just assert row path
    assert row["ok"] and row["reviewed"]
