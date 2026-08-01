"""Local UI prefs (QVAC SDK ack) — no Streamlit."""

from __future__ import annotations

import lib.ui_prefs as ui_prefs


def test_qvac_sdk_ack_roundtrip(tmp_path, monkeypatch):
    prefs = tmp_path / ".ui_prefs.json"
    monkeypatch.setattr(ui_prefs, "PREFS_FILE", prefs)
    assert ui_prefs.load_qvac_sdk_ack() is False
    ui_prefs.save_qvac_sdk_ack(True)
    assert prefs.is_file()
    assert ui_prefs.load_qvac_sdk_ack() is True
    raw = prefs.read_text(encoding="utf-8")
    assert "qvac_sdk_ack" in raw
    assert "sk-or" not in raw


def test_qvac_sdk_ack_corrupt_file(tmp_path, monkeypatch):
    prefs = tmp_path / ".ui_prefs.json"
    prefs.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(ui_prefs, "PREFS_FILE", prefs)
    assert ui_prefs.load_qvac_sdk_ack() is False
