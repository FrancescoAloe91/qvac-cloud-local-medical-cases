"""beta_rejudge CLI defaults to dry-run; --write requires --force-selective."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_rejudge_script():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "rejudge_beta_medpsy_na.py"
    spec = importlib.util.spec_from_file_location("rejudge_beta_medpsy_na", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rejudge_owner_defaults_to_dry_run(tmp_path, monkeypatch):
    from benchmark.beta_rejudge import rejudge_owner_beta_medpsy_na

    writes = []

    def fake_write(artifact, out_dir):
        writes.append((artifact.run_id, Path(out_dir)))
        return Path(out_dir) / f"{artifact.run_id}.json"

    monkeypatch.setattr("benchmark.beta_rejudge.write_artifact", fake_write)
    monkeypatch.setattr(
        "benchmark.beta_rejudge.iter_beta_medpsy_na_artifacts",
        lambda _owner: [],
    )
    summary = rejudge_owner_beta_medpsy_na(tmp_path)
    assert summary["dry_run"] is True
    assert summary["selective_medpsy_only"] is True
    assert writes == []


def test_cli_requires_owner(monkeypatch):
    mod = _load_rejudge_script()
    monkeypatch.setattr("sys.argv", ["rejudge_beta_medpsy_na.py"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code not in (0, None)


def test_cli_default_is_dry_run_requires_write(monkeypatch, tmp_path, capsys):
    mod = _load_rejudge_script()
    called = {}

    def fake_rejudge(owner, *, api_key="", dry_run=True, limit=None):
        called["dry_run"] = dry_run
        called["owner"] = owner
        return {
            "owner": str(owner),
            "n_artifacts": 0,
            "dry_run": dry_run,
            "has_api_key": False,
            "selective_medpsy_only": True,
            "reports": [],
            "recovered_total": 0,
            "still_na_total": 0,
            "attempted_total": 0,
        }

    monkeypatch.setattr(mod, "rejudge_owner_beta_medpsy_na", fake_rejudge)
    monkeypatch.setattr(mod, "resolve_openrouter_key", lambda: "")
    monkeypatch.setattr(
        "sys.argv",
        ["rejudge_beta_medpsy_na.py", "--owner", str(tmp_path)],
    )
    assert mod.main() == 0
    assert called["dry_run"] is True


def test_cli_write_without_force_selective_fails(monkeypatch, tmp_path):
    mod = _load_rejudge_script()
    monkeypatch.setattr(
        mod, "resolve_openrouter_key", lambda: "sk-or-v1-" + ("a" * 40)
    )
    monkeypatch.setattr(
        "sys.argv",
        ["rejudge_beta_medpsy_na.py", "--write", "--owner", str(tmp_path)],
    )
    assert mod.main() == 2


def test_cli_write_flag_disables_dry_run(monkeypatch, tmp_path):
    mod = _load_rejudge_script()
    called = {}

    def fake_rejudge(owner, *, api_key="", dry_run=True, limit=None):
        called["dry_run"] = dry_run
        return {
            "owner": str(owner),
            "n_artifacts": 0,
            "dry_run": dry_run,
            "has_api_key": True,
            "selective_medpsy_only": True,
            "reports": [],
            "recovered_total": 0,
            "still_na_total": 0,
            "attempted_total": 0,
        }

    monkeypatch.setattr(mod, "rejudge_owner_beta_medpsy_na", fake_rejudge)
    monkeypatch.setattr(
        mod, "resolve_openrouter_key", lambda: "sk-or-v1-" + ("a" * 40)
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "rejudge_beta_medpsy_na.py",
            "--write",
            "--force-selective",
            "--owner",
            str(tmp_path),
        ],
    )
    assert mod.main() == 0
    assert called["dry_run"] is False


def test_cli_no_hardcoded_owner_fingerprint():
    root = Path(__file__).resolve().parents[1]
    src = (root / "scripts" / "rejudge_beta_medpsy_na.py").read_text(encoding="utf-8")
    assert "893e6a29cf690fbef4d6aee2" not in src
    assert 'required=True' in src or 'required = True' in src
    assert "--force-selective" in src
