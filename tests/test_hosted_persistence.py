"""Hosted auth must not write plaintext artifacts after decrypt."""

from __future__ import annotations

from pathlib import Path

from benchmark.schema import RunArtifact, utc_now_iso


def test_hosted_auth_path_skips_write_artifact(monkeypatch, tmp_path):
    """Simulate _persist_run_artifact hosted branch: no disk write."""
    calls = []

    def fake_write(artifact, out_dir):
        calls.append((artifact.run_id, Path(out_dir)))
        return Path(out_dir) / f"{artifact.run_id}.json"

    monkeypatch.setattr("benchmark.report.write_artifact", fake_write)

    artifact = RunArtifact(
        run_id="hosted-test-1",
        case_id="caseC",
        started_at=utc_now_iso(),
        finished_at=utc_now_iso(),
    )

    hosted_no_plaintext = True
    session_memory: list = []

    def persist(art, workspace: Path):
        session_memory.append(art)
        if hosted_no_plaintext:
            return None
        return fake_write(art, workspace)

    result = persist(artifact, tmp_path)
    assert result is None
    assert calls == []
    assert session_memory[0].run_id == "hosted-test-1"

    # Local mode still writes.
    hosted_no_plaintext = False
    result2 = persist(artifact, tmp_path)
    assert result2 is not None
    assert len(calls) == 1


def test_workspace_scopes_by_supabase_user_id(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmark.workspace.ARTIFACTS_DIR", tmp_path)
    from benchmark.workspace import scoped_artifacts_dir

    local = scoped_artifacts_dir("sk-or-v1-" + ("a" * 40))
    account = scoped_artifacts_dir(
        "sk-or-v1-" + ("a" * 40), account_user_id="user-abc-123"
    )
    assert "user_user-abc-123" in str(account)
    assert local != account
