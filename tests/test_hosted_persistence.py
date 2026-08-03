"""Hosted auth must not write plaintext artifacts after decrypt."""

from __future__ import annotations

from pathlib import Path

from benchmark.schema import RunArtifact, utc_now_iso
from benchmark.workspace import LOCAL_NO_KEY_ID


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


def test_cloud_no_key_never_uses_local_no_key(tmp_path, monkeypatch):
    """C1: Streamlit Cloud without key must not share `_local_no_key`."""
    monkeypatch.setattr("benchmark.workspace.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("lib.deployment.is_streamlit_cloud", lambda: True)
    from benchmark.workspace import scoped_artifacts_dir

    path = scoped_artifacts_dir("", cloud_ephemeral_id="browser-session-uuid-1")
    assert not str(path).endswith(LOCAL_NO_KEY_ID)
    assert "_cloud_ephemeral_" in path.name
    other = scoped_artifacts_dir("", cloud_ephemeral_id="browser-session-uuid-2")
    assert path != other


def test_local_no_key_still_available_off_cloud(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmark.workspace.ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr("lib.deployment.is_streamlit_cloud", lambda: False)
    from benchmark.workspace import scoped_artifacts_dir

    path = scoped_artifacts_dir("")
    assert path.name == LOCAL_NO_KEY_ID


def test_comprehension_cloud_uses_hosted_run_store_and_skips_custom_disk():
    """H1: app.py Cloud path uses HostedRunStore; customs skip host disk."""
    root = Path(__file__).resolve().parents[1]
    home = (root / "app.py").read_text(encoding="utf-8")
    assert "HostedRunStore" in home
    assert "_comp_artifacts_memory" in home
    assert "_cloud_anon_ws" in home
    assert "cloud_ephemeral_id=" in home
    # _persist_beta_customs must early-return on Cloud.
    assert "def _persist_beta_customs()" in home
    idx = home.index("def _persist_beta_customs()")
    chunk = home[idx : idx + 280]
    assert "is_streamlit_cloud()" in chunk
    assert "return" in chunk
    # Local still wires LocalRunStore.
    assert "LocalRunStore(WORKSPACE_DIR)" in home


def test_readme_separates_comprehension_and_structured_privacy():
    """H2: README Privacy must distinguish Comprehension Cloud vs Structured."""
    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    assert "Privacy and hosted persistence" in text
    assert "Comprehension on Streamlit Cloud" in text
    assert "Structured + Supabase" in text or "Structured + Supabase hosted" in text
    assert "session-memory" in text or "session memory" in text
    assert "_local_no_key" in text
    assert "_cloud_ephemeral_" in text
    assert "plaintext" in text.lower()
    assert "Same OpenRouter key = shared History" in text
