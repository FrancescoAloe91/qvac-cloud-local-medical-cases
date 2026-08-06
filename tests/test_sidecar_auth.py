"""Sidecar auth headers + bind guards (source + client helpers)."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sidecar_server_requires_auth_and_loopback_guard():
    src = (ROOT / "sidecar" / "qvac_server.mjs").read_text(encoding="utf-8")
    assert "requireAuth" in src
    assert "Authorization" in src
    assert "QVAC_SIDECAR_ALLOW_REMOTE" in src
    assert "isLoopbackHost" in src
    assert "auth_required" in src
    # Mutating endpoints must call requireAuth before body work.
    load_idx = src.index('urlPath === "/load"')
    gen_idx = src.index('urlPath === "/generate"')
    assert "requireAuth(req, res)" in src[load_idx : load_idx + 200]
    assert "requireAuth(req, res)" in src[gen_idx : gen_idx + 200]


def test_qvac_bridge_sends_bearer_token(tmp_path, monkeypatch):
    from benchmark import qvac_bridge as qb

    token_file = tmp_path / "tok"
    token_file.write_text("secret-token-abc\n", encoding="utf-8")
    monkeypatch.setattr(qb, "_SIDECAR_TOKEN_FILE", token_file)
    monkeypatch.delenv("QVAC_SIDECAR_TOKEN", raising=False)
    assert qb.sidecar_token() == "secret-token-abc"
    headers = qb._sidecar_headers(json_body=True)
    assert headers["Authorization"] == "Bearer secret-token-abc"
    assert headers["Content-Type"] == "application/json"


def test_ensure_sidecar_token_creates_file(tmp_path, monkeypatch):
    from benchmark import qvac_bridge as qb

    token_file = tmp_path / "new.token"
    monkeypatch.setattr(qb, "_SIDECAR_TOKEN_FILE", token_file)
    monkeypatch.delenv("QVAC_SIDECAR_TOKEN", raising=False)
    tok = qb.ensure_sidecar_token()
    assert len(tok) >= 32
    assert token_file.read_text(encoding="utf-8").strip() == tok
    assert qb.sidecar_token() == tok
