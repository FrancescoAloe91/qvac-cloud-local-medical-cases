"""Unified Streamlit Cloud detection (deployment ↔ runtime_env)."""

from __future__ import annotations

import lib.deployment as deployment
import lib.runtime_env as runtime_env


def _clear_cloud_env(monkeypatch) -> None:
    for key in (
        "STREAMLIT_RUNTIME_ENVIRONMENT",
        "STREAMLIT_CLOUD",
        "STREAMLIT_SHARING_MODE",
        "HOSTNAME",
        "STREAMLIT_SERVER_URL",
        "STREAMLIT_SERVER_ADDRESS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(runtime_env.Path, "is_dir", lambda self: False)


def test_deployment_reexports_runtime_env_implementation():
    assert deployment.is_streamlit_cloud is runtime_env.is_streamlit_cloud


def test_cloud_true_for_sharing_mode_and_streamlit_cloud(monkeypatch):
    _clear_cloud_env(monkeypatch)
    assert runtime_env.is_streamlit_cloud() is False

    monkeypatch.setenv("STREAMLIT_SHARING_MODE", "1")
    assert runtime_env.is_streamlit_cloud() is True
    assert deployment.is_streamlit_cloud() is True

    monkeypatch.delenv("STREAMLIT_SHARING_MODE")
    monkeypatch.setenv("STREAMLIT_CLOUD", "true")
    assert runtime_env.is_streamlit_cloud() is True

    monkeypatch.delenv("STREAMLIT_CLOUD")
    monkeypatch.setenv("STREAMLIT_RUNTIME_ENVIRONMENT", "cloud")
    assert runtime_env.is_streamlit_cloud() is True


def test_cloud_true_for_streamlit_app_host(monkeypatch):
    _clear_cloud_env(monkeypatch)
    monkeypatch.setenv("HOSTNAME", "qvac-cloud-local-medical-cases.streamlit.app")
    assert runtime_env.is_streamlit_cloud() is True
    assert deployment.is_streamlit_cloud() is True


def test_cloud_true_for_mount_src(monkeypatch):
    _clear_cloud_env(monkeypatch)

    def _is_dir(self):
        return str(self) == "/mount/src"

    monkeypatch.setattr(runtime_env.Path, "is_dir", _is_dir)
    assert runtime_env.is_streamlit_cloud() is True
    assert deployment.is_streamlit_cloud() is True
