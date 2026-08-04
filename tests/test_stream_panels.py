"""Stream panel shell/body — fullscreen overlay without Streamlit rerun."""

from __future__ import annotations

import html

from lib.stream_panels import stream_body_html, stream_shell_html, stream_uid


def test_stream_shell_has_client_fullscreen_chrome():
    shell = stream_shell_html(title="ChatGPT", panel_id="chatgpt", lang="en")
    uid = stream_uid("chatgpt")
    assert f'for="fs_{uid}"' in shell
    assert f'id="fs_{uid}"' in shell
    assert 'class="fs-ck"' in shell
    assert 'class="fs-overlay"' in shell
    assert 'class="fs-pre"' in shell
    assert f'data-fs="fs_{uid}"' in shell
    assert "Full screen" in shell
    assert "ChatGPT" in shell
    # Hidden by default — must not leak into page flow
    assert "display:none" in shell.replace(" ", "")


def test_stream_shell_italian_labels():
    shell = stream_shell_html(title="MedPsy", panel_id="qvac", lang="it")
    assert "Schermo intero" in shell
    assert 'aria-label="Chiudi"' in shell


def test_stream_body_escapes_model_text():
    nasty = '<script>alert(1)</script> & "x"'
    body = stream_body_html(nasty, live=True, panel_id="m1")
    assert "<script>" not in body
    assert html.escape(nasty) in body
    assert 'class="caret"' in body
    assert f'data-panel="{stream_uid("m1")}"' in body


def test_comprehension_and_structured_use_shared_shell():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    home = (root / "app.py").read_text(encoding="utf-8")
    structured = (root / "pages" / "structured_graded.py").read_text(encoding="utf-8")
    panels = (root / "lib" / "stream_panels.py").read_text(encoding="utf-8")
    assert "stream_shell_html" in home
    assert "stream_shell_html as _stream_shell_html_shared" in structured
    assert "fs-overlay" in panels
    assert "stream.fullscreen" in panels
    assert "syncOpenFullscreenText" in (
        root / "assets" / "dashboard_portal.js"
    ).read_text(encoding="utf-8")
