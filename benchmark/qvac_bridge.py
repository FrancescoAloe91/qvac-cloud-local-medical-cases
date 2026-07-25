"""HTTP client for the local QVAC SDK sidecar (Node @qvac/sdk)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from benchmark.config import qvac_sidecar_url
from benchmark.schema import ModelCallMeta

TokenCallback = Optional[Callable[[str], None]]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SIDECAR_DIR = _REPO_ROOT / "sidecar"
_SIDECAR_LOG = Path(os.environ.get("QVAC_SIDECAR_LOG", "/tmp/qvac-sidecar.log"))
_SIDECAR_PID = Path(os.environ.get("QVAC_SIDECAR_PID", "/tmp/qvac-sidecar.pid"))


def health(timeout: float = 1.5) -> Dict[str, Any]:
    url = f"{qvac_sidecar_url()}/health"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def reachable(timeout: float = 1.5) -> bool:
    """True when the sidecar HTTP /health endpoint responds (model may still be loading)."""
    return "modelLoaded" in health(timeout=timeout)


def available(timeout: float = 1.5) -> bool:
    """True when MedPsy is loaded and ready for inference via the QVAC SDK sidecar."""
    return bool(health(timeout=timeout).get("modelLoaded"))


def _spawn_sidecar() -> None:
    """Start the real QVAC SDK sidecar (GPU/Metal preferred) in the background."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js not found on PATH (need ≥22 for the QVAC sidecar).")
    server = _SIDECAR_DIR / "qvac_server.mjs"
    if not server.is_file():
        raise RuntimeError(f"Sidecar missing at {server}")
    if not (_SIDECAR_DIR / "node_modules" / "@qvac" / "sdk").exists():
        raise RuntimeError(
            "QVAC SDK not installed. Run: ./scripts/setup_qvac_sidecar.sh"
        )
    env = os.environ.copy()
    env.setdefault("QVAC_DEVICE", "gpu")
    env.setdefault("QVAC_GPU_LAYERS", "99")
    env.setdefault("QVAC_WARM_LOAD", "1")
    # Prefer space-free symlink when present (SDK file:// breaks on spaces).
    link = Path.home() / ".local" / "qvac-models" / "medpsy-4b-q4_k_m-imat.gguf"
    if link.exists() and "QVAC_MODEL_PATH" not in env:
        env["QVAC_MODEL_PATH"] = str(link.resolve() if link.is_symlink() else link)
        # resolve() follows symlink into the spaced path — keep the link path itself
        env["QVAC_MODEL_PATH"] = str(link)
    _SIDECAR_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(_SIDECAR_LOG, "ab", buffering=0)
    proc = subprocess.Popen(
        [node, str(server)],
        cwd=str(_SIDECAR_DIR),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _SIDECAR_PID.write_text(str(proc.pid), encoding="utf-8")
    except OSError:
        pass


def ensure_sidecar(
    wait_s: float = 90.0,
    *,
    start_if_down: bool = True,
) -> Dict[str, Any]:
    """Make sure the real MedPsy sidecar is up (start it if needed) and return health.

    This is the live QVAC SDK path — not a mock/demo stub.
    """
    h = health()
    if h.get("modelLoaded"):
        return h
    if reachable() and not start_if_down:
        return h
    if not reachable() and start_if_down:
        try:
            _spawn_sidecar()
        except Exception as exc:
            out = health()
            out["ensure_error"] = str(exc)
            return out
    deadline = time.time() + wait_s
    while time.time() < deadline:
        h = health()
        if h.get("modelLoaded"):
            return h
        # Sidecar up but still loading — keep waiting
        if "modelLoaded" in h and h.get("modelLoaded") is False:
            time.sleep(1.0)
            continue
        time.sleep(0.8)
    h = health()
    if not h.get("modelLoaded"):
        h = dict(h)
        h.setdefault(
            "ensure_error",
            f"Sidecar did not become ready within {wait_s:.0f}s. "
            f"Check {_SIDECAR_LOG}",
        )
    return h


def generate(
    prompt: str,
    timeout: float = 300.0,
    on_token: TokenCallback = None,
    display_label: str = "",
) -> Tuple[str, ModelCallMeta]:
    """Ask the QVAC sidecar to generate a completion (with timing KPIs)."""
    # Prefer NDJSON stream so on_token can fire live when the caller supports it
    if on_token is not None:
        return generate_streaming(
            prompt, timeout=timeout, on_token=on_token, display_label=display_label
        )

    url = f"{qvac_sidecar_url()}/generate"
    body = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return "", ModelCallMeta(
            model="medpsy-4b",
            provider="qvac",
            display_label=display_label,
            error=f"QVAC sidecar unreachable at {qvac_sidecar_url()}: {exc}",
            latency_s=round(time.time() - t0, 2),
            cost_usd=0.0,
        )
    except Exception as exc:
        return "", ModelCallMeta(
            model="medpsy-4b",
            provider="qvac",
            display_label=display_label,
            error=str(exc),
            latency_s=round(time.time() - t0, 2),
            cost_usd=0.0,
        )

    if data.get("error"):
        return "", ModelCallMeta(
            model=data.get("model") or "medpsy-4b",
            provider="qvac",
            display_label=display_label,
            error=str(data["error"]),
            latency_s=round(time.time() - t0, 2),
            cost_usd=0.0,
        )

    text = (data.get("content") or "").strip()
    latency = data.get("latency_s")
    if latency is None:
        latency = round(time.time() - t0, 2)
    else:
        latency = float(latency)

    return text, ModelCallMeta(
        model=data.get("model") or "medpsy-4b",
        provider="qvac",
        prompt_tokens=int(data.get("prompt_tokens") or 0),
        completion_tokens=int(data.get("completion_tokens") or 0),
        cost_usd=0.0,
        latency_s=latency,
        ttft_s=float(data["ttft_s"]) if data.get("ttft_s") is not None else None,
        tps=float(data["tps"]) if data.get("tps") is not None else None,
        display_label=display_label,
    )


def generate_streaming(
    prompt: str,
    timeout: float = 300.0,
    on_token: TokenCallback = None,
    display_label: str = "",
) -> Tuple[str, ModelCallMeta]:
    """Stream NDJSON tokens from the sidecar; call on_token for each chunk."""
    url = f"{qvac_sidecar_url()}/generate/stream"
    body = json.dumps({"prompt": prompt, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    content_parts: list[str] = []
    done_meta: Dict[str, Any] = {}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                et = evt.get("type")
                if et == "token":
                    tok = evt.get("token")
                    if not isinstance(tok, str):
                        # Reject object leftovers ("[object Object]") — not scorable text
                        continue
                    if tok and tok != "[object Object]":
                        content_parts.append(tok)
                        if on_token:
                            on_token(tok)
                elif et == "done":
                    done_meta = evt
                    if evt.get("content") and not content_parts:
                        content_parts.append(str(evt["content"]))
                elif et == "error":
                    return "", ModelCallMeta(
                        model="medpsy-4b",
                        provider="qvac",
                        display_label=display_label,
                        error=str(evt.get("error") or "stream error"),
                        latency_s=round(time.time() - t0, 2),
                        cost_usd=0.0,
                    )
    except urllib.error.URLError as exc:
        # Older sidecar without /generate/stream — fall back to blocking JSON
        text, meta = _generate_blocking(prompt, timeout=timeout, display_label=display_label)
        if on_token and text and not meta.error:
            on_token(text)
        if meta.error and "unreachable" in (meta.error or ""):
            meta.error = f"QVAC sidecar unreachable at {qvac_sidecar_url()}: {exc}"
        return text, meta
    except Exception as exc:
        return "", ModelCallMeta(
            model="medpsy-4b",
            provider="qvac",
            display_label=display_label,
            error=str(exc),
            latency_s=round(time.time() - t0, 2),
            cost_usd=0.0,
        )

    text = (done_meta.get("content") or "".join(content_parts)).strip()
    if on_token and text and not content_parts:
        on_token(text)

    latency = done_meta.get("latency_s")
    if latency is None:
        latency = round(time.time() - t0, 2)
    else:
        latency = float(latency)

    return text, ModelCallMeta(
        model=done_meta.get("model") or "medpsy-4b",
        provider="qvac",
        prompt_tokens=int(done_meta.get("prompt_tokens") or 0),
        completion_tokens=int(
            done_meta.get("completion_tokens") or max(1, len(text.split()))
        ),
        cost_usd=0.0,
        latency_s=latency,
        ttft_s=float(done_meta["ttft_s"]) if done_meta.get("ttft_s") is not None else None,
        tps=float(done_meta["tps"]) if done_meta.get("tps") is not None else None,
        display_label=display_label,
    )


def _generate_blocking(
    prompt: str,
    timeout: float = 300.0,
    display_label: str = "",
) -> Tuple[str, ModelCallMeta]:
    url = f"{qvac_sidecar_url()}/generate"
    body = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(detail)
            detail = str(parsed.get("error") or detail)
        except Exception:
            detail = detail or str(exc)
        return "", ModelCallMeta(
            model="medpsy-4b",
            provider="qvac",
            display_label=display_label,
            error=detail,
            latency_s=round(time.time() - t0, 2),
            cost_usd=0.0,
        )
    except Exception as exc:
        return "", ModelCallMeta(
            model="medpsy-4b",
            provider="qvac",
            display_label=display_label,
            error=str(exc),
            latency_s=round(time.time() - t0, 2),
            cost_usd=0.0,
        )
    if data.get("error"):
        return "", ModelCallMeta(
            model=data.get("model") or "medpsy-4b",
            provider="qvac",
            display_label=display_label,
            error=str(data["error"]),
            latency_s=round(time.time() - t0, 2),
            cost_usd=0.0,
        )
    text = (data.get("content") or "").strip()
    latency = float(data["latency_s"]) if data.get("latency_s") is not None else round(
        time.time() - t0, 2
    )
    return text, ModelCallMeta(
        model=data.get("model") or "medpsy-4b",
        provider="qvac",
        prompt_tokens=int(data.get("prompt_tokens") or 0),
        completion_tokens=int(data.get("completion_tokens") or 0),
        cost_usd=0.0,
        latency_s=latency,
        ttft_s=float(data["ttft_s"]) if data.get("ttft_s") is not None else None,
        tps=float(data["tps"]) if data.get("tps") is not None else None,
        display_label=display_label,
    )


def iter_tokens(
    prompt: str,
    timeout: float = 300.0,
) -> Iterator[Dict[str, Any]]:
    """Yield {type: token|done|error, ...} for UI live updates on the main thread.

    Prefers POST /generate/stream (NDJSON). If the sidecar is older and returns
    404, falls back to blocking POST /generate and emits a single token + done.
    QVAC SDK only — no Ollama fallback.
    """
    url = f"{qvac_sidecar_url()}/generate/stream"
    body = json.dumps({"prompt": prompt, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    stream_err: str | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if evt.get("type") == "error":
                    stream_err = str(evt.get("error") or "stream error")
                    break
                yield evt
                if evt.get("type") == "done":
                    return
            if stream_err is None:
                return
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            stream_err = "HTTP Error 404: Not Found"
        else:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
                parsed = json.loads(detail)
                detail = str(parsed.get("error") or detail)
            except Exception:
                detail = detail or f"HTTP Error {exc.code}: {exc.reason}"
            stream_err = detail
    except Exception as exc:
        stream_err = str(exc)

    # Fallback: blocking /generate (older sidecars without /generate/stream)
    text, meta = _generate_blocking(prompt, timeout=timeout, display_label="")
    if not meta.error:
        if text:
            yield {"type": "token", "token": text}
        yield {
            "type": "done",
            "content": text,
            "model": meta.model,
            "latency_s": meta.latency_s,
            "ttft_s": meta.ttft_s,
            "tps": meta.tps,
            "prompt_tokens": meta.prompt_tokens,
            "completion_tokens": meta.completion_tokens,
            "cost_usd": 0.0,
        }
        return

    yield {"type": "error", "error": stream_err or meta.error or "QVAC generate failed"}
