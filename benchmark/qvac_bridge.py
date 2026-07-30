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


def _opt_float(data: Dict[str, Any], key: str) -> Optional[float]:
    if data.get(key) is None:
        return None
    try:
        return float(data[key])
    except (TypeError, ValueError):
        return None


def _opt_int(data: Dict[str, Any], key: str) -> Optional[int]:
    if data.get(key) is None:
        return None
    try:
        return int(data[key])
    except (TypeError, ValueError):
        return None


def _runtime_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map real sidecar runtime fields onto ModelCallMeta kwargs."""
    out: Dict[str, Any] = {
        "device": str(data.get("device") or ""),
        "gpu_layers": _opt_int(data, "gpu_layers"),
        "ctx_size": _opt_int(data, "ctx_size"),
        "predict": _opt_int(data, "predict"),
        "gguf_sha256": str(data.get("gguf_sha256") or ""),
    }
    if data.get("temperature") is not None:
        out["temperature"] = _opt_float(data, "temperature")
    if data.get("top_k") is not None:
        out["top_k"] = _opt_int(data, "top_k")
    if data.get("top_p") is not None:
        out["top_p"] = _opt_float(data, "top_p")
    if data.get("seed") is not None:
        out["seed"] = _opt_int(data, "seed")
    return out


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


def space_free_gguf_path(gguf: Path | str) -> Path:
    """Symlink into ~/.local/qvac-models when the real path contains spaces."""
    src = Path(gguf).expanduser().resolve()
    if " " not in str(src):
        return src
    link_dir = Path.home() / ".local" / "qvac-models"
    link_dir.mkdir(parents=True, exist_ok=True)
    link = link_dir / src.name
    try:
        if link.is_symlink() or link.exists():
            try:
                link.unlink()
            except OSError:
                pass
        link.symlink_to(src)
        return link
    except OSError:
        return src


def file_sha256(path: Path | str) -> str:
    """Hex digest of a local file (GGUF pin / reproducibility)."""
    import hashlib

    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(
    gguf_path: str | Path,
    timeout: float = 360.0,
    *,
    sampling: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Hot-swap the sidecar GGUF via POST /load (for multi-QVAC compare)."""
    src = Path(gguf_path).expanduser()
    if not src.is_file():
        return {"ok": False, "error": f"GGUF not found: {src}"}
    safe = space_free_gguf_path(src)
    digest = ""
    try:
        digest = file_sha256(safe)
    except OSError:
        digest = ""
    url = f"{qvac_sidecar_url()}/load"
    body = json.dumps(
        {"model_path": str(safe), "sampling": sampling or {}}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if not isinstance(data, dict):
                return {"ok": False, "error": "Invalid /load response"}
            data.setdefault("ok", True)
            data["gguf_sha256"] = digest
            data["gguf_path"] = str(safe)
            data["sampling"] = sampling or {}
            return data
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(detail)
            detail = str(parsed.get("error") or detail)
        except Exception:
            detail = detail or f"HTTP {exc.code}"
        return {"ok": False, "error": detail, "gguf_sha256": digest}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "gguf_sha256": digest}


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
    messages: Optional[list] = None,
    sampling: Optional[Dict[str, Any]] = None,
) -> Tuple[str, ModelCallMeta]:
    """Ask the QVAC sidecar to generate a completion (with timing KPIs)."""
    # Prefer NDJSON stream so on_token can fire live when the caller supports it
    if on_token is not None:
        return generate_streaming(
            prompt,
            timeout=timeout,
            on_token=on_token,
            display_label=display_label,
            messages=messages,
            sampling=sampling,
        )

    url = f"{qvac_sidecar_url()}/generate"
    payload: Dict[str, Any] = {"prompt": prompt}
    if messages:
        payload["messages"] = messages
    if sampling:
        payload.update({k: v for k, v in sampling.items() if v is not None})
    body = json.dumps(payload).encode("utf-8")
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
            **_runtime_fields(data),
        )

    text = (data.get("content") or "").strip()
    latency = data.get("latency_s")
    if latency is None:
        latency = round(time.time() - t0, 2)
    else:
        latency = float(latency)

    err = data.get("error")
    if not text and not err:
        err = "Empty generation (0 tokens)"

    runtime = _runtime_fields(data)
    if sampling:
        if sampling.get("temp") is not None and runtime.get("temperature") is None:
            runtime["temperature"] = float(sampling["temp"])
        if sampling.get("temperature") is not None and runtime.get("temperature") is None:
            runtime["temperature"] = float(sampling["temperature"])
        for src, dst in (("top_k", "top_k"), ("top_p", "top_p"), ("seed", "seed")):
            if sampling.get(src) is not None and runtime.get(dst) is None:
                runtime[dst] = sampling[src]

    return text, ModelCallMeta(
        model=data.get("model") or "medpsy-4b",
        provider="qvac",
        prompt_tokens=int(data.get("prompt_tokens") or 0),
        completion_tokens=int(data.get("completion_tokens") or 0),
        finish_reason=str(data.get("finish_reason") or ""),
        cost_usd=0.0,
        latency_s=latency,
        ttft_s=_opt_float(data, "ttft_s"),
        tps=_opt_float(data, "tps"),
        ram_mb=_opt_float(data, "ram_mb"),
        gguf_mb=_opt_float(data, "gguf_mb"),
        display_label=display_label,
        error=str(err) if err else None,
        **runtime,
    )


def generate_streaming(
    prompt: str,
    timeout: float = 300.0,
    on_token: TokenCallback = None,
    display_label: str = "",
    messages: Optional[list] = None,
    sampling: Optional[Dict[str, Any]] = None,
) -> Tuple[str, ModelCallMeta]:
    """Stream NDJSON tokens from the sidecar; call on_token for each chunk."""
    url = f"{qvac_sidecar_url()}/generate/stream"
    payload: Dict[str, Any] = {"prompt": prompt, "stream": True}
    if messages:
        payload["messages"] = messages
    if sampling:
        payload.update({k: v for k, v in sampling.items() if v is not None})
    body = json.dumps(payload).encode("utf-8")
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

    n_tok = int(done_meta.get("completion_tokens") or 0)
    if n_tok <= 0 and text:
        n_tok = max(1, len(text.split()))

    err = done_meta.get("error")
    if not text and not err:
        err = "Empty generation (0 tokens)"

    runtime = _runtime_fields(done_meta)
    if sampling:
        if sampling.get("temp") is not None and runtime.get("temperature") is None:
            runtime["temperature"] = float(sampling["temp"])
        if sampling.get("temperature") is not None and runtime.get("temperature") is None:
            runtime["temperature"] = float(sampling["temperature"])
        for src, dst in (("top_k", "top_k"), ("top_p", "top_p"), ("seed", "seed")):
            if sampling.get(src) is not None and runtime.get(dst) is None:
                runtime[dst] = sampling[src]

    return text, ModelCallMeta(
        model=done_meta.get("model") or "medpsy-4b",
        provider="qvac",
        prompt_tokens=int(done_meta.get("prompt_tokens") or 0),
        completion_tokens=n_tok,
        finish_reason=str(done_meta.get("finish_reason") or ""),
        cost_usd=0.0,
        latency_s=latency,
        ttft_s=_opt_float(done_meta, "ttft_s"),
        tps=_opt_float(done_meta, "tps"),
        ram_mb=_opt_float(done_meta, "ram_mb"),
        gguf_mb=_opt_float(done_meta, "gguf_mb"),
        display_label=display_label,
        error=str(err) if err else None,
        **runtime,
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
        finish_reason=str(data.get("finish_reason") or ""),
        cost_usd=0.0,
        latency_s=latency,
        ttft_s=_opt_float(data, "ttft_s"),
        tps=_opt_float(data, "tps"),
        ram_mb=_opt_float(data, "ram_mb"),
        gguf_mb=_opt_float(data, "gguf_mb"),
        display_label=display_label,
    )


def iter_tokens(
    prompt: str,
    timeout: float = 300.0,
    messages: Optional[list] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield {type: token|done|error, ...} for UI live updates on the main thread.

    Prefers POST /generate/stream (NDJSON). If the sidecar is older and returns
    404, falls back to blocking POST /generate and emits a single token + done.
    QVAC SDK only — no Ollama fallback.
    """
    url = f"{qvac_sidecar_url()}/generate/stream"
    payload: Dict[str, Any] = {"prompt": prompt, "stream": True}
    if messages:
        payload["messages"] = messages
    body = json.dumps(payload).encode("utf-8")
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
            "finish_reason": meta.finish_reason,
            "cost_usd": 0.0,
            "ram_mb": meta.ram_mb,
            "gguf_mb": meta.gguf_mb,
        }
        return

    yield {"type": "error", "error": stream_err or meta.error or "QVAC generate failed"}
