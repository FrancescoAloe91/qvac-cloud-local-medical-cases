"""Local Tether QVAC MedPsy 4B inference engine.

Runs REAL on-device inference against the actual qvac/MedPsy-4B model
(https://huggingface.co/qvac/MedPsy-4B-GGUF) via a local Ollama server.
No canned/simulated answers: if the local engine is unreachable, this
module reports that clearly instead of fabricating a result, so the
benchmark stays honest — QVAC must earn its answer exactly like the
cloud models do when the user pastes their real response.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Generator, Optional

MODEL_NAME = "Tether QVAC MedPsy 4B"

OLLAMA_HOST = os.environ.get("QVAC_OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("QVAC_OLLAMA_MODEL", "medpsy-4b-cpu")
GGUF_REPO = "qvac/MedPsy-4B-GGUF"
GGUF_FILE = os.environ.get("MEDPSY_QUANT", "medpsy-4b-q4_k_m-imat.gguf")


def quant_label() -> str:
    """Human-readable quant tag from the GGUF filename (e.g. Q4_K_M)."""
    lower = GGUF_FILE.lower()
    if "q4_k_m" in lower:
        return "Q4_K_M"
    if "q8_0" in lower:
        return "Q8_0"
    if "q5_k_m" in lower:
        return "Q5_K_M"
    stem = GGUF_FILE.replace(".gguf", "").split("-")[-1]
    return stem.upper().replace("_", "_") if stem else "GGUF"


def runtime_tier_label() -> str:
    """Full on-device stack label for cards, charts and screenshot headers."""
    try:
        from benchmark.qvac_bridge import available as qvac_sdk_up
    except Exception:
        qvac_sdk_up = lambda: False  # noqa: E731
    if qvac_sdk_up():
        return f"MedPsy-4B-GGUF {quant_label()} · QVAC SDK · CPU"
    return f"MedPsy-4B-GGUF {quant_label()} · Ollama {OLLAMA_MODEL} · CPU"


def runtime_header_chip() -> str:
    """Short label for the top header live chip."""
    try:
        from benchmark.qvac_bridge import available as qvac_sdk_up
    except Exception:
        qvac_sdk_up = lambda: False  # noqa: E731
    if qvac_sdk_up():
        return f"MedPsy-4B {quant_label()} · QVAC SDK · on-device"
    return f"MedPsy-4B {quant_label()} · Ollama {OLLAMA_MODEL} · on-device"

# Inference uses Ollama Modelfile defaults only (temperature 0.6, top_k 20, top_p 0.95).
# The dashboard does not override sampling — honest demo = same stack as setup_medpsy.sh.

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def ollama_available(timeout: float = 2.0) -> bool:
    """Check whether the local Ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/version")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def model_ready(timeout: float = 2.0) -> bool:
    """Check whether the medpsy model has actually been created in Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = {m.get("name", "").split(":")[0] for m in data.get("models", [])}
        return OLLAMA_MODEL.split(":")[0] in names
    except Exception:
        return False


def get_ram_usage_gb() -> Optional[float]:
    """Real memory footprint of the loaded model, from Ollama's /api/ps."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/ps")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for m in data.get("models", []):
            if m.get("model", "").startswith(OLLAMA_MODEL):
                size = m.get("size_vram") or m.get("size") or 0
                return round(size / 1e9, 2) if size else None
    except Exception:
        return None
    return None


def _split_thinking(text: str) -> tuple[str, str]:
    """Separate the model's real chain-of-thought from its final answer."""
    match = _THINK_RE.search(text)
    if match:
        thinking = match.group(1).strip()
        final = text[match.end():].strip()
        return thinking, final
    return "", text.strip()


def stream_inference(prompt: str) -> Generator[dict, None, None]:
    """Stream real tokens from the local MedPsy model as they are generated.

    Yields {"delta": str} events while generating, and a final
    {"done": True, "content", "thinking", "stats", ...} event with the
    complete answer and genuinely measured performance stats.
    On any failure, yields a single {"done": True, "error": str} event —
    never a fabricated diagnosis.
    """
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
        }
    ).encode("utf-8")

    t_start = time.time()
    ttft_s = None
    raw_chunks = []
    final_payload = {}

    try:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line.decode("utf-8"))
                if "error" in chunk:
                    yield {"done": True, "error": chunk["error"]}
                    return
                piece = chunk.get("response", "")
                if piece:
                    if ttft_s is None:
                        ttft_s = round(time.time() - t_start, 2)
                    raw_chunks.append(piece)
                    yield {"delta": piece}
                if chunk.get("done"):
                    final_payload = chunk
                    break
    except urllib.error.URLError as exc:
        yield {"done": True, "error": f"Cannot reach local Ollama server: {exc}"}
        return
    except Exception as exc:
        yield {"done": True, "error": str(exc)}
        return

    total_s = round(time.time() - t_start, 2)
    raw_text = "".join(raw_chunks)
    thinking, final_text = _split_thinking(raw_text)

    eval_count = final_payload.get("eval_count") or 0
    eval_duration_s = (final_payload.get("eval_duration") or 0) / 1e9
    prompt_eval_count = final_payload.get("prompt_eval_count") or 0
    tps = round(eval_count / eval_duration_s, 1) if eval_duration_s > 0 else 0.0

    yield {
        "done": True,
        "content": final_text or raw_text.strip(),
        "thinking": thinking,
        "raw": raw_text,
        "stats": {
            "ttft_s": ttft_s if ttft_s is not None else 0.0,
            "tps": tps,
            "tokens_per_second": tps,
            "latency_s": total_s,
            "ram_gb": get_ram_usage_gb(),
            "completion_tokens": eval_count,
            "prompt_tokens": prompt_eval_count,
            "total_tokens": eval_count + prompt_eval_count,
            "source": "local",
            "measurable": True,
            "real_inference": True,
        },
        "model": MODEL_NAME,
        "tier": "local",
    }


def run_inference(prompt: str, lang: str = "en") -> dict:
    """Blocking helper: run real local inference and return the final result.

    Prefer `stream_inference` in the UI so the user can watch real tokens
    arrive; this wrapper just drains the generator for callers that don't
    need progressive rendering.
    """
    result = {}
    for event in stream_inference(prompt):
        if event.get("done"):
            result = event
    if not result:
        result = {"error": "No response from local engine.", "content": "", "stats": {"measurable": False}}
    result.setdefault("model", MODEL_NAME)
    result.setdefault("tier", "local")
    return result
