"""OpenRouter chat client with streaming, TTFT/TPS, and cost accounting."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from benchmark.config import openrouter_api_key
from benchmark.schema import ModelCallMeta

OPENROUTER_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"
)

# Fallback $/1M when usage.cost is missing
_FALLBACK_PRICES = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-5.5": (5.00, 30.00),
    "openai/gpt-5.4": (2.50, 15.00),
    "openai/gpt-5.4-mini": (0.75, 4.50),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-3.5-flash": (1.50, 9.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "deepseek/deepseek-chat": (0.25, 0.95),
    "deepseek/deepseek-r1": (0.70, 2.50),
}


def estimate_tokens_from_text(*parts: str) -> int:
    """Rough token count: ~4 chars/token for mixed EN clinical text."""
    n = sum(len(p or "") for p in parts)
    return max(1, n // 4)


def estimate_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    pin, pout = _FALLBACK_PRICES.get(model, (2.0, 10.0))
    return (prompt_tokens / 1e6) * pin + (completion_tokens / 1e6) * pout


def model_prices_per_mtok(model: str) -> tuple[float, float]:
    return _FALLBACK_PRICES.get(model, (2.0, 10.0))


TokenCallback = Optional[Callable[[str], None]]


def _headers(key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/FrancescoAloe91/qvac-vs-cloud-llms-health-test",
        "X-Title": "QVAC Health Benchmark",
    }


def _format_http_error(exc: urllib.error.HTTPError) -> str:
    err_body = exc.read().decode("utf-8", errors="replace")[:800]
    if exc.code == 401:
        return (
            "OpenRouter 401 — API key rejected (invalid, truncated, or revoked). "
            "Paste a full sk-or-v1-… key in the sidebar (no '…' placeholder). "
            f"Detail: {err_body[:200]}"
        )
    return f"HTTP {exc.code}: {err_body}"


def chat(
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    response_format: Optional[Dict[str, Any]] = None,
    timeout: float = 180.0,
    on_token: TokenCallback = None,
    display_label: str = "",
) -> Tuple[str, ModelCallMeta]:
    """Non-streaming chat (used by judge). Still records latency."""
    key = openrouter_api_key()
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format:
        payload["response_format"] = response_format

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL, data=body, method="POST", headers=_headers(key)
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return "", ModelCallMeta(
            model=model,
            provider="openrouter",
            display_label=display_label,
            error=_format_http_error(exc),
            latency_s=round(time.time() - t0, 2),
        )
    except Exception as exc:
        return "", ModelCallMeta(
            model=model,
            provider="openrouter",
            display_label=display_label,
            error=str(exc),
            latency_s=round(time.time() - t0, 2),
        )

    latency = round(time.time() - t0, 2)
    choices = data.get("choices") or []
    content = ""
    if choices:
        content = (choices[0].get("message") or {}).get("content") or ""
    if on_token and content:
        on_token(content)

    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cost = usage.get("cost")
    cost = float(cost) if cost is not None else estimate_cost_usd(
        model, prompt_tokens, completion_tokens
    )
    tps = round(completion_tokens / latency, 1) if latency > 0 and completion_tokens else None

    return content.strip(), ModelCallMeta(
        model=model,
        provider="openrouter",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=round(cost, 6),
        latency_s=latency,
        ttft_s=None,
        tps=tps,
        display_label=display_label,
    )


def chat_stream(
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: float = 180.0,
    on_token: TokenCallback = None,
    display_label: str = "",
) -> Tuple[str, ModelCallMeta]:
    """Streaming chat — measures TTFT and TPS from first token."""
    key = openrouter_api_key()
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL, data=body, method="POST", headers=_headers(key)
    )
    t0 = time.time()
    ttft_s: Optional[float] = None
    chunks: List[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    cost: Optional[float] = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("error"):
                    err = data["error"]
                    msg = err if isinstance(err, str) else json.dumps(err)[:400]
                    return "", ModelCallMeta(
                        model=model,
                        provider="openrouter",
                        display_label=display_label,
                        error=msg,
                        latency_s=round(time.time() - t0, 2),
                    )
                usage = data.get("usage") or {}
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    completion_tokens = int(
                        usage.get("completion_tokens") or completion_tokens
                    )
                    if usage.get("cost") is not None:
                        cost = float(usage["cost"])
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content") or ""
                if not delta:
                    # some providers put full content in message
                    delta = (choices[0].get("message") or {}).get("content") or ""
                if delta:
                    if ttft_s is None:
                        ttft_s = round(time.time() - t0, 3)
                    chunks.append(delta)
                    if on_token:
                        on_token(delta)
    except urllib.error.HTTPError as exc:
        return "", ModelCallMeta(
            model=model,
            provider="openrouter",
            display_label=display_label,
            error=_format_http_error(exc),
            latency_s=round(time.time() - t0, 2),
        )
    except Exception as exc:
        return "", ModelCallMeta(
            model=model,
            provider="openrouter",
            display_label=display_label,
            error=str(exc),
            latency_s=round(time.time() - t0, 2),
        )

    text = "".join(chunks).strip()
    latency = round(time.time() - t0, 2)
    if completion_tokens <= 0 and text:
        # rough fallback when stream usage missing
        completion_tokens = max(1, len(text.split()))
    if cost is None:
        cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
    gen_s = (latency - (ttft_s or 0)) if latency else 0
    tps = (
        round(completion_tokens / gen_s, 1)
        if gen_s > 0.05 and completion_tokens
        else (round(completion_tokens / latency, 1) if latency > 0 and completion_tokens else None)
    )
    return text, ModelCallMeta(
        model=model,
        provider="openrouter",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=round(cost, 6),
        latency_s=latency,
        ttft_s=ttft_s,
        tps=tps,
        display_label=display_label,
    )
