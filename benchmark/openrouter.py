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


def _resolve_api_key(api_key: Optional[str] = None) -> str:
    """Prefer explicit session key; never fall back to host env on Cloud."""
    key = (api_key or "").strip()
    if key:
        return key
    try:
        from lib.deployment import is_streamlit_cloud

        if is_streamlit_cloud():
            raise RuntimeError(
                "OpenRouter API key required in this session (BYOK). "
                "Host OPENROUTER_API_KEY is not used on Streamlit Cloud."
            )
    except ImportError:
        pass
    return openrouter_api_key()

# Fallback $/1M when usage.cost is missing (catalog snapshot; billed truth = usage.cost)
_FALLBACK_PRICES = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-5.5": (5.00, 30.00),
    "openai/gpt-5.4": (2.50, 15.00),
    "openai/gpt-5.4-mini": (0.75, 4.50),
    "openai/gpt-5-mini": (0.25, 2.00),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-3.5-flash": (1.50, 9.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "anthropic/claude-sonnet-5": (2.00, 10.00),
    "qwen/qwen3.6-flash": (0.1875, 1.125),
    # Whole-run verifier (models.yaml judge.verifier_model)
    "qwen/qwen3.5-397b-a17b": (0.39, 2.34),
    "openai/gpt-oss-20b:free": (0.0, 0.0),
    "nvidia/nemotron-3-super-120b-a12b:free": (0.0, 0.0),
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


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _content_to_text(content: Any) -> str:
    """Normalize OpenRouter content (str | list of parts | null) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("content"), str):
                    parts.append(part["content"])
        return "".join(parts)
    return str(content)


def _message_text(message: Any) -> str:
    """Extract assistant text; fall back to reasoning fields when content is empty."""
    if not isinstance(message, dict):
        return _content_to_text(message)
    text = _content_to_text(message.get("content"))
    if text.strip():
        return text
    for key in ("reasoning_content", "reasoning", "refusal"):
        alt = _content_to_text(message.get(key))
        if alt.strip():
            return alt
    return text


def is_retryable_error(err: str) -> bool:
    e = (err or "").lower()
    return any(
        x in e
        for x in (
            "incompleteread",
            "incomplete read",
            "empty body",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "temporarily",
            "remote end closed",
            "remote disconnected",
            "http 502",
            "http 503",
            "http 504",
            "502",
            "503",
            "504",
            "429",
            "rate limit",
        )
    )


def _meta_error(
    model: str,
    display_label: str,
    error: str,
    t0: float,
) -> Tuple[str, ModelCallMeta]:
    return "", ModelCallMeta(
        model=model,
        provider="openrouter",
        display_label=display_label,
        error=error,
        latency_s=round(time.time() - t0, 2),
    )


def _provider_prefs(
    *,
    allowed_providers: Optional[List[str]] = None,
    require_parameters: bool = False,
    allow_fallbacks: bool = True,
) -> Optional[Dict[str, Any]]:
    """OpenRouter provider routing prefs.

    - best-effort / ``controlled``: prefer order, allow_fallbacks=True
    - ``strict_controlled``: allow_fallbacks=False + require_parameters when asked
    """
    if not allowed_providers and not require_parameters:
        return None
    prefs: Dict[str, Any] = {}
    if allowed_providers:
        prefs["order"] = list(allowed_providers)
        prefs["allow_fallbacks"] = bool(allow_fallbacks)
    if require_parameters:
        prefs["require_parameters"] = True
    return prefs or None


def _parse_chat_payload(
    data: Any,
    *,
    model: str,
    display_label: str,
    t0: float,
    on_token: TokenCallback,
    allowed_providers: Optional[List[str]] = None,
) -> Tuple[str, ModelCallMeta]:
    """Safe post-HTTP normalization — never raise after a paid call."""
    try:
        if not isinstance(data, dict):
            return _meta_error(
                model, display_label, "OpenRouter returned non-object JSON", t0
            )
        latency = round(time.time() - t0, 2)
        choices = data.get("choices") or []
        content = ""
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            content = _message_text(first.get("message") or {})
        else:
            first = {}
        if on_token and content:
            on_token(content)

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        prompt_tokens = _as_int(usage.get("prompt_tokens"), 0)
        completion_tokens = _as_int(usage.get("completion_tokens"), 0)
        details = usage.get("completion_tokens_details")
        reasoning_tokens = _as_int(usage.get("reasoning_tokens"), 0)
        if not reasoning_tokens and isinstance(details, dict):
            reasoning_tokens = _as_int(details.get("reasoning_tokens"), 0)
        cost = _as_float(usage.get("cost"), None)
        if cost is None:
            cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
        tps = (
            round(completion_tokens / latency, 1)
            if latency > 0 and completion_tokens
            else None
        )
        routed_provider = str(data.get("provider") or "")
        allowed = [str(p).strip().lower() for p in (allowed_providers or []) if str(p).strip()]
        deviation = bool(
            allowed
            and routed_provider
            and routed_provider.strip().lower() not in allowed
        )
        return content.strip(), ModelCallMeta(
            model=str(data.get("model") or model),
            provider="openrouter",
            requested_model=model,
            routed_model=str(data.get("model") or ""),
            routed_provider=routed_provider,
            finish_reason=str(first.get("finish_reason") or ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=round(float(cost), 6),
            latency_s=latency,
            ttft_s=None,
            tps=tps,
            display_label=display_label,
            configuration_deviation=deviation,
            requested_providers=list(allowed_providers or []),
        )
    except Exception as exc:
        return _meta_error(
            model,
            display_label,
            f"OpenRouter response parse error: {type(exc).__name__}: {exc}",
            t0,
        )


def _chat_once(
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    response_format: Optional[Dict[str, Any]],
    timeout: float,
    on_token: TokenCallback,
    display_label: str,
    api_key: Optional[str] = None,
    allowed_providers: Optional[List[str]] = None,
    require_parameters: bool = False,
    allow_fallbacks: bool = True,
) -> Tuple[str, ModelCallMeta]:
    key = _resolve_api_key(api_key)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format:
        payload["response_format"] = response_format
    prefs = _provider_prefs(
        allowed_providers=allowed_providers,
        require_parameters=require_parameters,
        allow_fallbacks=allow_fallbacks,
    )
    if prefs:
        payload["provider"] = prefs

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL, data=body, method="POST", headers=_headers(key)
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_body = resp.read().decode("utf-8")
            if not (raw_body or "").strip():
                return _meta_error(model, display_label, "Empty body from OpenRouter", t0)
            data = json.loads(raw_body)
    except urllib.error.HTTPError as exc:
        return _meta_error(model, display_label, _format_http_error(exc), t0)
    except Exception as exc:
        return _meta_error(model, display_label, str(exc), t0)

    return _parse_chat_payload(
        data,
        model=model,
        display_label=display_label,
        t0=t0,
        on_token=on_token,
        allowed_providers=allowed_providers,
    )


def chat(
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 3000,
    response_format: Optional[Dict[str, Any]] = None,
    timeout: float = 180.0,
    on_token: TokenCallback = None,
    display_label: str = "",
    max_attempts: int = 3,
    api_key: Optional[str] = None,
    allowed_providers: Optional[List[str]] = None,
    require_parameters: bool = False,
    allow_fallbacks: bool = True,
) -> Tuple[str, ModelCallMeta]:
    """Non-streaming chat (used by judge). Retries truncated / empty transport."""
    last: Tuple[str, ModelCallMeta] = ("", ModelCallMeta(model=model, provider="openrouter"))
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        text, meta = _chat_once(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            timeout=timeout,
            on_token=on_token,
            display_label=display_label,
            api_key=api_key,
            allowed_providers=allowed_providers,
            require_parameters=require_parameters,
            allow_fallbacks=allow_fallbacks,
        )
        last = (text, meta)
        ok_text = bool((text or "").strip())
        if ok_text and not meta.error:
            return text, meta
        retry = False
        if meta.error and is_retryable_error(meta.error):
            retry = True
        elif not meta.error and not ok_text:
            retry = True
        if retry and attempt + 1 < attempts:
            time.sleep(1.2 * (attempt + 1))
            continue
        return text, meta
    return last


def chat_stream(
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: Optional[float] = 0.3,
    max_tokens: int = 3000,
    timeout: float = 180.0,
    on_token: TokenCallback = None,
    display_label: str = "",
    api_key: Optional[str] = None,
    allowed_providers: Optional[List[str]] = None,
    require_parameters: bool = False,
    allow_fallbacks: bool = True,
) -> Tuple[str, ModelCallMeta]:
    """Streaming chat — measures TTFT and TPS from first token."""
    key = _resolve_api_key(api_key)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    prefs = _provider_prefs(
        allowed_providers=allowed_providers,
        require_parameters=require_parameters,
        allow_fallbacks=allow_fallbacks,
    )
    if prefs:
        payload["provider"] = prefs
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL, data=body, method="POST", headers=_headers(key)
    )
    t0 = time.time()
    ttft_s: Optional[float] = None
    chunks: List[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    cost: Optional[float] = None
    routed_model = ""
    routed_provider = ""
    finish_reason = ""

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
                if not isinstance(data, dict):
                    continue
                if data.get("model"):
                    routed_model = str(data.get("model"))
                if data.get("provider"):
                    routed_provider = str(data.get("provider"))
                if data.get("error"):
                    err = data["error"]
                    msg = err if isinstance(err, str) else json.dumps(err)[:400]
                    return _meta_error(model, display_label, msg, t0)
                usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                if usage:
                    prompt_tokens = _as_int(usage.get("prompt_tokens"), prompt_tokens)
                    completion_tokens = _as_int(
                        usage.get("completion_tokens"), completion_tokens
                    )
                    details = usage.get("completion_tokens_details")
                    if isinstance(details, dict) and details.get("reasoning_tokens") is not None:
                        reasoning_tokens = _as_int(
                            details.get("reasoning_tokens"), reasoning_tokens
                        )
                    elif usage.get("reasoning_tokens") is not None:
                        reasoning_tokens = _as_int(
                            usage.get("reasoning_tokens"), reasoning_tokens
                        )
                    if usage.get("cost") is not None:
                        cost = _as_float(usage.get("cost"), cost)
                choices = data.get("choices") or []
                if not isinstance(choices, list) or not choices:
                    continue
                first = choices[0] if isinstance(choices[0], dict) else {}
                if first.get("finish_reason"):
                    finish_reason = str(first.get("finish_reason"))
                delta = _content_to_text((first.get("delta") or {}).get("content"))
                if not delta:
                    delta = _message_text(first.get("message") or {})
                if delta:
                    if ttft_s is None:
                        ttft_s = round(time.time() - t0, 3)
                    chunks.append(delta)
                    if on_token:
                        on_token(delta)
    except urllib.error.HTTPError as exc:
        return _meta_error(model, display_label, _format_http_error(exc), t0)
    except Exception as exc:
        return _meta_error(model, display_label, str(exc), t0)

    try:
        text = "".join(chunks).strip()
        latency = round(time.time() - t0, 2)
        if completion_tokens <= 0 and text:
            completion_tokens = max(1, len(text.split()))
        if cost is None:
            cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
        gen_s = (latency - (ttft_s or 0)) if latency else 0
        tps = (
            round(completion_tokens / gen_s, 1)
            if gen_s > 0.05 and completion_tokens
            else (
                round(completion_tokens / latency, 1)
                if latency > 0 and completion_tokens
                else None
            )
        )
        allowed = [
            str(p).strip().lower() for p in (allowed_providers or []) if str(p).strip()
        ]
        deviation = bool(
            allowed
            and routed_provider
            and routed_provider.strip().lower() not in allowed
        )
        return text, ModelCallMeta(
            model=routed_model or model,
            provider="openrouter",
            requested_model=model,
            routed_model=routed_model,
            routed_provider=routed_provider,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=round(float(cost or 0), 6),
            latency_s=latency,
            ttft_s=ttft_s,
            tps=tps,
            display_label=display_label,
            configuration_deviation=deviation,
            requested_providers=list(allowed_providers or []),
        )
    except Exception as exc:
        return _meta_error(
            model,
            display_label,
            f"OpenRouter stream finalize error: {type(exc).__name__}: {exc}",
            t0,
        )
