"""Canonical LLM display: brand/name + model/version for charts and tables."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Full History / chart keyset (≤12). Default live roster is 9; three optional/legacy
# slots (Gemma, Llama, MedPsy Q8) stay labeled here so History still resolves them.
CURRENT_ROSTER_KEYS: Tuple[str, ...] = (
    "chatgpt",
    "claude",
    "gemini",
    "local_gemma",
    "local_llama",
    "local_phi",
    "local_medgemma",
    "local_med42",
    "local_ultramedical",
    "qvac_1_7b",
    "qvac",
    "qvac_4b_q8",
)
CURRENT_ROSTER_SET = frozenset(CURRENT_ROSTER_KEYS)

# Default active live roster (bands ON, optional/legacy OFF).
DEFAULT_ACTIVE_ROSTER_KEYS: Tuple[str, ...] = (
    "chatgpt",
    "claude",
    "gemini",
    "local_phi",
    "local_medgemma",
    "local_med42",
    "local_ultramedical",
    "qvac_1_7b",
    "qvac",
)
DEFAULT_ACTIVE_ROSTER_SET = frozenset(DEFAULT_ACTIVE_ROSTER_KEYS)

# Opt-in slots (UI expander); never deleted from GGUFs / artifacts / History labels.
OPTIONAL_LEGACY_SLOT_KEYS: Tuple[str, ...] = (
    "local_gemma",
    "local_llama",
    "qvac_4b_q8",
)
OPTIONAL_LEGACY_SLOT_SET = frozenset(OPTIONAL_LEGACY_SLOT_KEYS)


def is_current_roster_key(key: Optional[str]) -> bool:
    return (key or "") in CURRENT_ROSTER_SET


def filter_current_roster_rows(
    rows: Optional[Iterable[Any]],
    *,
    key_field: str = "key",
) -> List[Any]:
    """Keep only current-roster models (dict rows or objects with key/candidate_key)."""
    out: List[Any] = []
    for r in rows or []:
        if isinstance(r, dict):
            k = r.get(key_field) or r.get("candidate_key") or r.get("key")
        else:
            k = (
                getattr(r, key_field, None)
                or getattr(r, "candidate_key", None)
                or getattr(r, "key", None)
            )
        if is_current_roster_key(str(k) if k is not None else ""):
            out.append(r)
    return out


def rerank_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    score_field: str = "accuracy",
) -> List[Dict[str, Any]]:
    """Sort scored rows, preserve exact ties, leave technical N/A unranked.

    ``partial`` rows (incomplete Multi/Portfolio coverage) stay ranked by mean
    of scored runs — only hard N/A / missing scores are unranked.
    """

    def _rankable(r: Dict[str, Any]) -> bool:
        if r.get("eligible") is False:
            return False
        status = str(r.get("status") or "ok").lower()
        if status in {"n/a", "na", "failed", "error"}:
            return False
        # ok / partial / empty status with a score remain competitive
        if r.get(score_field) is None:
            return False
        return True

    ranked = sorted(
        (dict(r) for r in rows),
        key=lambda r: (
            0 if _rankable(r) else 1,
            -float(r.get(score_field) or 0),
        ),
    )
    last_score = None
    last_rank = 0
    rankable_i = 0
    for r in ranked:
        if not _rankable(r):
            r["rank"] = None
            continue
        rankable_i += 1
        score = float(r[score_field])
        if last_score is None or score != last_score:
            last_score = score
            last_rank = rankable_i
        r["rank"] = last_rank
    return ranked


# name = product / brand line · version = concrete model id / quant
MODEL_LABELS: Dict[str, Dict[str, str]] = {
    "chatgpt": {
        "name": "OpenAI API",
        "version": "GPT-5.5",
    },
    "claude": {
        "name": "Anthropic API",
        "version": "Claude Sonnet 5",
    },
    "gemini": {
        "name": "Google API",
        "version": "Gemini 3.5 Flash",
    },
    "local_gemma": {
        "name": "Local Gemma",
        "version": "Gemma-2-2B-IT Q4",
    },
    "local_llama": {
        "name": "Local Llama",
        "version": "Llama-3.2-3B Instruct Q4",
    },
    "local_phi": {
        "name": "Local Phi",
        "version": "Phi-3.5-mini Instruct Q4",
    },
    "local_medgemma": {
        "name": "MedGemma",
        "version": "MedGemma-1.5-4B-IT Q4",
    },
    "local_med42": {
        "name": "Med42",
        "version": "Med42-8B Q4",
    },
    "local_ultramedical": {
        "name": "UltraMedical",
        "version": "Llama-3-8B-UltraMedical Q4",
    },
    # legacy Band B peer (older artifacts)
    "local_qwen": {
        "name": "Local Qwen",
        "version": "Qwen2.5 (legacy peer)",
    },
    "qvac": {
        "name": "QVAC MedPsy",
        "version": "MedPsy-4B Q4",
    },
    "qvac_1_7b": {
        "name": "QVAC MedPsy",
        "version": "MedPsy-1.7B Q4",
    },
    "qvac_4b_q8": {
        "name": "QVAC MedPsy",
        "version": "MedPsy-4B Q8",
    },
    # legacy cloud Band B keys (older artifacts)
    "chatgpt_mini": {
        "name": "ChatGPT Mini",
        "version": "OpenAI GPT-5.4 Mini",
    },
    "claude_haiku": {
        "name": "Claude Haiku",
        "version": "Anthropic Haiku 4.5",
    },
    "qwen": {
        "name": "Qwen Flash",
        "version": "Alibaba Qwen3.6 Flash",
    },
}


def name_and_version(
    key: str,
    *,
    label: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, str]:
    """Return (name, version) for a candidate key."""
    meta = MODEL_LABELS.get(key or "")
    if meta:
        name = meta["name"]
        version = meta["version"]
    else:
        # Fall back: split "Brand · … · detail" display_label when present
        raw = (label or key or "").strip()
        if " · " in raw:
            parts = [p.strip() for p in raw.split(" · ") if p.strip()]
            name = parts[0]
            version = " · ".join(parts[1:]) if len(parts) > 1 else (model or key or "")
        else:
            name = raw or (key or "?")
            version = (model or "").strip() or name
    # Prefer explicit model id when it adds info (OpenRouter path / gguf tag)
    mid = (model or "").strip()
    if mid and mid not in version and "/" in mid:
        # openai/gpt-5.5 → keep friendly version; append slug only if unknown key
        if key not in MODEL_LABELS:
            version = mid
    return name, version


def full_model_label(
    key: str,
    *,
    label: Optional[str] = None,
    model: Optional[str] = None,
    sep: str = " · ",
) -> str:
    """Single-line label: Name · Model/version."""
    name, version = name_and_version(key, label=label, model=model)
    if not version or version == name:
        return name
    return f"{name}{sep}{version}"


def chart_model_label(
    key: str,
    *,
    label: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Y-axis label for histograms (two lines via newline)."""
    name, version = name_and_version(key, label=label, model=model)
    if not version or version == name:
        return name
    return f"{name}\n{version}"


def ranking_row_label(row: Dict[str, Any]) -> str:
    """Full label from a ranking / summary dict."""
    return full_model_label(
        str(row.get("key") or ""),
        label=row.get("label"),
        model=row.get("model"),
    )


def ranking_chart_label(row: Dict[str, Any]) -> str:
    return chart_model_label(
        str(row.get("key") or ""),
        label=row.get("label"),
        model=row.get("model"),
    )
