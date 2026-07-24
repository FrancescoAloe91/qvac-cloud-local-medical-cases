"""Load models.yaml and resolve paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parent
CASES_DIR = ROOT / "cases"
DEFAULT_MODELS = ROOT / "models.yaml"
ARTIFACTS_DIR = ROOT.parent / "artifacts"


def load_models_config(path: Path | None = None) -> Dict[str, Any]:
    cfg_path = path or Path(os.environ.get("BENCHMARK_MODELS", DEFAULT_MODELS))
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    judge_env = os.environ.get("BENCHMARK_JUDGE_MODEL")
    if judge_env:
        data.setdefault("judge", {})["model"] = judge_env
    return data


def is_usable_openrouter_key(key: str | None) -> bool:
    """True only for a complete OpenRouter key (rejects 'sk-or-v1-abc...xyz' placeholders)."""
    k = (key or "").strip()
    if not k.startswith("sk-or-"):
        return False
    if "..." in k or "…" in k:
        return False
    return len(k) >= 40


def openrouter_api_key() -> str:
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it or put it in a local .env "
            "(never commit the key)."
        )
    if not is_usable_openrouter_key(key):
        raise RuntimeError(
            "OPENROUTER_API_KEY looks invalid (truncated placeholder or too short). "
            "Paste the full sk-or-v1-… key from https://openrouter.ai/keys "
            "(must not contain '…' / '...')."
        )
    return key


def qvac_sidecar_url() -> str:
    return os.environ.get("QVAC_SIDECAR_URL", "http://127.0.0.1:8787").rstrip("/")
