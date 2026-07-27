"""Session-scoped cooperative cancellation for Streamlit reruns."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Dict


@dataclass
class RunToken:
    run_id: str
    cancelled: threading.Event


_LOCK = threading.Lock()
_TOKENS: Dict[str, RunToken] = {}


def start_run(scope: str) -> RunToken:
    token = RunToken(run_id=uuid.uuid4().hex, cancelled=threading.Event())
    with _LOCK:
        previous = _TOKENS.get(scope)
        if previous:
            previous.cancelled.set()
        _TOKENS[scope] = token
    return token


def cancel_run(scope: str) -> bool:
    with _LOCK:
        token = _TOKENS.get(scope)
    if not token:
        return False
    token.cancelled.set()
    return True


def is_cancelled(scope: str) -> bool:
    with _LOCK:
        token = _TOKENS.get(scope)
    return bool(token and token.cancelled.is_set())


def finish_run(scope: str, run_id: str) -> None:
    with _LOCK:
        current = _TOKENS.get(scope)
        if current and current.run_id == run_id:
            _TOKENS.pop(scope, None)
