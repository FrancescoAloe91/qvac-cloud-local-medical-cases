"""Run artifact persistence for local FS vs hosted (memory + encrypted account).

Hosted mode must never write plaintext artifacts/summaries/case text to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from benchmark.report import (
    list_run_artifacts as _fs_list_run_artifacts,
    load_artifact as _fs_load_artifact,
    write_artifact as _fs_write_artifact,
    write_summary as _fs_write_summary,
)
from benchmark.schema import MultiRunSummary, RunArtifact


class RunStore(Protocol):
    def persist_artifact(self, artifact: RunArtifact) -> Optional[Path]:
        ...

    def list_artifacts(self) -> List[Tuple[Optional[Path], RunArtifact]]:
        ...

    def persist_summary(self, summary: MultiRunSummary) -> Optional[Path]:
        ...

    @property
    def writes_plaintext(self) -> bool:
        ...


class LocalRunStore:
    """Filesystem JSON under a workspace directory."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)

    @property
    def writes_plaintext(self) -> bool:
        return True

    def persist_artifact(self, artifact: RunArtifact) -> Optional[Path]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        return _fs_write_artifact(artifact, self.workspace)

    def list_artifacts(self) -> List[Tuple[Optional[Path], RunArtifact]]:
        out: List[Tuple[Optional[Path], RunArtifact]] = []
        for path in _fs_list_run_artifacts(self.workspace):
            try:
                out.append((path, _fs_load_artifact(path)))
            except Exception:
                continue
        return out

    def persist_summary(self, summary: MultiRunSummary) -> Optional[Path]:
        self.workspace.mkdir(parents=True, exist_ok=True)
        return _fs_write_summary(summary, self.workspace)


class HostedRunStore:
    """Session-memory + optional encrypted cloud; never plaintext disk."""

    def __init__(
        self,
        *,
        memory: Optional[List[RunArtifact]] = None,
        memory_setter: Optional[Any] = None,
        account_session: Any = None,
        save_cloud: Optional[Any] = None,
        summaries: Optional[List[MultiRunSummary]] = None,
        summaries_setter: Optional[Any] = None,
    ):
        self._memory = list(memory or [])
        self._memory_setter = memory_setter
        self._account_session = account_session
        self._save_cloud = save_cloud
        self._summaries = list(summaries or [])
        self._summaries_setter = summaries_setter

    @property
    def writes_plaintext(self) -> bool:
        return False

    def persist_artifact(self, artifact: RunArtifact) -> Optional[Path]:
        self._memory.append(artifact)
        self._memory = self._memory[-200:]
        if self._memory_setter:
            self._memory_setter(self._memory)
        if self._save_cloud and self._account_session is not None:
            try:
                self._save_cloud(self._account_session, artifact)
            except Exception:
                pass
        return None

    def list_artifacts(self) -> List[Tuple[Optional[Path], RunArtifact]]:
        # Newest last in memory append order → reverse for History-style newest-first
        return [(None, a) for a in reversed(self._memory)]

    def persist_summary(self, summary: MultiRunSummary) -> Optional[Path]:
        self._summaries.append(summary)
        self._summaries = self._summaries[-50:]
        if self._summaries_setter:
            self._summaries_setter(self._summaries)
        return None


def artifacts_only(
    pairs: Sequence[Tuple[Optional[Path], RunArtifact]],
) -> List[RunArtifact]:
    return [a for _, a in pairs]
