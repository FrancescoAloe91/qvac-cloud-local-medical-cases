"""Hosted vs local run store — no plaintext writes in hosted mode."""

from __future__ import annotations

from pathlib import Path

from benchmark.schema import (
    CandidateAnswer,
    JudgeResult,
    ModelCallMeta,
    MultiRunSummary,
    RunArtifact,
)
from lib.run_store import HostedRunStore, LocalRunStore


def _art(run_id: str = "r1") -> RunArtifact:
    return RunArtifact(
        run_id=run_id,
        case_id="caseC",
        started_at="t0",
        finished_at="t1",
        cohort_id="c1",
        candidates=[
            CandidateAnswer(
                candidate_key="chatgpt",
                label="c",
                blind_id="b",
                answers={},
                meta=ModelCallMeta(model="m", provider="openrouter"),
            )
        ],
        judgments=[
            JudgeResult(
                candidate_key="chatgpt",
                blind_id="b",
                question_scores=[],
                weighted_accuracy=50,
                judge_model="j",
                judge_meta=ModelCallMeta(model="j", provider="openrouter"),
            )
        ],
        ranking=[{"key": "chatgpt", "accuracy": 50, "status": "ok"}],
    )


def test_local_store_writes_fs(tmp_path: Path):
    store = LocalRunStore(tmp_path)
    assert store.writes_plaintext
    path = store.persist_artifact(_art())
    assert path is not None and path.exists()
    listed = store.list_artifacts()
    assert len(listed) == 1
    s = MultiRunSummary(case_id="caseC", n=1, run_ids=["r1"])
    sp = store.persist_summary(s)
    assert sp is not None and sp.exists()


def test_hosted_store_no_fs_writes(tmp_path: Path):
    mem: list = []
    sums: list = []
    store = HostedRunStore(
        memory=mem,
        memory_setter=lambda a: mem.clear() or mem.extend(a),
        summaries=sums,
        summaries_setter=lambda s: sums.clear() or sums.extend(s),
    )
    assert not store.writes_plaintext
    assert store.persist_artifact(_art("h1")) is None
    assert store.persist_summary(MultiRunSummary(case_id="c", n=1)) is None
    assert len(store.list_artifacts()) == 1
    # Workspace dir must stay empty
    assert list(tmp_path.glob("*.json")) == []
