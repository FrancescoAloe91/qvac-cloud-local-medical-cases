"""Production cost aggregation: run vs batch-shared extraction."""

from __future__ import annotations

from benchmark.costing import (
    batch_cost_breakdown,
    batch_shared_cost_usd,
    batch_total_cost_usd,
    cost_breakdown_for_run,
    run_cost_usd,
)
from benchmark.schema import (
    CandidateAnswer,
    JudgeResult,
    ModelCallMeta,
    RunArtifact,
)


def _cand(key: str, cost: float) -> CandidateAnswer:
    return CandidateAnswer(
        candidate_key=key,
        label=key,
        blind_id=f"b-{key}",
        answers={},
        meta=ModelCallMeta(model=key, provider="openrouter", cost_usd=cost),
    )


def _judge(key: str, cost: float) -> JudgeResult:
    return JudgeResult(
        candidate_key=key,
        blind_id=f"b-{key}",
        status="valid",
        weighted_accuracy=50.0,
        question_scores=[],
        judge_model="judge",
        judge_meta=ModelCallMeta(model="judge", provider="openrouter", cost_usd=cost),
    )


def test_run_cost_excludes_extraction():
    cands = [_cand("a", 0.10), _cand("b", 0.05)]
    judges = [_judge("a", 0.20), _judge("b", 0.20)]
    assert run_cost_usd(cands, judges) == 0.55
    bd = cost_breakdown_for_run(cands, judges, extraction_cost_usd=0.12)
    assert bd["total_usd"] == 0.55
    assert bd["run_cost_usd"] == 0.55
    assert bd["extractor_usd"] == 0.12
    assert bd["batch_shared_cost_usd"] == 0.12


def test_batch_total_extraction_once_for_n_gt_1():
    arts = []
    for i in range(5):
        cands = [_cand("a", 0.10)]
        judges = [_judge("a", 0.20)]
        bd = cost_breakdown_for_run(cands, judges, extraction_cost_usd=0.12)
        arts.append(
            RunArtifact(
                run_id=f"r{i}",
                case_id="caseC",
                started_at="t0",
                finished_at="t1",
                n_index=i + 1,
                total_cost_usd=bd["run_cost_usd"],
                cost_breakdown=bd,
                models_config={"extraction_cost_usd": 0.12},
            )
        )
    # 5 * 0.30 run + 0.12 shared once = 1.62
    assert batch_total_cost_usd(arts) == 1.62
    assert batch_shared_cost_usd(0.12) == 0.12
    bb = batch_cost_breakdown(arts)
    assert bb["batch_shared_cost_usd"] == 0.12
    assert bb["run_cost_usd"] == 1.5
    assert bb["batch_total_cost_usd"] == 1.62
    assert bb["n_runs"] == 5


def test_batch_total_n1_includes_extraction_once():
    cands = [_cand("a", 0.10)]
    judges = [_judge("a", 0.20)]
    bd = cost_breakdown_for_run(cands, judges, extraction_cost_usd=0.05)
    art = RunArtifact(
        run_id="r0",
        case_id="caseC",
        started_at="t0",
        finished_at="t1",
        total_cost_usd=bd["run_cost_usd"],
        cost_breakdown=bd,
    )
    assert batch_total_cost_usd([art]) == 0.35
