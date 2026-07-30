"""Unified actual-cost accounting for Streamlit + CLI.

Cost model
----------
- ``run_cost_usd``: candidates (+ retries) + primary judge + section repair + verifier
  for one iteration. Never includes gold extraction.
- ``batch_shared_cost_usd``: gold extraction charged exactly once per Prepare/batch.
- ``batch_total_cost_usd = sum(run_cost_usd) + batch_shared_cost_usd``.

Artifact ``total_cost_usd`` stores **run_cost only**. Extraction lives in
``cost_breakdown["extractor_usd"]`` / ``models_config["extraction_cost_usd"]``
for audit; multi-run summaries add it once via ``batch_total_cost_usd``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from benchmark.schema import CandidateAnswer, JudgeResult, RunArtifact


def _f(x: Any) -> float:
    try:
        return float(x or 0.0)
    except (TypeError, ValueError):
        return 0.0


def run_cost_usd(
    candidates: Sequence[CandidateAnswer],
    judgments: Sequence[JudgeResult],
) -> float:
    """Paid spend for one iteration (no extraction)."""
    total = 0.0
    for c in candidates:
        total += _f(getattr(getattr(c, "meta", None), "cost_usd", 0.0))
    for j in judgments:
        total += _f(getattr(getattr(j, "judge_meta", None), "cost_usd", 0.0))
    return round(total, 6)


def batch_shared_cost_usd(extraction_cost: float = 0.0) -> float:
    return round(_f(extraction_cost), 6)


def cost_breakdown_for_run(
    candidates: Sequence[CandidateAnswer],
    judgments: Sequence[JudgeResult],
    *,
    extraction_cost_usd: float = 0.0,
) -> Dict[str, Any]:
    """Itemized actual-cost dict stored on each artifact.

    ``total_usd`` equals run cost only (matches artifact.total_cost_usd).
    ``extractor_usd`` is recorded for audit but not included in ``total_usd``.
    """
    cand = round(sum(_f(getattr(getattr(c, "meta", None), "cost_usd", 0.0)) for c in candidates), 6)
    judge = round(
        sum(_f(getattr(getattr(j, "judge_meta", None), "cost_usd", 0.0)) for j in judgments),
        6,
    )
    run = round(cand + judge, 6)
    shared = batch_shared_cost_usd(extraction_cost_usd)
    return {
        "candidates_usd": cand,
        "judge_usd": judge,
        "extractor_usd": shared,
        "run_cost_usd": run,
        "batch_shared_cost_usd": shared,
        "total_usd": run,  # run only — batch adds shared once
    }


def extraction_from_artifact(artifact: RunArtifact) -> float:
    bd = getattr(artifact, "cost_breakdown", None) or {}
    if isinstance(bd, dict) and bd.get("extractor_usd") is not None:
        return batch_shared_cost_usd(bd.get("extractor_usd"))
    mc = getattr(artifact, "models_config", None) or {}
    if isinstance(mc, dict):
        return batch_shared_cost_usd(mc.get("extraction_cost_usd"))
    return 0.0


def batch_total_cost_usd(
    artifacts: Sequence[RunArtifact],
    *,
    extraction_cost_usd: Optional[float] = None,
) -> float:
    """Sum of run costs + extraction once (from arg or first artifact metadata)."""
    runs = round(sum(_f(getattr(a, "total_cost_usd", 0.0)) for a in artifacts), 6)
    if extraction_cost_usd is not None:
        shared = batch_shared_cost_usd(extraction_cost_usd)
    else:
        shared = 0.0
        for a in artifacts:
            shared = extraction_from_artifact(a)
            if shared > 0:
                break
    return round(runs + shared, 6)


def batch_cost_breakdown(
    artifacts: Sequence[RunArtifact],
    *,
    extraction_cost_usd: Optional[float] = None,
) -> Dict[str, Any]:
    run_sum = round(sum(_f(getattr(a, "total_cost_usd", 0.0)) for a in artifacts), 6)
    cand_sum = 0.0
    judge_sum = 0.0
    for a in artifacts:
        bd = getattr(a, "cost_breakdown", None) or {}
        if isinstance(bd, dict):
            cand_sum += _f(bd.get("candidates_usd"))
            judge_sum += _f(bd.get("judge_usd"))
        else:
            # Fallback: treat total as run
            pass
    if extraction_cost_usd is not None:
        shared = batch_shared_cost_usd(extraction_cost_usd)
    else:
        shared = 0.0
        for a in artifacts:
            shared = extraction_from_artifact(a)
            if shared > 0:
                break
    total = round(run_sum + shared, 6)
    return {
        "candidates_usd": round(cand_sum, 6) if cand_sum else run_sum,
        "judge_usd": round(judge_sum, 6),
        "extractor_usd": shared,
        "run_cost_usd": run_sum,
        "batch_shared_cost_usd": shared,
        "batch_total_cost_usd": total,
        "n_runs": len(artifacts),
        "total_usd": total,
    }
