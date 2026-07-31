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

from typing import Any, Dict, List, Optional, Sequence, Union

from benchmark.schema import CandidateAnswer, JudgeResult, RunArtifact

ArtifactLike = Union[RunArtifact, Dict[str, Any]]


def _f(x: Any) -> float:
    try:
        return float(x or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _i(x: Any) -> int:
    try:
        return int(x or 0)
    except (TypeError, ValueError):
        return 0


def _median(xs: Sequence[float]) -> float:
    vals = sorted(float(x) for x in xs)
    if not vals:
        return 0.0
    return float(vals[len(vals) // 2])


def _percentile(xs: Sequence[float], p: float) -> float:
    vals = sorted(float(x) for x in xs)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    p = min(1.0, max(0.0, float(p)))
    idx = int(round((len(vals) - 1) * p))
    return float(vals[idx])


def _artifact_candidates(art: ArtifactLike) -> Sequence[Any]:
    if isinstance(art, dict):
        return art.get("candidates") or []
    return getattr(art, "candidates", None) or []


def _artifact_judgments(art: ArtifactLike) -> Sequence[Any]:
    if isinstance(art, dict):
        return art.get("judgments") or []
    return getattr(art, "judgments", None) or []


def _meta_of(obj: Any, field: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(field) or {}
    return getattr(obj, field, None) or {}


def _meta_get(meta: Any, key: str, default: Any = None) -> Any:
    if isinstance(meta, dict):
        return meta.get(key, default)
    return getattr(meta, key, default)


def cost_estimate_priors_from_artifacts(
    artifacts: Sequence[ArtifactLike],
    *,
    scored_keys: int,
    openrouter_keys: int,
    min_samples: int = 3,
    lookback: int = 60,
) -> Optional[Dict[str, Any]]:
    """Derive typical/high token + run-cost priors from recent History.

    Matches runs with the same judged roster size and OpenRouter candidate count
    so Multi ×N estimates track real ``usage.cost`` instead of max_tokens caps.
    Returns None when too few comparable runs exist.
    """
    want_scored = max(0, int(scored_keys))
    want_or = max(0, int(openrouter_keys))
    if want_scored < 1:
        return None

    run_costs: List[float] = []
    judge_comp: List[float] = []
    judge_prompt: List[float] = []
    cand_comp: List[float] = []
    cand_prompt: List[float] = []
    n_repair_calls = 0
    n_verifier_calls = 0
    n_judge_calls = 0

    for art in list(artifacts)[: max(1, int(lookback))]:
        cands = list(_artifact_candidates(art))
        judgments = list(_artifact_judgments(art))
        if not judgments and not cands:
            continue
        n_or = 0
        for c in cands:
            meta = _meta_of(c, "meta")
            if str(_meta_get(meta, "provider") or "") == "openrouter":
                n_or += 1
        n_scored = len(judgments) if judgments else len(cands)
        if n_scored != want_scored or n_or != want_or:
            continue
        total = _f(
            art.get("total_cost_usd")
            if isinstance(art, dict)
            else getattr(art, "total_cost_usd", 0.0)
        )
        if total <= 0:
            continue
        run_costs.append(total)
        for c in cands:
            meta = _meta_of(c, "meta")
            if str(_meta_get(meta, "provider") or "") != "openrouter":
                continue
            if _meta_get(meta, "error"):
                continue
            cout = _i(_meta_get(meta, "completion_tokens"))
            cin = _i(_meta_get(meta, "prompt_tokens"))
            if cout > 0:
                cand_comp.append(float(cout))
            if cin > 0:
                cand_prompt.append(float(cin))
        for j in judgments:
            meta = _meta_of(j, "judge_meta")
            cout = _i(_meta_get(meta, "completion_tokens"))
            cin = _i(_meta_get(meta, "prompt_tokens"))
            if cout > 0:
                judge_comp.append(float(cout))
            if cin > 0:
                judge_prompt.append(float(cin))
            n_judge_calls += 1
            for attempt in _meta_get(meta, "paid_attempts") or []:
                role = str(
                    attempt.get("role") if isinstance(attempt, dict) else ""
                )
                if "repair" in role or role == "corrective_retry":
                    n_repair_calls += 1
                if role == "verifier":
                    n_verifier_calls += 1

    if len(run_costs) < int(min_samples):
        return None

    repair_rate = (n_repair_calls / n_judge_calls) if n_judge_calls else 0.0
    verifier_rate = (n_verifier_calls / n_judge_calls) if n_judge_calls else 0.0
    return {
        "n_samples": len(run_costs),
        "run_cost_usd_typical": round(_median(run_costs), 6),
        "run_cost_usd_high": round(_percentile(run_costs, 0.9), 6),
        "judge_completion_tokens_typical": int(round(_median(judge_comp)))
        if judge_comp
        else None,
        "judge_completion_tokens_high": int(round(_percentile(judge_comp, 0.9)))
        if judge_comp
        else None,
        "judge_prompt_tokens_typical": int(round(_median(judge_prompt)))
        if judge_prompt
        else None,
        "candidate_completion_tokens_typical": int(round(_median(cand_comp)))
        if cand_comp
        else None,
        "candidate_prompt_tokens_typical": int(round(_median(cand_prompt)))
        if cand_prompt
        else None,
        "repair_rate_per_judge_call": round(repair_rate, 4),
        "verifier_rate_per_judge_call": round(verifier_rate, 4),
        "scored_keys": want_scored,
        "openrouter_keys": want_or,
    }


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


def _extraction_per_distinct_batch(
    artifacts: Sequence[RunArtifact],
    *,
    extraction_cost_usd: Optional[float] = None,
) -> float:
    """Charge gold extraction once per distinct batch_id (or prepare).

    Same-batch Multi → ×1. Portfolio / multi-batch history → sum once per batch.
    Empty batch_id groups collapse to a single ``__none__`` bucket (legacy).
    """
    if extraction_cost_usd is not None:
        return batch_shared_cost_usd(extraction_cost_usd)
    seen: Dict[str, float] = {}
    for a in artifacts:
        bid = str(getattr(a, "batch_id", None) or "").strip() or "__none__"
        if bid in seen:
            continue
        seen[bid] = extraction_from_artifact(a)
    return round(sum(seen.values()), 6)


def batch_total_cost_usd(
    artifacts: Sequence[RunArtifact],
    *,
    extraction_cost_usd: Optional[float] = None,
) -> float:
    """Sum of run costs + extraction once per distinct batch_id."""
    runs = round(sum(_f(getattr(a, "total_cost_usd", 0.0)) for a in artifacts), 6)
    shared = _extraction_per_distinct_batch(
        artifacts, extraction_cost_usd=extraction_cost_usd
    )
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
    shared = _extraction_per_distinct_batch(
        artifacts, extraction_cost_usd=extraction_cost_usd
    )
    total = round(run_sum + shared, 6)
    n_batches = len(
        {
            str(getattr(a, "batch_id", None) or "").strip() or "__none__"
            for a in artifacts
        }
    )
    return {
        "candidates_usd": round(cand_sum, 6) if cand_sum else run_sum,
        "judge_usd": round(judge_sum, 6),
        "extractor_usd": shared,
        "run_cost_usd": run_sum,
        "batch_shared_cost_usd": shared,
        "batch_total_cost_usd": total,
        "n_runs": len(artifacts),
        "n_batches": n_batches,
        "total_usd": total,
    }
