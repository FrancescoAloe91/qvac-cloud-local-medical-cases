"""Run one or N benchmark iterations with parallel candidates + event callbacks."""

from __future__ import annotations

import random
import hashlib
import json
import os
import platform
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from benchmark import openrouter, qvac_bridge
from benchmark.cases_loader import load_case
from benchmark.config import load_models_config
from benchmark.gold import (
    SCORING_VERSION,
    cohort_id as build_cohort_id,
    execution_cohort_id as build_execution_cohort_id,
    load_confirmed_gold,
    uses_controlled_sampling,
)
from benchmark.judge import (
    build_ranking,
    judge_candidates_parallel,
    systemic_judge_failure,
)
from benchmark.prompts import (
    CANDIDATE_MAX_OUTPUT_TOKENS,
    candidate_system,
    candidate_user,
    format_repair_messages,
    is_prompt_template_echo,
    local_chat_messages,
    missing_section_ids,
    parse_candidate_answers,
)
from benchmark.qvac_variants import is_qvac_key, merge_roster
from benchmark.report import summarize_runs, write_artifact, write_summary
from benchmark.schema import (
    CandidateAnswer,
    Case,
    JudgeResult,
    ModelCallMeta,
    MultiRunSummary,
    RunArtifact,
    utc_now_iso,
)

# Blind IDs for the judge — never reuse Case A/B/C letters (those are clinical cases).
BLIND_LABELS = [
    "Candidate 1",
    "Candidate 2",
    "Candidate 3",
    "Candidate 4",
    "Candidate 5",
    "Candidate 6",
    "Candidate 7",
    "Candidate 8",
    "Candidate 9",
]

EventCallback = Optional[Callable[[Dict[str, Any]], None]]


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _git_sha() -> str:
    """Resolve HEAD without invoking git in deployed workers."""
    head = Path(__file__).resolve().parent.parent / ".git" / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
        if value.startswith("ref: "):
            ref = head.parent / value[5:]
            return ref.read_text(encoding="utf-8").strip()
        return value
    except OSError:
        return os.environ.get("GIT_COMMIT", "")


def _rehydrate_model(model_cls: type, value: Any):
    """Re-bind Pydantic instances to the current class (Streamlit reload safe)."""
    if value is None:
        return None
    if isinstance(value, model_cls):
        return model_cls.model_validate(value.model_dump())
    if isinstance(value, dict):
        return model_cls.model_validate(value)
    if hasattr(value, "model_dump"):
        return model_cls.model_validate(value.model_dump())
    raise TypeError(f"Cannot rehydrate {model_cls.__name__} from {type(value)!r}")


def build_run_artifact(
    *,
    config_snapshot: Dict[str, Any],
    blind_seed: Optional[int] = None,
    judge_temperature: float = 0.0,
    **artifact_fields: Any,
) -> RunArtifact:
    """Build equivalent Streamlit/CLI artifacts with one reproducibility manifest."""
    existing = dict(artifact_fields.pop("reproducibility", {}) or {})
    models_config = dict(artifact_fields.get("models_config") or {})
    # Streamlit script reloads can leave CandidateAnswer/JudgeResult instances from
    # a previous module identity; re-validate so RunArtifact construction never
    # raises model_type on live instances that are otherwise valid.
    candidates = [
        _rehydrate_model(CandidateAnswer, item)
        for item in list(artifact_fields.get("candidates") or [])
    ]
    judgments = [
        _rehydrate_model(JudgeResult, item)
        for item in list(artifact_fields.get("judgments") or [])
    ]
    artifact_fields["candidates"] = candidates
    artifact_fields["judgments"] = judgments
    # Callers (esp. Beta) may omit wall-clock stamps — never fail validation on that.
    now_iso = utc_now_iso()
    if not artifact_fields.get("started_at"):
        artifact_fields["started_at"] = now_iso
    if not artifact_fields.get("finished_at"):
        artifact_fields["finished_at"] = now_iso
    # RunArtifact has no n_total; tolerate Multi/Beta passing it as metadata.
    artifact_fields.pop("n_total", None)
    track = str(artifact_fields.get("benchmark_track") or "controlled")
    judge_cfg = models_config.get("judge") or config_snapshot.get("judge") or {}
    configured_candidates = {
        str(candidate.get("key") or ""): candidate
        for candidate in (models_config.get("candidates") or [])
        if isinstance(candidate, dict)
    }
    primary_judge = str(judge_cfg.get("model") or "")
    effective_judges = sorted(
        {str(judgment.judge_model) for judgment in judgments if judgment.judge_model}
    )
    verifier_activated = bool(
        primary_judge
        and any(model != primary_judge for model in effective_judges)
    )
    manifest = {
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "models_config_sha256": hashlib.sha256(
            json.dumps(config_snapshot, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "prompts_sha256": _file_sha256(
            Path(__file__).resolve().parent / "prompts.py"
        ),
        "scoring_sha256": _file_sha256(
            Path(__file__).resolve().parent / "scoring.py"
        ),
        "blind_seed": blind_seed,
        "benchmark_track": track,
        "candidate_temperature": 0.2 if uses_controlled_sampling(track) else None,
        "candidate_sampling": (
            "controlled_temperature_0.2"
            if uses_controlled_sampling(track)
            else "provider_parameters_omitted_where_supported"
        ),
        "candidate_max_output_tokens": CANDIDATE_MAX_OUTPUT_TOKENS,
        "judge_temperature": judge_temperature,
        "primary_judge": primary_judge,
        "effective_judge": (
            effective_judges[0]
            if len(effective_judges) == 1
            else ("mixed" if effective_judges else primary_judge)
        ),
        "effective_judges": effective_judges,
        "verifier_activated": verifier_activated,
        "retry_count": sum(judgment.retry_count for judgment in judgments),
        "failure_categories": {
            status: sum(1 for judgment in judgments if judgment.status == status)
            for status in sorted({judgment.status for judgment in judgments})
            if status != "valid"
        },
        "candidate_calls": [
            {
                "key": candidate.candidate_key,
                "requested_model": candidate.meta.requested_model
                or candidate.meta.model,
                "routed_model": candidate.meta.routed_model or candidate.meta.model,
                "routed_provider": candidate.meta.routed_provider
                or candidate.meta.provider,
                "finish_reason": candidate.meta.finish_reason,
                "prompt_tokens": candidate.meta.prompt_tokens,
                "completion_tokens": candidate.meta.completion_tokens,
                "context": (
                    configured_candidates.get(candidate.candidate_key, {}).get(
                        "context"
                    )
                    or configured_candidates.get(candidate.candidate_key, {}).get(
                        "n_ctx"
                    )
                    or configured_candidates.get(candidate.candidate_key, {}).get(
                        "context_size"
                    )
                ),
                "configured_sampling": configured_candidates.get(
                    candidate.candidate_key, {}
                ).get("sampling"),
                "ram_mb": candidate.meta.ram_mb,
                "gguf_mb": candidate.meta.gguf_mb,
                "gguf_sha256": getattr(candidate.meta, "gguf_sha256", "") or "",
                "configuration_deviation": bool(
                    getattr(candidate.meta, "configuration_deviation", False)
                ),
                # Real QVAC device (cpu/gpu/…); OS stays under reproducibility.platform
                "device": str(getattr(candidate.meta, "device", "") or ""),
                "gpu_layers": getattr(candidate.meta, "gpu_layers", None),
                "ctx_size": getattr(candidate.meta, "ctx_size", None),
                "predict": getattr(candidate.meta, "predict", None),
                "seed": getattr(candidate.meta, "seed", None),
                "temperature": getattr(candidate.meta, "temperature", None),
            }
            for candidate in candidates
        ],
    }
    manifest.update(existing)
    artifact_fields["reproducibility"] = manifest
    # Fill execution_cohort_id when gold JSON is available and not already set.
    if not artifact_fields.get("execution_cohort_id"):
        gold_ref = str(models_config.get("gold_reference") or "")
        case_stem = str(models_config.get("case_stem") or "")
        if gold_ref.strip().startswith("{"):
            try:
                from benchmark.gold import (
                    execution_cohort_id as _exec_cid,
                    load_confirmed_gold as _load_g,
                )

                artifact_fields["execution_cohort_id"] = _exec_cid(
                    case_stem=case_stem,
                    gold=_load_g(gold_ref),
                    prompt_version=str(
                        artifact_fields.get("prompt_version") or "gold-only-v1"
                    ),
                    benchmark_track=track,
                    candidates=candidates,
                    judgments=judgments,
                    scoring_version=str(
                        artifact_fields.get("scoring_version") or "graded-clinical-v4"
                    ),
                )
            except Exception:
                pass
    return RunArtifact(**artifact_fields)


def _emit(on_event: EventCallback, event: Dict[str, Any]) -> None:
    if on_event:
        on_event(event)


def _validate_judge_separation(
    cfg: Dict[str, Any], candidates_cfg: Sequence[Dict[str, Any]]
) -> None:
    judge_cfg = cfg.get("judge") or {}
    verifier = str(judge_cfg.get("verifier_model") or "").strip()
    if not verifier:
        return
    primary = str(judge_cfg.get("model") or "").strip()
    candidate_models = {
        str(candidate.get("model") or "").strip() for candidate in candidates_cfg
    }
    extractor = os.environ.get(
        "BENCHMARK_GOLD_EXTRACTOR_MODEL", "openai/gpt-4o-mini"
    ).strip()
    if verifier == primary or verifier in candidate_models:
        raise ValueError(
            "Verifier must be outside the primary judge and candidate roster"
        )
    if verifier.split("/", 1)[0] == extractor.split("/", 1)[0]:
        raise ValueError("Verifier must be outside the gold extractor model family")


def estimate_run_cost_usd(
    cfg: Dict[str, Any],
    case: Case,
    include_qvac: bool,
    *,
    triple_qvac: bool = False,
) -> float:
    return float(
        estimate_cost_breakdown(
            cfg, case, include_qvac=include_qvac, triple_qvac=triple_qvac
        )["total_usd"]
    )


def estimate_cost_breakdown(
    cfg: Dict[str, Any],
    case: Case,
    *,
    include_qvac: bool,
    gold_reference: str = "",
    n: int = 1,
    triple_qvac: bool = False,
    local_only: bool = False,
    include_local_peers: Optional[bool] = None,
    include_medical_peers: Optional[bool] = None,
    include_optional_legacy: bool = False,
    optional_legacy_keys: Optional[Sequence[str]] = None,
    include_extractor: bool = True,
    extraction_cost_usd: Optional[float] = None,
    history_artifacts: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Rough OpenRouter spend forecast (USD) — often slightly over.

    Baseline uses **typical** completion sizes (not ``max_tokens`` caps). The
    API cap (16k judge / 4k repair) is a ceiling for rare retries, not the
    expected bill. When recent History has comparable runs, baseline/upper are
    calibrated from actual ``total_cost_usd`` / native token usage.

    ``local_only`` = on-device GGUFs ($0 collect) + paid extractor/judge path.
    Extractor is charged once per batch when ``include_extractor``; candidates
    and judge scale with ``n``.
    """
    from benchmark.costing import cost_estimate_priors_from_artifacts
    from benchmark.gold import extraction_messages
    from benchmark.judge import _JUDGE_PRIMARY_MAX_TOKENS, _JUDGE_SECTION_MAX_TOKENS

    est = cfg.get("estimate") or {}
    sys_u = candidate_system()
    user_u = candidate_user(case)
    gold = gold_reference or ""
    n_runs = max(1, int(n))

    # Live length signal: stem/gold growth moves the estimate immediately.
    base_in = openrouter.estimate_tokens_from_text(sys_u, user_u, gold)
    cin = base_in + 80  # small framing overhead only

    # Typical cloud answer length (~2k observed); never treat max_tokens as billable.
    cout_base = int(est.get("candidate_output_tokens", 2000))
    cout = max(400, int(cout_base * (0.65 + 0.35 * min(2.2, cin / 1000))))

    n_q = max(1, len(case.questions))
    # Judge sees stem + gold + rubric-ish prompt + one answer slice per Q
    judge_ctx = openrouter.estimate_tokens_from_text(case.stem, gold, sys_u)
    per_q_answer = max(200, cout // n_q)
    judge_in = judge_ctx + 350 + n_q * (per_q_answer + 220)
    # Typical primary judge completion (~5.5k observed). Cap 16k is NOT expected.
    judge_out_typical = int(est.get("judge_output_tokens", 5500))
    judge_out_high = int(
        est.get("judge_output_tokens_high", max(judge_out_typical, 8500))
    )
    judge_out_typical = max(800, min(int(_JUDGE_PRIMARY_MAX_TOKENS), judge_out_typical))
    judge_out_high = max(
        judge_out_typical, min(int(_JUDGE_PRIMARY_MAX_TOKENS), judge_out_high)
    )

    _opt_kw = dict(
        include_optional_legacy=bool(include_optional_legacy),
        optional_legacy_keys=optional_legacy_keys,
    )
    if local_only:
        roster = merge_roster(
            [],
            triple_qvac=bool(triple_qvac),
            include_qvac=True,
            include_local_peers=(
                True if include_local_peers is None else bool(include_local_peers)
            ),
            include_medical_peers=(
                False
                if include_medical_peers is None
                else bool(include_medical_peers)
            ),
            **_opt_kw,
        )
    else:
        roster = merge_roster(
            list(cfg.get("candidates") or []),
            triple_qvac=bool(triple_qvac),
            include_qvac=bool(include_qvac),
            include_local_peers=include_local_peers,
            include_medical_peers=include_medical_peers,
            **_opt_kw,
        )

    per_model: List[Dict[str, Any]] = []
    scored_keys = 0
    openrouter_keys = 0
    candidates_usd = 0.0
    for c in roster:
        key = c.get("key")
        provider = c.get("provider")
        if provider == "qvac":
            per_model.append(
                {
                    "key": key,
                    "label": c.get("display_label") or c.get("label") or key,
                    "model": c.get("model"),
                    "provider": "qvac",
                    "estimated_usd": 0.0,
                    "prompt_tokens": cin,
                    "completion_tokens": cout,
                    "note": "local · $0 API",
                }
            )
            scored_keys += 1  # still judged
            continue
        if provider != "openrouter":
            continue
        mid = c["model"]
        cost = openrouter.estimate_cost_usd(mid, cin, cout)
        candidates_usd += cost
        scored_keys += 1
        openrouter_keys += 1
        pin, pout = openrouter.model_prices_per_mtok(mid)
        per_model.append(
            {
                "key": key,
                "label": c.get("display_label") or c.get("label") or key,
                "model": mid,
                "provider": "openrouter",
                "estimated_usd": round(cost, 6),
                "prompt_tokens": cin,
                "completion_tokens": cout,
                "price_in_per_mtok": pin,
                "price_out_per_mtok": pout,
            }
        )

    priors = cost_estimate_priors_from_artifacts(
        history_artifacts or [],
        scored_keys=scored_keys,
        openrouter_keys=openrouter_keys,
    )
    calibrated = bool(priors)
    if priors:
        if priors.get("candidate_completion_tokens_typical"):
            cout = int(priors["candidate_completion_tokens_typical"])
            # Recompute cloud candidate line items with calibrated outs.
            candidates_usd = 0.0
            for row in per_model:
                if row.get("provider") != "openrouter":
                    row["completion_tokens"] = cout
                    continue
                mid = str(row["model"])
                pin_tok = int(row.get("prompt_tokens") or cin)
                if priors.get("candidate_prompt_tokens_typical"):
                    pin_tok = int(priors["candidate_prompt_tokens_typical"])
                    row["prompt_tokens"] = pin_tok
                row["completion_tokens"] = cout
                cost = openrouter.estimate_cost_usd(mid, pin_tok, cout)
                row["estimated_usd"] = round(cost, 6)
                candidates_usd += cost
        if priors.get("judge_completion_tokens_typical"):
            judge_out_typical = int(priors["judge_completion_tokens_typical"])
        if priors.get("judge_completion_tokens_high"):
            judge_out_high = int(priors["judge_completion_tokens_high"])
        if priors.get("judge_prompt_tokens_typical"):
            judge_in = int(priors["judge_prompt_tokens_typical"])

    # Gold extractor: once per Prepare/Confirm batch (skip if already paid).
    extractor_model = os.environ.get(
        "BENCHMARK_GOLD_EXTRACTOR_MODEL", "openai/gpt-4o-mini"
    )
    extract_msgs = extraction_messages(gold if gold.strip() else "(empty reference)")
    extract_in = openrouter.estimate_tokens_from_text(
        *(str(m.get("content") or "") for m in extract_msgs)
    )
    extract_out = int(est.get("extractor_output_tokens", 900))
    if extraction_cost_usd is not None:
        extractor_one = max(0.0, float(extraction_cost_usd))
    elif include_extractor:
        extractor_one = openrouter.estimate_cost_usd(
            extractor_model, extract_in, extract_out
        )
    else:
        extractor_one = 0.0
    epin, epout = openrouter.model_prices_per_mtok(extractor_model)
    extractor_block = {
        "model": extractor_model,
        "label": f"Gold extractor · {extractor_model}",
        "calls": 1 if extractor_one > 0 else 0,
        "estimated_usd": round(extractor_one, 6),
        "prompt_tokens": extract_in,
        "completion_tokens": extract_out,
        "price_in_per_mtok": epin,
        "price_out_per_mtok": epout,
        "note": (
            "already paid / omitted"
            if extractor_one <= 0
            else "once per prepared/confirmed reference batch"
        ),
    }

    judge_cfg = cfg.get("judge") or {}
    judge_model = judge_cfg.get("model", "deepseek/deepseek-r1")
    n_judge_calls = scored_keys
    judge_one = openrouter.estimate_cost_usd(
        judge_model, judge_in, judge_out_typical
    )
    judge_total = judge_one * n_judge_calls
    judge_one_high = openrouter.estimate_cost_usd(
        judge_model, judge_in, judge_out_high
    )
    judge_total_high = judge_one_high * n_judge_calls
    jpin, jpout = openrouter.model_prices_per_mtok(judge_model)
    judge_block = {
        "model": judge_model,
        "label": judge_cfg.get("display_label") or judge_model,
        "calls": n_judge_calls,
        "estimated_usd_per_call": round(judge_one, 6),
        "estimated_usd": round(judge_total, 6),
        "prompt_tokens_per_call": judge_in,
        "completion_tokens_per_call": judge_out_typical,
        "completion_tokens_high_per_call": judge_out_high,
        "completion_tokens_cap": int(_JUDGE_PRIMARY_MAX_TOKENS),
        "price_in_per_mtok": jpin,
        "price_out_per_mtok": jpout,
        "note": (
            f"typical ~{judge_out_typical} completion tokens/call "
            f"(API cap {_JUDGE_PRIMARY_MAX_TOKENS} is not the expected bill)"
        ),
    }

    # Repair/verifier are rare; upper uses observed-ish rates, not full worst case.
    repair_out = int(_JUDGE_SECTION_MAX_TOKENS)
    repair_one_section = openrouter.estimate_cost_usd(
        judge_model, judge_in, repair_out
    )
    repair_rate = float(
        (priors or {}).get("repair_rate_per_judge_call")
        if priors
        else est.get("repair_rate_per_judge_call", 0.08)
    )
    repair_rate = min(1.0, max(0.0, repair_rate))
    # At most one section-repair call expected per judge call at that rate.
    repair_expected = repair_one_section * n_judge_calls * repair_rate
    repair_ceiling = repair_one_section * n_q * n_judge_calls
    repair_block = {
        "model": judge_model,
        "label": "Section repair (possible)",
        "calls_max": n_q * n_judge_calls,
        "sections_per_candidate": n_q,
        "estimated_usd": round(repair_expected, 6),
        "estimated_usd_ceiling": round(repair_ceiling, 6),
        "prompt_tokens_per_call": judge_in,
        "completion_tokens_per_call": repair_out,
        "price_in_per_mtok": jpin,
        "price_out_per_mtok": jpout,
        "optional": True,
        "note": (
            f"upper uses ~{repair_rate:.0%} of judge calls needing one "
            f"{repair_out}-token repair (not every section × every model)"
        ),
    }

    verifier_model = (
        (judge_cfg.get("verifier_model") or "").strip()
        or "qwen/qwen3.5-397b-a17b"
    )
    verifier_one = openrouter.estimate_cost_usd(
        verifier_model, judge_in, judge_out_typical
    )
    verifier_full = verifier_one * n_judge_calls
    verifier_rate = float(
        (priors or {}).get("verifier_rate_per_judge_call")
        if priors
        else est.get("verifier_rate_per_judge_call", 0.05)
    )
    verifier_rate = min(1.0, max(0.0, verifier_rate))
    verifier_expected = verifier_full * verifier_rate
    vpin, vpout = openrouter.model_prices_per_mtok(verifier_model)
    verifier_block = {
        "model": verifier_model,
        "label": f"Whole-run verifier (optional) · {verifier_model}",
        "calls": n_judge_calls,
        "estimated_usd_per_call": round(verifier_one, 6),
        "estimated_usd": round(verifier_expected, 6),
        "estimated_usd_full_cohort": round(verifier_full, 6),
        "prompt_tokens_per_call": judge_in,
        "completion_tokens_per_call": judge_out_typical,
        "price_in_per_mtok": vpin,
        "price_out_per_mtok": vpout,
        "optional": True,
        "note": (
            f"upper folds ~{verifier_rate:.0%} chance of systemic re-judge "
            "(not billed on every run)"
        ),
    }

    # Formula path: typical candidates + typical primary judge.
    per_iteration = candidates_usd + judge_total
    per_iteration_high = candidates_usd + judge_total_high + repair_expected + verifier_expected
    source = "formula_typical_tokens"
    if priors and priors.get("run_cost_usd_typical"):
        # Prefer actual OpenRouter spend from comparable History runs.
        per_iteration = float(priors["run_cost_usd_typical"])
        per_iteration_high = max(
            float(priors.get("run_cost_usd_high") or per_iteration),
            per_iteration,
        )
        # Keep a small buffer for case-length drift when not already above.
        per_iteration_high = max(
            per_iteration_high, per_iteration * 1.15 + repair_expected
        )
        source = "history_calibrated"

    baseline_one = per_iteration + extractor_one
    upper_one = per_iteration_high + extractor_one
    total_for_n = extractor_one + per_iteration * n_runs
    upper_for_n = extractor_one + per_iteration_high * n_runs

    chars = len(case.stem or "") + len(gold)
    return {
        "per_model": per_model,
        "extractor": extractor_block,
        "judge": judge_block,
        "section_repair": repair_block,
        "verifier": verifier_block,
        "total_usd": round(baseline_one, 6),
        "total_usd_upper": round(upper_one, 6),
        "n": n_runs,
        "total_usd_for_n": round(total_for_n, 6),
        "total_usd_upper_for_n": round(upper_for_n, 6),
        "input_tokens_used_for_estimate": cin,
        "completion_tokens_used_for_estimate": cout,
        "chars_case_plus_gold": chars,
        "calibrated": calibrated,
        "calibration_n": int((priors or {}).get("n_samples") or 0),
        "estimate_source": source,
        "reliability": "rough_estimate_often_over",
        "priors": priors,
        "note": (
            "Rough estimate · often over. Baseline uses typical completion sizes "
            f"(judge ~{judge_out_typical} tok, not the {_JUDGE_PRIMARY_MAX_TOKENS} "
            "API cap). Upper adds higher judge outs + uncommon repair/verifier. "
            + (
                f"Calibrated from {int(priors['n_samples'])} comparable History runs."
                if priors
                else "No comparable History yet — formula priors only. "
            )
            + " Billed truth = OpenRouter usage.cost."
        ),
    }


def dry_run_estimate(
    case_ids: Sequence[str],
    n: int,
    *,
    models_path: Optional[Path] = None,
    skip_qvac: bool = False,
    gold_reference: str = "",
    case_stem_override: str = "",
    triple_qvac: bool = False,
) -> Dict[str, Any]:
    cfg = load_models_config(models_path)
    include_qvac = (not skip_qvac) and (
        qvac_bridge.available() or qvac_bridge.reachable()
    )
    roster = merge_roster(
        list(cfg.get("candidates") or []),
        triple_qvac=bool(triple_qvac),
        include_qvac=include_qvac,
    )
    per_case = {}
    breakdowns = {}
    total = 0.0
    total_upper = 0.0
    for cid in case_ids:
        case = load_case(cid)
        if case_stem_override.strip():
            case = case.model_copy(update={"stem": case_stem_override.strip()})
        bd = estimate_cost_breakdown(
            cfg,
            case,
            include_qvac=include_qvac,
            gold_reference=gold_reference,
            n=n,
            triple_qvac=triple_qvac,
        )
        breakdowns[cid] = bd
        per_case[cid] = bd["total_usd"]
        total += bd["total_usd_for_n"]
        total_upper += float(bd.get("total_usd_upper_for_n") or bd["total_usd_for_n"])
    return {
        "per_case_per_run_usd": per_case,
        "breakdowns": breakdowns,
        "n": n,
        "estimated_total_usd": round(total, 4),
        "estimated_total_usd_upper": round(total_upper, 4),
        "qvac_included": include_qvac,
        "triple_qvac": bool(triple_qvac) and include_qvac,
        "profile": cfg.get("profile"),
        "candidates": [
            {
                "key": c.get("key"),
                "model": c.get("model"),
                "display_label": c.get("display_label"),
                "site": c.get("site"),
            }
            for c in roster
        ],
        "note": (
            "Baseline excludes possible section repair and optional whole-run "
            "verifier; see estimated_total_usd_upper and per-case breakdowns. "
            "Length-aware tokens + fallback/OpenRouter-ish prices; actual billed "
            "usage may differ."
        ),
        "judge": (cfg.get("judge") or {}).get("display_label")
        or (cfg.get("judge") or {}).get("model"),
    }


# Local section recovery must fail fast — sequential per-gap regenerations can
# hang the live UI on "Recovering sections…" for slow local GGUFs.
LOCAL_RECOVERY_TIMEOUT_S = 90.0


def _collect_candidate_once(
    case: Case,
    cand_cfg: Dict[str, Any],
    blind_id: str,
    on_event: EventCallback = None,
    benchmark_track: str = "controlled",
    api_key: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    timeout: Optional[float] = None,
    answer_parser: Optional[Callable[[Case, str], Dict[str, str]]] = None,
) -> CandidateAnswer:
    key = cand_cfg["key"]
    label = cand_cfg.get("label") or key
    display = cand_cfg.get("display_label") or label
    vendor = cand_cfg.get("vendor") or ""
    site = cand_cfg.get("site") or ""
    provider = cand_cfg.get("provider")
    model_id = str(cand_cfg.get("model") or "")

    _emit(
        on_event,
        {
            "type": "candidate_start",
            "key": key,
            "display_label": display,
            "vendor": vendor,
            "site": site,
            "model": model_id,
            "provider": provider,
        },
    )

    def on_token(delta: str) -> None:
        _emit(
            on_event,
            {"type": "candidate_token", "key": key, "delta": delta},
        )

    if messages is None:
        messages = [
            {"role": "system", "content": candidate_system()},
            {"role": "user", "content": candidate_user(case)},
        ]
    if provider == "qvac":
        messages = local_chat_messages(messages, cand_cfg)

    if provider == "openrouter":
        from benchmark.gold import is_strict_track, uses_controlled_sampling

        temperature = 0.2 if uses_controlled_sampling(benchmark_track) else None
        allowed = (
            list(cand_cfg.get("allowed_providers") or [])
            if uses_controlled_sampling(benchmark_track)
            else None
        )
        strict = is_strict_track(benchmark_track)
        raw, meta = openrouter.chat_stream(
            model_id,
            messages,
            temperature=temperature,
            max_tokens=CANDIDATE_MAX_OUTPUT_TOKENS,
            on_token=on_token,
            display_label=display,
            api_key=api_key,
            allowed_providers=allowed,
            require_parameters=strict,
            allow_fallbacks=not strict,
        )
        if strict and getattr(meta, "configuration_deviation", False):
            meta = meta.model_copy(
                update={
                    "error": (
                        meta.error
                        or "strict_controlled: routed provider outside allowed pin"
                    )
                }
            )
            raw = ""
    elif provider == "qvac":
        from benchmark.gold import is_strict_track, uses_controlled_sampling

        gguf = cand_cfg.get("gguf_path")
        loaded: Dict[str, Any] = {}
        sampling: Dict[str, Any] = {}
        if uses_controlled_sampling(benchmark_track):
            sampling = {"temp": 0.2, "top_k": 20, "top_p": 0.95}
            if is_strict_track(benchmark_track):
                # Deterministic recorded seed per candidate key within the run.
                seed_basis = f"{blind_id}:{key}:{model_id}"
                sampling["seed"] = int(
                    hashlib.sha256(seed_basis.encode("utf-8")).hexdigest()[:8], 16
                ) % (2**31 - 1)
        if gguf:
            loaded = qvac_bridge.load_model(
                gguf,
                sampling=sampling,
            )
            if not loaded.get("ok"):
                raw, meta = "", ModelCallMeta(
                    model=model_id,
                    provider="qvac",
                    display_label=display,
                    error=str(loaded.get("error") or f"Failed to load {gguf}"),
                    cost_usd=0.0,
                )
                answers = {}
                cand = CandidateAnswer(
                    candidate_key=key,
                    label=label,
                    display_label=display,
                    vendor=vendor,
                    site=site,
                    blind_id=blind_id,
                    answers=answers,
                    raw_response=raw,
                    meta=meta,
                )
                _emit(
                    on_event,
                    {
                        "type": "candidate_done",
                        "key": key,
                        "error": meta.error,
                        "meta": meta.model_dump(),
                        "text": raw,
                    },
                )
                return cand
        sys_p = str(messages[0].get("content") or "") if messages else candidate_system()
        user_p = (
            str(messages[1].get("content") or "")
            if messages and len(messages) > 1
            else candidate_user(case)
        )
        prompt = sys_p + "\n\n" + user_p
        gen_kwargs: Dict[str, Any] = {
            "on_token": on_token,
            "display_label": display,
            "messages": messages,
            "sampling": sampling or None,
        }
        if timeout is not None:
            gen_kwargs["timeout"] = float(timeout)
        raw, meta = qvac_bridge.generate(prompt, **gen_kwargs)
        digest = str(loaded.get("gguf_sha256") or "")
        updates = {}
        if digest:
            updates["gguf_sha256"] = digest
        if sampling.get("seed") is not None and meta.seed is None:
            updates["seed"] = sampling["seed"]
        if updates:
            meta = meta.model_copy(update=updates)
    else:
        raw, meta = "", ModelCallMeta(
            model=model_id,
            provider=str(provider),
            display_label=display,
            error=f"Unknown provider: {provider}",
        )

    parser = answer_parser or parse_candidate_answers
    answers = parser(case, raw) if raw else {}
    cand = CandidateAnswer(
        candidate_key=key,
        label=label,
        display_label=display,
        vendor=vendor,
        site=site,
        blind_id=blind_id,
        answers=answers,
        raw_response=raw,
        meta=meta,
    )
    _emit(
        on_event,
        {
            "type": "candidate_done",
            "key": key,
            "error": meta.error,
            "meta": meta.model_dump(),
            "text": raw,
        },
    )
    return cand


def is_retryable_local_error(err: str) -> bool:
    """Technical on-device failures worth one more attempt.

    A GGUF hot-swap, sidecar transport, or worker start-up can fail for reasons
    that clear on a second try; those are infrastructure faults, not evidence
    about the model's clinical ability.
    """
    text = (err or "").lower()
    return any(
        marker in text
        for marker in (
            "failed to load",
            "load failed",
            "gguf not found",
            "sidecar unreachable",
            "sidecar outdated",
            "did not become ready",
            "empty generation",
            "stream error",
            "worker",
            "rpc",
            "broken pipe",
            "connection refused",
            "urlopen error",
        )
    )


def _merge_collect_meta(first: CandidateAnswer, second: CandidateAnswer) -> None:
    first_cost = float(first.meta.cost_usd or 0.0)
    second_cost = float(second.meta.cost_usd or 0.0)
    second.meta.cost_usd = round(first_cost + second_cost, 8)
    second.meta.prompt_tokens += first.meta.prompt_tokens
    second.meta.completion_tokens += first.meta.completion_tokens
    second.meta.reasoning_tokens = int(second.meta.reasoning_tokens or 0) + int(
        first.meta.reasoning_tokens or 0
    )
    second.meta.latency_s = round(
        float(first.meta.latency_s or 0.0) + float(second.meta.latency_s or 0.0),
        3,
    )
    second.meta.retry_count = max(1, int(first.meta.retry_count or 0) + 1)
    prior = list(first.meta.prior_attempts or [])
    prior.append(
        {
            "error": first.meta.error or "",
            "status": "error" if first.meta.error else "superseded",
            "model": first.meta.model,
            "provider": first.meta.provider,
            "requested_model": first.meta.requested_model or "",
            "routed_model": first.meta.routed_model or first.meta.model or "",
            "routed_provider": first.meta.routed_provider or first.meta.provider or "",
            "latency_s": first.meta.latency_s,
            "finish_reason": first.meta.finish_reason or "",
            "prompt_tokens": int(first.meta.prompt_tokens or 0),
            "completion_tokens": int(first.meta.completion_tokens or 0),
            "reasoning_tokens": int(first.meta.reasoning_tokens or 0),
            "cost_usd": first_cost,
            "requested_providers": list(first.meta.requested_providers or []),
        }
    )
    second.meta.prior_attempts = prior + list(second.meta.prior_attempts or [])
    # Keep first-attempt paid_attempts trail when present.
    first_paid = list(first.meta.paid_attempts or [])
    if first_paid:
        second.meta.paid_attempts = first_paid + list(second.meta.paid_attempts or [])


def _collect_candidate(
    case: Case,
    cand_cfg: Dict[str, Any],
    blind_id: str,
    on_event: EventCallback = None,
    benchmark_track: str = "controlled",
    api_key: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    answer_parser: Optional[Callable[[Case, str], Dict[str, str]]] = None,
    allow_format_repair: bool = True,
) -> CandidateAnswer:
    """Collect once, then spend at most one bounded retry on a recoverable fault.

    Recoverable means transport failure, a local sidecar/GGUF fault, explicit
    truncation, or sections the model left unwritten. When the first reply has
    content but almost no parseable A# markers, one format-repair pass asks the
    same model to re-label that text (no new clinical facts). Otherwise missing
    sections regenerate only the affected questions in one targeted call.
    Local (qvac) recovery is hard-capped at one extra generate — format-repair
    XOR one multi-gap targeted call — never N sequential per-section regenerations.

    Beta comprehension sets ``allow_format_repair=False`` and supplies free-form
    ``messages`` + ``answer_parser`` so A1–A5 repair never runs.
    """
    first = _collect_candidate_once(
        case,
        cand_cfg,
        blind_id,
        on_event,
        benchmark_track,
        api_key,
        messages=messages,
        answer_parser=answer_parser,
    )
    return maybe_retry_candidate(
        case,
        first,
        cand_cfg,
        blind_id,
        on_event=on_event,
        benchmark_track=benchmark_track,
        api_key=api_key,
        messages=messages,
        answer_parser=answer_parser,
        allow_format_repair=allow_format_repair,
    )


def _empty_recovery_candidate(
    template: CandidateAnswer, *, error: str
) -> CandidateAnswer:
    """Synthetic empty recovery result used when a timed recovery call fails."""
    return template.model_copy(
        update={
            "answers": {},
            "raw_response": "",
            "meta": template.meta.model_copy(
                update={
                    "error": error,
                    "cost_usd": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_s": 0.0,
                    "finish_reason": "recovery_timeout",
                    "prior_attempts": [],
                    "retry_count": 0,
                }
            ),
        }
    )


def _recover_collect_once(
    case: Case,
    cand_cfg: Dict[str, Any],
    blind_id: str,
    on_event: EventCallback,
    benchmark_track: str,
    api_key: Optional[str],
    *,
    messages: Optional[List[Dict[str, str]]] = None,
    timeout: Optional[float] = None,
    template: CandidateAnswer,
    answer_parser: Optional[Callable[[Case, str], Dict[str, str]]] = None,
) -> CandidateAnswer:
    """One recovery generate with optional hard wall-clock timeout (local)."""
    if timeout is None:
        return _collect_candidate_once(
            case,
            cand_cfg,
            blind_id,
            on_event,
            benchmark_track,
            api_key,
            messages=messages,
            answer_parser=answer_parser,
        )
    # Wall-clock cap so a slow/hung GGUF cannot leave the UI on Recovering forever.
    # urllib timeout alone can stall if tokens trickle; FuturesTimeout aborts wait.
    # shutdown(wait=False) is required — default wait=True would re-hang on timeout.
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        fut = pool.submit(
            _collect_candidate_once,
            case,
            cand_cfg,
            blind_id,
            on_event,
            benchmark_track,
            api_key,
            messages,
            timeout,
            answer_parser,
        )
        try:
            # Wall clock ≈ recovery budget; small grace for thread scheduling only.
            return fut.result(timeout=float(timeout) + 1.0)
        except TimeoutError:
            return _empty_recovery_candidate(
                template, error="recovery_timeout: section recovery exceeded budget"
            )
        except Exception as exc:  # noqa: BLE001 — recovery must never hang the UI
            return _empty_recovery_candidate(
                template, error=f"recovery_failed: {exc}"[:200]
            )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def maybe_retry_candidate(
    case: Case,
    first: CandidateAnswer,
    cand_cfg: Dict[str, Any],
    blind_id: str,
    on_event: EventCallback = None,
    benchmark_track: str = "controlled",
    api_key: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    answer_parser: Optional[Callable[[Case, str], Dict[str, str]]] = None,
    allow_format_repair: bool = True,
) -> CandidateAnswer:
    """Run format-repair / section recovery on an already-collected candidate.

    Used by the CLI collector and by the live UI after the first streamed reply
    so both paths share the same gap-filling policy.

    Local (qvac) honesty cap: at most one recovery generate total — either one
    format-repair pass or one multi-gap targeted call (never both, never
    sequential per-section). Remaining gaps stay missing / N/A.
    """
    is_local = cand_cfg.get("provider") == "qvac"
    recovery_timeout = LOCAL_RECOVERY_TIMEOUT_S if is_local else None
    error_text = first.meta.error or ""
    transport_failure = bool(
        error_text
        and (
            openrouter.is_retryable_error(error_text)
            or is_retryable_local_error(error_text)
        )
    )
    truncation = (first.meta.finish_reason or "").lower() in {
        "length",
        "max_tokens",
    }
    missing = missing_section_ids(case, first.answers or {})
    section_gap = bool(missing) and not error_text
    raw_blob = (first.raw_response or "").strip()
    # Short local replies still need re-labeling when almost no sections parsed.
    # Skip format-repair when the model mostly echoed the prompt template — that
    # content has no clinical substance to re-label; targeted regen is better.
    needs_format_repair = (
        allow_format_repair
        and section_gap
        and len(first.answers or {}) < 2
        and len(raw_blob) > 40
        and not is_prompt_template_echo(raw_blob, case)
    )
    # Beta / free-form: never chase graded A# gaps; only retry transport/truncation.
    if not allow_format_repair:
        section_gap = False
    if not (transport_failure or truncation or section_gap):
        return first

    if needs_format_repair:
        _emit(
            on_event,
            {
                "type": "candidate_retry",
                "key": first.candidate_key,
                "reason": "format repair",
                "missing_sections": list(missing),
            },
        )
        repaired = _recover_collect_once(
            case,
            cand_cfg,
            blind_id,
            on_event,
            benchmark_track,
            api_key,
            messages=format_repair_messages(case, raw_blob),
            timeout=recovery_timeout,
            template=first,
        )
        _merge_collect_meta(first, repaired)
        repaired.raw_response = (
            raw_blob
            + "\n\n[FORMAT REPAIR]\n"
            + (repaired.raw_response or "").lstrip()
        ).strip()
        if not missing_section_ids(case, repaired.answers or {}):
            return repaired
        if is_local:
            # Local budget spent: do not stack a second targeted call.
            if repaired.answers:
                return repaired
            return first.model_copy(
                update={
                    "meta": first.meta.model_copy(
                        update={
                            "cost_usd": repaired.meta.cost_usd,
                            "prompt_tokens": repaired.meta.prompt_tokens,
                            "completion_tokens": repaired.meta.completion_tokens,
                            "latency_s": repaired.meta.latency_s,
                            "retry_count": repaired.meta.retry_count,
                        }
                    )
                }
            )
        if repaired.answers:
            # Cloud: partial labels recovered — keep and try one targeted fill.
            first = repaired
            missing = missing_section_ids(case, first.answers or {})
            section_gap = bool(missing) and not (first.meta.error or "")
            truncation = False
            transport_failure = False
            if not section_gap:
                return first
        else:
            # Format repair failed; fall through to clinical gap recovery.
            first = first.model_copy(
                update={
                    "meta": first.meta.model_copy(
                        update={
                            "cost_usd": repaired.meta.cost_usd,
                            "prompt_tokens": repaired.meta.prompt_tokens,
                            "completion_tokens": repaired.meta.completion_tokens,
                            "latency_s": repaired.meta.latency_s,
                            "retry_count": repaired.meta.retry_count,
                        }
                    )
                }
            )

    if transport_failure:
        reason = "transport"
    elif truncation:
        reason = "truncation"
    else:
        reason = "missing sections"
    _emit(
        on_event,
        {
            "type": "candidate_retry",
            "key": first.candidate_key,
            "reason": reason,
            "missing_sections": list(missing),
        },
    )
    targeted = (truncation or section_gap) and not transport_failure
    if not targeted:
        second = _recover_collect_once(
            case,
            cand_cfg,
            blind_id,
            on_event,
            benchmark_track,
            api_key,
            messages=messages,
            timeout=recovery_timeout,
            template=first,
            answer_parser=answer_parser,
        )
        _merge_collect_meta(first, second)
        return second

    target_questions = [
        question for question in case.questions if question.id in set(missing)
    ] or [case.questions[-1]]
    # One multi-gap targeted call for everyone (cloud and local). Sequential
    # per-section local recovery was removed — it hung the UI on slow GGUFs and
    # violated the ≤1-retry honesty disclosure.
    recovery_case = case.model_copy(update={"questions": target_questions})
    target_question_ids = {question.id for question in target_questions}
    second = _recover_collect_once(
        recovery_case,
        cand_cfg,
        blind_id,
        on_event,
        benchmark_track,
        api_key,
        messages=messages,
        timeout=recovery_timeout,
        template=first,
        answer_parser=answer_parser,
    )
    _merge_collect_meta(first, second)
    # Timed-out / failed recovery: keep first answers; leave gaps as N/A.
    if second.meta.error and not (second.answers or {}):
        first.meta.cost_usd = second.meta.cost_usd
        first.meta.prompt_tokens = second.meta.prompt_tokens
        first.meta.completion_tokens = second.meta.completion_tokens
        first.meta.latency_s = second.meta.latency_s
        first.meta.retry_count = second.meta.retry_count
        first.meta.prior_attempts = second.meta.prior_attempts
        return first
    merged_answers = dict(first.answers or {})
    for question_id in target_question_ids:
        recovered = (second.answers or {}).get(question_id)
        if (recovered or "").strip():
            merged_answers[question_id] = recovered
    second.answers = merged_answers
    second.raw_response = (
        (first.raw_response or "").rstrip()
        + "\n\n[TARGETED SECTION RECOVERY]\n"
        + (second.raw_response or "").lstrip()
    ).strip()
    if not missing_section_ids(case, second.answers):
        second.meta.finish_reason = first.meta.finish_reason
    # Clear recovery transport error if we salvaged any sections from first.
    if second.meta.error and merged_answers:
        second.meta.error = None
    return second


def iter_collect_parallel(
    case: Case,
    candidates_cfg: List[Dict[str, Any]],
    blind_map: Dict[str, str],
    on_event: EventCallback = None,
    benchmark_track: str = "controlled",
    api_key: Optional[str] = None,
):
    """Yield CandidateAnswer as workers finish.

    Cloud candidates run in parallel; QVAC slots run sequentially (one GGUF at a time).
    """
    cloud = [c for c in candidates_cfg if c.get("provider") != "qvac"]
    qvac_list = [c for c in candidates_cfg if c.get("provider") == "qvac"]

    with ThreadPoolExecutor(max_workers=max(1, len(cloud) or 1)) as pool:
        futures = {
            pool.submit(
                _collect_candidate,
                case,
                c,
                blind_map[c["key"]],
                on_event,
                benchmark_track,
                api_key,
            ): c["key"]
            for c in cloud
        }
        for fut in as_completed(futures):
            yield fut.result()

    for c in qvac_list:
        yield _collect_candidate(
            case, c, blind_map[c["key"]], on_event, benchmark_track, api_key
        )


def iter_collect_live(
    case: Case,
    candidates_cfg: List[Dict[str, Any]],
    blind_map: Dict[str, str],
    benchmark_track: str = "controlled",
    api_key: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    answer_parser: Optional[Callable[[Case, str], Dict[str, str]]] = None,
    allow_format_repair: bool = True,
):
    """Cloud parallel + sequential QVAC with live token events for UI.

    Yields dicts:
      {"type": "token", "key", "delta", "chars", "ttft_s", "elapsed_s", "tps_live"}
      {"type": "done", "candidate": CandidateAnswer}
    """
    import queue
    import threading
    import time as _time

    q: queue.Queue = queue.Queue()
    t0_global = _time.time()
    first_token_at: Dict[str, float] = {}
    start_at: Dict[str, float] = {}
    char_count: Dict[str, int] = {}
    lock = threading.Lock()

    def worker(cand_cfg: Dict[str, Any]) -> None:
        key = cand_cfg["key"]
        run_cfg = dict(cand_cfg)

        # Live TTFT must exclude GGUF load (can be minutes). Load first, then start clock.
        if cand_cfg.get("provider") == "qvac":
            gguf = cand_cfg.get("gguf_path")
            if gguf:
                loaded = qvac_bridge.load_model(
                    gguf,
                    sampling=(
                        {"temp": 0.2, "top_k": 20, "top_p": 0.95}
                        if uses_controlled_sampling(benchmark_track)
                        else {}
                    ),
                )
                if not loaded.get("ok"):
                    from benchmark.schema import CandidateAnswer, ModelCallMeta

                    q.put(
                        {
                            "type": "done",
                            "candidate": CandidateAnswer(
                                candidate_key=key,
                                label=str(cand_cfg.get("label") or key),
                                display_label=str(
                                    cand_cfg.get("display_label")
                                    or cand_cfg.get("label")
                                    or ""
                                ),
                                vendor=str(cand_cfg.get("vendor") or ""),
                                site=str(cand_cfg.get("site") or ""),
                                blind_id=blind_map[key],
                                answers={},
                                raw_response="",
                                meta=ModelCallMeta(
                                    model=str(cand_cfg.get("model") or ""),
                                    provider="qvac",
                                    display_label=str(
                                        cand_cfg.get("display_label")
                                        or cand_cfg.get("label")
                                        or ""
                                    ),
                                    error=str(
                                        loaded.get("error") or f"Failed to load {gguf}"
                                    ),
                                    cost_usd=0.0,
                                ),
                            ),
                        }
                    )
                    return
                # Already loaded — skip second /load inside _collect_candidate
                run_cfg["gguf_path"] = None
            with lock:
                start_at[key] = _time.time()
        else:
            with lock:
                start_at[key] = _time.time()

        def on_event(evt: Dict[str, Any]) -> None:
            if evt.get("type") == "candidate_retry":
                with lock:
                    first_token_at.pop(key, None)
                    char_count[key] = 0
                    start_at[key] = _time.time()
                q.put(
                    {
                        "type": "retry",
                        "key": key,
                        "reason": evt.get("reason") or "retryable failure",
                    }
                )
                return
            if evt.get("type") != "candidate_token":
                return
            delta = evt.get("delta") or ""
            if not delta:
                return
            now = _time.time()
            with lock:
                if key not in first_token_at:
                    first_token_at[key] = now
                char_count[key] = char_count.get(key, 0) + len(delta)
                ft = first_token_at[key]
                started = start_at.get(key, t0_global)
                chars = char_count[key]
            ttft = round(ft - started, 3)
            elapsed = max(now - started, 0.05)
            # Rough live TPS from chars/4 tokens after first token
            gen = max(now - ft, 0.05)
            approx_tok = max(1, chars // 4)
            tps_live = round(approx_tok / gen, 1)
            q.put(
                {
                    "type": "token",
                    "key": key,
                    "delta": delta,
                    "chars": chars,
                    "ttft_s": ttft,
                    "elapsed_s": round(elapsed, 2),
                    "tps_live": tps_live,
                }
            )

        try:
            cand = _collect_candidate(
                case,
                run_cfg,
                blind_map[cand_cfg["key"]],
                on_event,
                benchmark_track,
                api_key,
                messages=messages,
                answer_parser=answer_parser,
                allow_format_repair=allow_format_repair,
            )
            q.put({"type": "done", "candidate": cand})
        except Exception as exc:
            # Surface as failed candidate shell
            from benchmark.schema import CandidateAnswer, ModelCallMeta

            q.put(
                {
                    "type": "done",
                    "candidate": CandidateAnswer(
                        candidate_key=cand_cfg["key"],
                        label=str(cand_cfg.get("label") or cand_cfg["key"]),
                        display_label=str(
                            cand_cfg.get("display_label") or cand_cfg.get("label") or ""
                        ),
                        vendor=str(cand_cfg.get("vendor") or ""),
                        site=str(cand_cfg.get("site") or ""),
                        blind_id=blind_map[cand_cfg["key"]],
                        answers={},
                        raw_response="",
                        meta=ModelCallMeta(
                            model=str(cand_cfg.get("model") or ""),
                            provider=str(cand_cfg.get("provider") or ""),
                            error=str(exc),
                            cost_usd=0.0,
                        ),
                    ),
                }
            )

    cloud = [c for c in candidates_cfg if c.get("provider") != "qvac"]
    qvac_list = [c for c in candidates_cfg if c.get("provider") == "qvac"]

    def qvac_sequence() -> None:
        for c in qvac_list:
            worker(c)

    threads = [
        threading.Thread(target=worker, args=(c,), daemon=True) for c in cloud
    ]
    if qvac_list:
        threads.append(threading.Thread(target=qvac_sequence, daemon=True))
    for t in threads:
        t.start()

    remaining = len(candidates_cfg)
    while remaining > 0:
        try:
            evt = q.get(timeout=0.15)
        except queue.Empty:
            continue
        if evt.get("type") == "done":
            remaining -= 1
        yield evt

    for t in threads:
        t.join(timeout=1.0)


def prepare_run(
    case_id: str,
    *,
    models_path: Optional[Path] = None,
    skip_qvac: bool = False,
    require_qvac: bool = False,
    seed: Optional[int] = None,
    triple_qvac: bool = False,
    include_local_peers: Optional[bool] = None,
    include_medical_peers: Optional[bool] = None,
    include_optional_legacy: bool = False,
    optional_legacy_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Resolve case + candidate list + blind map for UI-driven runs."""
    case = load_case(case_id)
    cfg = load_models_config(models_path)
    yaml_cands = list(cfg.get("candidates") or [])
    # Sidecar HTTP up is enough — /load can hot-swap GGUFs before generate.
    qvac_sidecar = qvac_bridge.reachable() or qvac_bridge.available()
    include_medpsy = (not skip_qvac) and qvac_sidecar
    candidates_cfg = merge_roster(
        yaml_cands,
        triple_qvac=bool(triple_qvac) and include_medpsy,
        include_qvac=include_medpsy,
        include_local_peers=include_local_peers,
        include_medical_peers=include_medical_peers,
        include_optional_legacy=bool(include_optional_legacy),
        optional_legacy_keys=optional_legacy_keys,
    )
    candidates_cfg = [
        c
        for c in candidates_cfg
        if c.get("provider") != "qvac" or c.get("gguf_ready", True)
    ]
    _validate_judge_separation(cfg, candidates_cfg)
    has_qvac_cfg = any(c.get("provider") == "qvac" for c in candidates_cfg)

    if require_qvac and not has_qvac_cfg:
        raise RuntimeError(
            "QVAC SDK sidecar is required for demo mode but is offline. "
            "Start it with: cd sidecar && npm start"
        )

    rng = random.Random(seed if seed is not None else uuid.uuid4().int)
    order = list(range(len(candidates_cfg)))
    rng.shuffle(order)
    blind_map = {
        candidates_cfg[i]["key"]: BLIND_LABELS[j % len(BLIND_LABELS)]
        for j, i in enumerate(order)
    }
    return {
        "case": case,
        "cfg": cfg,
        "candidates_cfg": candidates_cfg,
        "blind_map": blind_map,
        "has_qvac_cfg": has_qvac_cfg,
        "triple_qvac": bool(triple_qvac) and include_medpsy,
        "include_local_peers": include_local_peers,
        "include_medical_peers": include_medical_peers,
        "optional_legacy_keys": list(optional_legacy_keys or ())
        if optional_legacy_keys is not None
        else (["local_gemma", "local_llama", "qvac_4b_q8"] if include_optional_legacy else []),
    }


def run_once(
    case_id: str,
    *,
    models_path: Optional[Path] = None,
    skip_qvac: bool = False,
    require_qvac: bool = False,
    n_index: int = 1,
    seed: Optional[int] = None,
    on_event: EventCallback = None,
    gold_reference: str = "",
    case_stem_override: str = "",
    triple_qvac: bool = False,
    benchmark_track: str = "controlled",
    api_key: Optional[str] = None,
    batch_id: str = "",
) -> RunArtifact:
    gold_contract = load_confirmed_gold(gold_reference)
    prep = prepare_run(
        case_id,
        models_path=models_path,
        skip_qvac=skip_qvac,
        require_qvac=require_qvac,
        seed=seed,
        triple_qvac=triple_qvac,
    )
    case = prep["case"]
    if case_stem_override.strip():
        case = case.model_copy(update={"stem": case_stem_override.strip()})
    cfg = prep["cfg"]
    candidates_cfg = prep["candidates_cfg"]
    blind_map = prep["blind_map"]
    has_qvac_cfg = prep["has_qvac_cfg"]
    started = utc_now_iso()
    run_id = f"{case_id}-{uuid.uuid4().hex[:10]}"
    cohort = build_cohort_id(
        case_stem=case.stem,
        gold=gold_contract,
        prompt_version="gold-only-v1",
        model_config={
            "candidates": candidates_cfg,
            "judge": cfg.get("judge") or {},
        },
        benchmark_track=benchmark_track,
    )

    _emit(
        on_event,
        {
            "type": "phase",
            "phase": "collecting",
            "message": "Collecting answers from cloud LLMs via OpenRouter + QVAC…",
            "candidates": [
                {
                    "key": c["key"],
                    "display_label": c.get("display_label") or c.get("label"),
                    "vendor": c.get("vendor"),
                    "site": c.get("site"),
                    "model": c.get("model"),
                    "provider": c.get("provider"),
                }
                for c in candidates_cfg
            ],
            "judge": (cfg.get("judge") or {}).get("display_label")
            or (cfg.get("judge") or {}).get("model"),
        },
    )

    collected_map: Dict[str, CandidateAnswer] = {}
    for cand in iter_collect_parallel(
        case,
        candidates_cfg,
        blind_map,
        on_event,
        benchmark_track=benchmark_track,
        api_key=api_key,
    ):
        collected_map[cand.candidate_key] = cand

    collected = [
        collected_map[c["key"]] for c in candidates_cfg if c["key"] in collected_map
    ]

    judge_cfg = cfg.get("judge") or {}
    judge_model = judge_cfg.get("model", "deepseek/deepseek-r1")
    judge_temp = float(judge_cfg.get("temperature", 0))

    _emit(
        on_event,
        {
            "type": "phase",
            "phase": "judging",
            "message": (
                f"Judging blind ({', '.join(blind_map.values())}) with "
                f"{judge_cfg.get('display_label') or judge_model}…"
            ),
        },
    )

    judgments: List[JudgeResult] = judge_candidates_parallel(
        case,
        collected,
        judge_model,
        temperature=judge_temp,
        gold_reference=gold_reference,
        api_key=api_key,
        verifier_model=str(judge_cfg.get("verifier_model") or ""),
        benchmark_track=benchmark_track,
        judge_allowed_providers=list(judge_cfg.get("allowed_providers") or []) or None,
        verifier_allowed_providers=list(
            judge_cfg.get("verifier_allowed_providers") or []
        )
        or None,
    )
    for j in judgments:
        _emit(
            on_event,
            {
                "type": "judge_done",
                "key": j.candidate_key,
                "blind_id": j.blind_id,
                "accuracy": j.weighted_accuracy,
            },
        )

    ranking = build_ranking(judgments)
    label_by_key = {c.candidate_key: c.display_label or c.label for c in collected}
    meta_by_key = {c.candidate_key: c.meta for c in collected}
    for row in ranking:
        row["label"] = label_by_key.get(row["key"], row["key"])
        m = meta_by_key.get(row["key"])
        if m:
            row["ttft_s"] = m.ttft_s
            row["tps"] = m.tps
            row["latency_s"] = m.latency_s
            row["cost_usd"] = m.cost_usd
            row["model"] = m.model
            if m.ram_mb is not None:
                row["ram_mb"] = m.ram_mb
            if m.gguf_mb is not None:
                row["gguf_mb"] = m.gguf_mb

    extraction_cost = 0.0
    try:
        if gold_reference and gold_reference.strip().startswith("{"):
            gold_obj = load_confirmed_gold(gold_reference)
            extraction_cost = float(getattr(gold_obj, "extraction_cost_usd", 0.0) or 0.0)
    except Exception:
        extraction_cost = 0.0
    from benchmark.costing import cost_breakdown_for_run, run_cost_usd

    # Artifact total = run cost only; extraction is batch-shared (once per Prepare).
    total_cost = run_cost_usd(collected, judgments)
    cost_breakdown = cost_breakdown_for_run(
        collected, judgments, extraction_cost_usd=extraction_cost
    )

    notes = ""
    if has_qvac_cfg and not any(is_qvac_key(c.candidate_key) for c in collected):
        notes = "QVAC skipped (sidecar unavailable). Start sidecar for full compare."

    exec_cohort = ""
    try:
        if gold_reference and gold_reference.strip().startswith("{"):
            exec_cohort = build_execution_cohort_id(
                case_stem=case.stem,
                gold=load_confirmed_gold(gold_reference),
                prompt_version="gold-only-v1",
                benchmark_track=benchmark_track,
                candidates=collected,
                judgments=judgments,
            )
    except Exception:
        exec_cohort = ""

    artifact = build_run_artifact(
        config_snapshot=cfg,
        blind_seed=seed,
        judge_temperature=judge_temp,
        run_id=run_id,
        case_id=case_id,
        started_at=started,
        finished_at=utc_now_iso(),
        n_index=n_index,
        batch_id=batch_id or uuid.uuid4().hex,
        models_config={
            "profile": cfg.get("profile"),
            "candidates": candidates_cfg,
            "judge": judge_cfg,
            "blind_map": blind_map,
            "gold_reference": gold_reference.strip() if gold_reference else "",
            "case_stem": case.stem,
            "extraction_cost_usd": extraction_cost,
        },
        candidates=collected,
        judgments=judgments,
        ranking=ranking,
        total_cost_usd=round(total_cost, 6),
        cost_breakdown=cost_breakdown,
        notes=notes,
        cohort_id=cohort,
        execution_cohort_id=exec_cohort,
        scoring_version=SCORING_VERSION,
        prompt_version="gold-only-v1",
        benchmark_track=benchmark_track,
        run_status=(
            "cancelled"
            if any(j.status == "cancelled" for j in judgments)
            else (
                "complete"
                if all(j.status == "valid" for j in judgments)
                else "partial"
            )
        ),
        reproducibility={
            "git_sha": _git_sha(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "models_config_sha256": hashlib.sha256(
                json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
            "prompts_sha256": _file_sha256(
                Path(__file__).resolve().parent / "prompts.py"
            ),
            "scoring_sha256": _file_sha256(
                Path(__file__).resolve().parent / "scoring.py"
            ),
            "blind_seed": seed,
            "benchmark_track": benchmark_track,
            "candidate_temperature": (
                0.2 if uses_controlled_sampling(benchmark_track) else None
            ),
            "judge_temperature": judge_temp,
        },
    )

    _emit(
        on_event,
        {
            "type": "phase",
            "phase": "done",
            "message": "Ranking ready — raw weighted scores, no winner-to-100 rescale.",
            "ranking": ranking,
            "total_cost_usd": artifact.total_cost_usd,
        },
    )
    return artifact


def run_n(
    case_id: str,
    n: int = 1,
    *,
    models_path: Optional[Path] = None,
    skip_qvac: bool = False,
    require_qvac: bool = False,
    out_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    on_event: EventCallback = None,
    gold_reference: str = "",
    case_stem_override: str = "",
    triple_qvac: bool = False,
    benchmark_track: str = "controlled",
    api_key: Optional[str] = None,
) -> tuple[List[RunArtifact], MultiRunSummary]:
    load_confirmed_gold(gold_reference)
    if out_dir is None:
        from benchmark.workspace import scoped_artifacts_dir

        out = scoped_artifacts_dir()
    else:
        out = out_dir
    out.mkdir(parents=True, exist_ok=True)
    artifacts: List[RunArtifact] = []
    batch_id = uuid.uuid4().hex
    base_seed = seed if seed is not None else random.randint(0, 10**9)
    for i in range(1, n + 1):
        art = run_once(
            case_id,
            models_path=models_path,
            skip_qvac=skip_qvac,
            require_qvac=require_qvac,
            n_index=i,
            seed=base_seed + i,
            on_event=on_event,
            gold_reference=gold_reference,
            case_stem_override=case_stem_override,
            triple_qvac=triple_qvac,
            benchmark_track=benchmark_track,
            api_key=api_key,
            batch_id=batch_id,
        )
        write_artifact(art, out)
        artifacts.append(art)
        if n > 1 and systemic_judge_failure(art.judgments):
            art.notes = (
                (art.notes + " | " if art.notes else "")
                + f"Multi aborted after run {i}/{n}: systemic judge failure"
            )
            write_artifact(art, out)
            _emit(
                on_event,
                {
                    "type": "phase",
                    "phase": "aborted",
                    "message": art.notes,
                },
            )
            break
    summary = summarize_runs(artifacts)
    write_summary(summary, out)
    return artifacts, summary
