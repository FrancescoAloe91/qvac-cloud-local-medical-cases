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
from benchmark.gold import SCORING_VERSION, cohort_id as build_cohort_id, load_confirmed_gold
from benchmark.judge import (
    build_ranking,
    judge_candidates_parallel,
    systemic_judge_failure,
)
from benchmark.prompts import (
    CANDIDATE_MAX_OUTPUT_TOKENS,
    candidate_system,
    candidate_user,
    parse_candidate_answers,
)
from benchmark.qvac_variants import is_qvac_key, local_only_roster, merge_roster
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
    candidates = list(artifact_fields.get("candidates") or [])
    judgments = list(artifact_fields.get("judgments") or [])
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
        "candidate_temperature": 0.2 if track == "controlled" else None,
        "candidate_sampling": (
            "controlled_temperature_0.2"
            if track == "controlled"
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
            }
            for candidate in candidates
        ],
    }
    manifest.update(existing)
    artifact_fields["reproducibility"] = manifest
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
        "BENCHMARK_GOLD_EXTRACTOR_MODEL", "google/gemini-3.5-flash"
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
) -> Dict[str, Any]:
    """Length-aware per-model + judge cost estimate (USD).

    Scales with clinical case + gold text length (no high yaml floor that
    freezes the estimate for typical case sizes).
    ``local_only`` = 6 on-device GGUFs ($0 collect) + judge calls only.
    """
    est = cfg.get("estimate") or {}
    sys_u = candidate_system()
    user_u = candidate_user(case)
    gold = gold_reference or ""

    # Live length signal: stem/gold growth moves the estimate immediately.
    base_in = openrouter.estimate_tokens_from_text(sys_u, user_u, gold)
    cin = base_in + 80  # small framing overhead only

    cout_base = int(est.get("candidate_output_tokens", 900))
    # Longer cases → modestly longer answers (still an estimate)
    cout = max(400, int(cout_base * (0.65 + 0.35 * min(2.2, cin / 1000))))

    n_q = max(1, len(case.questions))
    # Judge sees stem + gold + rubric-ish prompt + one answer slice per Q
    judge_ctx = openrouter.estimate_tokens_from_text(case.stem, gold, sys_u)
    per_q_answer = max(200, cout // n_q)
    judge_in = judge_ctx + 350 + n_q * (per_q_answer + 220)
    judge_out = int(est.get("judge_output_tokens_per_question", 400)) * n_q

    if local_only:
        roster = local_only_roster()
    else:
        roster = merge_roster(
            list(cfg.get("candidates") or []),
            triple_qvac=bool(triple_qvac),
            include_qvac=bool(include_qvac),
        )

    per_model: List[Dict[str, Any]] = []
    cloud_keys = 0
    total = 0.0
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
            cloud_keys += 1  # still judged
            continue
        if provider != "openrouter":
            continue
        mid = c["model"]
        cost = openrouter.estimate_cost_usd(mid, cin, cout)
        total += cost
        cloud_keys += 1
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

    judge_cfg = cfg.get("judge") or {}
    judge_model = judge_cfg.get("model", "deepseek/deepseek-r1")
    # One judge call per scored candidate (cloud + qvac if included)
    n_judge_calls = cloud_keys
    judge_one = openrouter.estimate_cost_usd(judge_model, judge_in, judge_out)
    judge_total = judge_one * n_judge_calls
    total += judge_total
    jpin, jpout = openrouter.model_prices_per_mtok(judge_model)

    per_run = round(total, 6)
    chars = len(case.stem or "") + len(gold)
    return {
        "per_model": per_model,
        "judge": {
            "model": judge_model,
            "label": judge_cfg.get("display_label") or judge_model,
            "calls": n_judge_calls,
            "estimated_usd_per_call": round(judge_one, 6),
            "estimated_usd": round(judge_total, 6),
            "prompt_tokens_per_call": judge_in,
            "completion_tokens_per_call": judge_out,
            "price_in_per_mtok": jpin,
            "price_out_per_mtok": jpout,
        },
        "total_usd": per_run,
        "n": n,
        "total_usd_for_n": round(per_run * n, 6),
        "input_tokens_used_for_estimate": cin,
        "completion_tokens_used_for_estimate": cout,
        "chars_case_plus_gold": chars,
        "note": "Estimate tracks case/gold length live; actual billed cost from OpenRouter usage.",
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
    return {
        "per_case_per_run_usd": per_case,
        "breakdowns": breakdowns,
        "n": n,
        "estimated_total_usd": round(total, 4),
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
        "judge": (cfg.get("judge") or {}).get("display_label")
        or (cfg.get("judge") or {}).get("model"),
        "note": "Estimate uses length-aware tokens + models.yaml prices; actual may differ.",
    }


def _collect_candidate_once(
    case: Case,
    cand_cfg: Dict[str, Any],
    blind_id: str,
    on_event: EventCallback = None,
    benchmark_track: str = "controlled",
    api_key: Optional[str] = None,
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

    messages = [
        {"role": "system", "content": candidate_system()},
        {"role": "user", "content": candidate_user(case)},
    ]

    if provider == "openrouter":
        temperature = 0.2 if benchmark_track == "controlled" else None
        raw, meta = openrouter.chat_stream(
            model_id,
            messages,
            temperature=temperature,
            max_tokens=CANDIDATE_MAX_OUTPUT_TOKENS,
            on_token=on_token,
            display_label=display,
            api_key=api_key,
        )
    elif provider == "qvac":
        gguf = cand_cfg.get("gguf_path")
        if gguf:
            loaded = qvac_bridge.load_model(
                gguf,
                sampling=(
                    {"temp": 0.2, "top_k": 20, "top_p": 0.95}
                    if benchmark_track == "controlled"
                    else {}
                ),
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
        # Prefer structured chat messages so Instruct GGUFs (Qwen/Llama/Phi)
        # apply their embedded chat_template correctly.
        sys_p = candidate_system()
        user_p = candidate_user(case)
        prompt = sys_p + "\n\n" + user_p
        raw, meta = qvac_bridge.generate(
            prompt,
            on_token=on_token,
            display_label=display,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": user_p},
            ],
        )
    else:
        raw, meta = "", ModelCallMeta(
            model=model_id,
            provider=str(provider),
            display_label=display,
            error=f"Unknown provider: {provider}",
        )

    answers = parse_candidate_answers(case, raw) if raw else {}
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


def _collect_candidate(
    case: Case,
    cand_cfg: Dict[str, Any],
    blind_id: str,
    on_event: EventCallback = None,
    benchmark_track: str = "controlled",
    api_key: Optional[str] = None,
) -> CandidateAnswer:
    """Collect once, retrying only transport failure or explicit truncation."""
    first = _collect_candidate_once(
        case,
        cand_cfg,
        blind_id,
        on_event,
        benchmark_track,
        api_key,
    )
    transport_failure = bool(
        first.meta.error and openrouter.is_retryable_error(first.meta.error)
    )
    truncation = (first.meta.finish_reason or "").lower() in {
        "length",
        "max_tokens",
    }
    if not (transport_failure or truncation):
        return first
    _emit(
        on_event,
        {
            "type": "candidate_retry",
            "key": first.candidate_key,
            "reason": "transport" if transport_failure else "truncation",
        },
    )
    second = _collect_candidate_once(
        case,
        cand_cfg,
        blind_id,
        on_event,
        benchmark_track,
        api_key,
    )
    first_cost = float(first.meta.cost_usd or 0.0)
    second_cost = float(second.meta.cost_usd or 0.0)
    second.meta.cost_usd = round(first_cost + second_cost, 8)
    second.meta.prompt_tokens += first.meta.prompt_tokens
    second.meta.completion_tokens += first.meta.completion_tokens
    second.meta.latency_s = round(
        float(first.meta.latency_s or 0.0) + float(second.meta.latency_s or 0.0),
        3,
    )
    second.meta.retry_count = 1
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
                        if benchmark_track == "controlled"
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
) -> Dict[str, Any]:
    """Resolve case + candidate list + blind map for UI-driven runs."""
    case = load_case(case_id)
    cfg = load_models_config(models_path)
    yaml_cands = list(cfg.get("candidates") or [])
    # Sidecar HTTP up is enough — /load can hot-swap GGUFs before generate.
    qvac_sidecar = qvac_bridge.reachable() or qvac_bridge.available()
    include_qvac = (not skip_qvac) and qvac_sidecar
    candidates_cfg = merge_roster(
        yaml_cands,
        triple_qvac=bool(triple_qvac) and include_qvac,
        include_qvac=include_qvac,
    )
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
        "triple_qvac": bool(triple_qvac) and include_qvac,
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

    total_cost = 0.0
    for c in collected:
        if c.meta.cost_usd:
            total_cost += c.meta.cost_usd
    for j in judgments:
        if j.judge_meta.cost_usd:
            total_cost += j.judge_meta.cost_usd

    notes = ""
    if has_qvac_cfg and not any(is_qvac_key(c.candidate_key) for c in collected):
        notes = "QVAC skipped (sidecar unavailable). Start sidecar for full compare."

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
        },
        candidates=collected,
        judgments=judgments,
        ranking=ranking,
        total_cost_usd=round(total_cost, 6),
        notes=notes,
        cohort_id=cohort,
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
            "candidate_temperature": 0.2 if benchmark_track == "controlled" else None,
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
