"""Run one or N benchmark iterations with parallel candidates + event callbacks."""

from __future__ import annotations

import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from benchmark import openrouter, qvac_bridge
from benchmark.cases_loader import load_case
from benchmark.config import ARTIFACTS_DIR, load_models_config
from benchmark.judge import build_ranking, judge_candidates_parallel
from benchmark.prompts import candidate_system, candidate_user, parse_candidate_answers
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
]

EventCallback = Optional[Callable[[Dict[str, Any]], None]]


def _emit(on_event: EventCallback, event: Dict[str, Any]) -> None:
    if on_event:
        on_event(event)


def estimate_run_cost_usd(cfg: Dict[str, Any], case: Case, include_qvac: bool) -> float:
    return float(estimate_cost_breakdown(cfg, case, include_qvac=include_qvac)["total_usd"])


def estimate_cost_breakdown(
    cfg: Dict[str, Any],
    case: Case,
    *,
    include_qvac: bool,
    gold_reference: str = "",
    n: int = 1,
) -> Dict[str, Any]:
    """Length-aware per-model + judge cost estimate (USD).

    Scales with clinical case + gold text length (no high yaml floor that
    freezes the estimate for typical case sizes).
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

    per_model: List[Dict[str, Any]] = []
    cloud_keys = 0
    total = 0.0
    for c in cfg.get("candidates") or []:
        key = c.get("key")
        provider = c.get("provider")
        if provider == "qvac":
            if not include_qvac:
                continue
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
) -> Dict[str, Any]:
    cfg = load_models_config(models_path)
    include_qvac = (not skip_qvac) and qvac_bridge.available()
    per_case = {}
    breakdowns = {}
    total = 0.0
    for cid in case_ids:
        case = load_case(cid)
        if case_stem_override.strip():
            case = case.model_copy(update={"stem": case_stem_override.strip()})
        bd = estimate_cost_breakdown(
            cfg, case, include_qvac=include_qvac, gold_reference=gold_reference, n=n
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
        "profile": cfg.get("profile"),
        "candidates": [
            {
                "key": c.get("key"),
                "model": c.get("model"),
                "display_label": c.get("display_label"),
                "site": c.get("site"),
            }
            for c in (cfg.get("candidates") or [])
            if c.get("provider") != "qvac" or include_qvac
        ],
        "judge": (cfg.get("judge") or {}).get("display_label")
        or (cfg.get("judge") or {}).get("model"),
        "note": "Estimate uses length-aware tokens + models.yaml prices; actual may differ.",
    }


def _collect_candidate(
    case: Case,
    cand_cfg: Dict[str, Any],
    blind_id: str,
    on_event: EventCallback = None,
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
        raw, meta = openrouter.chat_stream(
            model_id,
            messages,
            temperature=0.3,
            max_tokens=2048,
            on_token=on_token,
            display_label=display,
        )
    elif provider == "qvac":
        prompt = candidate_system() + "\n\n" + candidate_user(case)
        raw, meta = qvac_bridge.generate(
            prompt, on_token=on_token, display_label=display
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


def iter_collect_parallel(
    case: Case,
    candidates_cfg: List[Dict[str, Any]],
    blind_map: Dict[str, str],
    on_event: EventCallback = None,
):
    """Yield CandidateAnswer as each parallel worker finishes (main-thread friendly)."""
    with ThreadPoolExecutor(max_workers=max(1, len(candidates_cfg))) as pool:
        futures = {
            pool.submit(
                _collect_candidate, case, c, blind_map[c["key"]], on_event
            ): c["key"]
            for c in candidates_cfg
        }
        for fut in as_completed(futures):
            yield fut.result()


def iter_collect_live(
    case: Case,
    candidates_cfg: List[Dict[str, Any]],
    blind_map: Dict[str, str],
):
    """Parallel collect with live token events for UI (main-thread consumer).

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
        with lock:
            start_at[key] = _time.time()

        def on_event(evt: Dict[str, Any]) -> None:
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
                case, cand_cfg, blind_map[cand_cfg["key"]], on_event
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

    threads = [
        threading.Thread(target=worker, args=(c,), daemon=True) for c in candidates_cfg
    ]
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
) -> Dict[str, Any]:
    """Resolve case + candidate list + blind map for UI-driven runs."""
    case = load_case(case_id)
    cfg = load_models_config(models_path)
    candidates_cfg = list(cfg.get("candidates") or [])
    has_qvac_cfg = any(c.get("provider") == "qvac" for c in candidates_cfg)
    qvac_up = qvac_bridge.available()

    if require_qvac and has_qvac_cfg and not qvac_up:
        raise RuntimeError(
            "QVAC SDK sidecar is required for demo mode but is offline. "
            "Start it with: cd sidecar && npm start"
        )
    if skip_qvac:
        candidates_cfg = [c for c in candidates_cfg if c.get("provider") != "qvac"]
    elif has_qvac_cfg and not qvac_up:
        candidates_cfg = [c for c in candidates_cfg if c.get("provider") != "qvac"]

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
) -> RunArtifact:
    prep = prepare_run(
        case_id,
        models_path=models_path,
        skip_qvac=skip_qvac,
        require_qvac=require_qvac,
        seed=seed,
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
    for cand in iter_collect_parallel(case, candidates_cfg, blind_map, on_event):
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

    total_cost = 0.0
    for c in collected:
        if c.meta.cost_usd:
            total_cost += c.meta.cost_usd
    for j in judgments:
        if j.judge_meta.cost_usd:
            total_cost += j.judge_meta.cost_usd

    notes = ""
    if has_qvac_cfg and not any(c.candidate_key == "qvac" for c in collected):
        notes = "QVAC skipped (sidecar unavailable). Start sidecar for full compare."

    artifact = RunArtifact(
        run_id=run_id,
        case_id=case_id,
        started_at=started,
        finished_at=utc_now_iso(),
        n_index=n_index,
        models_config={
            "profile": cfg.get("profile"),
            "candidates": candidates_cfg,
            "judge": judge_cfg,
            "blind_map": blind_map,
            "gold_reference": gold_reference.strip() if gold_reference else "",
        },
        candidates=collected,
        judgments=judgments,
        ranking=ranking,
        total_cost_usd=round(total_cost, 6),
        notes=notes,
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
) -> tuple[List[RunArtifact], MultiRunSummary]:
    out = out_dir or ARTIFACTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    artifacts: List[RunArtifact] = []
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
        )
        write_artifact(art, out)
        artifacts.append(art)
    summary = summarize_runs(artifacts)
    write_summary(summary, out)
    return artifacts, summary
