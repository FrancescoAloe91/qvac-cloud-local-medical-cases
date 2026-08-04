"""OFFLINE ops tool — paid rejudge for Comprehension MedPsy N/A rows only.

Not part of the Streamlit dashboard path. Do **not** use for public posts or
to selectively boost MedPsy in published means.

Re-cleans candidate answers (strip ``<think>``), calls DeepSeek judge again,
and may persist the artifact in place when ``dry_run=False`` (CLI ``--write``).
Default is dry-run. Does not invent scores offline.
Run only intentionally from CLI/scripts against a private workspace.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmark.beta_prompts import parse_beta_candidate_answers
from benchmark.beta_protocol import CASE_ID, SCORING_VERSION
from benchmark.cases_loader import load_case
from benchmark.costing import cost_breakdown_for_run
from benchmark.judge import build_ranking, is_failed_judgment, judge_candidate
from benchmark.report import load_artifact, write_artifact
from benchmark.schema import CandidateAnswer, JudgeResult, RunArtifact, utc_now_iso

MEDPSY_KEYS = ("qvac", "qvac_1_7b")
REJUDGEABLE_STATUSES = {
    "judge_evidence_invalid",
    "judge_schema_invalid",
    "timed_out",
    "judge_transport_failed",
}


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def resolve_openrouter_key(*, env_file: Optional[Path] = None) -> str:
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if key.startswith("sk-or-"):
        return key
    root = Path(__file__).resolve().parents[1]
    _load_dotenv(env_file or (root / ".env"))
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    return key if key.startswith("sk-or-") else ""


def is_rejudgeable_medpsy_na(judgment: JudgeResult) -> bool:
    if judgment.candidate_key not in MEDPSY_KEYS:
        return False
    if judgment.status in REJUDGEABLE_STATUSES:
        return True
    if is_failed_judgment(judgment) and judgment.status not in {
        "collect_failed",
        "candidate_empty",
        "candidate_partial",
        "cancelled",
    }:
        reason = (judgment.failure_reason or "").lower()
        return any(
            token in reason
            for token in (
                "empty or unusable json",
                "schema",
                "evidence",
                "wall-clock",
                "timeout",
            )
        )
    return False


def cleaned_beta_candidate(case, candidate: CandidateAnswer) -> CandidateAnswer:
    """Rebuild section answers from raw with think-stripping Beta parser."""
    answers = parse_beta_candidate_answers(case, candidate.raw_response or "")
    if not answers:
        # Fall back to sanitizing stored sections (already photocopied prose).
        from benchmark.prompts import sanitize_candidate_answers

        answers = sanitize_candidate_answers(candidate.answers or {})
    return candidate.model_copy(update={"answers": answers})


def case_from_beta_artifact(art: RunArtifact):
    base = load_case("caseC")
    cfg = art.models_config or {}
    stem = str(cfg.get("case_stem") or base.stem or "").strip() or base.stem
    return base.model_copy(
        update={
            "id": art.case_id or CASE_ID,
            "stem": stem,
            "title": base.title or "Comprehension",
        }
    )


def rejudge_medpsy_on_artifact(
    art: RunArtifact,
    *,
    api_key: str,
    keys: Sequence[str] = MEDPSY_KEYS,
    allow_verifier: bool = False,
) -> Tuple[RunArtifact, Dict[str, Any]]:
    """Rejudge eligible MedPsy N/A rows; return updated artifact + report row."""
    case = case_from_beta_artifact(art)
    cfg = art.models_config or {}
    judge_cfg = cfg.get("judge") or {}
    judge_model = str(judge_cfg.get("model") or "deepseek/deepseek-r1")
    judge_temp = float(judge_cfg.get("temperature", 0) or 0)
    gold_ref = str(cfg.get("gold_reference") or "")
    allowed = list(judge_cfg.get("allowed_providers") or []) or None
    verifier = str(judge_cfg.get("verifier_model") or "")

    cand_by_key = {c.candidate_key: c for c in art.candidates}
    new_candidates: List[CandidateAnswer] = []
    cleaned_keys: List[str] = []
    for cand in art.candidates:
        if cand.candidate_key in keys:
            cleaned = cleaned_beta_candidate(case, cand)
            if cleaned.answers != cand.answers:
                cleaned_keys.append(cand.candidate_key)
            new_candidates.append(cleaned)
        else:
            new_candidates.append(cand)

    report: Dict[str, Any] = {
        "run_id": art.run_id,
        "cleaned_keys": cleaned_keys,
        "attempted": [],
        "recovered": [],
        "still_na": [],
        "skipped": [],
    }

    new_judgments: List[JudgeResult] = []
    for j in art.judgments:
        if j.candidate_key not in keys or not is_rejudgeable_medpsy_na(j):
            if j.candidate_key in keys and is_failed_judgment(j):
                report["skipped"].append(
                    {
                        "key": j.candidate_key,
                        "status": j.status,
                        "reason": (j.failure_reason or "")[:160],
                    }
                )
            new_judgments.append(j)
            continue

        cand = next(
            (c for c in new_candidates if c.candidate_key == j.candidate_key),
            cand_by_key.get(j.candidate_key),
        )
        if cand is None or not any((cand.answers or {}).values()):
            report["still_na"].append(
                {"key": j.candidate_key, "status": j.status, "note": "no clean answers"}
            )
            new_judgments.append(j)
            continue

        report["attempted"].append(
            {"key": j.candidate_key, "prior_status": j.status}
        )
        fresh = judge_candidate(
            case,
            cand,
            judge_model,
            temperature=judge_temp,
            gold_reference=gold_ref,
            api_key=api_key,
            verifier_model=verifier,
            allow_verifier=allow_verifier,
            allowed_providers=allowed,
            require_parameters=False,
            allow_fallbacks=True,
        )
        # Never clobber a historical N/A with a transient transport/proxy failure.
        if (
            is_failed_judgment(fresh)
            and fresh.status == "judge_transport_failed"
            and j.status != "judge_transport_failed"
        ):
            report["still_na"].append(
                {
                    "key": j.candidate_key,
                    "status": j.status,
                    "reason": f"rejudge transport aborted; kept prior: {(fresh.failure_reason or '')[:160]}",
                    "prior_status": j.status,
                    "kept_prior": True,
                }
            )
            new_judgments.append(j)
            continue
        # Keep audit trail of the failed observation.
        attempts = list(fresh.prior_attempts or [])
        attempts.insert(
            0,
            {
                "status": j.status,
                "failure_reason": j.failure_reason,
                "weighted_accuracy": j.weighted_accuracy,
                "judge_model": j.judge_model,
                "raw_judge_json": (j.raw_judge_json or "")[:500],
                "rejudge_source": "beta_think_strip_v1",
            },
        )
        fresh.prior_attempts = attempts
        if is_failed_judgment(fresh):
            report["still_na"].append(
                {
                    "key": j.candidate_key,
                    "status": fresh.status,
                    "reason": (fresh.failure_reason or "")[:200],
                    "prior_status": j.status,
                }
            )
        else:
            report["recovered"].append(
                {
                    "key": j.candidate_key,
                    "prior_status": j.status,
                    "accuracy": fresh.weighted_accuracy,
                }
            )
        new_judgments.append(fresh)

    clone = art.model_copy(deep=True)
    clone.candidates = new_candidates
    clone.judgments = new_judgments
    clone.ranking = build_ranking(new_judgments)
    bd = cost_breakdown_for_run(
        new_candidates,
        new_judgments,
        extraction_cost_usd=float(
            (clone.cost_breakdown or {}).get("extractor_usd")
            or (cfg.get("extraction_cost_usd") or 0.0)
        ),
    )
    clone.cost_breakdown = bd
    clone.total_cost_usd = float(bd.get("total_usd") or 0.0)
    repro = dict(clone.reproducibility or {})
    prior_events = list(repro.get("beta_paid_rejudge") or [])
    prior_events.append(
        {
            "at": utc_now_iso(),
            "formula": "beta-think-strip-rejudge-v1",
            "scoring_version": SCORING_VERSION,
            "attempted": report["attempted"],
            "recovered": report["recovered"],
            "still_na": report["still_na"],
            "cleaned_keys": cleaned_keys,
        }
    )
    repro["beta_paid_rejudge"] = prior_events
    clone.reproducibility = repro
    if report["recovered"]:
        note = (clone.notes or "").strip()
        tag = "beta-rejudged: " + ", ".join(
            f"{r['key']}->{r['accuracy']:.1f}" for r in report["recovered"]
        )
        if tag not in note:
            clone.notes = (note + " | " + tag).strip(" |")
    return clone, report


def iter_beta_medpsy_na_artifacts(
    owner_dir: Path,
) -> List[Tuple[Path, RunArtifact, List[str]]]:
    out: List[Tuple[Path, RunArtifact, List[str]]] = []
    for path in sorted(owner_dir.glob("beta-*.json")):
        try:
            art = load_artifact(path)
        except Exception:
            continue
        if (art.case_id or "") != CASE_ID and not str(
            art.scoring_version or ""
        ).startswith("beta"):
            continue
        keys = [
            j.candidate_key
            for j in art.judgments
            if is_rejudgeable_medpsy_na(j)
        ]
        if keys:
            out.append((path, art, keys))
    return out


def rejudge_owner_beta_medpsy_na(
    owner_dir: Path,
    *,
    api_key: Optional[str] = None,
    dry_run: bool = True,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Rejudge MedPsy N/A rows under ``owner_dir``.

    Default ``dry_run=True`` lists targets only. Pass ``dry_run=False`` (CLI
    ``--write``) to call the judge API and rewrite finished ``beta-*.json``
    artifacts in place. Never mutates when the API key is missing.
    """
    key = (api_key or resolve_openrouter_key()).strip()
    targets = iter_beta_medpsy_na_artifacts(owner_dir)
    if limit is not None:
        targets = targets[: max(0, int(limit))]
    summary: Dict[str, Any] = {
        "owner": str(owner_dir),
        "n_artifacts": len(targets),
        "dry_run": dry_run,
        "has_api_key": bool(key.startswith("sk-or-")),
        "reports": [],
        "recovered_total": 0,
        "still_na_total": 0,
        "attempted_total": 0,
    }
    if not targets:
        return summary
    if dry_run or not key.startswith("sk-or-"):
        for path, art, keys in targets:
            summary["reports"].append(
                {
                    "run_id": art.run_id,
                    "path": str(path),
                    "keys": keys,
                    "dry_run": True,
                }
            )
            summary["attempted_total"] += len(keys)
        return summary

    for path, art, _keys in targets:
        clone, report = rejudge_medpsy_on_artifact(art, api_key=key)
        recovered = report.get("recovered") or []
        still = report.get("still_na") or []
        # Persist when we recovered scores, cleaned answers, or got a real
        # non-transport rejudge outcome. Skip write if every attempt was a
        # kept-prior transport abort (no judgment change).
        only_kept_prior = bool(still) and all(
            s.get("kept_prior") for s in still
        ) and not recovered
        answers_changed = bool(report.get("cleaned_keys"))
        if recovered or answers_changed or (still and not only_kept_prior):
            write_artifact(clone, path.parent)
        summary["reports"].append(report)
        summary["recovered_total"] += len(recovered)
        summary["still_na_total"] += len(still)
        summary["attempted_total"] += len(report.get("attempted") or [])
    return summary
