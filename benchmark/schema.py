"""Pydantic models for cases, answers, judge results, and run artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class QuestionRubric(BaseModel):
    """Legacy Demo/rubric reader only. New runs never score from this object."""

    acceptable: List[str] = Field(default_factory=list)
    must_include: List[str] = Field(default_factory=list)
    must_not: List[str] = Field(default_factory=list)
    notes: str = ""


class Question(BaseModel):
    id: str
    text: str
    weight: float = 0.2
    kind: Literal["diagnosis", "tests", "urgency", "safety", "plan", "other"] = "other"
    rubric: QuestionRubric = Field(default_factory=QuestionRubric)


class GoldClaim(BaseModel):
    """One frozen, source-linked statement in the user-supplied reference."""

    id: str
    text: str
    source_quote: str = ""
    critical: bool = False


class GoldSection(BaseModel):
    """Confirmed reference for one of the five benchmark sections."""

    summary: str
    claims: List[GoldClaim] = Field(default_factory=list)


class ConfirmedGold(BaseModel):
    """Gold-only scoring contract for a real/custom case."""

    raw_text: str
    sections: Dict[
        Literal["diagnosis", "tests", "urgency", "safety", "plan"], GoldSection
    ]
    confirmed_at: str
    extraction_model: str = ""
    extraction_prompt_version: str = "gold-extract-v1"
    extraction_cost_usd: float = 0.0


class Case(BaseModel):
    id: str
    title: str
    specialty: str = ""
    stem: str
    questions: List[Question]
    language: str = "en"
    # teaching remains accepted only so archived artifacts can still be opened.
    mode: Literal["teaching", "custom_real"] = "custom_real"
    # Short human label of the intended teaching target (not sent to candidates)
    gold_summary: str = ""


class ModelCallMeta(BaseModel):
    model: str
    provider: str
    requested_model: str = ""
    routed_model: str = ""
    routed_provider: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: Optional[float] = None
    latency_s: Optional[float] = None
    ttft_s: Optional[float] = None
    tps: Optional[float] = None
    # On-device QVAC: process-tree RSS (sidecar + llama worker), megabytes
    ram_mb: Optional[float] = None
    gguf_mb: Optional[float] = None
    gguf_sha256: str = ""
    # Real sidecar runtime (not OS platform string)
    device: str = ""
    gpu_layers: Optional[int] = None
    ctx_size: Optional[int] = None
    predict: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    display_label: str = ""
    retry_count: int = 0
    error: Optional[str] = None
    # Append-only paid OpenRouter attempts (primary / corrective / verifier).
    paid_attempts: List[Dict[str, Any]] = Field(default_factory=list)
    # Prior failed/superseded attempts kept for audit (not only retry_count).
    prior_attempts: List[Dict[str, Any]] = Field(default_factory=list)
    configuration_deviation: bool = False
    requested_providers: List[str] = Field(default_factory=list)


class CandidateAnswer(BaseModel):
    candidate_key: str
    label: str
    display_label: str = ""
    vendor: str = ""
    site: str = ""
    blind_id: str  # Candidate 1/2/… (never Case A/B/C)
    answers: Dict[str, str]  # question_id -> answer text
    raw_response: str = ""
    meta: ModelCallMeta


class QuestionScore(BaseModel):
    question_id: str
    score: float  # 0-100
    rationale: str = ""
    evidence: str = ""
    errors: List[str] = Field(default_factory=list)
    matched_claim_ids: List[str] = Field(default_factory=list)
    missed_claim_ids: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    claim_coverage: Dict[str, float] = Field(default_factory=dict)
    added_content: List[Dict[str, Any]] = Field(default_factory=list)
    precision: Optional[float] = None
    recall: Optional[float] = None
    quality: Optional[float] = None


class JudgeResult(BaseModel):
    blind_id: str
    candidate_key: str
    question_scores: List[QuestionScore]
    weighted_accuracy: float  # 0-100
    coverage_score: Optional[float] = None
    quality_score: Optional[float] = None
    discipline_score: Optional[float] = None
    retry_count: int = 0
    # Prior failed/superseded judge observations (error, status, model, …).
    prior_attempts: List[Dict[str, Any]] = Field(default_factory=list)
    judge_model: str  # effective judge that produced this observation
    primary_judge_model: str = ""  # requested primary; may differ after verifier
    judge_meta: ModelCallMeta
    raw_judge_json: str = ""
    status: Literal[
        "valid",
        "collect_failed",
        "candidate_empty",
        "candidate_partial",
        "judge_transport_failed",
        "judge_schema_invalid",
        "judge_evidence_invalid",
        "cancelled",
        "timed_out",
    ] = "valid"
    failure_reason: str = ""


class RunArtifact(BaseModel):
    schema_version: str = "2.0"
    run_id: str
    case_id: str
    started_at: str
    finished_at: str
    n_index: int = 1  # 1-based index within a multi-run
    batch_id: str = ""
    models_config: Dict[str, Any] = Field(default_factory=dict)
    candidates: List[CandidateAnswer] = Field(default_factory=list)
    judgments: List[JudgeResult] = Field(default_factory=list)
    ranking: List[Dict[str, Any]] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    cost_breakdown: Dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    cohort_id: str = ""
    scoring_version: str = "graded-clinical-v4"
    prompt_version: str = "gold-only-v1"
    benchmark_track: Literal[
        "controlled", "native_defaults", "legacy", "strict_controlled"
    ] = "controlled"
    run_status: Literal["complete", "partial", "cancelled", "failed"] = "complete"
    reproducibility: Dict[str, Any] = Field(default_factory=dict)
    execution_cohort_id: str = ""


class MultiRunSummary(BaseModel):
    case_id: str
    n: int
    candidate_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # key -> {mean, std, cv_pct, reliability, min, max, n}
    ranking_mean: List[Dict[str, Any]] = Field(default_factory=list)
    paired_ranking: List[Dict[str, Any]] = Field(default_factory=list)
    paired_n: int = 0
    run_ids: List[str] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    outliers: List[str] = Field(default_factory=list)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
