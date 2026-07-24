"""Pydantic models for cases, answers, judge results, and run artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class QuestionRubric(BaseModel):
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


class Case(BaseModel):
    id: str
    title: str
    specialty: str = ""
    stem: str
    questions: List[Question]
    language: str = "en"
    # teaching = preset vignette + built-in answer grid; custom_real = user pastes a real case
    mode: Literal["teaching", "custom_real"] = "teaching"
    # Short human label of the intended teaching target (not sent to candidates)
    gold_summary: str = ""


class ModelCallMeta(BaseModel):
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: Optional[float] = None
    latency_s: Optional[float] = None
    ttft_s: Optional[float] = None
    tps: Optional[float] = None
    display_label: str = ""
    error: Optional[str] = None


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


class JudgeResult(BaseModel):
    blind_id: str
    candidate_key: str
    question_scores: List[QuestionScore]
    weighted_accuracy: float  # 0-100
    judge_model: str
    judge_meta: ModelCallMeta
    raw_judge_json: str = ""


class RunArtifact(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    case_id: str
    started_at: str
    finished_at: str
    n_index: int = 1  # 1-based index within a multi-run
    models_config: Dict[str, Any] = Field(default_factory=dict)
    candidates: List[CandidateAnswer] = Field(default_factory=list)
    judgments: List[JudgeResult] = Field(default_factory=list)
    ranking: List[Dict[str, Any]] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    notes: str = ""


class MultiRunSummary(BaseModel):
    case_id: str
    n: int
    candidate_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    # key -> {mean, std, min, max, n}
    ranking_mean: List[Dict[str, Any]] = Field(default_factory=list)
    run_ids: List[str] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    outliers: List[str] = Field(default_factory=list)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
