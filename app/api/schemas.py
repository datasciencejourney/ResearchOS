"""API request/response models."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    approval_required: bool = False
    max_research_iterations: int | None = None
    max_sources_per_subquestion: int | None = None
    sufficiency_threshold: float | None = None


class SessionCreate(BaseModel):
    question: str = Field(min_length=8, max_length=2000)
    config: RunConfig = Field(default_factory=RunConfig)


class RunRequest(BaseModel):
    """Start (or resume) a research run for the session."""

    start: bool = True


class PlanItemOut(BaseModel):
    id: uuid.UUID
    position: int
    subquestion: str
    rationale: str
    priority: int
    evidence_goal: str
    status: str
    attempts: int
    sufficiency: float | None


class PlanOut(BaseModel):
    version: int
    rationale: str
    status: str
    approval_required: bool
    approved_at: datetime | None
    created_at: datetime
    items: list[PlanItemOut]


class SessionSummary(BaseModel):
    id: uuid.UUID
    question: str
    status: str
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None
    error: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)


class SessionDetail(SessionSummary):
    config: dict[str, Any] = Field(default_factory=dict)
    latest_report_version: int | None = None
    verification_summary: dict[str, Any] = Field(default_factory=dict)
    ratings: list[dict[str, Any]] = Field(default_factory=list)


class SourceOut(BaseModel):
    id: uuid.UUID
    url: str
    canonical_url: str
    domain: str
    title: str
    search_snippet: str
    http_status: int | None
    retrieved_at: datetime | None
    is_excluded: bool
    quality_score: float | None
    quality_notes: str
    used_for: list[str]
    content_hash: str
    text_chars: int = 0


class ClaimOut(BaseModel):
    id: uuid.UUID
    source_id: uuid.UUID
    plan_item_id: uuid.UUID | None
    text: str
    claim_type: str
    importance: int
    subject: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class ContradictionOut(BaseModel):
    id: uuid.UUID
    plan_item_id: uuid.UUID | None
    topic: str
    claim_a_text: str
    claim_b_text: str
    source_a_domain: str
    source_b_domain: str
    context: str
    severity: str


class ReportOut(BaseModel):
    id: uuid.UUID
    version: int
    summary: str
    markdown: str
    revision_reason: str
    verification_summary: dict[str, Any]
    created_at: datetime


class VerificationOut(BaseModel):
    claim_text: str
    verdict: str
    cited_source_ids: list[Any]
    passage: str
    notes: str


class EventOut(BaseModel):
    id: uuid.UUID
    node: str
    kind: str
    message: str
    payload: dict[str, Any]
    created_at: datetime
