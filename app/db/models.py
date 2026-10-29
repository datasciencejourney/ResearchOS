"""SQLAlchemy models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> uuid.UUID:
    return uuid.uuid4()


class Base(DeclarativeBase):
    pass


# JSON that works on Postgres (JSONB) and in unit tests on SQLite.
JSONType = JSON().with_variant(JSONB(), "postgresql")


class SessionStatus:
    CREATED = "created"
    PLANNING = "planning"
    RESEARCHING = "researching"
    WRITING = "writing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_APPROVAL = "awaiting_approval"


class ResearchSession(Base):
    __tablename__ = "research_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default=SessionStatus.CREATED, index=True)
    # model + knobs used for this run; part of the reproducibility record
    config: Mapped[dict] = mapped_column(JSONType, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    plan_versions: Mapped[list[PlanVersion]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="PlanVersion.version"
    )
    sources: Mapped[list[Source]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    claims: Mapped[list[Claim]] = relationship(back_populates="session", cascade="all, delete-orphan")
    contradictions: Mapped[list[Contradiction]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    reports: Mapped[list[Report]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Report.version"
    )
    events: Mapped[list[RunEvent]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class PlanVersion(Base):
    __tablename__ = "research_plans"
    __table_args__ = (UniqueConstraint("session_id", "version", name="uq_plan_session_version"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    rationale: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="active")  # active | superseded
    approval_required: Mapped[bool] = mapped_column(Boolean, default=False)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped[ResearchSession] = relationship(back_populates="plan_versions")
    items: Mapped[list[PlanItem]] = relationship(
        back_populates="plan", cascade="all, delete-orphan", order_by="PlanItem.position"
    )


class PlanItem(Base):
    __tablename__ = "research_plan_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    plan_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_plans.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer, default=0)
    subquestion: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, default="")
    priority: Mapped[int] = mapped_column(Integer, default=2)  # 1 = highest
    evidence_goal: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    sufficiency: Mapped[float | None] = mapped_column(Float, nullable=True)

    plan: Mapped[PlanVersion] = relationship(back_populates="items")


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (UniqueConstraint("session_id", "url", name="uq_source_session_url"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_url: Mapped[str] = mapped_column(Text, default="")
    domain: Mapped[str] = mapped_column(String(255), default="")
    title: Mapped[str] = mapped_column(Text, default="")
    search_snippet: Mapped[str] = mapped_column(Text, default="")
    # Verbatim snapshot of the retrieved document (reproducibility).
    text: Mapped[str] = mapped_column(Text, default="")
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_excluded: Mapped[bool] = mapped_column(Boolean, default=False)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_notes: Mapped[str] = mapped_column(Text, default="")
    used_for: Mapped[list[str]] = mapped_column(JSONType, default=list)  # subquestion ids
    extra: Mapped[dict] = mapped_column(JSONType, default=dict)
    search_vector: Mapped[str | None] = mapped_column(TSVECTOR(), nullable=True)

    session: Mapped[ResearchSession] = relationship(back_populates="sources")
    claims: Mapped[list[Claim]] = relationship(back_populates="source")


class Claim(Base):
    __tablename__ = "claims"
    __table_args__ = (Index("ix_claims_session_plan", "session_id", "plan_item_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE"), index=True
    )
    plan_item_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_plan_items.id", ondelete="SET NULL"), nullable=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    claim_type: Mapped[str] = mapped_column(String(32), default="factual")
    importance: Mapped[int] = mapped_column(Integer, default=2)  # 1 = load-bearing for the report
    subject: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    source: Mapped[Source] = relationship(back_populates="claims")
    session: Mapped[ResearchSession] = relationship(back_populates="claims")
    evidence: Mapped[list[Evidence]] = relationship(back_populates="claim", cascade="all, delete-orphan")


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    claim_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("claims.id", ondelete="CASCADE"), index=True
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE"), index=True
    )
    # Verbatim excerpt from the source snapshot that supports the claim.
    passage: Mapped[str] = mapped_column(Text, nullable=False)
    position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    claim: Mapped[Claim] = relationship(back_populates="evidence")


class Contradiction(Base):
    __tablename__ = "contradictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True
    )
    plan_item_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_plan_items.id", ondelete="SET NULL"), nullable=True
    )
    topic: Mapped[str] = mapped_column(Text, default="")
    claim_a_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("claims.id", ondelete="SET NULL"), nullable=True
    )
    claim_b_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("claims.id", ondelete="SET NULL"), nullable=True
    )
    claim_a_text: Mapped[str] = mapped_column(Text, default="")
    claim_b_text: Mapped[str] = mapped_column(Text, default="")
    source_a_domain: Mapped[str] = mapped_column(String(255), default="")
    source_b_domain: Mapped[str] = mapped_column(String(255), default="")
    context: Mapped[str] = mapped_column(Text, default="")  # why they can both be true
    severity: Mapped[str] = mapped_column(String(16), default="moderate")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped[ResearchSession] = relationship(back_populates="contradictions")


class Report(Base):
    __tablename__ = "reports"
    __table_args__ = (UniqueConstraint("session_id", "version", name="uq_report_session_version"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    markdown: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    revision_reason: Mapped[str] = mapped_column(Text, default="")
    verification_summary: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped[ResearchSession] = relationship(back_populates="reports")
    verifications: Mapped[list[ReportVerification]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )


class ReportVerification(Base):
    __tablename__ = "report_verifications"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    report_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("reports.id", ondelete="CASCADE"), index=True
    )
    claim_text: Mapped[str] = mapped_column(Text, default="")
    verdict: Mapped[str] = mapped_column(String(16), default="unsupported")
    cited_source_ids: Mapped[list] = mapped_column(JSONType, default=list)
    passage: Mapped[str] = mapped_column(Text, default="")
    notes: Mapped[str] = mapped_column(Text, default="")

    report: Mapped[Report] = relationship(back_populates="verifications")


class RunEvent(Base):
    """Append-only trace of the workflow: what ran, when, and what it produced."""

    __tablename__ = "run_events"
    __table_args__ = (Index("ix_run_events_session_created", "session_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True
    )
    node: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), default="info")
    message: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    session: Mapped[ResearchSession] = relationship(back_populates="events")
