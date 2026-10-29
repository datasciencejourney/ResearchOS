"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-18

The langgraph checkpointer tables are not here, AsyncPostgresSaver.setup()
creates those on startup.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
UUIDType = sa.Uuid()


def upgrade() -> None:
    op.create_table(
        "research_sessions",
        sa.Column("id", UUIDType, primary_key=True),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("config", JSONType, nullable=False),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_research_sessions_status", "research_sessions", ["status"])

    op.create_table(
        "research_plans",
        sa.Column("id", UUIDType, primary_key=True),
        sa.Column("session_id", UUIDType, sa.ForeignKey("research_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("rationale", sa.Text, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("approval_required", sa.Boolean, nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("session_id", "version", name="uq_plan_session_version"),
    )
    op.create_index("ix_research_plans_session_id", "research_plans", ["session_id"])

    op.create_table(
        "research_plan_items",
        sa.Column("id", UUIDType, primary_key=True),
        sa.Column("plan_id", UUIDType, sa.ForeignKey("research_plans.id", ondelete="CASCADE"), nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("subquestion", sa.Text, nullable=False),
        sa.Column("rationale", sa.Text, nullable=False),
        sa.Column("priority", sa.Integer, nullable=False),
        sa.Column("evidence_goal", sa.Text, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("attempts", sa.Integer, nullable=False),
        sa.Column("sufficiency", sa.Float, nullable=True),
    )
    op.create_index("ix_research_plan_items_plan_id", "research_plan_items", ["plan_id"])

    op.create_table(
        "sources",
        sa.Column("id", UUIDType, primary_key=True),
        sa.Column("session_id", UUIDType, sa.ForeignKey("research_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("canonical_url", sa.Text, nullable=False),
        sa.Column("domain", sa.String(255), nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("search_snippet", sa.Text, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("http_status", sa.Integer, nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_excluded", sa.Boolean, nullable=False),
        sa.Column("quality_score", sa.Float, nullable=True),
        sa.Column("quality_notes", sa.Text, nullable=False),
        sa.Column("used_for", JSONType, nullable=False),
        sa.Column("extra", JSONType, nullable=False),
        sa.Column("search_vector", postgresql.TSVECTOR(), nullable=True),
        sa.UniqueConstraint("session_id", "url", name="uq_source_session_url"),
    )
    op.create_index("ix_sources_session_id", "sources", ["session_id"])

    op.create_table(
        "claims",
        sa.Column("id", UUIDType, primary_key=True),
        sa.Column("session_id", UUIDType, sa.ForeignKey("research_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_id", UUIDType, sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("plan_item_id", UUIDType, sa.ForeignKey("research_plan_items.id", ondelete="SET NULL"), nullable=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("claim_type", sa.String(32), nullable=False),
        sa.Column("importance", sa.Integer, nullable=False),
        sa.Column("subject", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_claims_session_id", "claims", ["session_id"])
    op.create_index("ix_claims_source_id", "claims", ["source_id"])
    op.create_index("ix_claims_session_plan", "claims", ["session_id", "plan_item_id"])

    op.create_table(
        "evidence",
        sa.Column("id", UUIDType, primary_key=True),
        sa.Column("claim_id", UUIDType, sa.ForeignKey("claims.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_id", UUIDType, sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("passage", sa.Text, nullable=False),
        sa.Column("position", sa.Integer, nullable=True),
        sa.Column("relevance_score", sa.Float, nullable=True),
    )
    op.create_index("ix_evidence_claim_id", "evidence", ["claim_id"])
    op.create_index("ix_evidence_source_id", "evidence", ["source_id"])

    op.create_table(
        "contradictions",
        sa.Column("id", UUIDType, primary_key=True),
        sa.Column("session_id", UUIDType, sa.ForeignKey("research_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("plan_item_id", UUIDType, sa.ForeignKey("research_plan_items.id", ondelete="SET NULL"), nullable=True),
        sa.Column("topic", sa.Text, nullable=False),
        sa.Column("claim_a_id", UUIDType, sa.ForeignKey("claims.id", ondelete="SET NULL"), nullable=True),
        sa.Column("claim_b_id", UUIDType, sa.ForeignKey("claims.id", ondelete="SET NULL"), nullable=True),
        sa.Column("claim_a_text", sa.Text, nullable=False),
        sa.Column("claim_b_text", sa.Text, nullable=False),
        sa.Column("source_a_domain", sa.String(255), nullable=False),
        sa.Column("source_b_domain", sa.String(255), nullable=False),
        sa.Column("context", sa.Text, nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_contradictions_session_id", "contradictions", ["session_id"])

    op.create_table(
        "reports",
        sa.Column("id", UUIDType, primary_key=True),
        sa.Column("session_id", UUIDType, sa.ForeignKey("research_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("markdown", sa.Text, nullable=False),
        sa.Column("summary", sa.Text, nullable=False),
        sa.Column("revision_reason", sa.Text, nullable=False),
        sa.Column("verification_summary", JSONType, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("session_id", "version", name="uq_report_session_version"),
    )
    op.create_index("ix_reports_session_id", "reports", ["session_id"])

    op.create_table(
        "report_verifications",
        sa.Column("id", UUIDType, primary_key=True),
        sa.Column("report_id", UUIDType, sa.ForeignKey("reports.id", ondelete="CASCADE"), nullable=False),
        sa.Column("claim_text", sa.Text, nullable=False),
        sa.Column("verdict", sa.String(16), nullable=False),
        sa.Column("cited_source_ids", JSONType, nullable=False),
        sa.Column("passage", sa.Text, nullable=False),
        sa.Column("notes", sa.Text, nullable=False),
    )
    op.create_index("ix_report_verifications_report_id", "report_verifications", ["report_id"])

    op.create_table(
        "run_events",
        sa.Column("id", UUIDType, primary_key=True),
        sa.Column("session_id", UUIDType, sa.ForeignKey("research_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("node", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("payload", JSONType, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_run_events_session_id", "run_events", ["session_id"])
    op.create_index("ix_run_events_created_at", "run_events", ["created_at"])
    op.create_index("ix_run_events_session_created", "run_events", ["session_id", "created_at"])


def downgrade() -> None:
    op.drop_table("run_events")
    op.drop_table("report_verifications")
    op.drop_table("reports")
    op.drop_table("contradictions")
    op.drop_table("evidence")
    op.drop_table("claims")
    op.drop_table("sources")
    op.drop_table("research_plan_items")
    op.drop_table("research_plans")
    op.drop_table("research_sessions")
