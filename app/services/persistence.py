"""DB writes for the graph (plans, sources, claims, reports, events)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from sqlalchemy import delete, select

from app.db.models import (
    Claim,
    Contradiction,
    Evidence,
    PlanItem,
    PlanVersion,
    Report,
    ReportVerification,
    ResearchSession,
    RunEvent,
    SessionStatus,
    Source,
)
from app.db.session import session_scope


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# sessions / events
def get_session(session_id: uuid.UUID) -> ResearchSession | None:
    with session_scope() as db:
        return db.get(ResearchSession, session_id)


def set_status(session_id: uuid.UUID, status: str, *, error: str | None = None, finished: bool = False) -> None:
    with session_scope() as db:
        row = db.get(ResearchSession, session_id)
        if row is None:
            return
        row.status = status
        row.updated_at = _utcnow()
        if error is not None:
            row.error = error
        if finished:
            row.finished_at = _utcnow()


def log_event(
    session_id: uuid.UUID,
    node: str,
    message: str,
    payload: dict[str, Any] | None = None,
    kind: str = "info",
) -> None:
    with session_scope() as db:
        db.add(
            RunEvent(
                session_id=session_id,
                node=node,
                message=message,
                kind=kind,
                payload=payload or {},
                created_at=_utcnow(),
            )
        )


# plan
def save_plan(session_id: uuid.UUID, plan: dict[str, Any], approval_required: bool) -> list[dict]:
    """Persist a new plan version and return its items (with ids)."""
    with session_scope() as db:
        existing = db.execute(
            select(PlanVersion).where(PlanVersion.session_id == session_id)
        ).scalars().all()
        for old in existing:
            old.status = "superseded"
        version = max([p.version for p in existing], default=0) + 1
        row = PlanVersion(
            session_id=session_id,
            version=version,
            rationale=plan.get("rationale", ""),
            approval_required=approval_required,
        )
        db.add(row)
        db.flush()
        items: list[dict] = []
        for position, item in enumerate(plan.get("subquestions", []), start=1):
            db.add(
                PlanItem(
                    plan_id=row.id,
                    position=position,
                    subquestion=item["subquestion"],
                    rationale=item.get("rationale", ""),
                    priority=item.get("priority", 2),
                    evidence_goal=item.get("evidence_goal", ""),
                )
            )
        db.flush()
        for item in row.items:
            items.append(
                {
                    "id": str(item.id),
                    "position": item.position,
                    "subquestion": item.subquestion,
                    "rationale": item.rationale,
                    "priority": item.priority,
                    "evidence_goal": item.evidence_goal,
                }
            )
        return items


def load_approved_plan_items(session_id: uuid.UUID) -> list[dict] | None:
    """Return plan items when an approved plan already exists (re-run case)."""
    with session_scope() as db:
        plan = db.execute(
            select(PlanVersion)
            .where(PlanVersion.session_id == session_id, PlanVersion.approved_at.is_not(None))
            .order_by(PlanVersion.version.desc())
            .limit(1)
        ).scalars().first()
        if plan is None:
            return None
        return [
            {
                "id": str(item.id),
                "position": item.position,
                "subquestion": item.subquestion,
                "rationale": item.rationale,
                "priority": item.priority,
                "evidence_goal": item.evidence_goal,
            }
            for item in plan.items
        ]


def approve_plan(session_id: uuid.UUID) -> bool:
    with session_scope() as db:
        plan = db.execute(
            select(PlanVersion)
            .where(PlanVersion.session_id == session_id)
            .order_by(PlanVersion.version.desc())
            .limit(1)
        ).scalars().first()
        if plan is None:
            return False
        plan.approved_at = _utcnow()
        return True


def plan_approved(session_id: uuid.UUID) -> bool:
    with session_scope() as db:
        plan = db.execute(
            select(PlanVersion)
            .where(PlanVersion.session_id == session_id)
            .order_by(PlanVersion.version.desc())
            .limit(1)
        ).scalars().first()
        return bool(plan is not None and plan.approved_at is not None)


def update_plan_item(session_id: uuid.UUID, item_id: str, *, status: str, sufficiency: float) -> None:
    with session_scope() as db:
        row = db.get(PlanItem, uuid.UUID(item_id))
        if row is None:
            return
        row.status = status
        row.sufficiency = sufficiency
        row.attempts = (row.attempts or 0) + 1


# sources / claims / evidence
def excluded_urls(session_id: uuid.UUID) -> set[str]:
    with session_scope() as db:
        rows = db.execute(
            select(Source.url).where(Source.session_id == session_id, Source.is_excluded.is_(True))
        ).scalars().all()
        return set(rows)


def upsert_source(session_id: uuid.UUID, source: dict[str, Any]) -> str:
    """Insert or refresh a source snapshot; returns the source id."""
    with session_scope() as db:
        row = db.execute(
            select(Source).where(Source.session_id == session_id, Source.url == source["url"])
        ).scalars().first()
        if row is None:
            row = Source(session_id=session_id, url=source["url"])
            db.add(row)
        row.canonical_url = source.get("canonical_url", "")
        row.domain = source.get("domain", "")
        row.title = source.get("title", "")
        row.search_snippet = source.get("search_snippet", "")
        row.text = source.get("text", "")
        row.content_hash = source.get("content_hash", "")
        row.http_status = source.get("http_status")
        row.retrieved_at = _utcnow()
        row.quality_score = source.get("quality_score")
        row.quality_notes = source.get("quality_notes", "")
        row.extra = source.get("extra", {})
        used = set(row.used_for or [])
        used.update(source.get("used_for", []))
        row.used_for = sorted(used)
        db.flush()
        return str(row.id)


def replace_claims(session_id: uuid.UUID, source_id: str, plan_item_id: str, claims: list[dict]) -> None:
    """Replace the claims a source contributed to one subquestion (idempotent)."""
    with session_scope() as db:
        db.execute(
            delete(Claim).where(
                Claim.session_id == session_id,
                Claim.source_id == uuid.UUID(source_id),
                Claim.plan_item_id == uuid.UUID(plan_item_id),
            )
        )
        for claim in claims:
            row = Claim(
                session_id=session_id,
                source_id=uuid.UUID(source_id),
                plan_item_id=uuid.UUID(plan_item_id),
                text=claim["text"],
                claim_type=claim.get("claim_type", "factual"),
                importance=claim.get("importance", 2),
                subject=claim.get("subject", ""),
            )
            db.add(row)
            db.flush()
            for passage in claim.get("evidence", []):
                db.add(
                    Evidence(
                        claim_id=row.id,
                        source_id=uuid.UUID(source_id),
                        passage=passage.get("passage", ""),
                        position=passage.get("position"),
                        relevance_score=passage.get("relevance"),
                    )
                )


def save_contradictions(session_id: uuid.UUID, plan_item_id: str, pairs: list[dict]) -> None:
    with session_scope() as db:
        db.execute(delete(Contradiction).where(
            Contradiction.session_id == session_id,
            Contradiction.plan_item_id == uuid.UUID(plan_item_id),
        ))
        for pair in pairs:
            db.add(
                Contradiction(
                    session_id=session_id,
                    plan_item_id=uuid.UUID(plan_item_id),
                    topic=pair.get("topic", ""),
                    claim_a_text=pair.get("claim_a", ""),
                    claim_b_text=pair.get("claim_b", ""),
                    source_a_domain=pair.get("source_a_domain", ""),
                    source_b_domain=pair.get("source_b_domain", ""),
                    context=pair.get("context", ""),
                    severity=pair.get("severity", "moderate"),
                )
            )


# reports
def save_report(
    session_id: uuid.UUID,
    markdown: str,
    summary: str,
    revision_reason: str = "",
) -> int:
    with session_scope() as db:
        existing = db.execute(
            select(Report.version).where(Report.session_id == session_id)
        ).scalars().all()
        version = max(existing, default=0) + 1
        db.add(
            Report(
                session_id=session_id,
                version=version,
                markdown=markdown,
                summary=summary,
                revision_reason=revision_reason,
                created_at=_utcnow(),
            )
        )
        return version


def save_verifications(
    session_id: uuid.UUID,
    version: int,
    results: list[dict],
    summary: dict,
) -> str:
    with session_scope() as db:
        report = db.execute(
            select(Report).where(Report.session_id == session_id, Report.version == version)
        ).scalars().first()
        if report is None:
            return ""
        report.verification_summary = summary
        db.execute(delete(ReportVerification).where(ReportVerification.report_id == report.id))
        for result in results:
            db.add(
                ReportVerification(
                    report_id=report.id,
                    claim_text=result.get("sentence", ""),
                    verdict=result.get("verdict", "unsupported"),
                    cited_source_ids=result.get("sources", []),
                    passage=result.get("passage", ""),
                    notes=result.get("notes", ""),
                )
            )
        return str(report.id)
