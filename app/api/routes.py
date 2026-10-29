"""FastAPI routes for ResearchOS."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api import schemas
from app.api.deps import get_db
from app.config import get_settings
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
from app.services import persistence as store
from app.services import runner

router = APIRouter()
DbSession = Annotated[Session, Depends(get_db)]


def _client_id(db: Session) -> str | None:
    return db.info.get("client_id")


def _session_or_404(db: Session, session_id: uuid.UUID) -> ResearchSession:
    row = db.get(ResearchSession, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="research session not found")
    # Sessions created with a client id belong to that visitor. Older ones have no owner.
    owner = (row.config or {}).get("owner")
    if owner and owner != _client_id(db):
        raise HTTPException(status_code=404, detail="research session not found")
    return row


def _counts(db: Session, session_id: uuid.UUID) -> dict[str, int]:
    def count(model, *filters) -> int:
        return db.execute(select(func.count()).select_from(model).where(*filters)).scalar_one()

    return {
        "sources": count(Source, Source.session_id == session_id),
        "claims": count(Claim, Claim.session_id == session_id),
        "contradictions": count(Contradiction, Contradiction.session_id == session_id),
        "reports": count(Report, Report.session_id == session_id),
        "events": count(RunEvent, RunEvent.session_id == session_id),
    }


# health
@router.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok", "app": get_settings().app_name, "running_runs": runner.running_count()}


# sessions
@router.post("/sessions", response_model=schemas.SessionDetail, status_code=201, tags=["sessions"])
async def create_session(body: schemas.SessionCreate, db: DbSession) -> schemas.SessionDetail:
    config = body.config.model_dump(exclude_none=True)
    if _client_id(db):
        config["owner"] = _client_id(db)
    row = ResearchSession(question=body.question.strip(), status=SessionStatus.CREATED, config=config)
    db.add(row)
    db.commit()
    db.refresh(row)
    store.log_event(row.id, "api", "Session created", {"question": row.question})
    return _detail(db, row)


@router.get("/sessions", response_model=list[schemas.SessionSummary], tags=["sessions"])
async def list_sessions(
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[schemas.SessionSummary]:
    query = select(ResearchSession)
    owner_col = ResearchSession.config["owner"].as_string()
    if _client_id(db):
        query = query.where(owner_col == _client_id(db))
    else:
        query = query.where(owner_col.is_(None))
    rows = db.execute(
        query.order_by(ResearchSession.created_at.desc()).limit(limit).offset(offset)
    ).scalars().all()
    return [
        schemas.SessionSummary(
            id=row.id,
            question=row.question,
            status=row.status,
            created_at=row.created_at,
            updated_at=row.updated_at,
            finished_at=row.finished_at,
            error=row.error,
            counts=_counts(db, row.id),
        )
        for row in rows
    ]


@router.get("/sessions/{session_id}", response_model=schemas.SessionDetail, tags=["sessions"])
async def get_session(session_id: uuid.UUID, db: DbSession) -> schemas.SessionDetail:
    return _detail(db, _session_or_404(db, session_id))


def _detail(db: Session, row: ResearchSession) -> schemas.SessionDetail:
    plan = db.execute(
        select(PlanVersion)
        .where(PlanVersion.session_id == row.id)
        .order_by(PlanVersion.version.desc())
        .limit(1)
    ).scalars().first()
    latest_report = db.execute(
        select(Report).where(Report.session_id == row.id).order_by(Report.version.desc()).limit(1)
    ).scalars().first()
    # Sufficiency ratings are re-derived from plan items (authoritative).
    ratings = []
    if plan is not None:
        for item in plan.items:
            ratings.append(
                {
                    "plan_item_id": str(item.id),
                    "subquestion": item.subquestion,
                    "status": item.status,
                    "attempts": item.attempts,
                    "sufficiency": item.sufficiency,
                }
            )
    return schemas.SessionDetail(
        id=row.id,
        question=row.question,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
        finished_at=row.finished_at,
        error=row.error,
        config=row.config or {},
        counts=_counts(db, row.id),
        latest_report_version=latest_report.version if latest_report else None,
        verification_summary=(latest_report.verification_summary if latest_report else {}),
        ratings=ratings,
    )


@router.post("/sessions/{session_id}/run", tags=["sessions"], summary="Start or resume a research run")
async def run_session(session_id: uuid.UUID, db: DbSession, body: schemas.RunRequest | None = None) -> dict:
    row = _session_or_404(db, session_id)
    if not (body and body.start):
        return {"session_id": str(session_id), "status": row.status, "started": False}
    if runner.is_running(session_id):
        raise HTTPException(status_code=409, detail="a run is already in progress")
    try:
        runner.start_run(session_id)
    except RuntimeError as exc:  # pragma: no cover - guarded above
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"session_id": str(session_id), "status": "planning", "started": True}


@router.get("/sessions/{session_id}/plan", response_model=schemas.PlanOut, tags=["plan"])
async def get_plan(session_id: uuid.UUID, db: DbSession) -> schemas.PlanOut:
    _session_or_404(db, session_id)
    plan = db.execute(
        select(PlanVersion)
        .where(PlanVersion.session_id == session_id)
        .order_by(PlanVersion.version.desc())
        .limit(1)
    ).scalars().first()
    if plan is None:
        raise HTTPException(status_code=404, detail="no plan yet")
    items = db.execute(
        select(PlanItem).where(PlanItem.plan_id == plan.id).order_by(PlanItem.position)
    ).scalars().all()
    return schemas.PlanOut(
        version=plan.version,
        rationale=plan.rationale,
        status=plan.status,
        approval_required=plan.approval_required,
        approved_at=plan.approved_at,
        created_at=plan.created_at,
        items=[schemas.PlanItemOut.model_validate(item, from_attributes=True) for item in items],
    )


@router.post("/sessions/{session_id}/plan/approve", tags=["plan"], summary="Human-in-the-loop: approve the plan")
async def approve_plan(session_id: uuid.UUID, db: DbSession, start: bool = True) -> dict:
    _session_or_404(db, session_id)
    if not store.approve_plan(session_id):
        raise HTTPException(status_code=404, detail="no plan to approve")
    store.log_event(session_id, "api", "Research plan approved by user")
    started = False
    if start and not runner.is_running(session_id):
        runner.start_run(session_id)
        started = True
    return {"session_id": str(session_id), "approved": True, "started": started}


# evidence
@router.get("/sessions/{session_id}/sources", response_model=list[schemas.SourceOut], tags=["evidence"])
async def list_sources(session_id: uuid.UUID, db: DbSession) -> list[schemas.SourceOut]:
    _session_or_404(db, session_id)
    rows = db.execute(
        select(Source).where(Source.session_id == session_id).order_by(Source.retrieved_at)
    ).scalars().all()
    return [
        schemas.SourceOut(
            id=row.id,
            url=row.url,
            canonical_url=row.canonical_url,
            domain=row.domain,
            title=row.title,
            search_snippet=row.search_snippet,
            http_status=row.http_status,
            retrieved_at=row.retrieved_at,
            is_excluded=row.is_excluded,
            quality_score=row.quality_score,
            quality_notes=row.quality_notes,
            used_for=list(row.used_for or []),
            content_hash=row.content_hash,
            text_chars=len(row.text or ""),
        )
        for row in rows
    ]


@router.post(
    "/sessions/{session_id}/sources/{source_id}/exclude",
    tags=["evidence"],
    summary="Human-in-the-loop: exclude a source from future research rounds",
)
async def exclude_source(session_id: uuid.UUID, source_id: uuid.UUID, db: DbSession, excluded: bool = True) -> dict:
    _session_or_404(db, session_id)
    row = db.get(Source, source_id)
    if row is None or row.session_id != session_id:
        raise HTTPException(status_code=404, detail="source not found")
    row.is_excluded = excluded
    db.commit()
    store.log_event(session_id, "api", f"Source {'excluded' if excluded else 're-included'}: {row.url}")
    return {"source_id": str(source_id), "is_excluded": excluded}


@router.get("/sessions/{session_id}/claims", response_model=list[schemas.ClaimOut], tags=["evidence"])
async def list_claims(
    session_id: uuid.UUID,
    db: DbSession,
    plan_item_id: Annotated[uuid.UUID | None, Query()] = None,
    min_importance: Annotated[int | None, Query(ge=1, le=3)] = None,
) -> list[schemas.ClaimOut]:
    _session_or_404(db, session_id)
    query = select(Claim).where(Claim.session_id == session_id)
    if plan_item_id is not None:
        query = query.where(Claim.plan_item_id == plan_item_id)
    if min_importance is not None:
        query = query.where(Claim.importance <= min_importance)
    rows = db.execute(query.order_by(Claim.importance, Claim.created_at)).scalars().all()
    out: list[schemas.ClaimOut] = []
    for row in rows:
        passages = db.execute(select(Evidence).where(Evidence.claim_id == row.id)).scalars().all()
        out.append(
            schemas.ClaimOut(
                id=row.id,
                source_id=row.source_id,
                plan_item_id=row.plan_item_id,
                text=row.text,
                claim_type=row.claim_type,
                importance=row.importance,
                subject=row.subject,
                evidence=[
                    {
                        "passage": p.passage,
                        "position": p.position,
                        "relevance_score": p.relevance_score,
                        "source_id": str(p.source_id),
                    }
                    for p in passages
                ],
            )
        )
    return out


@router.get(
    "/sessions/{session_id}/contradictions", response_model=list[schemas.ContradictionOut], tags=["evidence"]
)
async def list_contradictions(session_id: uuid.UUID, db: DbSession) -> list[schemas.ContradictionOut]:
    _session_or_404(db, session_id)
    rows = db.execute(
        select(Contradiction).where(Contradiction.session_id == session_id).order_by(Contradiction.created_at)
    ).scalars().all()
    return [schemas.ContradictionOut.model_validate(row, from_attributes=True) for row in rows]


# reports
@router.get("/sessions/{session_id}/reports", response_model=list[schemas.ReportOut], tags=["reports"])
async def list_reports(session_id: uuid.UUID, db: DbSession) -> list[schemas.ReportOut]:
    _session_or_404(db, session_id)
    rows = db.execute(
        select(Report).where(Report.session_id == session_id).order_by(Report.version)
    ).scalars().all()
    return [schemas.ReportOut.model_validate(row, from_attributes=True) for row in rows]


@router.get("/sessions/{session_id}/report", response_model=schemas.ReportOut, tags=["reports"])
async def get_report(session_id: uuid.UUID, db: DbSession, version: int | None = None) -> schemas.ReportOut:
    _session_or_404(db, session_id)
    query = select(Report).where(Report.session_id == session_id)
    if version is not None:
        query = query.where(Report.version == version)
    row = db.execute(query.order_by(Report.version.desc()).limit(1)).scalars().first()
    if row is None:
        raise HTTPException(status_code=404, detail="no report yet")
    return schemas.ReportOut.model_validate(row, from_attributes=True)


@router.get(
    "/sessions/{session_id}/report/{version}/verifications",
    response_model=list[schemas.VerificationOut],
    tags=["reports"],
    summary="Claim-level citation audit for a report version",
)
async def report_verifications(session_id: uuid.UUID, version: int, db: DbSession) -> list[schemas.VerificationOut]:
    _session_or_404(db, session_id)
    report = db.execute(
        select(Report).where(Report.session_id == session_id, Report.version == version)
    ).scalars().first()
    if report is None:
        raise HTTPException(status_code=404, detail="report version not found")
    rows = db.execute(
        select(ReportVerification).where(ReportVerification.report_id == report.id)
    ).scalars().all()
    return [schemas.VerificationOut.model_validate(row, from_attributes=True) for row in rows]


# trace
@router.get("/sessions/{session_id}/events", response_model=list[schemas.EventOut], tags=["trace"])
async def list_events(
    session_id: uuid.UUID,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    node: Annotated[str | None, Query()] = None,
) -> list[schemas.EventOut]:
    _session_or_404(db, session_id)
    query = select(RunEvent).where(RunEvent.session_id == session_id)
    if node:
        query = query.where(RunEvent.node == node)
    rows = db.execute(query.order_by(RunEvent.created_at.desc()).limit(limit)).scalars().all()
    return [schemas.EventOut.model_validate(row, from_attributes=True) for row in reversed(rows)]