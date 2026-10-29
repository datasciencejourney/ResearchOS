"""Graph tests, using the fakes (no network, no real LLM). Needs Postgres."""

from __future__ import annotations

import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.db.models import Claim, Contradiction, Evidence, ResearchSession, RunEvent, Source
from app.db.session import session_scope
from app.graph.graph import build_graph
from app.services import persistence as store
from sqlalchemy import func, select


async def _run(session_id: uuid.UUID, config: dict | None = None) -> dict:
    session = store.get_session(session_id)
    graph = build_graph(InMemorySaver())
    return await graph.ainvoke(
        {
            "session_id": str(session_id),
            "question": session.question,
            "config": {**(session.config or {}), **(config or {})},
        },
        {"configurable": {"thread_id": str(session_id)}, "recursion_limit": 200},
    )


@pytest.mark.asyncio
async def test_full_run_produces_verified_cited_report(session_id, scripted_llm):
    final = await _run(session_id)

    assert final["final"] is True
    assert final["report_markdown"].startswith("# ")
    assert "[S1]" in final["report_markdown"]
    assert final["report_version"] >= 1

    # evidence trail
    unique_urls = {source["url"] for source in final["sources"]}
    assert len(unique_urls) >= 4
    assert len(final["claims"]) >= 6
    assert final["contradictions"], "contradiction detection should fire on conflicting sources"
    assert all(claim["evidence"] for claim in final["claims"]), "every claim carries verbatim evidence"

    # verification
    summary = final["verification_summary"]
    assert summary["checked"] > 0
    assert summary["supported"] >= 1
    assert summary["supported_ratio"] >= summary["threshold"]

    with session_scope() as db:
        session = db.get(ResearchSession, session_id)
        assert session.status == "completed"
        assert session.finished_at is not None
        # one row per unique source: parallel branches may re-discover a URL
        assert db.execute(select(func.count()).select_from(Source)).scalar_one() == len(unique_urls)
        assert db.execute(select(func.count()).select_from(Claim)).scalar_one() >= 6
        assert db.execute(select(func.count()).select_from(Evidence)).scalar_one() >= 6
        assert db.execute(select(func.count()).select_from(Contradiction)).scalar_one() >= 1
        # the workflow is fully traced
        nodes = set(
            db.execute(select(RunEvent.node).distinct()).scalars().all()
        )
    assert {"plan", "gather", "synthesize", "write", "verify", "finalize"} <= nodes


@pytest.mark.asyncio
async def test_adaptive_loop_researches_under_evidenced_subquestion(session_id, scripted_llm):
    final = await _run(session_id)
    # round 1 scored subquestion 0 as insufficient (0.1) with threshold 0.75,
    # so the workflow must have gone back to research before writing.
    assert final["round"] >= 2
    assert scripted_llm.sufficiency_calls >= 2
    with session_scope() as db:
        messages = db.execute(select(RunEvent.message)).scalars().all()
    assert any("routing back to research" in message for message in messages)
    # the failing item ends up flagged, and the report still gets written
    assert final["report_markdown"]


@pytest.mark.asyncio
async def test_citation_audit_sends_report_back_for_revision(session_id, scripted_llm):
    final = await _run(session_id)
    # the first draft contains a claim the cited passage cannot support
    assert scripted_llm.report_calls >= 2, "report should have been revised"
    assert final["revisions"] >= 1
    assert "UNSUPPORTED" not in final["report_markdown"]
    with session_scope() as db:
        messages = db.execute(select(RunEvent.message)).scalars().all()
    assert any("requesting revision" in message for message in messages)


@pytest.mark.asyncio
async def test_below_threshold_audit_finalises_honestly(session_id, monkeypatch):
    """If revisions cannot fix the citations, the run must not claim success."""
    from tests.fakes import FakeResearchTools, ResearchFakeModel, ScriptedChatModel

    handler = ResearchFakeModel(fail_revision=True)
    model = ScriptedChatModel(handler)
    monkeypatch.setattr("app.graph.nodes.get_chat_model", lambda *args, **kwargs: model)
    monkeypatch.setattr("app.graph.nodes.get_research_tools", lambda: FakeResearchTools())

    final = await _run(session_id)

    assert final["final"] is True
    assert final["revisions"] >= 1
    summary = final["verification_summary"]
    assert summary["passed"] is False
    assert summary["supported_ratio"] < summary["threshold"]
    with session_scope() as db:
        messages = db.execute(select(RunEvent.message)).scalars().all()
    assert any("revision budget exhausted" in message for message in messages)


@pytest.mark.asyncio
async def test_run_is_reproducible_reusing_approved_plan(session_id, scripted_llm):
    first = await _run(session_id)
    plan_items = [item["subquestion"] for item in first["plan"]["items"]]
    assert store.approve_plan(session_id)

    # a second run must reuse the stored approved plan instead of re-planning
    second = await _run(session_id)
    assert [item["subquestion"] for item in second["plan"]["items"]] == plan_items
    with session_scope() as db:
        messages = db.execute(select(RunEvent.message)).scalars().all()
    assert any("Reusing approved research plan" in message for message in messages)


@pytest.mark.asyncio
async def test_approval_gate_pauses_before_research(session_id, scripted_llm):
    final = await _run(session_id, config={"approval_required": True})
    assert not final.get("final")
    assert final["queue"], "plan is queued but research must not start"
    assert store.get_session(session_id).status == "awaiting_approval"
    assert not final.get("sources")

    # after approval the same thread continues into research
    store.approve_plan(session_id)
    resumed = await _run(session_id, config={"approval_required": True})
    assert resumed["final"] is True
    assert resumed["sources"]


@pytest.mark.asyncio
async def test_excluded_source_is_not_used_again(session_id, scripted_llm, fake_tools):
    first = await _run(session_id)
    assert first["sources"]
    excluded_url = first["sources"][0]["url"]

    with session_scope() as db:
        row = db.execute(select(Source).where(Source.url == excluded_url)).scalars().first()
        row.is_excluded = True
    assert store.excluded_urls(session_id) == {excluded_url}

    # a follow-up run must not fetch the excluded source again
    fake_tools.fetches.clear()
    await _run(session_id)
    assert excluded_url not in fake_tools.fetches
