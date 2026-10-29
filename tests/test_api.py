"""API tests: the full HTTP surface, driving a real run through the runner."""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services import runner

QUESTION = (
    "Is PostgreSQL a better choice than MongoDB for a SaaS application? "
    "Compare performance, scalability, consistency, operational complexity and cost."
)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://researchos.test") as http:
        yield http


async def _create_session(client, **config) -> str:
    response = await client.post("/api/v1/sessions", json={"question": QUESTION, "config": config})
    assert response.status_code == 201, response.text
    return response.json()["id"]


async def _run_to_completion(client, session_id: str) -> None:
    response = await client.post(f"/api/v1/sessions/{session_id}/run", json={"start": True})
    assert response.status_code == 200, response.text
    await runner.wait_for_run(uuid.UUID(session_id), timeout=180)


async def test_health(client):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_create_and_list_sessions(client):
    session_id = await _create_session(client)
    listing = await client.get("/api/v1/sessions")
    assert listing.status_code == 200
    payload = listing.json()
    assert [item["id"] for item in payload] == [session_id]
    assert payload[0]["status"] == "created"
    assert payload[0]["counts"]["sources"] == 0


async def test_invalid_question_is_rejected(client):
    response = await client.post("/api/v1/sessions", json={"question": "short"})
    assert response.status_code == 422


async def test_unknown_session_returns_404(client):
    response = await client.get("/api/v1/sessions/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


async def test_report_is_404_before_a_run(client):
    session_id = await _create_session(client)
    assert (await client.get(f"/api/v1/sessions/{session_id}/report")).status_code == 404
    assert (await client.get(f"/api/v1/sessions/{session_id}/plan")).status_code == 404


@pytest.mark.asyncio
async def test_end_to_end_research_session(client, scripted_llm):
    session_id = await _create_session(client, sufficiency_threshold=0.75, max_research_iterations=2)
    await _run_to_completion(client, session_id)

    detail = (await client.get(f"/api/v1/sessions/{session_id}")).json()
    assert detail["status"] == "completed"
    assert detail["counts"]["sources"] >= 4
    assert detail["counts"]["claims"] >= 6
    assert detail["counts"]["contradictions"] >= 1
    assert detail["counts"]["reports"] >= 1
    assert detail["latest_report_version"] >= 1
    assert detail["verification_summary"]["supported"] >= 1
    # reproducibility metadata
    assert detail["config"]["sufficiency_threshold"] == 0.75
    assert any(rating["sufficiency"] is not None for rating in detail["ratings"])

    plan = (await client.get(f"/api/v1/sessions/{session_id}/plan")).json()
    assert len(plan["items"]) >= 3
    assert plan["items"][0]["status"] in {"sufficient", "insufficient"}

    sources = (await client.get(f"/api/v1/sessions/{session_id}/sources")).json()
    assert all(source["url"].startswith("http") for source in sources)
    assert all(0 <= (source["quality_score"] or 0) <= 1 for source in sources)
    assert all(source["content_hash"] for source in sources)  # snapshot hash for reproducibility
    assert all(source["text_chars"] > 0 for source in sources)

    claims = (await client.get(f"/api/v1/sessions/{session_id}/claims")).json()
    assert all(claim["evidence"] for claim in claims), "claims must carry verbatim evidence"
    assert all(claim["evidence"][0]["passage"] for claim in claims)

    contradictions = (await client.get(f"/api/v1/sessions/{session_id}/contradictions")).json()
    assert contradictions
    assert all(item["context"] for item in contradictions)

    report = (await client.get(f"/api/v1/sessions/{session_id}/report")).json()
    assert "[S1]" in report["markdown"]
    assert "## Comparison matrix" in report["markdown"]
    assert "## Limitations" in report["markdown"]
    assert report["verification_summary"]["checked"] > 0

    verifications = (
        await client.get(f"/api/v1/sessions/{session_id}/report/{plan and detail['latest_report_version']}/verifications")
    ).json()
    assert len(verifications) == detail["verification_summary"]["checked"]
    assert {v["verdict"] for v in verifications} <= {"supported", "partial", "unsupported"}

    versions = (await client.get(f"/api/v1/sessions/{session_id}/reports")).json()
    assert len(reports := versions) == detail["counts"]["reports"]
    assert any(report["revision_reason"] for report in reports)  # revision is recorded

    events = (await client.get(f"/api/v1/sessions/{session_id}/events")).json()
    assert {event["node"] for event in events} >= {"plan", "gather", "synthesize", "write", "verify", "finalize"}
    assert any(event["kind"] == "warning" for event in events)  # the research loop was triggered


async def test_source_can_be_excluded(client, scripted_llm):
    session_id = await _create_session(client)
    await _run_to_completion(client, session_id)
    sources = (await client.get(f"/api/v1/sessions/{session_id}/sources")).json()

    response = await client.post(
        f"/api/v1/sessions/{session_id}/sources/{sources[0]['id']}/exclude?excluded=true"
    )
    assert response.status_code == 200
    assert response.json()["is_excluded"] is True
    refreshed = (await client.get(f"/api/v1/sessions/{session_id}/sources")).json()
    assert any(source["is_excluded"] for source in refreshed)


@pytest.mark.asyncio
async def test_plan_approval_flow(client, scripted_llm):
    session_id = await _create_session(client, approval_required=True)
    response = await client.post(f"/api/v1/sessions/{session_id}/run", json={"start": True})
    assert response.status_code == 200
    await runner.wait_for_run(uuid.UUID(session_id), timeout=60)

    detail = (await client.get(f"/api/v1/sessions/{session_id}")).json()
    assert detail["status"] == "awaiting_approval"
    assert detail["counts"]["sources"] == 0

    approved = await client.post(f"/api/v1/sessions/{session_id}/plan/approve")
    assert approved.status_code == 200
    assert approved.json()["approved"] is True
    await runner.wait_for_run(uuid.UUID(session_id), timeout=180)

    detail = (await client.get(f"/api/v1/sessions/{session_id}")).json()
    assert detail["status"] == "completed"
    assert detail["counts"]["sources"] >= 4
    plan = (await client.get(f"/api/v1/sessions/{session_id}/plan")).json()
    assert plan["approved_at"] is not None


async def test_verifications_endpoint_validates_report_version(client, scripted_llm):
    session_id = await _create_session(client)
    await _run_to_completion(client, session_id)
    assert (await client.get(f"/api/v1/sessions/{session_id}/report/99/verifications")).status_code == 404
