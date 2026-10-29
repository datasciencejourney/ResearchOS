"""Test configuration: an isolated PostgreSQL database plus scripted agents."""

from __future__ import annotations

import os
import uuid

import pytest

TEST_DB = "researchos_test"
ADMIN_DSN = os.environ.get("RESEARCHOS_ADMIN_DSN", "postgresql://postgres@127.0.0.1:5432/postgres")
TEST_DSN = f"postgresql://researchos:researchos@127.0.0.1:5432/{TEST_DB}"

# Must be set before any app module reads settings.
os.environ["DATABASE_URL"] = TEST_DSN.replace("postgresql://", "postgresql+psycopg://")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("MCP_TRANSPORT", "stdio")

import psycopg  # noqa: E402
from sqlalchemy import text  # noqa: E402

from tests.fakes import FakeResearchTools, ResearchFakeModel, ScriptedChatModel  # noqa: E402

TABLES = (
    "run_events",
    "report_verifications",
    "reports",
    "contradictions",
    "evidence",
    "claims",
    "sources",
    "research_plan_items",
    "research_plans",
    "research_sessions",
)


@pytest.fixture(scope="session", autouse=True)
def _database() -> None:
    """Create a throwaway database and the schema once per test session."""
    with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{TEST_DB}" OWNER researchos')

    from app.config import get_settings
    from app.db.models import Base
    from app.db.session import get_engine, reset_engine

    get_settings.cache_clear()
    reset_engine()
    Base.metadata.create_all(get_engine())
    yield
    reset_engine()


@pytest.fixture(autouse=True)
def _clean_tables(_database) -> None:
    """Each test starts from empty tables."""
    from app.db.session import session_scope

    with session_scope() as db:
        db.execute(text(f"TRUNCATE {', '.join(TABLES)} CASCADE"))
    yield


@pytest.fixture(autouse=True)
def _reset_runtime():
    """Drop loop-bound runtime state between tests (checkpointer pool, tasks)."""
    yield
    from app.services import checkpointer, runner

    runner._ACTIVE.clear()
    checkpointer._saver = None
    checkpointer._cm = None


@pytest.fixture
def fake_tools() -> FakeResearchTools:
    return FakeResearchTools()


@pytest.fixture
def scripted_llm(monkeypatch, fake_tools):
    """Patch the agent layer with a deterministic model and offline toolbelt."""
    handler = ResearchFakeModel()
    model = ScriptedChatModel(handler)
    monkeypatch.setattr("app.graph.nodes.get_chat_model", lambda *args, **kwargs: model)
    monkeypatch.setattr("app.graph.nodes.get_research_tools", lambda: fake_tools)
    return handler


@pytest.fixture
def session_id() -> uuid.UUID:
    """A persisted research session ready to run."""
    from app.db.models import ResearchSession, SessionStatus
    from app.db.session import session_scope

    with session_scope() as db:
        row = ResearchSession(
            question=(
                "Is PostgreSQL a better choice than MongoDB for a SaaS application? "
                "Compare performance, scalability, consistency, operational complexity and cost."
            ),
            status=SessionStatus.CREATED,
            config={"sufficiency_threshold": 0.75, "max_research_iterations": 2},
        )
        db.add(row)
        db.flush()
        return row.id
