"""Runs the graph as a background task so the API call returns right away."""

from __future__ import annotations

import asyncio
import uuid

from app.config import get_settings
from app.graph.graph import build_graph
from app.services import persistence as store
from app.services.checkpointer import init_checkpointer

# One active run per session; guarded to avoid accidental double-starts.
_ACTIVE: dict[str, asyncio.Task] = {}


def is_running(session_id: uuid.UUID) -> bool:
    task = _ACTIVE.get(str(session_id))
    return task is not None and not task.done()


def running_count() -> int:
    """Return the number of research runs that are still active."""
    return sum(not task.done() for task in _ACTIVE.values())


async def run_session(session_id: uuid.UUID) -> None:
    settings = get_settings()
    session = store.get_session(session_id)
    if session is None:
        return
    store.set_status(session_id, "planning")
    try:
        checkpointer = await init_checkpointer()
        graph = build_graph(checkpointer)
        config = {
            "configurable": {"thread_id": str(session_id)},
            "recursion_limit": 400,
        }
        initial_state = {
            "session_id": str(session_id),
            "question": session.question,
            "config": session.config or {},
        }
        async for _ in graph.astream(initial_state, config):
            pass
    except Exception as exc:  # noqa: BLE001 - the failure must be visible to the user
        store.set_status(session_id, "failed", error=f"{type(exc).__name__}: {exc}", finished=True)
        store.log_event(session_id, "runner", f"run failed: {type(exc).__name__}: {exc}", kind="error")
        raise
    finally:
        _ACTIVE.pop(str(session_id), None)


def start_run(session_id: uuid.UUID) -> asyncio.Task:
    if is_running(session_id):
        raise RuntimeError("a run is already in progress for this session")
    task = asyncio.create_task(run_session(session_id))
    _ACTIVE[str(session_id)] = task
    return task


async def wait_for_run(session_id: uuid.UUID, timeout: float | None = None) -> None:
    """Test helper: block until the session's run finishes."""
    task = _ACTIVE.get(str(session_id))
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
