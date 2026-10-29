"""Postgres checkpointer for langgraph, keyed by session id."""

from __future__ import annotations

from typing import Any

from app.config import get_settings

_saver: Any = None
_cm: Any = None


async def init_checkpointer():
    """Create (once) the async Postgres saver and its tables."""
    global _saver, _cm
    if _saver is None:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        dsn = get_settings().dsn
        _cm = AsyncPostgresSaver.from_conn_string(dsn)
        _saver = await _cm.__aenter__()
        await _saver.setup()
    return _saver


def get_checkpointer():
    """Already-initialised saver (None when tests run without checkpointing)."""
    return _saver


async def close_checkpointer() -> None:
    global _saver, _cm
    if _cm is not None:
        await _cm.__aexit__(None, None, None)
    _saver = None
    _cm = None
