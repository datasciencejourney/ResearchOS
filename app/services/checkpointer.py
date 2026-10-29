"""Postgres checkpointer for langgraph, keyed by session id."""

from __future__ import annotations

from typing import Any

from app.config import get_settings

_saver: Any = None
_cm: Any = None  # the connection pool behind the saver


async def init_checkpointer():
    """Create (once) the async Postgres saver and its tables.

    The saver uses a connection pool that tests each connection before handing
    it out. A single long-lived connection breaks on hosts that drop idle
    connections (Neon suspends its compute after a few minutes).
    """
    global _saver, _cm
    if _saver is None:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        pool = AsyncConnectionPool(
            conninfo=get_settings().dsn,
            min_size=1,
            max_size=5,
            max_lifetime=240,
            check=AsyncConnectionPool.check_connection,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            open=False,
        )
        await pool.open(wait=True)
        saver = AsyncPostgresSaver(pool)
        try:
            await saver.setup()
        except Exception:
            await pool.close()
            raise
        _cm, _saver = pool, saver
    return _saver


def get_checkpointer():
    """Already-initialised saver (None when tests run without checkpointing)."""
    return _saver


async def close_checkpointer() -> None:
    global _saver, _cm
    if _cm is not None:
        await _cm.close()
    _saver = None
    _cm = None