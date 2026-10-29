"""FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Header
from sqlalchemy.orm import Session

from app.db.session import get_sessionmaker


def get_db(x_client_id: Annotated[str | None, Header()] = None) -> Iterator[Session]:
    session = get_sessionmaker()()
    # The browser sends a random id (kept in a cookie) so each visitor only sees their own sessions.
    session.info["client_id"] = (x_client_id or "").strip()[:64] or None
    try:
        yield session
    finally:
        session.close()