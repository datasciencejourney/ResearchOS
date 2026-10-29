"""ResearchOS application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.checkpointer import close_checkpointer, init_checkpointer

    try:
        await init_checkpointer()  # LangGraph checkpointer tables in Postgres
    except Exception as exc:  # noqa: BLE001 - degrade loudly but keep the API up
        app.state.checkpointer_error = str(exc)
    yield
    await close_checkpointer()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Multi-agent research and intelligence platform: adaptive planning, "
            "evidence gathering over MCP, contradiction detection, claim-level "
            "citation verification and reproducible research sessions."
        ),
        lifespan=lifespan,
    )
    app.include_router(router, prefix=settings.api_prefix)
    return app


app = create_app()
