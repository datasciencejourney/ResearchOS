"""Settings, read from env vars / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API
    app_name: str = "ResearchOS"
    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # PostgreSQL
    # Used both by SQLAlchemy and by the LangGraph Postgres checkpointer.
    database_url: str = "postgresql+psycopg://researchos:researchos@127.0.0.1:5432/researchos"
    # DSN understood by psycopg directly (no SQLAlchemy driver suffix).
    checkpointer_database_url: str = ""

    # LLM
    # ResearchOS is provider agnostic: any OpenAI-compatible chat endpoint
    # works.  Defaults to the Gemini OpenAI-compatibility layer.
    llm_provider: str = "openai_compatible"
    openai_api_key: str = ""
    google_api_key: str = ""
    openai_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-3.8-flash"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 8192
    # "none" keeps latency/cost low for the many small structured calls the
    # workflow makes; set to "low"/"medium"/"high" for harder questions.
    llm_reasoning_effort: str = "none"
    llm_timeout_seconds: int = 120
    llm_max_retries: int = 3
    # Client-side throttle for providers with strict quotas (e.g. Gemini's free
    # tier allows 5 requests/minute). 0 disables throttling. Calls are spaced so
    # that at most this many requests are started per rolling minute.
    llm_max_requests_per_minute: int = 0

    # MCP
    # The researcher talks to the outside world exclusively through MCP tools.
    mcp_transport: str = "stdio"  # "stdio" or "http"
    mcp_server_url: str = "http://127.0.0.1:8900/mcp"
    mcp_command: str = "python"
    mcp_server_module: str = "mcp_server"
    mcp_http_port: int = 8900

    # Research behaviour
    max_research_iterations: int = Field(default=3, ge=1, le=8)
    min_subquestions: int = 2
    max_subquestions: int = 5
    max_search_results_per_query: int = 5
    max_sources_per_subquestion: int = 4
    max_fetch_concurrency: int = 4
    max_chars_per_source: int = 24_000
    # A subquestion whose sufficiency score is below this gets re-researched.
    sufficiency_threshold: float = 0.6
    # Report revision loop: rewrite at most this many times when the
    # verification pass finds unsupported claims.
    max_report_revisions: int = 1
    # Maximum number of cited statements checked per report version.
    max_citations_to_verify: int = 14
    # Minimum share of supported citations required to accept a report.
    min_supported_citation_ratio: float = 0.75
    user_agent: str = "ResearchOS/0.1 (+https://example.invalid/researchos)"

    @property
    def dsn(self) -> str:
        """DSN for the LangGraph Postgres checkpointer (no driver suffix)."""
        if self.checkpointer_database_url:
            return self.checkpointer_database_url
        return self.database_url.replace("postgresql+psycopg://", "postgresql://")

    @property
    def resolved_api_key(self) -> str:
        return self.openai_api_key or self.google_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
