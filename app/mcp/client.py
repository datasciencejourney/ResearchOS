"""MCP client. All web access from the graph goes through here."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


class ResearchTools:
    """Thin async wrapper around the MCP research toolbelt."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._tools: list[Any] | None = None

    # -- connection -------------------------------------------------------
    def _client(self):
        from langchain_mcp_adapters.client import MultiServerMCPClient

        if self.settings.mcp_transport == "http":
            connection = {"transport": "http", "url": self.settings.mcp_server_url}
        else:
            env = {**os.environ, "PYTHONPATH": PROJECT_ROOT}
            connection = {
                "transport": "stdio",
                "command": self.settings.mcp_command,
                "args": ["-m", self.settings.mcp_server_module],
                "env": env,
                "cwd": PROJECT_ROOT,
            }
        return MultiServerMCPClient({"researchos": connection})

    async def tools(self) -> list[Any]:
        """Load (and cache) the MCP tools as LangChain tools."""
        if self._tools is None:
            client = self._client()
            self._tools = list(await client.get_tools())
        return self._tools

    async def close(self) -> None:
        self._tools = None

    # -- tool calls -------------------------------------------------------
    @staticmethod
    def _normalize(result: Any) -> Any:
        """MCP returns JSON payloads wrapped in content blocks; unwrap them.

        A list-returning tool comes back as one content block per element, so
        each block is parsed on its own.
        """
        if isinstance(result, list):
            blocks = [
                part for part in result if isinstance(part, dict) and part.get("type") == "text"
            ]
            if not blocks:
                return result
            parsed: list[Any] = []
            for block in blocks:
                text = block.get("text", "")
                try:
                    parsed.append(json.loads(text))
                except json.JSONDecodeError:
                    parsed.append(text)
            return parsed[0] if len(parsed) == 1 else parsed
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return result
        return result

    async def _call(self, name: str, **kwargs) -> Any:
        tools = {tool.name: tool for tool in await self.tools()}
        tool = tools.get(name)
        if tool is None:  # pragma: no cover - misconfigured server
            raise RuntimeError(f"MCP tool not found: {name}")
        return self._normalize(await tool.ainvoke(kwargs))

    async def web_search(self, query: str, max_results: int = 5) -> list[dict]:
        result = await self._call("web_search", query=query, max_results=max_results)
        if isinstance(result, dict):
            result = result.get("results") or []
        return [item for item in (result or []) if isinstance(item, dict)]

    async def fetch_page(self, url: str, max_chars: int = 24_000) -> dict:
        result = await self._call("fetch_page", url=url, max_chars=max_chars)
        if isinstance(result, dict):
            return result
        return {"url": url, "error": "unexpected_response", "text": ""}

    async def find_relevant_passages(
        self, text: str, focus: str, max_passages: int = 8
    ) -> list[dict]:
        result = await self._call(
            "find_relevant_passages", text=text, focus=focus, max_passages=max_passages
        )
        if isinstance(result, dict):
            result = result.get("passages") or []
        return [item for item in (result or []) if isinstance(item, dict)]


_default: ResearchTools | None = None


def get_research_tools() -> ResearchTools:
    global _default
    if _default is None:
        _default = ResearchTools()
    return _default
