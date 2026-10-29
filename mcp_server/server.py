"""MCP server exposing web_search, fetch_page and find_relevant_passages.

HTTP: python -m mcp_server.server (port from MCP_HTTP_PORT, default 8900)
stdio: python -m mcp_server
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from mcp_server import core

mcp = FastMCP(
    name="researchos-research-toolbelt",
    host="0.0.0.0",
    port=int(os.environ.get("MCP_HTTP_PORT", "8900")),
    instructions=(
        "Tools for collecting and reading web evidence. Use web_search to find "
        "candidate sources, fetch_page to retrieve readable document text, and "
        "find_relevant_passages to locate the parts of a document that answer a "
        "specific research subquestion."
    ),
)


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return ranked results (title, url, snippet, domain).

    Use this to discover candidate sources for a research subquestion. Snippets
    are short previews; call fetch_page on the most promising URLs to get the
    full text.
    """
    return await core.web_search(query, max_results=max_results)


@mcp.tool()
async def fetch_page(url: str, max_chars: int = 24000) -> dict:
    """Fetch a URL and return its readable text content plus metadata.

    Returns a dict with keys: url, final_url, title, text, http_status,
    content_hash, truncated. `text` is plain prose with navigation/boilerplate
    removed; it is what you should cite evidence from.
    """
    return await core.fetch_page(url, max_chars=max_chars)


@mcp.tool()
def find_relevant_passages(text: str, focus: str, max_passages: int = 6) -> list[dict]:
    """Rank sentences of `text` by relevance to `focus`, returning verbatim passages.

    Each result is {passage, score, position}. `position` is the sentence index
    in the source document, useful for citing the passage later. Passages are
    verbatim excerpts, so they can be quoted as evidence.
    """
    return core.find_relevant_passages(text, focus=focus, max_passages=max_passages)


if __name__ == "__main__":
    # Streamable-HTTP mode for standalone operation / docker-compose.
    mcp.run(transport="streamable-http")
