"""Search / fetch / passage ranking code, kept free of MCP imports so it can be tested directly."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import urllib.parse
from collections import deque

import httpx
from bs4 import BeautifulSoup

USER_AGENT = os.environ.get(
    "MCP_USER_AGENT",
    "ResearchOS/0.1 (+https://example.invalid/researchos)",
)
REQUEST_TIMEOUT = float(os.environ.get("MCP_REQUEST_TIMEOUT", "25"))
MAX_RETRIES = 2

DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"


async def _get(client: httpx.AsyncClient, url: str) -> httpx.Response | None:
    """GET with a couple of retries; returns None when the page is unreachable."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await client.get(url)
        except (httpx.HTTPError, httpx.InvalidURL):
            await asyncio.sleep(0.5 * (attempt + 1))
    return None


def parse_search_results(html: str) -> list[dict]:
    """Parse DuckDuckGo's HTML results page into {title, url, snippet, domain}."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []
    for anchor in soup.select("a.result__a"):
        href = anchor.get("href", "")
        parsed = urllib.parse.urlparse(href)
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
        if not target.startswith("http"):
            continue
        snippet = ""
        parent = anchor.find_parent(class_="result")
        if parent is not None:
            snippet_el = parent.select_one(".result__snippet")
            if snippet_el is not None:
                snippet = re.sub(r"\s+", " ", snippet_el.get_text(" ", strip=True))
        results.append(
            {
                "title": re.sub(r"\s+", " ", anchor.get_text(" ", strip=True)),
                "url": target,
                "snippet": snippet,
                "domain": urllib.parse.urlparse(target).netloc.removeprefix("www."),
            }
        )
    return results


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return ranked results (title, url, snippet, domain)."""
    max_results = max(1, min(int(max_results), 10))
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT, follow_redirects=True
    ) as client:
        response = await _get(client, f"{DDG_HTML_ENDPOINT}?q={urllib.parse.quote(query)}")
    if response is None or response.status_code != 200:
        return []
    return parse_search_results(response.text)[:max_results]


def clean_text(html: str) -> str:
    """Strip boilerplate from an HTML document and return readable prose."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "form", "header", "footer", "nav"]):
        tag.decompose()
    block = soup.find("article") or soup.body or soup
    pieces: list[str] = []
    for el in block.find_all(["p", "li", "h1", "h2", "h3", "h4", "blockquote", "td"]):
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if len(text) >= 40:  # skip nav crumbs / one-word labels
            pieces.append(text)
    seen: set[str] = set()
    out: list[str] = []
    for piece in pieces:
        if piece not in seen:
            seen.add(piece)
            out.append(piece)
    return "\n".join(out)


async def fetch_page(url: str, max_chars: int = 24_000) -> dict:
    """Fetch a URL and return readable text plus metadata."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT, follow_redirects=True
    ) as client:
        response = await _get(client, url)
    if response is None:
        return {"url": url, "error": "unreachable", "text": "", "http_status": 0}
    if response.status_code >= 400:
        return {
            "url": url,
            "final_url": str(response.url),
            "error": f"http_{response.status_code}",
            "text": "",
            "http_status": response.status_code,
        }

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else url
    text = clean_text(response.text)
    return {
        "url": url,
        "final_url": str(response.url),
        "title": title[:500],
        "text": text[:max_chars],
        "http_status": response.status_code,
        "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        "truncated": len(text) > max_chars,
    }


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _blocks(text: str, max_block_chars: int = 1200) -> list[str]:
    """Split cleaned document text into citable blocks (paragraph/li-sized)."""
    blocks: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_block_chars:
            blocks.append(line)
        else:
            blocks.extend(sentence.strip() for sentence in _SENTENCE_SPLIT.split(line) if sentence.strip())
    return blocks


def find_relevant_passages(
    text: str, focus: str, max_passages: int = 6, min_words: int = 8
) -> list[dict]:
    """Rank blocks of `text` by relevance to `focus`, returning verbatim passages.

    Uses TF-IDF style weighting (rare focus terms count for more) so a passage
    about latency outranks one that merely repeats the word "performance".
    Passages are verbatim excerpts and can therefore be quoted as evidence.
    """
    focus_terms = {term for term in _WORD_RE.findall(focus.lower()) if len(term) > 2}
    if not focus_terms:
        return []

    blocks = _blocks(text)
    tokenised = [_WORD_RE.findall(block.lower()) for block in blocks]
    document_frequency = {
        term: sum(1 for tokens in tokenised if term in tokens) for term in focus_terms
    }

    scored: list[tuple[float, int, str]] = []
    for index, (block, tokens) in enumerate(zip(blocks, tokenised, strict=True)):
        if len(tokens) < min_words:
            continue
        hits = focus_terms.intersection(tokens)
        if not hits:
            continue
        rarity = sum(1.0 / (1.0 + document_frequency[term]) for term in hits)
        coverage = len(hits) / len(focus_terms)
        scored.append((0.6 * rarity + 0.4 * coverage, index, block))
    scored.sort(key=lambda row: row[0], reverse=True)

    picked: list[dict] = []
    seen: deque[str] = deque(maxlen=16)
    for score, index, block in scored:
        marker = block.lower()[:120]
        if any(marker[:60] in previous or previous[:60] in marker for previous in seen):
            continue
        seen.append(marker)
        picked.append({"passage": block, "score": round(score, 4), "position": index})
        if len(picked) >= max(1, max_passages):
            break
    return picked
