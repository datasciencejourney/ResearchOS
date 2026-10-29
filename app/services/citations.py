"""Helpers for the [S1], [S2] ... citations in the report: building the source
registry, splitting the report into sentences and matching them to passages.
"""

from __future__ import annotations

import re
from collections import defaultdict

CITATION_RE = re.compile(r"\[S(\d+)\]")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "was", "were",
    "this", "that", "with", "from", "have", "has", "had", "its", "it's", "into",
    "than", "then", "them", "they", "their", "there", "which", "when", "what",
    "would", "should", "could", "about", "more", "most", "some", "such", "only",
    "also", "between", "while", "both", "each", "these", "those", "does", "did",
}


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2 and token not in _STOPWORDS
    }


def build_registry(sources: list[dict]) -> list[dict]:
    """Assign stable [S#] markers to sources.

    Sources that produced claims come first (they are the ones a report can
    actually cite), ordered by domain for determinism.  Parallel research
    branches may return the same URL more than once, so URLs are de-duplicated
    here: one source means exactly one [S#] marker.
    """
    best_by_url: dict[str, dict] = {}
    for source in sources:
        url = source.get("url", "")
        if not url:
            continue
        previous = best_by_url.get(url)
        if previous is None or source.get("claim_count", 0) > previous.get("claim_count", 0):
            best_by_url[url] = {**source}
    unique = list(best_by_url.values())
    with_claims = [s for s in unique if s.get("claim_count", 1) != 0]
    without = [s for s in unique if s.get("claim_count", 1) == 0]
    ordered = sorted(with_claims, key=lambda s: (s.get("domain", ""), s.get("url", "")))
    ordered += sorted(without, key=lambda s: (s.get("domain", ""), s.get("url", "")))
    registry = []
    for index, source in enumerate(ordered, start=1):
        registry.append(
            {
                "marker": f"S{index}",
                "index": index,
                "url": source.get("url", ""),
                "title": source.get("title", "") or source.get("url", ""),
                "domain": source.get("domain", ""),
                "published_date": source.get("published_date", ""),
                "quality_score": source.get("quality_score"),
                "quality_notes": source.get("quality_notes", ""),
            }
        )
    return registry


def marker_to_url(registry: list[dict]) -> dict[str, str]:
    return {entry["marker"]: entry["url"] for entry in registry}


def parse_report_citations(markdown: str) -> list[dict]:
    """Split the report into verifiable units: {sentence, markers}.

    Headings, table separator rows and the source list are skipped; every other
    line that carries at least one [S#] marker becomes evidence-checkable.
    """
    units: list[dict] = []
    in_sources_section = False
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _HEADING_RE.match(line):
            in_sources_section = bool(re.search(r"#+\s*sources\b", line, re.IGNORECASE))
            continue
        if in_sources_section or _TABLE_SEP_RE.match(line) or line.startswith(("[S", "- [S", "* [S")):
            continue
        if not CITATION_RE.search(line):
            continue
        # Split long lines into sentences so each check is fine-grained.
        text = re.sub(r"\[(?:S\d+(?:\]\[S\d+)*)\]", lambda m: m.group(0), line)
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            markers = [f"S{match}" for match in CITATION_RE.findall(sentence)]
            if not markers:
                continue
            cleaned = CITATION_RE.sub("", sentence).strip().strip("|").strip()
            cleaned = re.sub(r"\s{2,}", " ", cleaned)
            cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
            if len(cleaned.split()) < 4:
                continue
            units.append({"sentence": cleaned, "markers": list(dict.fromkeys(markers))})
    return units


def best_matching_claims(sentence: str, claims: list[dict]) -> list[dict]:
    """Rank claims of one source by token overlap with the sentence."""
    wanted = _tokens(sentence)
    if not wanted:
        return []
    scored = []
    for claim in claims:
        overlap = len(wanted & _tokens(claim.get("text", "")))
        if overlap:
            scored.append((overlap, claim))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [claim for _, claim in scored[:2]]


def passages_for_sentence(sentence: str, claims_by_url: dict[str, list[dict]], urls: list[str]) -> tuple[str, list[str]]:
    """Best verbatim evidence passages backing a cited sentence."""
    chunks: list[str] = []
    used: list[str] = []
    for url in urls:
        for claim in best_matching_claims(sentence, claims_by_url.get(url, [])):
            for evidence in claim.get("evidence", []):
                passage = (evidence.get("passage") or "").strip()
                if passage and passage not in chunks:
                    chunks.append(passage)
                    used.append(url)
            if chunks:
                break
    return "\n---\n".join(chunks[:4]), list(dict.fromkeys(used))


def group_claims_by_subquestion(claims: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for claim in claims:
        grouped[claim.get("plan_item_id", "")].append(claim)
    return dict(grouped)
