"""Unit tests for the citation machinery."""

from __future__ import annotations

from app.services import citations as cit


def test_build_registry_orders_sources_with_claims_first():
    sources = [
        {"url": "https://b.example/2", "domain": "b.example", "claim_count": 0},
        {"url": "https://a.example/1", "domain": "a.example", "claim_count": 3},
        {"url": "https://c.example/3", "domain": "c.example", "claim_count": 1},
    ]
    registry = cit.build_registry(sources)
    assert [entry["marker"] for entry in registry] == ["S1", "S2", "S3"]
    assert [entry["url"] for entry in registry] == [
        "https://a.example/1",
        "https://c.example/3",
        "https://b.example/2",
    ]
    assert cit.marker_to_url(registry)["S2"] == "https://c.example/3"


def test_parse_report_citations_extracts_sentences_and_markers():
    markdown = """# Title

## Executive summary
PostgreSQL leads in mixed workloads [S1][S2]. MongoDB wins on document ingestion [S3].

## Comparison matrix
| Dimension | PG | Mongo |
| --- | --- | --- |
| Perf | faster [S1] | slower [S3] |

## Sources
- [S1] a - https://a.example
"""
    units = cit.parse_report_citations(markdown)
    sentences = [unit["sentence"] for unit in units]
    assert "PostgreSQL leads in mixed workloads." in sentences
    assert any(unit["markers"] == ["S1", "S2"] for unit in units)
    assert any(unit["markers"] == ["S3"] for unit in units)
    # headings and the source list are not claims
    assert all(not unit["sentence"].startswith("#") for unit in units)
    assert all("https://a.example" not in unit["sentence"] for unit in units)
    assert len(sentences) == 3


def test_parse_report_citations_ignores_short_fragments():
    assert cit.parse_report_citations("## X\nok [S1]\n") == []


def test_passages_for_sentence_picks_best_matching_claim():
    claims_by_url = {
        "https://a.example": [
            {
                "text": "PostgreSQL sustained 41,000 operations per second with 19 ms tail latency.",
                "evidence": [{"passage": "PostgreSQL sustained 41,000 operations per second."}],
            },
            {
                "text": "PostgreSQL requires tuned autovacuum settings for write-heavy workloads.",
                "evidence": [{"passage": "Tuned autovacuum settings were required."}],
            },
        ]
    }
    passage, used = cit.passages_for_sentence(
        "PostgreSQL sustained roughly 41,000 operations per second.",
        claims_by_url,
        ["https://a.example"],
    )
    assert "41,000 operations per second" in passage
    assert used == ["https://a.example"]


def test_passages_for_sentence_reports_missing_evidence():
    passage, used = cit.passages_for_sentence("Unrelated sentence about penguins.", {}, ["https://x.example"])
    assert passage == ""
    assert used == []
