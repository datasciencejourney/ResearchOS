"""Tests for mcp_server.core, using saved HTML in tests/fixtures."""

from __future__ import annotations

from pathlib import Path

from mcp_server.core import clean_text, find_relevant_passages, parse_search_results

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_search_results_on_captured_page():
    results = parse_search_results((FIXTURES / "ddg_search.html").read_text())
    assert len(results) >= 5
    for result in results:
        assert result["url"].startswith("http")
        assert result["domain"]
        assert result["title"]
        assert "duckduckgo.com/l/" not in result["url"]  # redirects are unwrapped


def test_parse_search_results_wraps_redirect_links():
    html = """
    <div class="result results_links">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fbenchmarks&amp;rut=abc">Benchmarks</a>
      <a class="result__snippet">Independent benchmarks of both systems.</a>
    </div>
    """
    results = parse_search_results(html)
    assert results == [
        {
            "title": "Benchmarks",
            "url": "https://example.org/benchmarks",
            "snippet": "Independent benchmarks of both systems.",
            "domain": "example.org",
        }
    ]


def test_parse_search_results_ignores_relative_links():
    html = '<a class="result__a" href="/internal/page">Relative</a>'
    assert parse_search_results(html) == []


def test_clean_text_keeps_prose_and_drops_boilerplate():
    text = clean_text((FIXTURES / "benchmark_page.html").read_text())
    assert "62,000 transactions per second" in text
    assert "88,000 document inserts per second" in text
    assert "should not appear" not in text  # <script> removed
    assert "Copyright notice" not in text  # <footer> removed
    assert "Short." not in text  # too short to be a real paragraph
    assert "Home" not in text  # nav removed (also too short)


def test_clean_text_deduplicates_repeated_blocks():
    html = "<article><p>" + "Repeated paragraph about database performance." + "</p><p>Repeated paragraph about database performance.</p></article>"
    text = clean_text(html)
    assert text.count("Repeated paragraph") == 1


def test_find_relevant_passages_ranks_rare_terms_higher():
    text = (
        "Performance is primarily determined by the access pattern.\n"
        "PostgreSQL sustained 41,000 operations per second with 19 ms tail latency.\n"
        "Performance tuning is a broad topic with many aspects to consider carefully."
    )
    passages = find_relevant_passages(text, focus="tail latency milliseconds", max_passages=2)
    assert passages
    assert "41,000 operations per second" in passages[0]["passage"]
    assert passages[0]["score"] >= passages[-1]["score"]


def test_find_relevant_passages_returns_verbatim_blocks_with_positions():
    second = "The second block explains how throughput and sharding behave under sustained load in detail."
    text = "First unrelated block of text that is long enough to be considered content.\n" + second
    passages = find_relevant_passages(text, focus="throughput sharding", max_passages=3)
    assert passages == [{"passage": second, "score": passages[0]["score"], "position": 1}]


def test_find_relevant_passages_without_focus_terms():
    assert find_relevant_passages("Some document text.", focus="of to") == []


def test_find_relevant_passages_skips_blocks_without_hits():
    text = (
        "Nothing relevant in this sentence at all about the topic at hand here today.\n"
        "Another sentence that also fails to discuss the requested subject in any detail whatsoever."
    )
    assert find_relevant_passages(text, focus="throughput latency") == []
