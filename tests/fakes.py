"""Fake chat model and a small fixed web corpus for the tests.

The fake model picks its answer from the schema name that llm.structured
puts in the prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mcp_server.core import find_relevant_passages

# corpus
CORPUS: dict[str, dict[str, Any]] = {
    "https://benchlab.example.org/postgres-vs-mongodb-2025": {
        "title": "Independent benchmark: PostgreSQL 17 vs MongoDB 8.0",
        "domain": "benchlab.example.org",
        "keywords": ["performance", "benchmark", "throughput", "latency"],
        "snippet": "Independent, reproducible benchmark of transactional throughput and tail latency.",
        "text": (
            "We measured PostgreSQL 17 and MongoDB 8.0 on identical 16 vCPU instances using a "
            "mixed read/write workload of 50,000 ops per second for six hours.\n"
            "PostgreSQL sustained 62,000 transactions per second with p99 latency of 8 ms, while "
            "MongoDB sustained 41,000 operations per second with p99 latency of 19 ms in the same test.\n"
            "Under write-heavy document ingestion MongoDB performed better than PostgreSQL, reaching "
            "88,000 inserts per second without a schema migration.\n"
            "Both databases scaled linearly to four nodes; PostgreSQL required tuned autovacuum settings "
            "to avoid throughput collapse during the run.\n"
            "Our methodology and raw results are published so anyone can reproduce the benchmark."
        ),
    },
    "https://www.mongodb.com/docs/manual/scaling": {
        "title": "MongoDB manual: scaling strategies",
        "domain": "mongodb.com",
        "text": (
            "MongoDB provides horizontal scaling through sharding, distributing collections across "
            "multiple shards with no application changes required.\n"
            "Sharded clusters can scale writes linearly by choosing an appropriate shard key, which is a "
            "document-level operation that Postgres cannot match without sharding extensions.\n"
            "MongoDB Atlas handles failover automatically and offers multi-region clusters for latency "
            "sensitive applications.\n"
            "Operational overhead is reduced because Atlas manages backups, upgrades, and monitoring."
        ),
    },
    "https://www.postgresql.org/docs/current/performance-tips.html": {
        "title": "PostgreSQL documentation: performance tips",
        "domain": "postgresql.org",
        "text": (
            "PostgreSQL supports mature query planning, MVCC isolation levels and strong consistency "
            "guarantees per transaction.\n"
            "Read replicas and declarative partitioning provide scaling without changing the query model, "
            "and logical replication supports selective distribution of data.\n"
            "Latency for simple primary-key lookups is dominated by network round trips at high connection "
            "counts, which connection poolers such as PgBouncer mitigate.\n"
            "Operational complexity grows with table bloat and vacuum tuning in write-heavy workloads."
        ),
    },
    "https://blog.acme.dev/database-cost-at-scale": {
        "title": "Acme engineering: what databases really cost at scale",
        "domain": "blog.acme.dev",
        "text": (
            "We migrated 40 TB of analytics data from MongoDB to PostgreSQL and cut our monthly database "
            "bill from 18,000 USD to 6,500 USD at the same request volume.\n"
            "The migration took eleven weeks and required rewriting 30 percent of the aggregation code.\n"
            "Managed MongoDB pricing scales with cluster tier size, so cost grows faster than data volume "
            "for small datasets.\n"
            "PostgreSQL is the cheapest option when the workload is relational, but MongoDB's flexible "
            "schema saved us roughly 20 percent in development time on a greenfield product."
        ),
    },
    "https://forum.example.net/t/which-database-2026": {
        "title": "Forum thread: which database should I pick?",
        "domain": "forum.example.net",
        "text": (
            "A short community thread with one-line opinions and no measurements.\n"
            "Someone says Postgres is always fine, another says Mongo scales better, but nobody provides "
            "numbers or configuration details.\n"
            "This thread contains no reproducible evidence and should not be cited."
        ),
    },
    "https://cloud.google.com/blog/products/databases/managed-database-pricing": {
        "title": "Cloud pricing note: managed database tiers",
        "domain": "cloud.google.com",
        "text": (
            "Managed database pricing is billed per vCPU hour plus storage and network egress.\n"
            "A three-node highly available cluster multiplies the compute line item by roughly three, and "
            "cross-region replication adds egress costs that dominate for read heavy services.\n"
            "Our published pricing table lists PostgreSQL and MongoDB-compatible tiers at comparable "
            "per-vCPU rates, with MongoDB managed tiers costing approximately 15 percent more at equal vCPU."
        ),
    },
}

CORPUS_KEYWORDS: dict[str, tuple[str, ...]] = {
    "https://benchlab.example.org/postgres-vs-mongodb-2025": (
        "performance", "benchmark", "throughput", "latency", "comparison", "postgres", "postgresql",
        "mongodb", "scaling", "read", "write",
    ),
    "https://www.mongodb.com/docs/manual/scaling": (
        "scalability", "scale", "sharding", "horizontal", "mongodb", "availability", "operations",
        "operational", "consistency", "write", "cluster",
    ),
    "https://www.postgresql.org/docs/current/performance-tips.html": (
        "consistency", "scalability", "performance", "postgres", "postgresql", "replication",
        "operations", "operational", "latency", "transaction",
    ),
    "https://blog.acme.dev/database-cost-at-scale": (
        "cost", "pricing", "price", "operational", "complexity", "migration", "mongodb", "scale",
    ),
    "https://forum.example.net/t/which-database-2026": (
        "criticism", "limitations", "opinion", "which", "choose", "database",
    ),
    "https://cloud.google.com/blog/products/databases/managed-database-pricing": (
        "cost", "pricing", "managed", "cloud", "operational", "vendor", "cluster",
    ),
}


class FakeResearchTools:
    """Drop-in replacement for ``app.mcp.client.ResearchTools`` (same methods)."""

    def __init__(self, corpus: dict[str, dict] | None = None):
        self.corpus = corpus or CORPUS
        self.searches: list[str] = []
        self.fetches: list[str] = []

    def _domain(self, url: str) -> str:
        return url.split("//", 1)[-1].split("/")[0].removeprefix("www.")

    async def web_search(self, query: str, max_results: int = 5) -> list[dict]:
        self.searches.append(query)
        tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored: list[tuple[int, str]] = []
        for url in self.corpus:
            keywords = CORPUS_KEYWORDS.get(url, tuple(tokens))
            score = len(tokens & set(keywords))
            scored.append((score, url))
        scored.sort(key=lambda row: row[0], reverse=True)
        results = [
            {
                "title": self.corpus[url].get("title", url),
                "url": url,
                "snippet": self.corpus[url].get("text", "")[:180],
                "domain": self._domain(url),
            }
            for score, url in scored
            if score > 0
        ] or [
            {
                "title": self.corpus[url].get("title", url),
                "url": url,
                "snippet": self.corpus[url].get("text", "")[:180],
                "domain": self._domain(url),
            }
            for _, url in scored
        ]
        return results[:max_results]

    async def fetch_page(self, url: str, max_chars: int = 24_000) -> dict:
        self.fetches.append(url)
        entry = self.corpus.get(url)
        if entry is None:
            return {"url": url, "error": "http_404", "text": "", "http_status": 404}
        text = entry["text"]
        return {
            "url": url,
            "final_url": url,
            "title": entry["title"],
            "text": text[:max_chars],
            "http_status": 200,
            "content_hash": str(abs(hash(text)))[:32],
            "truncated": len(text) > max_chars,
        }

    async def find_relevant_passages(self, text: str, focus: str, max_passages: int = 8) -> list[dict]:
        return find_relevant_passages(text, focus=focus, max_passages=max_passages)


# scripted chat model
class ScriptedChatModel:
    """Minimal async chat model; dispatches on the requested JSON schema."""

    def __init__(self, handler):
        self.handler = handler
        self.prompts: list[str] = []

    async def ainvoke(self, messages, **_kwargs):
        prompt = messages[-1].content if messages else ""
        self.prompts.append(prompt if isinstance(prompt, str) else str(prompt))
        return _Message(self.handler(prompt))


# phrases that mark a sentence as an unverifiable overclaim in the scripted world
_OVERCLAIM_MARKERS = ("always", "under any circumstances", "never ", "ten times")


class _Message:
    def __init__(self, content: str):
        self.content = content

    def text(self) -> str:
        return self.content


_SCHEMA_RE = re.compile(r"matching this JSON schema:\s*(\{.*)", re.DOTALL)


def schema_name(prompt: str) -> str | None:
    match = _SCHEMA_RE.search(prompt)
    if not match:
        return None
    try:
        return json.loads(match.group(1)).get("title")
    except json.JSONDecodeError:
        # the prompt may continue after the schema; try a balanced-object parse
        try:
            from app.llm import extract_json

            return extract_json(match.group(1)).get("title")
        except ValueError:
            return None


class ResearchFakeModel:
    """Handler implementing every ResearchOS agent prompt deterministically."""

    def __init__(self, subquestions: list[str] | None = None, threshold_round_one: float = 0.1,
                 fail_revision: bool = False):
        self.subquestions = subquestions or [
            "How do PostgreSQL and MongoDB compare on throughput and latency in independent benchmarks?",
            "How do the two scale horizontally and what consistency guarantees do they provide?",
            "What are the operational complexity and cost implications of each option?",
        ]
        self.sufficiency_calls = 0
        self.report_calls = 0
        self.verify_calls: list[str] = []
        self.threshold_round_one = threshold_round_one
        # When True the revision pass is a no-op, so the audit stays below the
        # threshold and the run must finalise with flagged citations.
        self.fail_revision = fail_revision

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _subquestion(prompt: str) -> str:
        """First non-empty line after the 'Research subquestion:' label."""
        tail = prompt.split("Research subquestion:", 1)[-1]
        for line in tail.splitlines():
            line = line.strip()
            if line and not line.lower().startswith("research goal"):
                return line
        return ""

    @staticmethod
    def _is_overclaim(sentence: str) -> bool:
        lowered = sentence.lower()
        return any(marker in lowered for marker in _OVERCLAIM_MARKERS)

    @staticmethod
    def _candidate_urls(prompt: str) -> list[str]:
        return list(dict.fromkeys(re.findall(r"https?://[^\s)\]]+", prompt)))

    @staticmethod
    def _passages(prompt: str) -> list[str]:
        return [line.strip() for line in re.findall(r"^- \[[^\]]*\] (.+)$", prompt, re.MULTILINE)]

    @staticmethod
    def _markers(prompt: str) -> list[str]:
        return list(dict.fromkeys(re.findall(r"^\[(S\d+)\]", prompt, re.MULTILINE)))

    def __call__(self, prompt: str) -> str:
        name = schema_name(prompt)
        if name == "ResearchPlan":
            return json.dumps(
                {
                    "rationale": "Compare on performance, scalability, cost and operations.",
                    "subquestions": [
                        {"subquestion": question, "rationale": "core dimension", "priority": 1 if index == 0 else 2, "evidence_goal": "independent measurements and primary documentation"}
                        for index, question in enumerate(self.subquestions)
                    ],
                }
            )
        if name == "SearchQueries":
            subquestion = self._subquestion(prompt)
            return json.dumps(
                {
                    "queries": [
                        f"{subquestion} benchmark",
                        f"{subquestion} vendor documentation",
                        f"{subquestion} criticism limitations",
                    ]
                }
            )
        if name == "SourceSelection":
            # emulate a good librarian: one source per domain, best first
            urls: list[str] = []
            domains: set[str] = set()
            for url in self._candidate_urls(prompt):
                domain = url.split("//", 1)[-1].split("/")[0]
                if domain in domains:
                    continue
                domains.add(domain)
                urls.append(url)
                if len(urls) == 3:
                    break
            return json.dumps({"urls": urls})
        if name == "ClaimSet":
            passages = self._passages(prompt)
            claims = []
            for index, passage in enumerate(passages[:3]):
                claims.append(
                    {
                        "text": passage[:220],
                        "claim_type": "quantitative" if re.search(r"\d", passage) else "factual",
                        "importance": 1 if index == 0 else 2,
                        "subject": "comparison",
                        "evidence_passages": [passage],
                    }
                )
            return json.dumps({"claims": claims, "published_date": "2025-03-01"})
        if name == "ContradictionReport":
            numbered = re.findall(r"^\d+\. \[([^\]]+)\] (.+)$", prompt, re.MULTILINE)
            if len(numbered) < 2:
                return json.dumps({"contradictions": []})
            (domain_a, claim_a), (domain_b, claim_b) = numbered[0], numbered[1]
            return json.dumps(
                {
                    "contradictions": [
                        {
                            "topic": "relative throughput",
                            "claim_a": claim_a[:200],
                            "claim_b": claim_b[:200],
                            "context": f"{domain_a} and {domain_b} measured different workloads.",
                            "severity": "moderate",
                        }
                    ]
                }
            )
        if name == "SufficiencyRatings":
            self.sufficiency_calls += 1
            indices = [int(match) for match in re.findall(r"^\[(\d+)\]", prompt, re.MULTILINE)]
            ratings = []
            for index in indices:
                score = self.threshold_round_one if (self.sufficiency_calls == 1 and index == 0) else 0.95
                ratings.append({"index": index, "score": score, "justification": "scripted"})
            return json.dumps({"ratings": ratings})
        if name == "FindingsBatch":
            indices = [int(match) for match in re.findall(r"^\[(\d+)\]", prompt, re.MULTILINE)]
            items = [
                {
                    "index": index,
                    "answer": f"Scripted answer {index + 1}: the evidence supports a hedged conclusion.",
                    "summary": f"Scripted synthesis {index + 1} based on cited claims.",
                }
                for index, _ in enumerate(indices)
            ]
            return json.dumps({"items": items})
        if name == "CitationVerdict":
            self.verify_calls.append(prompt)
            sentence = prompt.split("Report sentence:", 1)[-1].split("Cited sources:", 1)[0]
            if "UNSUPPORTED" in sentence or self._is_overclaim(sentence):
                return json.dumps({"verdict": "unsupported", "notes": "passage does not support the sentence"})
            return json.dumps({"verdict": "supported", "notes": "entailed by the cited passage"})
        if "Failed verifications:" in prompt:
            self.report_calls += 1
            original = prompt.split("Original report:", 1)[-1].split("Failed verifications:", 1)[0]
            if self.fail_revision:
                return original
            failures_block = prompt.split("Failed verifications:", 1)[-1]
            # verify_node renders each failure as: - "<sentence>" (cited S1): <notes>
            failed = [
                sentence.strip()
                for sentence in re.findall(r'^-\s*"(.*?)"\s*\(cited', failures_block, re.MULTILINE)
            ]

            def is_unsupported(line: str) -> bool:
                norm = re.sub(r"\s+", " ", line).strip()
                if not norm:
                    return False
                if "UNSUPPORTED" in norm or self._is_overclaim(norm):
                    return True
                return any(sentence and sentence in norm for sentence in failed)

            cleaned = "\n".join(line for line in original.splitlines() if not is_unsupported(line))
            return cleaned
        if "Write the report as markdown." in prompt:
            self.report_calls += 1
            markers = self._markers(prompt) or ["S1"]
            first, second = markers[0], markers[min(1, len(markers) - 1)]
            unsupported_line = (
                "\nMongoDB is always ten times faster than PostgreSQL for every workload [S1][S2]."
                "\nPostgreSQL cannot scale horizontally under any circumstances [S2]."
                "\nMongoDB is always cheaper than PostgreSQL at every scale [S2]."
                if self.report_calls == 1
                else ""
            )
            return f"""# PostgreSQL vs MongoDB for a SaaS application

## Executive summary
Independent benchmarks show PostgreSQL sustaining higher transactional throughput with lower tail
latency than MongoDB in comparable tests [{first}][{second}]. MongoDB retains an advantage for
document-heavy write paths where schema flexibility matters [{second}].{unsupported_line}

## Comparison matrix
| Dimension | PostgreSQL | MongoDB |
| --- | --- | --- |
| Performance | Higher transactional throughput in independent tests [{first}] | Competitive on document ingestion [{second}] |
| Scalability | Read replicas and partitioning [ {first} ] | Sharding scales writes [{second}] |

## Findings by question
### Throughput and latency
PostgreSQL led on mixed read/write throughput in the cited benchmark [{first}], while MongoDB led on
bulk document inserts [{second}].

## Contradictions and caveats
Sources disagree about which engine is faster; the disagreement tracks workload shape [{first}][{second}].

## Limitations
Managed-service pricing evidence is thinner than the performance evidence [{first}].

## Sources
- [{first}] benchmark source
- [{second}] secondary source
"""

    async def ainvoke_for(self, prompt: str) -> str:  # pragma: no cover - convenience
        return self(prompt)
