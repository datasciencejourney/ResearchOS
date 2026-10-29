"""Graph nodes: plan, gather, synthesize, write, verify, finalize.

Web access only goes through the MCP tools (app.mcp.client), DB access only
through app.services.persistence.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections import defaultdict
from typing import Any
from urllib.parse import urlparse

from langgraph.types import Send

from app import prompts
from app.config import get_settings
from app.graph.state import GatherInput, ResearchState
from app.llm import complete, get_chat_model, structured
from app.mcp.client import get_research_tools
from app.schemas.agents import (
    ClaimSet,
    ContradictionReport,
    FindingsBatch,
    CitationVerdict,
    ResearchPlan,
    SearchQueries,
    SourceSelection,
    SufficiencyRatings,
)
from app.services import citations as cit
from app.services import persistence as store

# Serialises source upserts: two parallel branches may discover the same URL.
_SOURCE_LOCK = asyncio.Lock()

_AUTHORITATIVE_DOMAINS = (
    "postgresql.org", "mongodb.com", "docs.aws.amazon.com", "aws.amazon.com",
    "cloud.google.com", "learn.microsoft.com", "arxiv.org", "acm.org", "ieee.org",
    "github.io", "kernel.org", "ietf.org", "w3.org", "nvlpubs.nist.gov",
)
_VENDOR_DOMAINS = ("mongodb.com", "postgresql.org", "aws.amazon.com", "cloud.google.com", "azure.microsoft.com")
_CASUAL_DOMAINS = ("reddit.com", "quora.com", "stackoverflow.com", "news.ycombinator.com", "x.com", "twitter.com")
_DATA_HINT = re.compile(
    r"\b(benchmark|throughput|latency|p9[59]|iops|queries per second|tps|mb/s|gbps|"
    r"cost per|pricing|\$\d|%\b|percent|million|billion)\b",
    re.IGNORECASE,
)


def _domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _claims_block(claims: list[dict], url_markers: dict[str, str] | None = None, limit: int = 14) -> str:
    """Render claims for an LLM prompt, optionally with [S#] markers."""
    lines = []
    for index, claim in enumerate(claims[:limit], start=1):
        marker = ""
        if url_markers is not None:
            marker = f" {url_markers.get(claim.get('source_url', ''), '')}"
        lines.append(
            f"{index}. ({claim.get('claim_type', 'factual')}, importance {claim.get('importance', 2)})"
            f"{marker} {claim['text']}"
        )
    return "\n".join(lines) or "(none)"


# source quality (deterministic heuristics, documented in the report)
def score_source(doc: dict) -> tuple[float, str]:
    text = doc.get("text", "")
    domain = doc.get("domain", "") or _domain(doc.get("url", ""))
    words = len(text.split())
    score = 0.45
    notes: list[str] = []
    if words > 1500:
        score += 0.2
        notes.append("substantial document")
    elif words > 600:
        score += 0.1
        notes.append("moderate depth")
    else:
        notes.append("thin document")
    if any(domain.endswith(d) for d in _AUTHORITATIVE_DOMAINS):
        score += 0.15
        notes.append("authoritative/primary source")
    if any(d in domain for d in _VENDOR_DOMAINS):
        score -= 0.05
        notes.append("vendor-published (interested party)")
    if any(d in domain for d in _CASUAL_DOMAINS):
        score -= 0.15
        notes.append("community source, unverified")
    if _DATA_HINT.search(text[:12000]):
        score += 0.1
        notes.append("contains quantitative data")
    return _clamp(round(score, 3)), "; ".join(notes)


# plan
async def plan_node(state: ResearchState) -> dict:
    settings = get_settings()
    session_id = uuid.UUID(state["session_id"])
    question = state["question"]
    config = state.get("config", {})
    approval_required = bool(config.get("approval_required"))
    model = get_chat_model()

    def _pack(items: list[dict], rationale: str) -> dict:
        queue = sorted(items, key=lambda i: (i.get("priority", 2), i.get("position", 0)))
        return {
            "plan": {"rationale": rationale, "items": items},
            "queue": queue,
            "round": 1,
            "max_rounds": int(config.get("max_research_iterations", settings.max_research_iterations)),
        }

    # A re-run over an already approved plan must not re-plan (reproducibility).
    existing = store.load_approved_plan_items(session_id)
    if existing:
        store.log_event(session_id, "plan", "Reusing approved research plan", {"items": len(existing)})
        return {
            **_pack(existing, "Reused previously approved plan."),
            "events": [{"node": "plan", "message": f"reused approved plan ({len(existing)} subquestions)"}],
        }

    plan = await structured(
        prompts.PLANNER_USER.format(question=question),
        ResearchPlan,
        system=prompts.PLANNER_SYSTEM,
        model=model,
    )
    max_q = settings.max_subquestions
    items = [item.model_dump() for item in plan.subquestions][:max_q]
    if not items:  # pragma: no cover - defensive
        items = [{"subquestion": question, "rationale": "", "priority": 1, "evidence_goal": "general evidence"}]
    saved = store.save_plan(
        session_id,
        {"rationale": plan.rationale, "subquestions": items},
        approval_required,
    )
    store.log_event(
        session_id,
        "plan",
        f"Research plan created with {len(saved)} subquestions",
        {"subquestions": [i["subquestion"] for i in saved], "rationale": plan.rationale},
    )
    if approval_required:
        store.set_status(session_id, "awaiting_approval")
    else:
        store.set_status(session_id, "researching")
    return {
        **_pack(saved, plan.rationale),
        "events": [
            {
                "node": "plan",
                "message": f"planned {len(saved)} subquestions",
                "payload": {"subquestions": [i["subquestion"] for i in saved]},
            }
        ],
    }


def route_after_plan(state: ResearchState):
    """Apply the approval gate, then fan out one branch per subquestion."""
    session_id = uuid.UUID(state["session_id"])
    config = state.get("config", {})
    if config.get("approval_required") and not store.plan_approved(session_id):
        store.log_event(session_id, "plan", "Run paused: awaiting plan approval")
        return "__end__"
    return _fan_out(state)


def _fan_out(state: ResearchState) -> list[Send]:
    existing_urls = [source.get("url", "") for source in state.get("sources", [])]
    config = state.get("config", {})
    return [
        Send(
            "gather",
            GatherInput(
                session_id=state["session_id"],
                question=state["question"],
                config=config,
                plan_item=item,
                existing_urls=existing_urls,
            ),
        )
        for item in state.get("queue", [])
    ]


# gather (parallel branch, one per subquestion)
async def gather_branch(payload: GatherInput) -> dict:
    settings = get_settings()
    session_id = uuid.UUID(payload["session_id"])
    item = payload["plan_item"]
    question = payload["question"]
    config = payload.get("config", {}) or {}
    model = get_chat_model()
    tools = get_research_tools()

    max_sources = int(config.get("max_sources_per_subquestion", settings.max_sources_per_subquestion))
    events: list[dict] = []

    def log(message: str, kind: str = "info", **payload_kv: Any) -> None:
        events.append({"node": "gather", "message": message, "kind": kind, "payload": payload_kv})
        store.log_event(session_id, "gather", message, payload_kv, kind=kind)

    log(f"researching: {item['subquestion']}", subquestion=item["subquestion"], item_id=item["id"])

    # 1. search queries
    try:
        queries = await structured(
            prompts.QUERIES_USER.format(subquestion=item["subquestion"], evidence_goal=item.get("evidence_goal", "")),
            SearchQueries,
            system=prompts.QUERIES_SYSTEM,
            model=model,
        )
        query_list = queries.queries[:4]
    except ValueError:
        query_list = [f'{item["subquestion"]} evidence']
    log(f"search queries: {query_list}")

    # 2. search via MCP
    blocked = set(payload.get("existing_urls", [])) | store.excluded_urls(session_id)
    candidates: list[dict] = []
    seen: set[str] = set()
    for query in query_list:
        try:
            results = await tools.web_search(query, max_results=settings.max_search_results_per_query)
        except Exception as exc:  # network problems must not kill the run
            log(f"search failed for '{query}': {exc}", "warning")
            continue
        for result in results:
            url = (result.get("url") or "").strip()
            if not url or url in seen or url in blocked:
                continue
            seen.add(url)
            candidates.append(
                {
                    "url": url,
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", ""),
                    "domain": result.get("domain") or _domain(url),
                }
            )
        if len(candidates) >= max_sources * 3:
            break

    if not candidates:
        log("no new candidate sources found", "warning")
        return {"sources": [], "claims": [], "events": events}

    # 3. source selection
    listing = "\n".join(
        f"{index + 1}. [{c['domain']}] {c['title']} - {c['snippet'][:220]}\n   {c['url']}"
        for index, c in enumerate(candidates[:14])
    )
    try:
        selection = await structured(
            prompts.SELECTION_USER.format(
                subquestion=item["subquestion"],
                evidence_goal=item.get("evidence_goal", ""),
                candidates=listing,
                max_urls=max_sources,
            ),
            SourceSelection,
            system=prompts.SELECTION_SYSTEM,
            model=model,
        )
        by_url = {c["url"]: c for c in candidates}
        chosen = [url for url in selection.urls if url in by_url]
    except ValueError:
        by_url = {c["url"]: c for c in candidates}
        chosen = [c["url"] for c in candidates]
    # domain diversity: at most 2 sources per domain
    per_domain: dict[str, int] = defaultdict(int)
    picked: list[str] = []
    for url in chosen:
        domain = by_url[url]["domain"]
        if per_domain[domain] >= 2:
            continue
        per_domain[domain] += 1
        picked.append(url)
        if len(picked) >= max_sources:
            break
    if not picked:
        picked = [c["url"] for c in candidates[:max_sources]]
    log(f"selected {len(picked)} sources", urls=picked)

    # 4. fetch via MCP
    semaphore = asyncio.Semaphore(int(config.get("max_fetch_concurrency", settings.max_fetch_concurrency)))

    async def fetch(url: str) -> dict:
        async with semaphore:
            try:
                return await tools.fetch_page(url, max_chars=settings.max_chars_per_source)
            except Exception as exc:
                log(f"fetch failed {url}: {exc}", "warning")
                return {"url": url, "error": str(exc), "text": ""}

    docs = await asyncio.gather(*[fetch(url) for url in picked])
    docs = [
        {**doc, "domain": _domain(doc.get("final_url") or doc.get("url", ""))}
        for doc in docs
        if doc.get("text") and len(doc["text"]) > 400
    ]
    if not docs:
        log("no documents could be retrieved", "warning")
        return {"sources": [], "claims": [], "events": events}

    # 5. passages + claim extraction per document (parallel)
    async def extract(doc: dict) -> tuple[dict, list[dict], str]:
        try:
            passages = await tools.find_relevant_passages(
                doc["text"], focus=item["subquestion"], max_passages=8
            )
        except Exception as exc:
            log(f"passage mining failed for {doc['url']}: {exc}", "warning")
            passages = []
        passage_block = "\n".join(
            f"- [{p.get('position', '?')}] {p['passage']}" for p in passages
        ) or doc["text"][:2000]

        claims: list[dict] = []
        published = ""
        try:
            extracted = await structured(
                prompts.EXTRACT_USER.format(
                    subquestion=item["subquestion"],
                    title=doc.get("title", ""),
                    domain=doc.get("domain", ""),
                    published=doc.get("published", "unknown"),
                    passages=passage_block,
                    opening=doc["text"][:2500],
                ),
                ClaimSet,
                system=prompts.EXTRACT_SYSTEM,
                model=model,
            )
            published = extracted.published_date or ""
            for claim in extracted.claims[:6]:
                # A claim without a verbatim, actually-present passage is dropped:
                # evidence must be checkable, not asserted.
                evidence = []
                for passage in claim.evidence_passages:
                    position = doc["text"].find(passage.strip()[:120])
                    if position == -1:
                        continue
                    evidence.append(
                        {
                            "passage": passage.strip(),
                            "position": doc["text"][:position].count("\n") + 1,
                            "relevance": None,
                        }
                    )
                if evidence:
                    claims.append(
                        {
                            "text": claim.text.strip(),
                            "claim_type": claim.claim_type,
                            "importance": claim.importance,
                            "subject": claim.subject,
                            "evidence": evidence,
                        }
                    )
        except ValueError as exc:
            log(f"claim extraction failed for {doc['url']}: {exc}", "warning")

        quality, notes = score_source(doc)
        source_record = {
            "url": doc["url"],
            "canonical_url": doc.get("final_url", ""),
            "domain": doc.get("domain", ""),
            "title": doc.get("title", ""),
            "search_snippet": by_url.get(doc["url"], {}).get("snippet", ""),
            "text": doc["text"][:8000],  # state carries an excerpt; full text lives in Postgres
            "content_hash": doc.get("content_hash", ""),
            "http_status": doc.get("http_status"),
            "quality_score": quality,
            "quality_notes": notes,
            "published_date": published,
            "used_for": [item["id"]],
            "claim_count": len(claims),
        }
        return source_record, claims, published

    results = await asyncio.gather(*[extract(doc) for doc in docs])

    # 6. persist
    sources_out: list[dict] = []
    claims_out: list[dict] = []
    for source_record, claims, _published in results:
        async with _SOURCE_LOCK:
            source_id = store.upsert_source(session_id, source_record)
            store.replace_claims(
                session_id, source_id, item["id"],
                [{**claim, "evidence": claim["evidence"]} for claim in claims],
            )
        sources_out.append({**source_record, "id": source_id})
        for claim in claims:
            claims_out.append({**claim, "source_url": source_record["url"], "plan_item_id": item["id"]})

    log(
        f"gathered {len(sources_out)} sources / {len(claims_out)} claims",
        subquestion=item["subquestion"],
        sources=len(sources_out),
        claims=len(claims_out),
    )
    return {"sources": sources_out, "claims": claims_out, "events": events}


# synthesize
async def synthesize(state: ResearchState) -> dict:
    settings = get_settings()
    session_id = uuid.UUID(state["session_id"])
    question = state["question"]
    plan = state.get("plan", {})
    items = plan.get("items", [])
    all_claims = state.get("claims", [])
    sources = {source["url"]: source for source in state.get("sources", [])}
    model = get_chat_model()

    claims_by_item: dict[str, list[dict]] = defaultdict(list)
    for claim in all_claims:
        claims_by_item[claim.get("plan_item_id", "")].append(claim)

    events: list[dict] = []

    # contradiction detection per subquestion
    contradictions: list[dict] = []
    inventory_lines: list[str] = []
    for index, item in enumerate(items):
        item_claims = claims_by_item.get(item["id"], [])
        pairs: list[dict] = []
        if len(item_claims) >= 4:
            claim_lines = []
            for position, claim in enumerate(item_claims[:12], start=1):
                domain = sources.get(claim.get("source_url", ""), {}).get("domain", "?")
                claim_lines.append(f"{position}. [{domain}] {claim['text']}")
            try:
                report = await structured(
                    prompts.CONTRADICTIONS_USER.format(
                        subquestion=item["subquestion"], claims="\n".join(claim_lines)
                    ),
                    ContradictionReport,
                    system=prompts.CONTRADICTIONS_SYSTEM,
                    model=model,
                )
                by_position = {position: claim for position, claim in enumerate(item_claims[:12], start=1)}
                for pair in report.contradictions[:5]:
                    dumped = pair.model_dump()
                    contradictions.append({**dumped, "plan_item_id": item["id"], "plan_item": item["subquestion"]})
                    pairs.append(dumped)
            except ValueError as exc:
                events.append({"node": "synthesize", "message": f"contradiction pass failed: {exc}", "kind": "warning"})
        if pairs:
            store.save_contradictions(session_id, item["id"], pairs)
            events.append(
                {
                    "node": "synthesize",
                    "message": f"{len(pairs)} contradictions recorded for: {item['subquestion'][:70]}",
                    "payload": {"topics": [p["topic"] for p in pairs]},
                }
            )
        domains = {sources.get(c.get("source_url", ""), {}).get("domain", "") for c in item_claims}
        inventory_lines.append(
            f"[{index}] {item['subquestion']}\n"
            f"    evidence_goal: {item.get('evidence_goal', '')}\n"
            f"    claims: {len(item_claims)} from {len({d for d in domains if d})} distinct domains"
        )

    # sufficiency scoring (LLM + deterministic coverage)
    ratings_map: dict[int, float] = {}
    try:
        ratings = await structured(
            prompts.SUFFICIENCY_USER.format(question=question, inventory="\n".join(inventory_lines)),
            SufficiencyRatings,
            system=prompts.SUFFICIENCY_SYSTEM,
            model=model,
        )
        ratings_map = {rating.index: rating.score for rating in ratings.ratings}
    except ValueError as exc:
        events.append({"node": "synthesize", "message": f"sufficiency pass failed: {exc}", "kind": "warning"})

    threshold = float(
        state.get("config", {}).get("sufficiency_threshold") or get_settings().sufficiency_threshold
    )
    insufficient: list[dict] = []
    findings_input: list[dict] = []
    rating_rows: list[dict] = []
    for index, item in enumerate(items):
        item_claims = claims_by_item.get(item["id"], [])
        domains = {sources.get(c.get("source_url", ""), {}).get("domain", "") for c in item_claims} - {""}
        llm_score = ratings_map.get(index, 0.5 if item_claims else 0.1)
        coverage = min(1.0, len(domains) / 2)
        quantitative = any(c.get("claim_type") == "quantitative" for c in item_claims)
        if not item_claims:
            score = 0.0
        else:
            score = _clamp(0.6 * llm_score + 0.35 * coverage + (0.05 if quantitative else 0.0))
        status = "sufficient" if score >= threshold else "insufficient"
        rating_rows.append(
            {
                "plan_item_id": item["id"],
                "subquestion": item["subquestion"],
                "llm_score": llm_score,
                "coverage": coverage,
                "score": round(score, 3),
                "status": status,
            }
        )
        store.update_plan_item(session_id, item["id"], status=status, sufficiency=round(score, 3))
        if status == "insufficient":
            insufficient.append(item)
        findings_input.append(
            {
                "index": index,
                "subquestion": item["subquestion"],
                "sufficiency": round(score, 3),
                "claims": item_claims[:12],
                "contradictions": [
                    {k: v for k, v in pair.items() if k in ("topic", "claim_a", "claim_b", "context", "severity")}
                    for pair in contradictions
                    if pair.get("plan_item_id") == item["id"]
                ],
            }
        )

    # findings synthesis (single batched call)
    findings: list[dict] = []
    try:
        batch = await structured(
            prompts.FINDINGS_USER.format(
                question=question,
                subquestions_and_evidence="\n\n".join(
                    f"[{row['index']}] {row['subquestion']}\n{_claims_block(row['claims'])}"
                    for row in findings_input
                ),
            ),
            FindingsBatch,
            system=prompts.FINDINGS_SYSTEM,
            model=model,
        )
        findings = [item.model_dump() for item in batch.items]
    except Exception as exc:
        events.append({"node": "synthesize", "message": f"findings pass failed: {exc}", "kind": "warning"})
    found_by_index = {f.get("index"): f for f in findings}
    merged_findings = []
    for row in findings_input:
        found = found_by_index.get(row["index"], {})
        merged_findings.append(
            {
                "index": row["index"],
                "subquestion": row["subquestion"],
                "sufficiency": row["sufficiency"],
                "answer": found.get("answer", ""),
                "summary": found.get("summary", ""),
                "claims": row["claims"],
                "contradictions": row["contradictions"],
            }
        )

    # decide: another research round, or write the report
    round_no = int(state.get("round", 1))
    max_rounds = int(state.get("max_rounds", settings.max_research_iterations))
    loop = bool(insufficient) and round_no < max_rounds
    next_queue = sorted(insufficient, key=lambda i: (i.get("priority", 2), i.get("position", 0)))[:3] if loop else []
    if loop:
        store.log_event(
            session_id,
            "synthesize",
            f"Round {round_no} insufficient for {len(insufficient)} subquestions - routing back to research",
            {"subquestions": [i["subquestion"] for i in insufficient]},
            kind="warning",
        )
        events.append(
            {
                "node": "synthesize",
                "message": f"looping back to research for {len(insufficient)} subquestions (round {round_no + 1})",
                "kind": "warning",
            }
        )
    else:
        store.set_status(session_id, "writing")
        events.append({"node": "synthesize", "message": "evidence sufficient - moving to report writing"})
    store.log_event(
        session_id, "synthesize", "Evidence sufficiency scored",
        {"ratings": rating_rows, "contradictions": len(contradictions)},
    )

    return {
        "contradictions": contradictions,
        "ratings": rating_rows,
        "findings": merged_findings,
        "insufficient": [dict(i) for i in insufficient],
        "queue": next_queue,
        "round": round_no + 1 if loop else round_no,
        "events": events,
    }


def route_sufficiency(state: ResearchState):
    """Loop back to research when subquestions remain under-evidenced."""
    if state.get("insufficient") and state.get("queue"):
        return _fan_out(state)
    return "write"


# write / verify
def _registry_with_claims(state: ResearchState) -> tuple[list[dict], dict[str, str]]:
    sources = state.get("sources", [])
    claims_by_url: dict[str, list[dict]] = defaultdict(list)
    for claim in state.get("claims", []):
        claims_by_url[claim.get("source_url", "")].append(claim)
    decorated = [
        {**source, "claim_count": len(claims_by_url.get(source.get("url", ""), []))}
        for source in sources
    ]
    registry = cit.build_registry(decorated)
    return registry, cit.marker_to_url(registry)


async def write_node(state: ResearchState) -> dict:
    settings = get_settings()
    session_id = uuid.UUID(state["session_id"])
    model = get_chat_model()
    registry, url_markers = _registry_with_claims(state)
    claims_by_url: dict[str, list[dict]] = defaultdict(list)
    for claim in state.get("claims", []):
        claims_by_url[claim.get("source_url", "")].append(claim)

    registry_block = "\n".join(
        f"[{entry['marker']}] {entry['title']} - {entry['url']}"
        + (f" (published {entry['published_date']})" if entry.get("published_date") else "")
        + (f" (quality {entry['quality_score']}: {entry['quality_notes']})" if entry.get("quality_score") is not None else "")
        for entry in registry
    )
    plan_items = state.get("plan", {}).get("items", [])
    plan_block = "\n".join(
        f"{index + 1}. {item['subquestion']} (priority {item.get('priority', 2)})"
        for index, item in enumerate(plan_items)
    )
    findings_block = "\n\n".join(
        "### " + row["subquestion"]
        + f"\nEvidence sufficiency: {row['sufficiency']}"
        + (f"\nAnswer: {row['answer']}" if row.get("answer") else "")
        + (f"\nSynthesis: {row['summary']}" if row.get("summary") else "")
        + (f"\nClaims:\n{_claims_block(row['claims'], url_markers)}" if row.get("claims") else "\nClaims: (none - state this as a limitation)")
        + (
            "\nContradictions:\n"
            + "\n".join(
                f"- [{c.get('severity', 'moderate')}] {c['topic']}: \"{c['claim_a']}\" vs \"{c['claim_b']}\""
                + (f" - context: {c['context']}" if c.get("context") else "")
                for c in row.get("contradictions", [])
            )
            if row.get("contradictions")
            else ""
        )
        for row in state.get("findings", [])
    )
    insufficient = state.get("insufficient", [])
    if insufficient:
        findings_block += "\n\n## Under-evidenced subquestions (MUST be reported as limitations)\n" + "\n".join(
            f"- {item['subquestion']}" for item in insufficient
        )

    revision = state.get("report_feedback", "")
    if revision:
        markdown = await complete(
            prompts.REVISION_USER.format(report=state.get("report_markdown", ""), failures=revision),
            system=prompts.REVISION_SYSTEM,
            model=model,
        )
        reason = "revision after failed citation audit"
    else:
        markdown = await complete(
            prompts.REPORT_USER.format(
                question=state["question"],
                plan=plan_block,
                registry=registry_block,
                findings=findings_block,
            ),
            system=prompts.REPORT_SYSTEM,
            model=model,
        )
        reason = ""

    markdown = markdown.strip()
    if markdown.startswith("```"):
        markdown = re.sub(r"^```[a-z]*\n|\n```$", "", markdown).strip()

    summary = ""
    match = re.search(r"##\s*Executive summary\s*\n+(.*?)(?=\n##\s|\Z)", markdown, re.DOTALL)
    if match:
        summary = re.sub(r"\s{2,}", " ", match.group(1)).strip()[:1200]

    version = store.save_report(session_id, markdown, summary, reason)
    store.log_event(
        session_id,
        "write",
        f"Report v{version} written ({reason or 'initial draft'})",
        {"version": version, "sources": len(registry), "chars": len(markdown)},
    )
    return {
        "report_markdown": markdown,
        "report_version": version,
        "report_feedback": "",
        "events": [
            {"node": "write", "message": f"report v{version} written", "payload": {"chars": len(markdown)}}
        ],
    }


async def verify_node(state: ResearchState) -> dict:
    settings = get_settings()
    session_id = uuid.UUID(state["session_id"])
    model = get_chat_model()
    registry, url_markers = _registry_with_claims(state)
    url_by_marker = url_markers
    marker_by_url = {url: marker for marker, url in url_by_marker.items()}
    claims_by_url: dict[str, list[dict]] = defaultdict(list)
    for claim in state.get("claims", []):
        claims_by_url[claim.get("source_url", "")].append(claim)

    units = cit.parse_report_citations(state.get("report_markdown", ""))
    checked = units[: settings.max_citations_to_verify]

    async def check(unit: dict) -> dict:
        urls = [url_by_marker[m] for m in unit["markers"] if m in url_by_marker]
        passage, used_urls = cit.passages_for_sentence(unit["sentence"], claims_by_url, urls)
        cited = [marker_by_url.get(url, "S?") for url in used_urls] or unit["markers"]
        if not passage:
            return {
                "sentence": unit["sentence"],
                "verdict": "unsupported",
                "sources": cited,
                "passage": "",
                "notes": "no verbatim evidence passage was harvested for this statement",
            }
        try:
            verdict = await structured(
                prompts.VERIFY_USER.format(
                    sentence=unit["sentence"],
                    sources=", ".join(cited),
                    passages=passage,
                ),
                CitationVerdict,
                system=prompts.VERIFY_SYSTEM,
                model=model,
            )
        except ValueError as exc:
            verdict = CitationVerdict(verdict="partial", notes=f"verifier error: {exc}")
        return {
            "sentence": unit["sentence"],
            "verdict": verdict.verdict,
            "sources": cited,
            "passage": passage[:900],
            "notes": verdict.notes,
        }

    results = await asyncio.gather(*[check(unit) for unit in checked]) if checked else []
    results = list(results)
    supported = sum(1 for r in results if r["verdict"] == "supported")
    partial = sum(1 for r in results if r["verdict"] == "partial")
    unsupported = sum(1 for r in results if r["verdict"] == "unsupported")
    ratio = supported / len(results) if results else 1.0
    below_threshold = bool(results) and ratio < settings.min_supported_citation_ratio
    summary = {
        "checked": len(results),
        "supported": supported,
        "partial": partial,
        "unsupported": unsupported,
        "supported_ratio": round(ratio, 3),
        "threshold": settings.min_supported_citation_ratio,
        "passed": not below_threshold,
        "report_version": state.get("report_version"),
    }
    store.save_verifications(session_id, int(state.get("report_version", 1)), results, summary)

    events: list[dict] = [
        {"node": "verify", "message": f"verified {len(results)} cited statements", "payload": summary}
    ]
    revisions = int(state.get("revisions", 0))
    failures = [r for r in results if r["verdict"] == "unsupported"]
    if below_threshold and revisions < settings.max_report_revisions:
        feedback = "\n".join(
            f'- "{r["sentence"]}" (cited {", ".join(r["sources"])}): {r["notes"]}' for r in failures
        ) or "Overall citation support was below the acceptable ratio."
        store.log_event(
            session_id,
            "verify",
            f"Citation audit failed for report v{state.get('report_version')} - requesting revision",
            summary,
            kind="warning",
        )
        store.set_status(session_id, "writing")
        events.append({"node": "verify", "message": "citation audit failed - routing back to write", "kind": "warning"})
        return {
            "verification": results,
            "verification_summary": summary,
            "revisions": revisions + 1,
            "report_feedback": feedback,
            "events": events,
        }

    if below_threshold:
        # Honest outcome: the report is finalised but its citations did not clear
        # the bar and there is no revision budget left. Do not claim it passed.
        store.log_event(
            session_id,
            "verify",
            f"Citation audit below threshold for report v{state.get('report_version')} "
            "and revision budget exhausted - finalising with flagged citations",
            summary,
            kind="warning",
        )
        events.append(
            {
                "node": "verify",
                "message": "citation audit below threshold; revision budget exhausted - finalising anyway",
                "kind": "warning",
            }
        )
        store.set_status(session_id, "verifying")
        return {
            "verification": results,
            "verification_summary": summary,
            "revisions": revisions,
            "report_feedback": "",
            "events": events,
        }

    store.log_event(session_id, "verify", f"Citation audit passed for report v{state.get('report_version')}", summary)
    store.set_status(session_id, "verifying")
    return {
        "verification": results,
        "verification_summary": summary,
        "revisions": revisions,
        "report_feedback": "",
        "events": events,
    }


def route_verification(state: ResearchState):
    if state.get("report_feedback"):
        return "write"
    return "finalize"


def finalize_node(state: ResearchState) -> dict:
    session_id = uuid.UUID(state["session_id"])
    summary = state.get("verification_summary", {})
    store.set_status(session_id, "completed", finished=True)
    store.log_event(
        session_id,
        "finalize",
        "Research run completed",
        {
            "report_version": state.get("report_version"),
            "sources": len(state.get("sources", [])),
            "claims": len(state.get("claims", [])),
            "contradictions": len(state.get("contradictions", [])),
            "verification": summary,
            "rounds": state.get("round", 1),
        },
    )
    return {
        "final": True,
        "events": [{"node": "finalize", "message": "run completed", "payload": summary}],
    }
