"""Graph state. Only sources/claims/events have reducers, since the parallel
gather branches are the only ones writing them at the same time.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class ResearchState(TypedDict, total=False):
    # run identity
    session_id: str
    question: str
    config: dict[str, Any]

    # planning
    plan: dict[str, Any]          # {rationale, items: [{id, subquestion, ...}]}
    queue: list[dict[str, Any]]   # plan items to research in the current round
    round: int
    max_rounds: int

    # evidence (reduced: written by parallel gather branches)
    sources: Annotated[list[dict], operator.add]
    claims: Annotated[list[dict], operator.add]

    # synthesis (single writer: synthesize node)
    contradictions: list[dict]
    ratings: list[dict]
    findings: list[dict]
    insufficient: list[dict]

    # writing (single writer: write node)
    report_markdown: str
    report_version: int
    report_feedback: str

    # verification (single writer: verify node)
    verification: list[dict]
    verification_summary: dict
    revisions: int

    # bookkeeping (append-only, safe for parallel branches)
    events: Annotated[list[dict], operator.add]
    final: bool


class GatherInput(TypedDict, total=False):
    """Payload handed to each parallel research branch via ``Send``."""

    session_id: str
    question: str
    config: dict[str, Any]
    plan_item: dict[str, Any]
    existing_urls: list[str]
