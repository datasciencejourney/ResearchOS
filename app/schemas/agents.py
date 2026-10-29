"""Pydantic models for the structured LLM outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlannedSubquestion(BaseModel):
    subquestion: str = Field(description="A focused, researchable question")
    rationale: str = Field(default="", description="Why this subquestion matters")
    priority: int = Field(default=2, ge=1, le=3, description="1 = highest priority")
    evidence_goal: str = Field(
        default="", description="What kind of evidence would adequately answer this"
    )


class ResearchPlan(BaseModel):
    rationale: str = Field(default="", description="How the question was decomposed")
    subquestions: list[PlannedSubquestion]


class SearchQueries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=6)


class ExtractedClaim(BaseModel):
    text: str = Field(description="One factual, checkable statement")
    claim_type: Literal["factual", "comparative", "quantitative", "causal"] = "factual"
    importance: int = Field(default=2, ge=1, le=3, description="1 = load-bearing for the report")
    subject: str = Field(default="", description="Short label: which side/topic this is about")
    evidence_passages: list[str] = Field(
        default_factory=list,
        description="Verbatim excerpts from the document that support the claim",
    )


class SourceSelection(BaseModel):
    urls: list[str] = Field(min_length=1, description="Chosen source URLs, in priority order")


class ClaimSet(BaseModel):
    claims: list[ExtractedClaim]
    published_date: str = Field(
        default="", description="Publication or last-update date of the document if stated"
    )


class ContradictionPair(BaseModel):
    topic: str = Field(description="What the disagreement is about")
    claim_a: str
    claim_b: str
    context: str = Field(
        default="",
        description="Why both can be true: differing conditions, dates, workloads, editions",
    )
    severity: Literal["minor", "moderate", "major"] = "moderate"


class ContradictionReport(BaseModel):
    contradictions: list[ContradictionPair]


class ItemRating(BaseModel):
    index: int = Field(description="Index of the subquestion in the plan, 0-based")
    score: float = Field(ge=0.0, le=1.0, description="How well the evidence answers it")
    justification: str = ""


class SufficiencyRatings(BaseModel):
    ratings: list[ItemRating]


class SourceQuality(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Overall evidence quality of this document")
    notes: str = Field(default="", description="Recency, authority, benchmark rigour, bias")


class CitationVerdict(BaseModel):
    verdict: Literal["supported", "partial", "unsupported"]
    notes: str = Field(default="", description="Short reason for the verdict")


class FindingsSummary(BaseModel):
    summary: str = Field(description="2-3 sentence synthesis of the findings for one subquestion")
    answer: str = Field(description="Direct answer to the subquestion, hedged where evidence is thin")


class FindingItem(BaseModel):
    index: int = Field(description="Index of the subquestion in the plan, 0-based")
    summary: str
    answer: str


class FindingsBatch(BaseModel):
    items: list[FindingItem]


class ReportOutline(BaseModel):
    sections: list[str]
    matrix_dimensions: list[str] = Field(
        default_factory=list, description="Dimensions for the comparison matrix"
    )
