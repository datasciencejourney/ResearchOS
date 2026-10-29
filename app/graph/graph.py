"""Builds the graph. plan -> gather (one branch per subquestion) -> synthesize -> write -> verify -> finalize.

synthesize can loop back to gather, verify can loop back to write.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import ResearchState


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(ResearchState)

    builder.add_node("plan", nodes.plan_node)
    builder.add_node("gather", nodes.gather_branch)
    builder.add_node("synthesize", nodes.synthesize)
    builder.add_node("write", nodes.write_node)
    builder.add_node("verify", nodes.verify_node)
    builder.add_node("finalize", nodes.finalize_node)

    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", nodes.route_after_plan, ["gather", END])
    builder.add_edge("gather", "synthesize")
    builder.add_conditional_edges("synthesize", nodes.route_sufficiency, ["gather", "write"])
    builder.add_edge("write", "verify")
    builder.add_conditional_edges("verify", nodes.route_verification, ["write", "finalize"])
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)
