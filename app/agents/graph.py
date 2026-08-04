import os
import json
import uuid
from datetime import datetime, timezone
from typing import TypedDict, Dict, Any

from langgraph.graph import StateGraph, END

from app.config import TRAJECTORY_PATH
from app.guardrails.moderation import check_moderation
from app.guardrails.neutrality import is_contested_topic
from app.guardrails.output_safety import validate_output
from app.uncertainty.semantic_entropy import run_uncertainty_check
from app.rag.retrieval import retrieve_evidence

try:
    from app.agents.crew import run_verification_crew
    CREW_AVAILABLE = True
except Exception:
    CREW_AVAILABLE = False


class GraphState(TypedDict):
    query: str
    query_id: str
    trace: Dict[str, Any]
    blocked: bool
    final_result: Dict[str, Any]


def moderation_node(state: GraphState) -> GraphState:
    moderation = check_moderation(state["query"])
    state["trace"]["moderation"] = moderation
    if moderation["blocked"]:
        state["blocked"] = True
        state["final_result"] = {
            "status": "refused",
            "answer": moderation["reason"],
            "semantic_entropy_score": 0.0,
            "is_consistent": True,
            "confidence": 0.0,
            "sources": [],
            "is_neutral_assessment": True,
            "perspectives_considered": [],
        }
    else:
        state["blocked"] = False
    return state


def retrieval_node(state: GraphState) -> GraphState:
    if state["blocked"]:
        return state
    evidence = retrieve_evidence(state["query"])
    state["trace"]["evidence"] = evidence
    state["trace"]["retrieved_sources"] = [e["source"] for e in evidence]
    return state


def entropy_node(state: GraphState) -> GraphState:
    if state["blocked"]:
        return state
    evidence = state["trace"].get("evidence", [])
    evidence_text = "\n".join(e["text"] for e in evidence)[:6000]

    uncertainty = run_uncertainty_check(state["query"], evidence_text)
    state["trace"]["sampled_responses"] = uncertainty["sampled_responses"]
    state["trace"]["semantic_entropy_score"] = uncertainty["semantic_entropy_score"]
    state["trace"]["uncertainty"] = uncertainty
    return state


def verification_node(state: GraphState) -> GraphState:
    if state["blocked"]:
        return state

    uncertainty = state["trace"]["uncertainty"]
    majority_answer = uncertainty["majority_answer"]

    if not majority_answer:
        state["final_result"] = {
            "status": "insufficient_evidence",
            "answer": (
                "Unable to generate a confident answer — the model's sampled "
                "responses were inconsistent or the API call failed."
            ),
            "semantic_entropy_score": uncertainty["semantic_entropy_score"],
            "is_consistent": False,
            "confidence": 0.0,
            "sources": [],
            "is_neutral_assessment": True,
            "perspectives_considered": [],
        }
        return state

    evidence = state["trace"].get("evidence", [])
    contested = is_contested_topic(state["query"])
    state["trace"]["contested_topic"] = contested

    final_answer = majority_answer

    if CREW_AVAILABLE:
        try:
            crew_output = run_verification_crew(state["query"], majority_answer, evidence)
            state["trace"]["crew_output"] = crew_output
            if crew_output.get("final_verdict"):
                final_answer = crew_output["final_verdict"]
        except Exception as e:
            state["trace"]["crew_error"] = str(e)
    else:
        state["trace"]["crew_error"] = "CrewAI unavailable at import time; using majority answer directly."

    perspectives = []
    if contested:
        perspectives = [
            "Multiple viewpoints exist on this topic; the answer above attempts a balanced synthesis rather than a single verdict."
        ]

    safety = validate_output(final_answer, [e["source"] for e in evidence])
    if not safety["valid"]:
        final_answer = safety["fallback"]
        status = "insufficient_evidence"
    else:
        status = "answered"

    state["final_result"] = {
        "status": status,
        "answer": final_answer,
        "semantic_entropy_score": uncertainty["semantic_entropy_score"],
        "is_consistent": uncertainty["is_consistent"],
        "confidence": uncertainty["confidence"],
        "sources": [e["source"] for e in evidence],
        "is_neutral_assessment": True,
        "perspectives_considered": perspectives,
    }
    return state


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("moderation", moderation_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("entropy", entropy_node)
    graph.add_node("verification", verification_node)

    graph.set_entry_point("moderation")
    graph.add_conditional_edges(
        "moderation",
        lambda s: "end" if s["blocked"] else "continue",
        {"end": END, "continue": "retrieval"},
    )
    graph.add_edge("retrieval", "entropy")
    graph.add_edge("entropy", "verification")
    graph.add_edge("verification", END)
    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(query: str) -> dict:
    query_id = str(uuid.uuid4())
    initial_state: GraphState = {
        "query": query,
        "query_id": query_id,
        "trace": {
            "query_id": query_id,
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "blocked": False,
        "final_result": {},
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)

    result = dict(final_state["final_result"])
    result["reasoning_trace"] = final_state["trace"]
    result["query_id"] = query_id

    _save_trace(query_id, final_state["trace"])
    return result


def _save_trace(query_id: str, trace: dict):
    try:
        path = os.path.join(TRAJECTORY_PATH, f"{query_id}.json")
        with open(path, "w") as f:
            json.dump(trace, f, indent=2, default=str)
    except Exception:
        pass


    