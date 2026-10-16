"""
LangGraph state machine definition
"""
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    initialize_node,
    question_generation_node,
    llm_processing_node,
    logic_routing_node,
    progress_tracker_node,
    completion_node,
    update_resolution_node,
)


def should_continue(state: AgentState) -> str:
    """Determine next step based on state"""
    if state.get("all_questions_completed"):
        return "complete"
    return "generate"


def route_intent(state: AgentState) -> str:
    """Route based on NLU intent."""
    nlu_data = state.get("nlu_data", {})
    intent = nlu_data.get("intent")

    if intent == "update_response":
        return "update_resolution"

    return "logic_routing"


def route_entry_point(state: AgentState) -> str:
    """Route from entry based on whether we're initializing or processing user response."""
    if not state.get("user_message"):
        return "initialize"
    return "llm_processing"


def router_node(state: AgentState) -> AgentState:
    """Router node - just passes state through"""
    return state


def create_conversation_graph():
    """Create and compile the conversation graph"""

    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("llm_processing", llm_processing_node)
    workflow.add_node("logic_routing", logic_routing_node)
    workflow.add_node("update_resolution", update_resolution_node)
    workflow.add_node("progress", progress_tracker_node)
    workflow.add_node("question_generation", question_generation_node)
    workflow.add_node("complete", completion_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_entry_point,
        {
            "initialize": "initialize",
            "llm_processing": "llm_processing",
        },
    )

    # Initialize -> Question Generation (Start)
    workflow.add_edge("initialize", "question_generation")

    # Question Generation -> END (Wait for user)
    workflow.add_edge("question_generation", END)

    # LLM Processing -> Intent Routing
    workflow.add_conditional_edges(
        "llm_processing",
        route_intent,
        {
            "logic_routing": "logic_routing",
            "update_resolution": "update_resolution"
        }
    )

    # Logic Routing -> Progress Tracker
    workflow.add_edge("logic_routing", "progress")
    workflow.add_edge("update_resolution", "progress")

    # Progress Tracker -> Conditional (Generate or Complete)
    workflow.add_conditional_edges(
        "progress",
        should_continue,
        {
            "generate": "question_generation",
            "complete": "complete",
        }
    )

    workflow.add_edge("complete", END)

    return workflow.compile()
