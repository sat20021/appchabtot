"""
Conversation context management for efficient routing decisions.

This module provides efficient context building and management for the hybrid orchestrator,
ensuring optimal memory usage and fast context retrieval.
"""
import logging
from typing import Dict, List, Optional

from models.hybrid_schemas import ContextState, ConversationRequest
from hybrid_orchestrator.session_manager import get_session

logger = logging.getLogger(__name__)

# Maximum number of conversation turns to keep in context
MAX_CONTEXT_TURNS = 3


def build_context(req: ConversationRequest, session: Optional[Dict] = None) -> ContextState:
    """
    Build conversation context from request and session data.
    
    Args:
        req: The conversation request
        session: Optional pre-fetched session data (for optimization)
    
    Returns:
        ContextState with all necessary context for routing
    
    Raises:
        ValueError: If session_id is missing
    """
    if not req.session_id:
        raise ValueError("session_id is required for context building")
    
    # Fetch session if not provided
    if session is None:
        session = get_session(req.session_id) or {}
    
    # Extract relevant data
    metadata = req.metadata.dict() if req.metadata else {}
    last_three_turns = session.get("last_three_turns", [])
    active_flow = session.get("active_flow")
    form_data = session.get("form_data", {})  # Extract collected form data
    completed_forms = session.get("completed_forms", [])  # 🔴 NEW: Track completed forms
    
    # 🔴 ENTERPRISE DEBUG: Log form data availability for troubleshooting
    if form_data:
        logger.info(
            f"📋 CONTEXT: Session {req.session_id} has form_data: {list(form_data.keys())}"
        )
    else:
        logger.debug(f"📋 CONTEXT: Session {req.session_id} has NO form_data")
    
    if completed_forms:
        logger.info(
            f"✅ CONTEXT: Session {req.session_id} has completed_forms: {completed_forms}"
        )
    
    # Validate and limit turns
    if len(last_three_turns) > MAX_CONTEXT_TURNS:
        last_three_turns = last_three_turns[-MAX_CONTEXT_TURNS:]
        logger.debug(
            "Truncated conversation history to %d turns for session %s",
            MAX_CONTEXT_TURNS,
            req.session_id
        )
    
    return ContextState(
        session_id=req.session_id,
        message=req.message,
        session=session,
        metadata=metadata,
        last_three_turns=last_three_turns,
        active_flow=active_flow,
        form_data=form_data,  # Include form data in context
    )


def get_relevant_history(context: ContextState, max_turns: int = MAX_CONTEXT_TURNS) -> List[Dict[str, str]]:
    """
    Get relevant conversation history for LLM processing.
    
    Args:
        context: The conversation context
        max_turns: Maximum number of turns to return (default: 3)
    
    Returns:
        List of conversation turns (role, message pairs)
    """
    turns = context.last_three_turns
    if len(turns) > max_turns:
        return turns[-max_turns:]
    return turns


def validate_context(context: ContextState) -> bool:
    """
    Validate that context has required fields.
    
    Args:
        context: The context to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not context.session_id:
        logger.warning("Context missing session_id")
        return False
    
    if not context.message:
        logger.warning("Context missing message")
        return False
    
    if not isinstance(context.session, dict):
        logger.warning("Context session is not a dict")
        return False
    
    return True


def update_context_after_decision(
    context: ContextState,
    decision_result: Dict[str, any],
    response_message: Optional[str] = None
) -> Dict:
    """
    Update session context after a routing decision.
    
    Args:
        context: The conversation context
        decision_result: The decision result from a node
        response_message: Optional response message to append
    
    Returns:
        Updated session dict
    """
    from hybrid_orchestrator.session_manager import append_turn, upsert_session, append_conversation
    
    # Append user message if not already done
    if context.message:
        append_turn(context.session_id, "user", context.message)
    
    # Append assistant response if provided
    if response_message:
        append_turn(context.session_id, "assistant", response_message)
        
        # Also save to MongoDB history format (query/response pairs)
        if context.message:
            domain = context.metadata.get("domain", "domain_3") if context.metadata else "domain_3"
            append_conversation(context.session_id, context.message, response_message, domain)
    
    # Update session with any decision metadata
    updates = {}
    if decision_result.get("route") == "form_ml" or decision_result.get("route") == "form_llm":
        question_set = decision_result.get("response_data", {}).get("question_set")
        if question_set:
            updates["active_flow"] = question_set
            updates["graph_state_id"] = decision_result.get("response_data", {}).get("graph_state_id")
    
    if updates:
        upsert_session(context.session_id, updates)
        # Refresh session in context
        context.session = get_session(context.session_id) or context.session
    
    return context.session


