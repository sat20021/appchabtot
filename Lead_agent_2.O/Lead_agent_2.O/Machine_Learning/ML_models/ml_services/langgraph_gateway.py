"""
Helper utilities to interact with the LangGraph conversational agent programmatically.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Optional
from hybrid_orchestrator.session_manager import get_session, upsert_session

# Ensure the repository root is on sys.path so we can import Conversational_AI modules
REPO_ROOT = Path(__file__).resolve().parents[3]
TOP_LEVEL_ROOT = REPO_ROOT.parent

# CRITICAL: Insert TOP_LEVEL_ROOT at 0 to prioritize it.
# This prevents 'utils' package in Conversational_AI from shadowing the top-level 'utils' package.
if str(TOP_LEVEL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOP_LEVEL_ROOT))

CONV_ROOT = REPO_ROOT / "Conversational_AI"
if str(CONV_ROOT) not in sys.path:
    # Append CONV_ROOT so it doesn't shadow top-level packages
    sys.path.append(str(CONV_ROOT))

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

logger = logging.getLogger(__name__)

_sessions: Dict[str, "ConversationalAgent"] = {}
ConversationalAgent = None


def _get_agent_cls():
    global ConversationalAgent
    if ConversationalAgent is None:
        from Conversational_AI.main import ConversationalAgent as _Agent  # type: ignore

        ConversationalAgent = _Agent
    return ConversationalAgent


def start_form_session(
    question_set_title: str, 
    session_id: Optional[str] = None, 
    domain: str = "default",
    previous_message: str = None,
    selected_job: Optional[Dict] = None
) -> Dict:
    """
    Start a LangGraph session for the specified question set (e.g., lead or job form).
    
    Args:
        question_set_title: Title of the question set to use
        session_id: Optional session ID (will be generated if not provided)
        domain: Domain/tenant identifier for persona and RAG context
        previous_message: Optional previous user message to extract info from
        selected_job: Optional dict containing selected job details for JD-based applications
                     Keys: job_id, title, description, key_skills, location, job_type, experience_level
    """
    # 🔴 NEW: Retrieve persisted user context from session before creating agent
    session_form_data = {}
    if session_id:
        session = get_session(session_id)
        if session:
            session_form_data = session.get("form_data", {})
            if session_form_data:
                logger.info(
                    "🔄 Retrieving %d persisted form fields for session '%s': %s",
                    len(session_form_data),
                    session_id,
                    list(session_form_data.keys())
                )
    
    agent_cls = _get_agent_cls()
    
    # Determine if this is a JD-based job application (specific job selected)
    jd_selected = True if selected_job else None
    
    agent = agent_cls(question_set_title, domain=domain, jd_selected=jd_selected)  # ← Pass jd_selected
    
    # 🔴 NEW: Inject persisted form_data into agent state BEFORE starting
    # This allows initialize_node to pre-fill collected_responses
    agent.state["session_form_data"] = session_form_data
    
    # 🎯 NEW: Store original user inquiry for personalization throughout form
    # This is what the user asked BEFORE the form started (e.g., "chatbot", "Java", "Swift")
    if previous_message:
        agent.state["original_user_inquiry"] = previous_message
        logger.info(f"💾 Storing original inquiry: '{previous_message}'")
    
    greeting = agent.start(initial_context=previous_message)  # ← Pass previous context
    session_id = session_id or agent.state.get("session_id")
    if not session_id:
        raise RuntimeError("LangGraph agent did not produce a session_id.")
    agent.state["session_id"] = session_id
    
    # If selected_job is provided, store it in agent state for JD-based flows
    if selected_job:
        agent.state["selected_job_id"] = selected_job.get("job_id")
        agent.state["selected_job_details"] = selected_job
        agent.state["job_description_selected"] = True
        logger.info(
            "JD-based job application: job_id='%s', title='%s'",
            selected_job.get("job_id"),
            selected_job.get("title")
        )
    
    _sessions[session_id] = agent
    logger.info(
        "Started LangGraph session '%s' for question set '%s' with domain '%s'%s%s%s",
        session_id,
        question_set_title,
        domain,
        f" (with context: '{previous_message[:50]}...')" if previous_message else "",
        f" (job_id: {selected_job.get('job_id')})" if selected_job else "",
        f" (pre-filled: {list(session_form_data.keys())})" if session_form_data else ""
    )
    
    # Persist initial state to Redis (including original_user_inquiry for session restoration)
    upsert_session(session_id, {
        "graph_state": agent.get_state(),
        "original_user_inquiry": previous_message,  # 🔴 FIX: Persist for session restore
        "domain": domain  # 🔴 FIX: Persist domain for session restore
    })
    
    return {
        "session_id": session_id,
        "agent_message": greeting,
        "collected_data": agent.get_collected_data(),
        "all_questions_completed": agent.state.get("all_questions_completed", False),
        "state": agent.state,
    }


def send_form_message(session_id: str, message: str) -> Dict:
    """
    Send a user reply to an existing LangGraph session and return the agent's response.
    """
    agent = _sessions.get(session_id)
    
    # Restore from persistence if not in memory (e.g. after restart)
    if not agent:
        session = get_session(session_id)
        if session and session.get("graph_state"):
            logger.info(f"🔄 Restoring session '{session_id}' from Redis persistence")
            agent_cls = _get_agent_cls()
            
            # Re-instantiate agent with saved config
            graph_state = session["graph_state"]
            agent = agent_cls(
                graph_state.get("question_set_title", "lead_questions"),
                domain=session.get("domain", "default")
            )
            # Restore state
            agent.load_state(graph_state)
            
            # 🔴 FIX: Restore original_user_inquiry from Redis
            original_inquiry = session.get("original_user_inquiry")
            if original_inquiry:
                agent.state["original_user_inquiry"] = original_inquiry
                logger.info(f"📍 Restored original_user_inquiry: '{original_inquiry[:50]}...'")
            
            _sessions[session_id] = agent
        else:
            raise KeyError(f"Session '{session_id}' not found in LangGraph gateway or persistence.")
            
    response = agent.send_message(message)
    
    # Persist updated state to Redis
    upsert_session(session_id, {"graph_state": agent.get_state()})
    
    done = agent.state.get("all_questions_completed", False)
    if done:
        logger.info("LangGraph session '%s' completed.", session_id)
    return {
        "session_id": session_id,
        "agent_message": response,
        "collected_data": agent.get_collected_data(),
        "all_questions_completed": done,
        "state": agent.state,
    }


def end_form_session(session_id: str):
    """
    Cleanup and close an existing LangGraph session.
    """
    agent = _sessions.pop(session_id, None)
    if agent:
        agent.close()
        logger.info("Closed LangGraph session '%s'.", session_id)


def get_active_session(session_id: str) -> Optional[ConversationalAgent]:
    """
    Get the ConversationalAgent instance for a given session.
    """
    return _sessions.get(session_id)

