"""
LangGraph Agent Nodes - Enhanced for human-like conversation

🔒 ANTI-GRAVITY ARCHITECTURE:
The agent nodes now integrate with the governance module for:
- Intent-driven form start (only when ML intent approves)
- Consent state management via session_manager
- Audit trail for compliance
"""
print("🚀 NODES.PY MODULE LOADED - MULTI-FIELD EXTRACTION ENABLED 🚀")
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.runnables import RunnablePassthrough
from agent.tools import AGENT_TOOLS, create_lead_record_action
from agent.state import AgentState
from Conversational_AI.utils.conversational_helpers import (
    get_next_question_key,
    format_question_for_display,
    validate_email,
    validate_phone,
    normalize_email_answer,
    normalize_phone_answer
)
from Conversational_AI.database.lead_writer import (
    create_realtime_lead_id,
    upsert_realtime_lead,
    finalize_realtime_lead,
)
from Conversational_AI.database.mongodb import MongoDBHandler
import uuid
import random
import json
import re
import logging
from hybrid_orchestrator.question_registry import (
    get_question_registry,
    update_question_status,
    QuestionStatus,
    is_question_complete
)

# 🔒 GOVERNANCE: Import consent management for Anti-Gravity architecture
try:
    from hybrid_orchestrator.session_manager import (
        grant_consent,
        get_consent_state,
        has_consent,
        update_session_state,
    )
    CONSENT_MANAGEMENT_ENABLED = True
except ImportError:
    CONSENT_MANAGEMENT_ENABLED = False
    logging.warning("Consent management not available - running in legacy mode")

logger = logging.getLogger(__name__)


# ============================================================================
# SECTION D: CONVERSATION MODE ENUM (EXPLICIT, NOT INFERRED IN PROMPTS)
# ============================================================================
class ConversationMode(Enum):
    """Strict conversation mode - computed in logic nodes, consumed by prompts."""
    OPENING = "OPENING"
    MIDDLE = "MIDDLE"
    CLARIFICATION = "CLARIFICATION"
    OFF_TOPIC = "OFF_TOPIC"
    CLOSING = "CLOSING"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"


def compute_conversation_mode(state: AgentState) -> ConversationMode:
    """
    Deterministically compute conversation mode from state.
    Mode is computed here (logic), NOT inferred in prompts.
    """
    needs_clarification = state.get("needs_clarification", False)
    off_topic_handled = state.get("off_topic_handled", False)
    questions_asked_count = len(state.get("questions_asked", []))
    questions_queue = state.get("questions_queue", [])
    clarification_retry_count = state.get("clarification_retry_count", {})
    current_key = state.get("current_question_key")
    
    # Check for retry exhaustion
    if current_key and clarification_retry_count.get(current_key, 0) >= 3:
        return ConversationMode.RETRY_EXHAUSTED
    
    if needs_clarification:
        return ConversationMode.CLARIFICATION
    
    if off_topic_handled:
        return ConversationMode.OFF_TOPIC
    
    if questions_asked_count == 0:
        return ConversationMode.OPENING
    
    if len(questions_queue) <= 1:
        return ConversationMode.CLOSING
    
    return ConversationMode.MIDDLE


def get_mode_instruction(mode: ConversationMode) -> str:
    """Convert mode enum to human-readable instruction for prompt."""
    instructions = {
        ConversationMode.OPENING: "Make the user feel welcome and excited to chat",
        ConversationMode.MIDDLE: "Keep momentum, stay engaged",
        ConversationMode.CLARIFICATION: "Gently clarify without sounding like an error message",
        ConversationMode.OFF_TOPIC: "Answer their question helpfully, then transition back naturally",
        ConversationMode.CLOSING: "This is the last thing you need. Wrap up warmly.",
        ConversationMode.RETRY_EXHAUSTED: "We've asked multiple times. Skip gracefully and move on."
    }
    return instructions.get(mode, "Continue the conversation naturally")


# ============================================================================
# KEY NORMALIZATION & GUARDRAILS
# ============================================================================
KEY_ALIASES = {
    "name": ["name", "business_name", "full_name", "your_name", "user_name"],
    "email": ["email", "work_email", "email_address", "your_email"],
    "phone": ["phone", "mobile", "contact_number", "phone_number", "your_phone"],
    "company": ["company", "business_name", "company_name", "organization"],
    "user_persona": ["user_persona", "persona", "role", "student_or_parent", "who_are_you"],
    "source": ["source", "referral", "how_did_you_hear", "channel"],
}


def _normalize_key(key: str) -> str:
    """Normalize a key to its canonical form for comparison."""
    key_lower = key.lower()
    for canonical, aliases in KEY_ALIASES.items():
        if key_lower in aliases:
            return canonical
    return key_lower


def _key_is_answered(key: str, collected: Dict, session_form_data: Dict) -> bool:
    """Check if a key (or any of its aliases) is already answered."""
    normalized = _normalize_key(key)
    
    for k in collected.keys():
        if _normalize_key(k) == normalized:
            return True
    
    for k in session_form_data.keys():
        if _normalize_key(k) == normalized:
            return True
    
    return False


def ask_question_guardrail(state: AgentState, question_key: str) -> bool:
    """
    ENTERPRISE GUARDRAIL: Deterministically check if a question can be asked.
    
    Returns:
        bool: True if question can be asked, False if it should be skipped
    """
    collected = state.get("collected_responses", {})
    session_form_data = state.get("session_form_data", {})
    
    if _key_is_answered(question_key, collected, session_form_data):
        logger.warning(f"🚫 GUARDRAIL: Blocking re-ask of semantically answered question '{question_key}'")
        return False
    
    if question_key in state.get("definitively_completed_keys", []):
        logger.warning(f"🚫 GUARDRAIL: Blocking re-ask of completed question '{question_key}'")
        return False
    
    if question_key in collected:
        logger.warning(f"🚫 GUARDRAIL: Blocking re-ask of answered question '{question_key}'")
        return False
    
    if question_key in state.get("declined_questions", []):
        logger.warning(f"🚫 GUARDRAIL: Blocking re-ask of declined question '{question_key}'")
        return False
    
    registry = state.get("question_status_registry", {})
    entry = registry.get(question_key, {})
    status = entry.get("status")
    
    if status in ["answered", "declined", "skipped", "exhausted", "pre_extracted"]:
        logger.warning(f"🚫 GUARDRAIL: Blocking re-ask per registry status '{status}'")
        return False
    
    return True


# ============================================================================
# TOOL ACTION WRAPPER
# ============================================================================
class ToolAction:
    """Wrapper to make Gemini tool calls compatible with LangChain action format"""
    def __init__(self, tool_name: str, tool_input: dict):
        self.tool = tool_name
        self.tool_input = tool_input


def log_intermediate_steps(intermediate_steps: List) -> None:
    """Log agent executor activity similar to LangChain verbose output."""
    if not intermediate_steps:
        return
    print("\n> Entering new AgentExecutor chain...")
    for action, observation in intermediate_steps:
        tool_name = getattr(action, "tool", "unknown_tool")
        tool_input = getattr(action, "tool_input", {})
        try:
            formatted_input = json.dumps(tool_input, ensure_ascii=False)
        except TypeError:
            formatted_input = str(tool_input)
        print(f"\nInvoking: `{tool_name}` with `{formatted_input}`\n")
        print(observation)
    print("\n> Finished chain.\n")


def _advance_to_unasked_question(
    current_key: Optional[str],
    questions_queue: List[str],
    questions_asked: List[str],
    collected: Dict,
) -> Tuple[Optional[str], List[str], List[str], bool]:
    """
    Skip over any questions that already have collected responses before we ask them.
    """
    skipped = False
    while current_key and current_key in collected:
        skipped = True
        if current_key in questions_queue:
            questions_queue.remove(current_key)
        if current_key not in questions_asked:
            questions_asked.append(current_key)
        current_key = get_next_question_key(questions_queue)
    return current_key, questions_queue, questions_asked, skipped


# ============================================================================
# PERSISTENCE HELPERS (SECTION B FIX - SINGLE AUTHORITATIVE IMPLEMENTATION)
# ============================================================================
def _persist_partial_lead_data(state: AgentState, updates: Dict[str, Any]) -> Optional[str]:
    """Persist incremental answers to the realtime lead document.

    Behaviour:
    - Only persists for lead collection flows (`question_set_title == "lead_questions"`).
    - Lazily creates a `lead_id` on first successful answer instead of during
      initialization, so we never create empty leads.
    """
    if not updates:
        return None

    question_set = state.get("question_set_title", "lead_questions")
    if question_set != "lead_questions":
        return None

    lead_id = state.get("lead_id")
    if not lead_id:
        session_id = state.get("session_id")
        domain = state.get("domain", "default")
        try:
            lead_id = create_realtime_lead_id(
                session_id=session_id,
                question_set_title=question_set,
                domain=domain,
            )
            state["lead_id"] = lead_id
            print(f"[lead-storage] Bootstrapped lead {lead_id} for session {session_id}")
        except Exception as exc:
            print(f"[lead-storage] Failed to bootstrap lead id for partial save: {exc}")
            return None

    session_id = state.get("session_id") or lead_id
    domain = state.get("domain", "default")

    try:
        upsert_realtime_lead(
            lead_id=lead_id,
            session_id=session_id,
            updates=updates,
            question_set_title=question_set,
            domain=domain,
        )
        print(f"[lead-storage] Saved partial fields {list(updates.keys())} for {lead_id}")
    except Exception as exc:
        print(f"[lead-storage] Failed to save partial fields for {lead_id}: {exc}")
    
    return lead_id


def get_persona_for_domain(domain: str) -> Optional[Dict]:
    """Load persona for the given domain from persona service."""
    try:
        import sys
        from pathlib import Path
        
        repo_root = Path(__file__).resolve().parents[3]
        ml_root = repo_root / "Machine_Learning" / "ML_models"
        if str(ml_root) not in sys.path:
            sys.path.insert(0, str(ml_root))
        
        import importlib

        try:
            persona_module = importlib.import_module("services.persona_service")
        except ImportError:
            persona_module = importlib.import_module("ml_services.persona_service")
        
        persona_service = persona_module.get_persona_service()
        persona = persona_service.get_persona(domain)
        
        if persona:
            # 🔴 FIX: Return ALL persona fields, not just name/description/instructions
            return {
                "name": persona.name,
                "description": persona.description,
                "instructions": persona.instructions,
                "conditions": persona.conditions or {},         # ADD: For rule-based behavior
                "formatting_style": persona.formatting_style or {}  # ADD: For enhancer guidelines
            }
        return None
    except Exception as e:
        print(f"Warning: Could not load persona for domain '{domain}': {e}")
        return None


def format_previous_responses(collected_responses: Dict, max_items: int = 3) -> str:
    """Format previously collected responses for LLM context."""
    if not collected_responses:
        return "None yet - this is the first question."
    
    items = list(collected_responses.items())[-max_items:]
    
    formatted_lines = []
    for key, value in items:
        formatted_key = key.replace("_", " ").title()
        formatted_lines.append(f"  - {formatted_key}: {value}")
    
    return "\n".join(formatted_lines)


# ============================================================================
# SECTION E: CONSENT GATING
# ============================================================================
def check_consent_gate(state: AgentState) -> Tuple[bool, Optional[str]]:
    """
    Check if consent is required and granted before proceeding.
    
    Returns:
        (can_proceed, consent_request_message)
        - can_proceed: True if consent exists or not required
        - consent_request_message: Message to display if consent needed
    """
    if not CONSENT_MANAGEMENT_ENABLED:
        return True, None
    
    session_id = state.get("session_id")
    if not session_id:
        return True, None
    
    # Check if PII consent exists
    if has_consent(session_id, "pii"):
        return True, None
    
    # Consent not granted - return friendly request
    consent_message = (
        "Before we continue, I'd like to make sure you're comfortable sharing some "
        "basic information with us. Your data will be handled securely and used only "
        "to help you. Is that okay with you?"
    )
    return False, consent_message


def grant_pii_consent_if_needed(state: AgentState) -> None:
    """Auto-grant PII consent when form starts (user implicitly consents by starting form)."""
    if not CONSENT_MANAGEMENT_ENABLED:
        return
    
    session_id = state.get("session_id")
    question_set = state.get("question_set_title", "")
    
    if session_id and question_set:
        try:
            if not has_consent(session_id, "pii"):
                grant_consent(session_id, "pii")
                logger.info(f"🔒 Auto-granted PII consent for form '{question_set}' in session {session_id}")
        except Exception as e:
            logger.warning(f"Could not grant consent: {e}")


# ============================================================================
# MAIN NODES
# ============================================================================
def initialize_node(state: AgentState) -> Dict:
    """Initialize the conversation and load question set.
    
    IMPORTANT: This node is idempotent - it preserves existing state values
    to avoid resetting progress when the graph is re-invoked.
    """
    if not state.get("session_id"):
        state["session_id"] = str(uuid.uuid4())
    session_id = state.get("session_id")
    question_set_title = state.get("question_set_title", "lead_questions")
    domain = state.get("domain", "default")
    
    # SECTION E: Grant consent when form starts
    grant_pii_consent_if_needed(state)
    
    # 🔴 FIX: Handle nested 'questions' object if present (e.g. from JSON seed)
    raw_data = state["question_set_data"]
    if "questions" in raw_data and isinstance(raw_data["questions"], dict):
        print("🔄 Detected nested 'questions' object - flattening for graph usage")
        # Extract metadata
        metadata = {k: v for k, v in raw_data.items() if k != "questions"}
        # Merge questions to top level
        flat_data = {**metadata, **raw_data["questions"]}
        # Update state
        state["question_set_data"] = flat_data

    excluded_fields = [
        "_id", "title",
        "lead_category", "description", "questions"
    ]
    
    question_keys = [
        k
        for k in state["question_set_data"].keys()
        if k not in excluded_fields and not isinstance(state["question_set_data"][k], str)
    ]
    
    # Load question registry from session
    registry = get_question_registry(session_id)
    
    # Build queue from PENDING questions only
    definitively_completed = [
        key for key, entry in registry.items()
        if entry["status"] in [
            QuestionStatus.ANSWERED.value,
            QuestionStatus.DECLINED.value,
            QuestionStatus.SKIPPED.value,
            QuestionStatus.PRE_EXTRACTED.value,
            QuestionStatus.EXHAUSTED.value
        ]
    ]
    
    # CRITICAL FIX: Preserve existing state to avoid resetting progress
    existing_collected = state.get("collected_responses", {})
    existing_current_key = state.get("current_question_key")
    existing_queue = state.get("questions_queue", [])
    existing_asked = state.get("questions_asked", [])
    
    # Load persisted user context from previous form sessions
    session_form_data = state.get("session_form_data", {})
    
    # Merge strategy: session_form_data provides defaults, existing_collected takes precedence
    merged_collected = {**session_form_data, **existing_collected}
    
    is_first_init = not existing_collected and not existing_current_key
    
    if is_first_init:
        if session_form_data:
            print(f"🔄 First initialization with {len(session_form_data)} pre-filled fields: {list(session_form_data.keys())}")
        else:
            print("🆕 First initialization - using defaults")
    else:
        print(f"🔄 Re-initialization - preserving existing state (collected: {list(existing_collected.keys())})")

    # 🆕 PRE-FILL FROM INITIAL CONTEXT: Extract info from original_user_inquiry
    # If user said "I'm John and I want to book a meeting", pre-fill name
    original_inquiry = state.get("original_user_inquiry", "")
    if original_inquiry and is_first_init:
        print(f"🔍 Scanning initial message for pre-fill: '{original_inquiry[:60]}...'")
        
        # Extract name patterns: "I'm [Name]", "My name is [Name]", "[Name] here"
        name_patterns = [
            r"\bI'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",  # I'm John, I am John Smith
            r"\bMy\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",  # My name is John
            r"\bthis\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",  # This is John
            r"^([A-Z][a-z]+)\s+here\b",  # John here
        ]
        
        extracted_name = None
        for pattern in name_patterns:
            match = re.search(pattern, original_inquiry, re.IGNORECASE)
            if match:
                extracted_name = match.group(1).strip()
                break
        
        if extracted_name and len(extracted_name) > 1:
            # Find the name key in question_set_data
            for q_key in question_set_data.keys():
                if q_key.lower() == "name" or "name" in q_key.lower():
                    if q_key not in merged_collected:
                        merged_collected[q_key] = extracted_name
                        print(f"   ✅ Pre-filled '{q_key}' = '{extracted_name}'")
                    break
        
        # Extract email pattern
        email_match = re.search(r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', original_inquiry)
        if email_match:
            extracted_email = email_match.group(1)
            for q_key in question_set_data.keys():
                if "email" in q_key.lower():
                    if q_key not in merged_collected:
                        merged_collected[q_key] = extracted_email
                        print(f"   ✅ Pre-filled '{q_key}' = '{extracted_email}'")
                    break
        
        # Extract phone pattern (international format)
        phone_match = re.search(r'\b(\+?\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4})\b', original_inquiry)
        if phone_match:
            extracted_phone = phone_match.group(1)
            # Validate it's long enough to be a phone number
            digits_only = re.sub(r'\D', '', extracted_phone)
            if len(digits_only) >= 7:
                for q_key in question_set_data.keys():
                    if "phone" in q_key.lower() or "mobile" in q_key.lower():
                        if q_key not in merged_collected:
                            merged_collected[q_key] = extracted_phone
                            print(f"   ✅ Pre-filled '{q_key}' = '{extracted_phone}'")
                        break
    
    # 🆕 Update queue to remove pre-filled questions
    initial_queue = existing_queue if existing_queue else question_keys
    filtered_queue = [q for q in initial_queue if q not in merged_collected or q in existing_collected]

    # Determine current question (skip any that are already collected)
    if existing_current_key and existing_current_key not in merged_collected:
        final_current_key = existing_current_key
    elif filtered_queue:
        final_current_key = filtered_queue[0]
    else:
        final_current_key = question_keys[0] if question_keys else None
    
    return {
        "questions_queue": filtered_queue if is_first_init else (existing_queue if existing_queue else question_keys),
        "current_question_key": final_current_key,
        "questions_asked": existing_asked if existing_asked else [],
        "collected_responses": merged_collected,
        "declined_questions": state.get("declined_questions", []),
        "messages": state.get("messages", []),
        "lead_id": state.get("lead_id"),
        "session_form_data": session_form_data,
        # 🔴 FIX: Preserve user context during re-initialization
        "original_user_inquiry": state.get("original_user_inquiry"),
        "user_message": state.get("user_message"),
        "question_set_data": state["question_set_data"]
    }


# ============================================================================
# SECTION C: STATE SEPARATION - BUILD CLEAN PROMPT CONTEXT
# ============================================================================
def _build_known_user_summary(collected: Dict, session_context: Dict) -> str:
    """
    Build human-readable summary of known user info.
    NEVER expose raw state dicts to LLM.
    """
    known_parts = []
    if collected.get('name'):
        known_parts.append(f"Their name is {collected.get('name')}")
    if collected.get('email'):
        known_parts.append(f"email is {collected.get('email')}")
    if collected.get('phone'):
        known_parts.append(f"phone is {collected.get('phone')}")
    if session_context:
        for k, v in session_context.items():
            if k not in collected and k not in ['applicant_id', 'job_id', 'resume_link']:
                known_parts.append(f"{k} is {v}")
    return ". ".join(known_parts) if known_parts else "Nothing yet - this is a new conversation."


def _build_conversation_goal(
    needs_clarification: bool,
    off_topic_handled: bool,
    last_intent: Optional[str],
    last_key: Optional[str],
    last_value: Optional[str],
    question_text: str,
    mode: ConversationMode
) -> str:
    """
    Build human-readable conversation goal.
    Deterministic - no state dicts exposed.
    """
    if mode == ConversationMode.RETRY_EXHAUSTED:
        return f"We've asked about {last_key} multiple times. Skip gracefully with a friendly message and move on."
    
    if needs_clarification:
        return "The last answer wasn't quite clear. Gently ask for clarification without sounding like an error message."
    
    if off_topic_handled:
        user_question = last_value if last_value else "something unrelated"
        return f"""IMPORTANT: They asked an off-topic question: '{user_question}'.

STEP 1: Answer their question briefly (1-2 sentences max) using your knowledge or saying you're not sure
STEP 2: Then naturally transition back to ask: {question_text}

Example format:
"[Brief answer to their question]. Now, {question_text}"

DO NOT just say "that's interesting" - actually try to answer their question first."""
    
    
    if last_intent == "mark_question_declined":
        return f"They declined to share {last_key}. That's totally fine - respect it and move on naturally."
    
    return f"Get the answer to: {question_text}"


def question_generation_node(state: AgentState) -> Dict:
    """Generate the final conversational response based on state.
    
    Enhanced with anti-hallucination safeguards:
    1. Response sanitization (removes robotic phrases)
    2. Question validation (ensures correct field is asked)
    3. Acknowledgment prefixes (hardcoded, not LLM-generated)
    """
    
    current_key = state.get("current_question_key")
    needs_clarification = state.get("needs_clarification", False)
    off_topic_handled = state.get("off_topic_handled", False)
    collected = state.get("collected_responses", {})
    
    # Get NLU Data
    nlu_data = state.get("nlu_data", {})
    last_intent = nlu_data.get("intent")
    last_key = nlu_data.get("key")
    last_value = nlu_data.get("value")
    
    # Get Conversation History (Last 6 messages = 3 turns)
    messages = state.get("messages", [])
    history_text = ""
    if messages:
        def get_msg_info(m):
            if isinstance(m, dict):
                return m.get("type"), m.get("content")
            return getattr(m, "type", None), getattr(m, "content", None)

        relevant_msgs = []
        for m in messages:
            m_type, m_content = get_msg_info(m)
            if m_type in ["human", "ai"]:
                relevant_msgs.append((m_type, m_content))
                
        recent_msgs = relevant_msgs[-6:]
        history_text = "\n".join([f"{t.upper()}: {c}" for t, c in recent_msgs])
    
    # Get Question Data
    question_data = state["question_set_data"].get(current_key, {})
    if isinstance(question_data.get("name"), dict):
        question_text = question_data.get("name", {}).get("name", "")
    else:
        question_text = question_data.get("name", "")

    # 🔴 CUSTOM LOGIC: Enforce clear "Student or Parent" question
    if current_key == "user_persona":
        question_text = "Are you a Student or a Parent?"
        print(f"🔒 Enforcing FIXED question for user_persona: {question_text}")

    # Get fallback question if available
    fallback_question = question_data.get("fallback", question_text)
        
    # Get question description for tone guidance
    question_description = question_data.get("Description", "")
    
    options_text = ""
    if question_data.get("type") == "options":
        options = question_data.get("options", [])
        options_text = f"Options: {', '.join(options)}"

    # Get Persona
    domain = state.get("domain", "default")
    persona = get_persona_for_domain(domain)
    if not persona:
        persona = {
            "name": "Sara",
            "description": "A warm, friendly assistant.",
            "instructions": "Be warm and conversational."
        }

    # Build clean context - NO RAW STATE
    session_context = state.get("session_form_data", {})
    known_user_summary = _build_known_user_summary(collected, session_context)
    
    # Compute mode deterministically
    mode = compute_conversation_mode(state)
    mode_instruction = get_mode_instruction(mode)
    
    # Determine if user just answered something (for acknowledgment)
    has_previous_answer = bool(collected) and last_intent == "extract_answer"
    
    # Build previous field name for acknowledgment
    previous_field = ""
    if has_previous_answer and collected:
        previous_field = list(collected.keys())[-1]  # Get last collected field
    
    # Adjust sentence limit based on mode
    if mode == ConversationMode.OFF_TOPIC:
        sentence_limit = "2-3 sentences (answer their question first, then ask yours)"
    else:
        sentence_limit = "1-2 sentences"
    
    # Build acknowledgment instruction
    if has_previous_answer:
        acknowledgment_instruction = f"""
ACKNOWLEDGMENT REQUIRED:
- User just provided their {previous_field}
- Start with a brief acknowledgment: "Got it!", "Perfect!", "Great!", "Thanks!"
- Then ask the next question
- Keep it conversational and friendly

Example: "Got it! And **what number works best for you?**"
"""
    else:
        acknowledgment_instruction = ""
    
    # 🆕 Build clarification instruction when validation failed
    clarification_instruction = ""
    if needs_clarification:
        validation_error = state.get("validation_error")
        user_input = state.get("user_message", "")
        if validation_error:
            clarification_instruction = f"""
🚨 CLARIFICATION REQUIRED:
- User just typed: "{user_input}"
- Problem: {validation_error}
- Your job: Politely explain the issue and ask again in a friendly way

DO NOT just repeat the question. Acknowledge their input and explain what you need.

Example responses:
- "Hmm, 812 seems a bit short for a phone number! Could you share the full number with country code?"
- "I didn't quite catch that as a valid email format. Mind double-checking?"

Generate a friendly clarification that helps them understand what you need.
"""
            # Override acknowledgment with clarification
            acknowledgment_instruction = clarification_instruction
    
    # IMPROVED PROMPT: Simpler, more natural, with anti-hallucination anchoring
    system_prompt = f"""You're {persona['name']} - chatty, friendly, and helpful.

WHAT YOU NEED TO ASK: {current_key.upper()}
Standard phrasing: "{question_text}"
{f'Tone: {question_description}' if question_description else ''}
{f'{options_text}' if options_text else ''}

WHAT YOU KNOW: {known_user_summary}

MODE: {mode.value} - {mode_instruction}
{acknowledgment_instruction}
YOUR RESPONSE MUST:
1. Be {sentence_limit}
2. Actually ask for {current_key} (this is validated!)
3. Use contractions (what's, you're)
4. Sound like texting a friend, not filling a form

NEVER SAY:
- "To help me assist you"
- "Could you please share"
- "May I have your"
- "In order to"

EXAMPLES:
❌ "To help me assist you further, could you please share your phone number?"
✅ "What number works best for you?"

Generate ONLY your question - nothing else."""

    print(f"\n{'='*40}\nGENERATION PROMPT (key={current_key}):\n{system_prompt[:500]}...\n{'='*40}\n")

    from agent.llm_gateway import LLMGateway
    
    # Moderate temperature - more creative but still controlled
    llm = LLMGateway.get_llm(temperature=0.5)
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Generate the question now.")
        ]).content.strip()
    except Exception as e:
        print(f"Error generating question: {e}")
        response = fallback_question
    
    # =========================================================================
    # ANTI-HALLUCINATION: Validate, then Enhance with context
    # =========================================================================
    try:
        from Conversational_AI.utils.response_sanitizer import (
            sanitize_response,
            validate_question_response,
            get_random_acknowledgment,
        )
        
        # Step 1: Validate that response asks for correct field
        is_valid, validated_response = validate_question_response(response, current_key)
        
        if not is_valid:
            logger.warning(f"🚨 Response validation failed for {current_key}, using fallback")
            validated_response = fallback_question
        
        # Step 2: Sanitize (remove robotic phrases)
        sanitized_response = sanitize_response(validated_response)
        
        # Step 3: ENHANCE with context using second LLM
        try:
            from Conversational_AI.utils.response_enhancer import enhance_response, should_enhance
            
            
            # Get user's last message for context
            # Priority: state['user_message'] (actual input) > messages (form internal)
            user_last_message = state.get("user_message", None)
            
            if not user_last_message:
                # Fallback: parse from internal messages
                for msg in reversed(messages):
                    msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", None)
                    msg_content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                    if msg_type in ["human", "user"]:
                        user_last_message = msg_content
                        break
            
            if user_last_message:
                logger.info(f"📨 User's actual message: '{user_last_message[:50]}...'")
            
            # 🎯 GET ORIGINAL USER INQUIRY from state
            # This was stored when the form started (in langgraph_gateway.py)
            # It's what the user asked BEFORE the form (e.g., "chatbot", "Java", "Swift")
            original_user_inquiry = state.get("original_user_inquiry", None)
            
            # 🔴 FIX: Fallback - extract from conversation history if not in state
            if not original_user_inquiry:
                # Try to find the first substantive tech-related message from user
                tech_keywords = {
                    'python', 'java', 'react', 'node', 'ai', 'ml', 'blockchain', 
                    'mobile', 'app', 'web', 'development', 'software', 'it', 'tech',
                    'solution', 'service', 'project', 'build', 'create', 'need',
                    'chatbot', 'bot', 'api', 'cloud', 'devops', 'ios', 'android',
                    'swift', 'kotlin', 'flutter', 'angular', 'vue', 'django', 'flask'
                }
                form_keywords = {'email', 'phone', 'name', 'yes', 'no', 'sure', 'ok', 'thanks'}
                
                for msg in messages:
                    msg_type = msg.get("type") if isinstance(msg, dict) else getattr(msg, "type", None)
                    msg_content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                    
                    if msg_type in ["human", "user"] and msg_content:
                        msg_lower = msg_content.lower()
                        # Skip form answers (emails, phones, short responses)
                        if '@' in msg_content or msg_content.replace('+', '').replace('-', '').replace(' ', '').isdigit():
                            continue
                        if len(msg_content) < 4 or msg_lower in form_keywords:
                            continue
                        # Check if it contains tech keywords
                        if any(kw in msg_lower for kw in tech_keywords):
                            original_user_inquiry = msg_content
                            logger.info(f"📍 Extracted original inquiry from history: '{original_user_inquiry[:50]}...'")
                            break
            
            # 🔴 FIX: If still no original inquiry, use current user message if it's tech-related
            if not original_user_inquiry and user_last_message:
                msg_lower = user_last_message.lower()
                tech_indicators = ['solution', 'service', 'develop', 'build', 'create', 'need', 'help', 'project', 'it ', 'tech', 'software']
                if any(ind in msg_lower for ind in tech_indicators):
                    original_user_inquiry = user_last_message
                    logger.info(f"📍 Using current message as inquiry: '{original_user_inquiry[:50]}...'")
            
            if original_user_inquiry:
                logger.info(f"📍 Using original inquiry from state: '{original_user_inquiry[:50]}...'")
            else:
                logger.info("⚠️ No original inquiry found in state")
            
            # 🆕 Skip enhancement completely for clarification cases
            # When validation fails, the base question already contains the clarification message
            # Enhancing would override it with a generic response
            if needs_clarification:
                logger.info(f"🚫 Skipping enhancement: needs_clarification=True (validation error)")
                final_response = sanitized_response
            # Only enhance if we have context to reference
            elif should_enhance(current_key, len(collected)):
                # Get domain for RAG queries
                domain = state.get("domain", "tech_company")
                
                # 🏗️ UNIFIED CONTEXT ARCHITECTURE: Proactive RAG Query in Orchestrator
                # This fixes the "Split Brain" issue where enhancer used a stale isolated RAG instance
                rag_context = None
                query_text = user_last_message or original_user_inquiry
                
                if query_text:
                    try:
                        logger.info(f"🔄 [Orchestrator] Proactively querying RAG for: '{query_text[:50]}...'")
                        # Import from the shared API definition to use the SAME service singleton
                        # The path 'api.rag_routes' is available because main.py adds ML_models to sys.path
                        try:
                            from api.rag_routes import _get_query_service
                        except ImportError:
                            # Fallback if path setup is different (e.g. testing)
                            from Machine_Learning.ML_models.api.rag_routes import _get_query_service
                            
                        query_service = _get_query_service()
                        results = query_service.search(query=query_text, domain=domain, top_k=2)
                        
                        if results:
                            context_parts = []
                            for result in results:
                                intent = result.metadata.get("intent", "general") if result.metadata else "general"
                                content = result.content[:300]
                                context_parts.append(f"[{intent}] {content}")
                            rag_context = "\n".join(context_parts)
                            logger.info(f"✅ [Orchestrator] RAG Context injected: {len(rag_context)} chars")
                        else:
                            logger.info("ℹ️ [Orchestrator] No RAG results found")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ [Orchestrator] RAG query failed: {e}")

                enhanced_response = enhance_response(
                    base_question=sanitized_response,
                    persona=persona,
                    conversation_history=messages,
                    collected_data=collected,
                    current_key=current_key,
                    user_last_message=user_last_message,
                    rag_context=rag_context,  # 💉 INJECTED CONTEXT (Fixes Split Brain)
                    original_user_inquiry=original_user_inquiry,
                    off_topic_handled=off_topic_handled,  # 🆕 Pass off-topic flag for question answering
                )
                final_response = enhanced_response
                logger.info(f"✨ Enhanced response for {current_key}")
            else:
                # First question - just add simple acknowledgment
                if has_previous_answer and not sanitized_response.lower().startswith(('got', 'perfect', 'great', 'awesome', 'nice')):
                    ack = get_random_acknowledgment()
                    sanitized_response = f"{ack} {sanitized_response}"
                final_response = sanitized_response
                
        except ImportError as e:
            logger.warning(f"Response enhancer not available: {e}")
            # Fallback to simple acknowledgment
            if has_previous_answer and not sanitized_response.lower().startswith(('got', 'perfect', 'great', 'awesome', 'nice')):
                ack = get_random_acknowledgment()
                sanitized_response = f"{ack} {sanitized_response}"
            final_response = sanitized_response
        
    except ImportError as e:
        # Fallback if sanitizer not available
        logger.warning(f"Response sanitizer not available: {e}")
        final_response = response

    # 🔴 FIX: Prepend update acknowledgment if user updated a field
    update_ack = state.get("update_acknowledgment")
    if update_ack:
        final_response = f"{update_ack} {final_response}"
        # Clear the acknowledgment so it's not repeated
        state["update_acknowledgment"] = None

    return {
        "agent_message": final_response,
        "messages": [{"role": "assistant", "content": final_response}],
        "update_acknowledgment": None  # Clear in returned state
    }


def llm_processing_node(state: AgentState) -> Dict:
    """NLU Engine: Analyzes user input and returns structured intent via Text + Regex."""
    
    current_key = state["current_question_key"]
    current_question = state["question_set_data"][current_key]
    user_msg = state["user_message"]
    
    print(f"\n{'='*60}")
    print(f"🧠 NLU PROCESSING NODE CALLED")
    print(f"   Current Question: {current_key}")
    print(f"   User Message: '{user_msg[:80]}...' (len={len(user_msg)})" if len(user_msg) > 80 else f"   User Message: '{user_msg}'")
    print(f"{'='*60}")
    
    remaining_questions = state.get("questions_queue", [])
    
    # 🔴 DETERMINISTIC PRE-PARSER: Check for multi-field patterns BEFORE calling LLM
    # This catches patterns like "Name: John, email: john@test.com, company: Acme"
    multi_field_pattern = r'(?:name|email|phone|company|mobile|additional)[:\s]+([^,\n]+)'
    matches = re.findall(multi_field_pattern, user_msg, re.IGNORECASE)
    print(f"🔎 PRE-PARSER CHECK: Found {len(matches)} potential field-value pairs")
    
    if len(matches) >= 2:
        print(f"\n🔍 PRE-PARSER SCANNING: '{user_msg}'")
        extractions = []
        
        # Extract name - Improved regex to avoid matching "company name"
        # Look for Name at start of string OR preceded by comma/newline
        name_match = re.search(r'(?:^|,\s*|\n)name[:\s]+([^,\n@]+?)(?:,|email|phone|mobile|company|additional|$)', user_msg, re.IGNORECASE)
        if name_match:
            name_val = name_match.group(1).strip()
            print(f"   found name: '{name_val}'")
            if len(name_val) > 1 and not '@' in name_val and "company" not in name_val.lower():
                extractions.append({"key": "name", "value": name_val})
        
        # Extract email
        email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', user_msg)
        if email_match:
            email_val = email_match.group(1)
            print(f"   found email: '{email_val}'")
            extractions.append({"key": "email", "value": email_val})
        
        # Extract phone
        phone_match = re.search(r'(?:phone|mobile)[:\s]*(\+?\d[\d\s\-]{7,})', user_msg, re.IGNORECASE)
        if not phone_match:
            # Try alternative pattern for phone numbers (but be careful not to match dates/IDs)
            # Only match if 10+ digits
            phone_match = re.search(r'(?<!\d)(\d{10,})(?!\d)', user_msg)
        if phone_match:
            phone_val = re.sub(r'[^\d+]', '', phone_match.group(1))
            print(f"   found phone: '{phone_val}'")
            if len(phone_val) >= 7:
                extractions.append({"key": "phone", "value": phone_val})
        
        # Extract company
        company_match = re.search(r'company(?:\s*name)?[:\s]+([^,\n]+?)(?:,|phone|mobile|email|additional|$)', user_msg, re.IGNORECASE)
        if company_match:
            company_val = company_match.group(1).strip()
            print(f"   found company: '{company_val}'")
            if len(company_val) > 1:
                extractions.append({"key": "company_name", "value": company_val})
        
        # Extract additional message
        additional_match = re.search(r'additional(?:\s*message)?[:\s]+([^,\n]+?)(?:,|$)', user_msg, re.IGNORECASE)
        if additional_match:
            additional_val = additional_match.group(1).strip()
            print(f"   found additional: '{additional_val}'")
            if additional_val.lower() not in ['none', 'no', 'n/a', 'na']:
                extractions.append({"key": "additional_message", "value": additional_val})
        
        if len(extractions) >= 2:
            print(f"✅ PRE-PARSER: Detected {len(extractions)} fields - {extractions}")
            return {
                "nlu_data": {
                    "intent": "extract_multiple",
                    "extractions": extractions
                }
            }
        else:
             print(f"⚠️ PRE-PARSER: Only found {len(extractions)} valid fields, falling back to LLM")
    
    if isinstance(current_question.get("name"), dict):
        question_text = current_question.get("name", {}).get("name", "")
    else:
        question_text = current_question.get("name", "")
    
    from agent.llm_gateway import LLMGateway
    import os
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    llm = LLMGateway.get_llm(temperature=0.0)
    
    domain = state.get("domain", "default")
    
    # SECTION C: NLU prompt uses minimal context - only what's needed for parsing
    # NO retry counters, queue sizes, registry status, etc.
    # Get remaining questions for multi-field context - with options for robust mapping
    remaining_fields_info = []
    for rk in remaining_questions[:5]:
        rq = state["question_set_data"].get(rk, {})
        rf_str = rk
        if rq.get("type") == "options":
            opts = rq.get("options", [])
            rf_str += f" (options: {', '.join(opts)})"
        remaining_fields_info.append(rf_str)
    
    remaining_fields = ", ".join(remaining_fields_info) if remaining_fields_info else "none"

    # Get options for current question
    q_type = current_question.get("type", "text")
    options_info = ""
    if q_type == "options":
        options = current_question.get("options", [])
        options_info = f"- **Allowed Options for {current_key}**: {', '.join(options)}\n"
    
    system_prompt = f"""You are a strict NLU (Natural Language Understanding) engine.
Your job is to analyze the User's Input relative to the Current Question and output a structured command.

### CONTEXT
- **Domain**: {domain}
- **Current Question**: "{question_text}" (Key: {current_key})
{options_info}- **Remaining Fields**: {remaining_fields}

### OUTPUT FORMAT
You MUST output in this EXACT format (no markdown, no json, just text):

TOOL: [intent]
KEY: [key]
VALUE: [value]

For MULTIPLE fields in one message, use this format:
TOOL: extract_multiple
EXTRACTIONS:
- KEY: [key1], VALUE: [value1]
- KEY: [key2], VALUE: [value2]

### VALID INTENTS (TOOL)
1. **extract_answer**: User provided a valid answer FOR THE CURRENT QUESTION ONLY.
   - KEY: {current_key}
   - VALUE: The extracted answer.
2. **extract_multiple**: User provided MULTIPLE pieces of info (e.g., name AND email in one message).
   - Use EXTRACTIONS format with each KEY: value pair on its own line.
   - Only extract fields that are in the remaining fields list or current question.
3. **mark_question_declined**: User refused/declined (e.g. "no", "skip", "pass").
   - KEY: {current_key}
   - VALUE: declined
4. **update_response**: User wants to change PREVIOUS data (e.g. "change email to...").
   - KEY: The key to update (e.g. "email").
   - VALUE: The new value. **IF THE USER DID NOT PROVIDE THE NEW VALUE (just asked to change it), use "ASK_AGAIN".**
5. **request_clarification**: User input is unclear or ambiguous.
   - KEY: {current_key}
   - VALUE: reason_for_clarification
6. **handle_off_topic**: User asks a random question unrelated to the form.
   - KEY: {current_key}
   - VALUE: The user's question.
7. **cancel_form**: User explicitly wants to stop, cancel, or exit the form (e.g. "stop", "cancel", "I don't want to apply").
   - KEY: {current_key}
   - VALUE: cancel
8. **complete_all_remaining**: User wants to finish WITHOUT answering remaining questions (e.g. "I don't want to give any more details", "that's all", "finish up", "no more questions", "I'm done").
   - KEY: all
   - VALUE: complete

### EXAMPLES
User: "My email is john@doe.com"
TOOL: extract_answer
KEY: email
VALUE: john@doe.com

User: "My name is John and my email is john@example.com"
TOOL: extract_multiple
EXTRACTIONS:
- KEY: name, VALUE: John
- KEY: email, VALUE: john@example.com

User: "I'm Tushar, you can reach me at tushar@gmail.com or call me at 9876543210"
TOOL: extract_multiple
EXTRACTIONS:
- KEY: name, VALUE: Tushar
- KEY: email, VALUE: tushar@gmail.com
- KEY: phone, VALUE: 9876543210

User: "I don't want to say"
TOOL: mark_question_declined
KEY: phone
VALUE: declined

User: "Actually change my name to Bob"
TOOL: update_response
KEY: name
VALUE: Bob

User: "I gave you the wrong phone number, I want to change it"
TOOL: update_response
KEY: phone
VALUE: ASK_AGAIN

User: "What is your pricing?"
TOOL: handle_off_topic
KEY: {current_key}
VALUE: What is your pricing?

User: "I don't want to give any other detail"
TOOL: complete_all_remaining
KEY: all
VALUE: complete

User: "that's all from my side"
TOOL: complete_all_remaining
KEY: all
VALUE: complete

### USER INPUT
"{user_msg}"

### CRITICAL RULES:
1. **ALWAYS check if user provides MULTIPLE pieces of info** (name, email, phone, company, etc.)
2. If user message contains 2+ fields (e.g., "Name: John, email: john@test.com"), use **extract_multiple** - NOT extract_answer!
3. Only use extract_answer when user provides EXACTLY ONE piece of information.
4. Look for patterns like: "Name: X, email: Y" or "I'm X, my email is Y, company is Z"
5. **Categorical Mapping**: If **Allowed Options** are provided in the CONTEXT, you MUST map the user's answer to one of those exact English options. This applies even if the user answers in another language (e.g., Hindi 'छात्र' or 'पिता') or uses a synonym. Output the English option as the VALUE.

Analyze the input and use the correct intent.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg)
    ]
    
    print(f"\n> NLU Analysis for '{current_key}'...")
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        print(f"LLM Output:\n{content}\n")
        
        # Check for extract_multiple format first
        if "extract_multiple" in content.lower() and "EXTRACTIONS:" in content:
            extractions = []
            # Parse multi-line EXTRACTIONS format
            # Pattern: - KEY: name, VALUE: John
            extraction_pattern = r"-\s*KEY:\s*(\w+),?\s*VALUE:\s*(.+)"
            matches = re.findall(extraction_pattern, content, re.IGNORECASE)
            
            for key, value in matches:
                extractions.append({"key": key.strip(), "value": value.strip()})
            
            if extractions:
                print(f"✅ Parsed Multi-field NLU: {len(extractions)} extractions - {extractions}")
                return {
                    "nlu_data": {
                        "intent": "extract_multiple",
                        "extractions": extractions
                    }
                }
        
        # Standard single-field parsing
        pattern = r"TOOL:\s*(?P<intent>\w+)\s*KEY:\s*(?P<key>\w+)\s*VALUE:\s*(?P<value>.+)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            intent = match.group("intent")
            key = match.group("key")
            value = match.group("value").strip()
            
            print(f"✅ Parsed NLU: {{'intent': '{intent}', 'key': '{key}', 'value': '{value}'}}")
            
            return {
                "nlu_data": {
                    "intent": intent,
                    "key": key,
                    "value": value
                }
            }
        else:
            print(f"⚠️ Failed to parse NLU output: {content}")
            return {
                "nlu_data": {
                    "intent": "request_clarification",
                    "key": current_key,
                    "value": "failed_to_parse"
                }
            }
            
    except Exception as e:
        print(f"❌ NLU Error: {e}")
        return {
            "nlu_data": {
                "intent": "request_clarification",
                "key": current_key,
                "value": "error"
            }
        }


def update_resolution_node(state: AgentState) -> Dict:
    """Handle data update requests using the update_lead_data tool."""
    from agent.tools import update_lead_data
    
    nlu_data = state.get("nlu_data", {})
    key = nlu_data.get("key")
    value = nlu_data.get("value")
    
    collected = dict(state.get("collected_responses") or {})
    questions_queue = list(state.get("questions_queue") or [])
    questions_asked = list(state.get("questions_asked") or [])
    
    needs_clarification = False
    updated_queue = False
    
    print(f"\n> Update Resolution: Key='{key}', Value='{value}'")
    
    if key:
        if value == "ASK_AGAIN":
            print(f"🔄 User wants to change '{key}' but didn't provide value. Re-asking.")
            collected.pop(key, None)
            
            if key not in questions_queue:
                questions_queue.insert(0, key)
                updated_queue = True
            
            if key in questions_asked:
                questions_asked.remove(key)
                
        elif value:
            result = update_lead_data(key, value, state)
            
            if result["success"]:
                collected[key] = value
                print(f"✅ {result['message']}")
            else:
                print(f"❌ Update failed: {result['error']}")
                needs_clarification = True
    else:
        print("⚠️ Update requested but no key identified.")
        needs_clarification = True
        
    result = {
        "collected_responses": collected,
        "needs_clarification": needs_clarification,
    }
    
    if updated_queue:
        result["questions_queue"] = questions_queue
        result["questions_asked"] = questions_asked
        
    return result


def logic_routing_node(state: AgentState) -> Dict:
    """Execute business logic based on NLU intent."""
    
    nlu_data = state.get("nlu_data", {})
    intent = nlu_data.get("intent")
    key = nlu_data.get("key")
    value = nlu_data.get("value")
    
    # 🔴 STRICT FLOW ENFORCEMENT: Only allow answering the current question
    # "If user answers out-of-order -> ignore and re-ask current step"
    
    current_key = state["current_question_key"]
    normalized_current = _normalize_key(current_key)

    
    collected = dict(state.get("collected_responses") or {})
    previous_collected = collected.copy()
    declined = list(state.get("declined_questions") or [])
    questions_queue = list(state.get("questions_queue") or [])
    questions_asked = list(state.get("questions_asked") or [])
    
    newly_declined = []
    updated_queue = False
    needs_clarification = False
    off_topic_handled = False
    
    print(f"\n> Logic Routing: Intent='{intent}', Key='{key}', Value='{value}'")
    
    def validate_value(key, val, state):
        """
        Validate user input. Returns (is_valid, error_message).
        error_message is None if valid, or a human-friendly message if invalid.
        """
        if key in state["question_set_data"]:
            question_data = state["question_set_data"][key]
            q_type = question_data.get("type", "text")
            
            if q_type == "options":
                options = [opt.lower() for opt in question_data.get("options", [])]
                if val.lower() not in options:
                    options_str = ", ".join(question_data.get("options", []))
                    return False, f"Please choose one of these options: {options_str}"
            
            elif "email" in key.lower() or q_type == "email":
                # Strict email validation
                if not validate_email(val):
                    return False, "That doesn't look like a valid email address. Could you check and try again?"
            
            elif "phone" in key.lower() or "mobile" in key.lower() or q_type == "phone":
                # Strict phone validation using helper
                if not validate_phone(val):
                    return False, "That doesn't look like a valid phone number. Please enter a valid 10-15 digit number."
                
                phone_clean = re.sub(r"[^0-9+]", "", val)
                digits_only = re.sub(r"[^0-9]", "", phone_clean)
                
                # Additional checks
                if len(set(digits_only)) == 1:  # All same digit
                    return False, "That doesn't look like a real phone number. Could you share your actual number?"
                    
        return True, None

    # Handle Intents
    if intent == "extract_answer":
        if value:
            # Check if this value is a valid option (to prevent "No" being treated as decline if it's a valid choice)
            question_data = state["question_set_data"].get(key, {})
            options = [opt.lower() for opt in question_data.get("options", [])]
            is_valid_option = value.lower().strip() in options
            
            DECLINE_PHRASES = ["no", "nope", "nah", "nothing", "nothing else", "no thanks", "no thank you", "no, thanks", "no, thank you"]
            
            # Treat as decline ONLY if it's NOT a valid option for this question
            if (not is_valid_option) and (value.lower().strip() in DECLINE_PHRASES):
                print(f"ℹ️ Treating answer '{value}' for question '{key}' as an explicit decline.")
                if key and key not in declined:
                    # STRICT: Only allow declining optional fields?
                    # User rule: "Optional steps must allow 'skip' or empty response"
                    # "If required field invalid -> re-ask SAME step" - handled by validation
                    is_required = question_data.get("required", True)
                    
                    if not is_required:
                         newly_declined.append(key)
                         print(f"ℹ️ Marked optional '{key}' as declined")
                    else:
                         print(f"🚫 Cannot decline required field '{key}'. Re-answering.")
                         needs_clarification = True
                         state["validation_error"] = "This step is required. Please provide a valid answer."
            else:
                # STRICT: Ignore out-of-order answers
                if _normalize_key(key) != normalized_current and key != current_key:
                    print(f"🚫 STRICT FLOW: Ignoring answer for '{key}' while asking '{current_key}'")
                    # return without updating collected, effectively ignoring and re-asking
                    return {
                        "collected_responses": collected,
                        "needs_clarification": False, # Just re-ask nicely
                        "validation_error": None
                    }

                is_valid, validation_error_msg = validate_value(key, value, state)
                if is_valid:
                    # 🔴 FIX: Detect if this is an UPDATE (field already had a different value)
                    old_value = previous_collected.get(key)
                    if old_value and old_value != value:
                        # This is an update! Store acknowledgment for next response
                        field_label = key.replace("_", " ").title()
                        update_msg = f"Got it! I've updated your {field_label} from {old_value} to {value}."
                        state["update_acknowledgment"] = update_msg
                        print(f"📝 Update detected: {key} changed from '{old_value}' to '{value}'")
                    
                    collected[key] = value
                    
                    # 🔴 ENTRANCE EXAM LOGIC
                    # Check for "No" or declined logic for entrance exam
                    print(f"DEBUG: Processing '{key}'='{value}'. Queue: {questions_queue}")
                    if key == "entrance_exam" and value.lower() in ["no", "nope", "nah"]:
                        print("ℹ️ Entrance exam is NO, skipping exam details.")
                        for skipped_key in ["exam_name", "exam_score"]:
                            if skipped_key in questions_queue:
                                questions_queue.remove(skipped_key)
                                updated_queue = True
                            if skipped_key not in declined:
                                declined.append(skipped_key) # Mark as declined/skipped explicit

                else:
                    print(f"❌ Validation failed for '{key}': '{value}'")
                    needs_clarification = True
                    # 🆕 Store the validation error for clarification prompt
                    state["validation_error"] = validation_error_msg
    
    # 🆕 Handle multiple field extraction
    # STRICT MODE: Only accept the current key. Ignore others.
    elif intent == "extract_multiple":
        extractions = nlu_data.get("extractions", [])
        valid_count = 0
        for extraction in extractions:
            ext_key = extraction.get("key")
            ext_value = extraction.get("value")
            
            if ext_key and ext_value:
                # Normalization
                matched_key = None
                normalized_ext_key = _normalize_key(ext_key)
                
                # Find matching question key
                for q_key in state.get("question_set_data", {}).keys():
                     if _normalize_key(q_key) == normalized_ext_key:
                         matched_key = q_key
                         break
                
                # STRICT FLOW: Only process if it matches CURRENT key
                if matched_key and (_normalize_key(matched_key) == normalized_current):
                    is_valid, validation_error_msg = validate_value(matched_key, ext_value, state)
                    if is_valid:
                        collected[matched_key] = ext_value
                        # Queue update handled by progress tracker seeing collected presence
                        if matched_key in questions_queue:
                             questions_queue.remove(matched_key)
                             updated_queue = True
                        if matched_key not in questions_asked:
                             questions_asked.append(matched_key)
                        
                        valid_count += 1
                        print(f"✅ Strict Multi-extract: Accepted '{matched_key}' = '{ext_value}'")
                        
                        # Handle entrance exam logic here too if needed
                        if matched_key == "entrance_exam" and ext_value.lower() == "no":
                            print("ℹ️ Entrance exam is NO, skipping exam details.")
                            for skipped_key in ["exam_name", "exam_score"]:
                                if skipped_key in questions_queue:
                                    questions_queue.remove(skipped_key)
                                    updated_queue = True
                                if skipped_key not in declined:
                                    declined.append(skipped_key)
                    else:
                        print(f"❌ Strict Multi-extract validation failed for '{matched_key}': {validation_error_msg}")
                        needs_clarification = True
                        state["validation_error"] = validation_error_msg
                else:
                    print(f"🚫 STRICT FLOW: Ignoring '{ext_key}' (not current step '{current_key}')")
        
        if valid_count > 0:
            print(f"✅ Multi-field extraction: {valid_count} valid fields stored")
            updated_queue = True
                
    elif intent == "mark_question_declined":
        if key and key not in declined:
            print(f"🚫 Marked '{key}' as declined")
            newly_declined.append(key)
            
    elif intent == "update_response":
        if key:
            if value == "ASK_AGAIN":
                print(f"🔄 User wants to change '{key}' but didn't provide value. Re-asking.")
                collected.pop(key, None)
                
                if key not in questions_queue:
                    questions_queue.insert(0, key)
                    updated_queue = True
                
                if key in questions_asked:
                    questions_asked.remove(key)
                    
            elif value:
                is_valid, validation_error_msg = validate_value(key, value, state)
                if is_valid:
                    collected[key] = value
                    print(f"🔄 Updated '{key}' to '{value}'")
                else:
                    print(f"❌ Update Validation failed for '{key}': '{value}'")
                    needs_clarification = True
                    state["validation_error"] = validation_error_msg
            
    elif intent == "request_clarification":
        needs_clarification = True
        print(f"❓ Clarification needed for '{key}'")
        
    elif intent == "handle_off_topic":
        off_topic_handled = True
        print("🔀 Off-topic handled")
        
    elif intent == "cancel_form":
        print("⛔ User requested to CANCEL the form.")
        questions_queue = []  # Clear queue to stop asking
        updated_queue = True
        # We'll use a special flag in collected to indicate cancellation without schema change
        collected["_status"] = "cancelled"
    
    elif intent == "complete_all_remaining":
        # User wants to finish without providing more details
        print("✅ User requested to COMPLETE form without more details.")
        # Mark all remaining questions as declined
        for remaining_key in questions_queue:
            if remaining_key not in declined:
                declined.append(remaining_key)
        questions_queue = []  # Clear queue to complete
        updated_queue = True
        collected["_status"] = "completed_early"
        
    else:
        print(f"⚠️ Unknown intent '{intent}'")
        needs_clarification = True

    # Persistence & Resume Handling
    incremental_updates = {
        k: v for k, v in collected.items()
        if previous_collected.get(k) != v
    }
    
    lead_id = state.get("lead_id")

    # 🔴 STRICT RULE: Lead must be saved ONLY at the final step.
    # Disabled partial persistence.
    # UPDATE: Re-enabling partial persistence ONLY when essential contact info is captured.
    # This ensures we capture the lead as early as possible.
    if incremental_updates:
        # Trigger persistence if email or phone is present in this update
        should_persist = False
        for k in incremental_updates.keys():
            if k in ["email", "phone", "mobile"]:
                should_persist = True
                break
        
        if should_persist:
            print(f"💾 Triggering early lead persistence for contact info: {list(incremental_updates.keys())}")
            new_lead_id = _persist_partial_lead_data(state, incremental_updates)
            if new_lead_id:
                lead_id = new_lead_id
                # Update current state with the new lead_id
                state["lead_id"] = lead_id

    resume_path = None
    for k, v in incremental_updates.items():
        if "resume" in k.lower() and isinstance(v, str) and (v.startswith("http") or v.startswith("s3://")):
            resume_path = v
            break
            
    # Get validation error from state if it was set
    validation_error = state.get("validation_error")
    
    result = {
        "collected_responses": collected,
        "declined_questions": newly_declined,
        "needs_clarification": needs_clarification,
        "off_topic_handled": off_topic_handled,
        "lead_id": lead_id,
        "validation_error": validation_error,  # 🆕 Pass validation error to question generation
    }
    
    if updated_queue:
        result["questions_queue"] = questions_queue
        result["questions_asked"] = questions_asked
    
    if resume_path:
        result["resume_file_path"] = resume_path
        
    return result


# ============================================================================
# SECTION F: PROGRESS TRACKER WITH UX SAFETY
# ============================================================================
def progress_tracker_node(state: AgentState) -> Dict:
    """Track progress and move to next question - with clarification retry logic."""
    
    current_key = state["current_question_key"]
    questions_queue = list(state.get("questions_queue") or [])
    questions_asked = list(state.get("questions_asked") or [])
    needs_clarification = state.get("needs_clarification", False)
    clarification_retry_count = dict(state.get("clarification_retry_count") or {})
    collected = dict(state.get("collected_responses") or {})
    declined = list(state.get("declined_questions") or [])
    clear_llm_output = False
    
    # Deduplicate lists
    questions_asked = list(dict.fromkeys(questions_asked))
    declined = list(dict.fromkeys(declined))
    
    # Handle explicitly declined questions
    if current_key in declined:
        print(f"\n⚠️  Current question '{current_key}' was declined. Advancing flow.\n")
        if current_key in questions_queue:
            questions_queue.remove(current_key)
        if current_key not in questions_asked:
            questions_asked.append(current_key)
        clarification_retry_count.pop(current_key, None)
        needs_clarification = False
        clear_llm_output = True
        
    # Handle answered / clarification logic
    else:
        question_answered = current_key in collected
        
        if needs_clarification and not question_answered:
            if current_key not in clarification_retry_count:
                clarification_retry_count[current_key] = 0
            clarification_retry_count[current_key] += 1
            
            if clarification_retry_count[current_key] < 3:
                return {
                    "clarification_retry_count": clarification_retry_count,
                    "needs_clarification": True,
                }
            else:
                # SECTION F: User-visible skip message on retry exhaustion
                print(f"\n⚠️  Clarification retries exhausted for: {current_key}")
                print(f"   Marking as declined and moving to next question\n")
                
                if current_key not in declined:
                    declined.append(current_key)
                
                if current_key in questions_queue:
                    questions_queue.remove(current_key)
                if current_key not in questions_asked:
                    questions_asked.append(current_key)
                
                clarification_retry_count.pop(current_key, None)
                clear_llm_output = True
                
                # Return with mode set to trigger friendly skip message
                return {
                    "questions_queue": questions_queue,
                    "questions_asked": questions_asked,
                    "declined_questions": declined,
                    "clarification_retry_count": clarification_retry_count,
                    "needs_clarification": False,
                    "llm_result": None,
                    # SECTION F: Flag for human-friendly skip acknowledgment
                    "retry_exhausted_for": current_key,
                }
        elif question_answered:
            if current_key in questions_queue:
                questions_queue.remove(current_key)
            if current_key not in questions_asked:
                questions_asked.append(current_key)
            clarification_retry_count.pop(current_key, None)
            needs_clarification = False
            clear_llm_output = True

    # Determine Next Question
    # Determine Next Question - Case-insensitive filtering
    collected_keys_lower = {k.lower() for k in collected.keys()}
    declined_keys_lower = {k.lower() for k in declined}
    
    questions_queue = [
        q for q in questions_queue 
        if q.lower() not in collected_keys_lower 
        and q.lower() not in declined_keys_lower
    ]
    
    next_question_key = questions_queue[0] if questions_queue else None
        
    print(f"\n{'='*60}")
    print(f"PROGRESS TRACKER SUMMARY:")
    print(f"  Questions Queue: {questions_queue}")
    print(f"  Questions Asked: {questions_asked}")
    print(f"  Collected Responses: {list(collected.keys())}")
    print(f"  Declined Questions: {declined}")
    print(f"  Current Question: {current_key}")
    print(f"  Next Question: {next_question_key}")
    print(f"  ALL DONE? {not questions_queue}")
    print(f"{'='*60}\n")
    
    result = {
        "current_question_key": next_question_key,
        "questions_queue": questions_queue,
        "questions_asked": questions_asked,
        "all_questions_completed": not questions_queue,
        "needs_clarification": needs_clarification,
        "declined_questions": declined,
    }
    
    if clear_llm_output:
        result["llm_result"] = None
        
    return result


# ============================================================================
# SECTION B: COMPLETION NODE - SPLIT HANDLERS, SINGLE EXECUTION
# ============================================================================
def _handle_lead_completion(
    collected: Dict,
    session_id: str,
    question_set_title: str,
    domain: str,
    existing_lead_id: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Handle lead form completion persistence.
    
    Returns:
        (lead_id, error_message)
    """
    lead_id = existing_lead_id
    error = None
    
    if lead_id:
        try:
            lead_result = finalize_realtime_lead(
                lead_id=lead_id,
                responses=collected,
                session_id=session_id,
                question_set_title=question_set_title,
                domain=domain,
            )
            if lead_result.get("status") == "success":
                lead_id = lead_result.get("lead_id", lead_id)
            else:
                error = lead_result.get("error") or "Failed to store lead data."
                print(f"[lead-storage] Unable to save lead: {error}")
        except Exception as exc:
            error = str(exc)
            print(f"[lead-storage] Unable to finalize realtime lead {lead_id}: {error}")
    else:
        lead_result = create_lead_record_action(
            responses=collected,
            session_id=session_id,
            question_set_title=question_set_title,
            domain=domain,
        )
        if lead_result.get("status") == "success":
            lead_id = lead_result.get("lead_id")
        else:
            error = lead_result.get("error") or "Failed to store lead data."
            print(f"[lead-storage] Unable to save lead: {error}")
    
    return lead_id, error





def completion_node(state: AgentState) -> Dict:
    """Complete the conversation and save data.

    SECTION B FIX: Persistence executes exactly once via dedicated handlers.
    """
    collected = state.get("collected_responses", {}) or {}
    session_id = state.get("session_id", "")
    question_set_title = state.get("question_set_title", "lead_questions")
    domain = state.get("domain", "default")
    existing_lead_id = state.get("lead_id")

    if not collected:
        lead_save_error = "No responses collected for storage."
        print("[lead-storage] Skipping persistence: empty responses.")
    elif is_cancelled:
        print("[persistence] Form cancelled by user. Skipping save.")
    else:
        # SECTION B: Single execution path
        lead_id, lead_save_error = _handle_lead_completion(
            collected=collected,
            session_id=session_id,
            question_set_title=question_set_title,
            domain=domain,
            existing_lead_id=existing_lead_id
        )

    # Generate completion message using LLM
    from agent.llm_gateway import LLMGateway

    persona = get_persona_for_domain(domain)
    if not persona:
        persona = {
            "name": "Sara AI",
            "description": "A warm, friendly academic assistant.",
            "instructions": "You are Sara AI, a joyful and human-like academic assistant.",
        }

    user_msg = state.get("user_message", "")
    
    # SECTION C: Build clean context for completion - no raw state
    if is_cancelled:
        lead_status_context = (
            "The user explicitly CANCELLED the form (e.g. said 'stop' or 'cancel'). "
            "Acknowledge this politely without being pushy. "
            "Say something like 'No problem at all! I've stopped that process for you.' "
            "Then ask: 'Is there anything else I can help you with today?'"
        )

    else:
        if lead_id and not lead_save_error:
            lead_status_context = (
                f"The data has been saved successfully. ID: {lead_id}.\n"
                "MANDATORY INSTRUCTION: You MUST include this exact sentence in your response:\n"
                f"'Your ID is {lead_id}. You will receive an email from our side and our team will try to reach you soon!'"
            )
        else:
            lead_status_context = (
                "We could not save the data due to an internal issue. Apologize, let the user know "
                "the team will follow up manually, and avoid inventing an ID."
            )

    # Build user name for personalization
    user_name = collected.get("name", "")
    
    system_prompt = f"""You are {persona['name']}, a real, warm, emotionally intelligent human assistant.

The conversation is ending. Say goodbye warmly.

USER'S LAST MESSAGE: "{user_msg}"

WHAT YOU KNOW ABOUT THEM:
{f"Their name is {user_name}." if user_name else "You don't know their name yet."}

INSTRUCTIONS:
1. If the user said they want to stop/quit (e.g., "I am out", "stop", "cancel"), be understanding and polite. Say "No problem at all".
2. If the user completed the form normally, thank them enthusiastically and warmly.
3. Always mention that our team will get in touch with them very soon.
4. Use the user's name if we have it.
5. Keep it warm, human, generous, and friendly.
6. Express genuine appreciation for their time and trust in us.
7. {lead_status_context}
8. DO NOT say "If you ever change your mind or need anything in the future" - skip this phrase entirely.
9. For LEAD FORMS: End with "Is there anything you want to ask? We are always here to help you grow your business."

Generate ONLY the goodbye message. Be generous, warm, and make them feel valued.
"""

    # 🔴 STRICT: Use deterministic completion message for Lead Forms
    if lead_id and not lead_save_error:
        completion_msg = "Thanks! Your details have been submitted successfully."
    else:
        completion_msg = "Thanks! Your details have been submitted. (Warning: Save might have failed)"
        print(f"⚠️ Save failed: {lead_id} {lead_save_error}")

    result_state: Dict[str, Any] = {
        "agent_message": completion_msg,
        "messages": [{"role": "assistant", "content": completion_msg}],
        "lead_id": lead_id,
    }

    return result_state




