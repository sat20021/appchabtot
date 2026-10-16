"""
LangChain Expression Language Router for low-confidence intent handling.

When the classifier confidence is below the configured threshold we now let the LLM
craft the actual assistant response (or instruct the orchestrator to start a form).
This module builds a tiny LangGraph workflow with distinct nodes so the fallback flow
is easier to reason about and test.

Now enhanced with TOOL CALLING - the LLM can search MongoDB job listings dynamically
instead of relying on hardcoded patterns.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict
import logging

import json
import sys
from pathlib import Path
import importlib.util

# Import LLMGateway using absolute file path to avoid module path issues
llm_gateway_path = Path(__file__).parent.parent.parent.parent / "Conversational_AI" / "agent" / "llm_gateway.py"
spec = importlib.util.spec_from_file_location("llm_gateway", llm_gateway_path)
llm_gateway_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_gateway_module)
LLMGateway = llm_gateway_module.LLMGateway

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

# Module-level logger
logger = logging.getLogger(__name__)


# List of tools available to the router LLM
ROUTER_TOOLS = []

FORMAT_INSTRUCTIONS = (
    "Return valid JSON with the following fields:\\n"
    '{\\n'
    '  "action": "RESPOND_MESSAGE | START_LEAD_FORM | CLARIFY | HANDOFF_TO_HUMAN",\\n'
    '  "assistant_response": "Text reply when applicable (required for RESPOND_MESSAGE, CLARIFY, and HANDOFF_TO_HUMAN)",\\n'
    '  "question_set": "lead_questions | null",\\n'
    '  "reason": "Short explanation of your decision"\\n'
    '}}'
)


class LLMDecision(BaseModel):
    action: Literal[
        "RESPOND_MESSAGE", "START_LEAD_FORM", "CLARIFY", "HANDOFF_TO_HUMAN"
    ] = Field(
        ...,
        description="High-level action to take after reviewing the conversation history. "
        "Use HANDOFF_TO_HUMAN when user explicitly requests human agent or shows frustration. ",
    )
    assistant_response: Optional[str] = Field(
        None,
        description="If action is RESPOND_MESSAGE or CLARIFY, this contains the assistant reply.",
    )
    question_set: Optional[str] = Field(
        None,
        description="Name of the LangGraph question set (lead_questions, etc.).",
    )
    reason: str = Field(..., description="Short rationale for auditing/debugging.")


class LLMFallbackState(TypedDict, total=False):
    history: List[Dict[str, str]]
    user_message: str
    ml_context: Dict
    rag_context: str
    persona_instructions: str
    persona_conditions: Dict
    persona_formatting: Dict
    user_context: str  # User info from completed forms (name, email, etc.)
    decision: Dict
    # 🔴 NEW: Fields for validation
    session_data: Dict
    completed_forms: List[str]
    form_data: Dict
    active_flow: str  # 🔴 NEW: Current active form flow


def _format_history(turns: List[Dict[str, str]]) -> str:
    return "\\n".join(f"{t['role']}: {t['message']}" for t in turns) or "(no history)"


def _build_prompt():
    """
    Build a SIMPLIFIED prompt that prioritizes PERSONA instructions.
    
    Key changes from previous version:
    - Persona is now the PRIMARY guide (not buried in the middle)
    - RESPOND_MESSAGE is the default action for general questions
    - NO hardcoded persona names or strategies - all driven by persona config
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a routing decision engine for a College/University admissions agent. Return ONLY valid JSON.\\n\\n"
                    
                    "=== PERSONA (YOUR PRIMARY GUIDE) ===\\n"
                    "{persona_instructions}\\n\\n"
                    
                    "=== PERSONA CONDITIONS ===\\n"
                    "{persona_conditions}\\n\\n"
                    
                    "=== KNOWN USER INFO ===\\n"
                    "{user_context}\\n\\n"
                    
                    "=== COMPLETED FORMS ===\\n"
                    "{completed_forms_status}\\n\\n"
                    
                    "=== CURRENT FLOW ===\\n"
                    "{active_flow_status}\\n\\n"
                    
                    "=== AVAILABLE ACTIONS ===\\n"
                    "- RESPOND_MESSAGE: General conversation or info. DEFAULT action.\\n"
                    "- START_LEAD_FORM: Use when user expresses interest in ADMISSIONS, COURSES, or APPLYING. Set question_set='lead_questions'.\\n"
                    "- HANDOFF_TO_HUMAN: User asks for a human counselor or support person.\\n"
                    "- CLARIFY: Message is unintelligible.\\n\\n"
                    
                    "=== 🔴 ADMISSIONS ===\\n"
                    "User asks about courses, eligibility, software engineering, fees, or how to apply.\\n"
                    "   → If you have their interest (e.g. they said 'engineering') and you don't have their email/details, use START_LEAD_FORM.\\n"
                    "   → If you are just answering a question, use RESPOND_MESSAGE and ask a follow-up about their education.\\n\\n"
                    
                    "=== 🔴 CRITICAL: TRIGGERING THE FORM ===\\n"
                    "If the user provides an answer to an admissions question (like their interest in a program), \\n"
                    "ALWAYS try to move toward START_LEAD_FORM to capture their details officially.\\n\\n"
                    
                    "=== RAG CONTEXT ===\\n"
                    "{rag_context}\\n\\n"
                    
                    "{format_instructions}\\n"
                ),
            ),
            (
                "human",
                "History: {history}\\n"
                "User message: {user_message}\\n"
                "ML context: {ml_context}\\n\\n"
                "Return JSON."
            ),
        ]
    )
    return prompt.partial(format_instructions=FORMAT_INSTRUCTIONS)


_PROMPT = _build_prompt()
_MODEL = None
_MODEL_WITH_TOOLS = None
_CHAIN = None


def _get_model():
    """
    Lazy-load LLM model for performance optimization.
    Includes fallback mechanism for failed models.
    
    Returns:
        Initialized LLM model (without tool binding)
    """
    global _MODEL
    if _MODEL is None:
        import os
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        # Temperature 0.4: More natural than 0.2, but still controlled
        temperature = 0.8 if provider == "openai" else 0.4
        try:
            _MODEL = LLMGateway.get_llm(temperature=temperature)
        except Exception as e:
            logger.error(f"Failed to initialize primary LLM: {e}")
            # Try fallback model
            if LLMGateway.is_retryable_error(e):
                _MODEL = LLMGateway.get_fallback_llm(temperature=temperature)
            if _MODEL is None:
                raise
    return _MODEL


def _get_model_with_tools():
    """
    Get LLM model with tools bound for function calling.
    
    Returns:
        LLM model with search_available_jobs tool bound
    """
    global _MODEL_WITH_TOOLS
    if _MODEL_WITH_TOOLS is None:
        model = _get_model()
        _MODEL_WITH_TOOLS = model.bind_tools(ROUTER_TOOLS)
        logger.info("🔧 LLM Router initialized with tools: %s", [t.name for t in ROUTER_TOOLS])
    return _MODEL_WITH_TOOLS


def _get_chain():
    """
    Lazy-load LLM chain for performance optimization.
    
    The LLM model is only initialized when first needed, reducing startup time.
    This is important since LLM is only used as a fallback (FAQ and ML handle most cases).
    
    Returns:
        Initialized LangChain pipeline for LLM routing (prompt → model)
    """
    global _CHAIN
    if _CHAIN is None:
        # Use model without tool binding for the chain - tool handling is done in decision_node
        _CHAIN = _PROMPT | _get_model() | RunnableLambda(
            lambda msg: getattr(msg, "content", str(msg))
        )
    return _CHAIN


def _prepare_node(state: LLMFallbackState) -> LLMFallbackState:
    state.setdefault("history", [])
    state.setdefault("ml_context", {})
    state.setdefault("rag_context", "")
    state.setdefault("persona_instructions", "You are a helpful assistant.")
    state.setdefault("persona_conditions", {})
    state.setdefault("persona_formatting", {})
    state.setdefault("user_context", "No user information available yet.")
    # 🔴 NEW: Defaults for validation
    state.setdefault("session_data", {})
    state.setdefault("completed_forms", [])
    state.setdefault("form_data", {})
    state.setdefault("active_flow", "")  # 🔴 NEW: Current active flow
    return state


def _decision_node(state: LLMFallbackState) -> LLMFallbackState:
    """
    The actual decision-making node that calls the LLM.
    """
    chain = _get_chain()
    
    # 🔴 Build completed_forms_status for LLM context
    completed_forms = state.get("completed_forms", [])
    if "lead_questions" in completed_forms:
        completed_forms_status = (
            "🔴 CRITICAL: Lead form is ALREADY COMPLETED!\n"
            "The user has already provided their contact details.\n"
            "DO NOT use START_LEAD_FORM - they don't need to fill it again!\n"
            "Instead, acknowledge their previous submission and help with their query."
        )
    else:
        completed_forms_status = "No forms completed yet - you may start a form if appropriate."
    
    # 🔴 Build active_flow_status for LLM context
    active_flow = state.get("active_flow", "")
    if active_flow:
        active_flow_status = f"User is currently in form: {active_flow} - DO NOT switch to a different form!"
    else:
        active_flow_status = "No active form - user is in free conversation"
    
    # 🔴 Format persona_formatting dict into readable guidelines
    persona_formatting_dict = state.get("persona_formatting", {})
    if persona_formatting_dict and isinstance(persona_formatting_dict, dict):
        guidelines = persona_formatting_dict.get("guidelines", [])
        if guidelines:
            persona_formatting_text = "When crafting assistant_response, use rich markdown formatting:\n" + "\n".join([f"- {g}" for g in guidelines])
        else:
            persona_formatting_text = "Use clear markdown formatting in assistant_response."
    else:
        persona_formatting_text = "Use clear markdown formatting in assistant_response."
    
    raw_decision_msg = chain.invoke(
        {
            "history": _format_history(state.get("history", [])),
            "user_message": state["user_message"],
            "ml_context": state.get("ml_context", {}),
            "rag_context": state.get("rag_context", ""),
            "persona_instructions": state.get("persona_instructions", ""),
            "persona_conditions": json.dumps(state.get("persona_conditions", {}), indent=2),
            "persona_formatting": persona_formatting_text,  # ← Now formatted as readable text
            "user_context": state.get("user_context", "No user information available yet."),
            "completed_forms_status": completed_forms_status,  # 🔴 NEW: Inform LLM about completed forms
            "active_flow_status": active_flow_status,  # 🔴 NEW
        }
    )
    
    # Ensure we get string content from the message object
    raw_decision = getattr(raw_decision_msg, "content", str(raw_decision_msg))

    # Strip markdown code blocks if present (Gemini sometimes wraps JSON in ```json ... ```)
    raw_str = str(raw_decision).strip()
    if raw_str.startswith("```json"):
        raw_str = raw_str[7:]  # Remove ```json
    elif raw_str.startswith("```"):
        raw_str = raw_str[3:]  # Remove ```
    if raw_str.endswith("```"):
        raw_str = raw_str[:-3]  # Remove trailing ```
    raw_str = raw_str.strip()
    
    logger.info(f"🔍 LLM raw response: {raw_str}")
    
    try:
        decision_dict = json.loads(raw_str)
        
        # Validate and reject RESPOND_GREETING if present
        action = decision_dict.get("action", "").upper()
        if action == "RESPOND_GREETING":
            logger.warning(
                "LLM returned RESPOND_GREETING which is not allowed. "
                "Converting to RESPOND_MESSAGE. Original response: %s",
                raw_str[:200]
            )
            decision_dict["action"] = "RESPOND_MESSAGE"
            if not decision_dict.get("assistant_response"):
                decision_dict["assistant_response"] = "Hello! How can I help you today?"
            decision_dict["reason"] = (
                decision_dict.get("reason", "") + 
                " [Note: RESPOND_GREETING was converted to RESPOND_MESSAGE - greetings are handled by ML]"
            )
            action = "RESPOND_MESSAGE"
        
        # 🔴 VALIDATION GUARD: Prevent restarting completed lead form
        completed_forms = state.get("completed_forms", [])
        if action == "START_LEAD_FORM" and "lead_questions" in completed_forms:
            logger.warning(
                "🚫 GUARD: LLM tried to restart lead form but it's already completed. "
                "Converting to RESPOND_MESSAGE."
            )
            # Override action to acknowledge they already submitted
            action = "RESPOND_MESSAGE"
            decision_dict["action"] = "RESPOND_MESSAGE"
            decision_dict["assistant_response"] = (
                "Thank you for your interest! I see you've already provided your information. "
                "Our team will be in touch with you shortly. Is there anything else I can help you with?"
            )
            decision_dict["reason"] = (
                f"[OVERRIDDEN] Lead form already completed - prevented restart. "
                f"Original: {decision_dict.get('reason', '')}"
            )

        # =====================================================================
        # TOOL EXECUTION: Removed.
        # =====================================================================

        # Validate with Pydantic model (allows extra fields)
        decision_obj = LLMDecision(**decision_dict)
        state["decision"] = decision_obj.dict()
        
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(
            f"Failed to parse LLM response as JSON or validate with Pydantic: {e}. Raw response (first 200 chars): {raw_str[:200]}"
        )
        # Fallback decision
        state["decision"] = {
            "action": "CLARIFY",
            "assistant_response": "I'm not entirely sure I understood that. Could you clarify?",
            "question_set": None,
            "reason": f"Unable to parse router response: {str(e)}",
        }
    return state


def _finalize_node(state: LLMFallbackState) -> LLMFallbackState:
    if "decision" not in state:
        state["decision"] = {
            "action": "CLARIFY",
            "assistant_response": "I'm not entirely sure I understood that. Could you clarify?",
            "reason": "LLM fallback failed to respond",
        }
    return state


_WORKFLOW = StateGraph(LLMFallbackState)
_WORKFLOW.add_node("prepare", _prepare_node)
_WORKFLOW.add_node("decide", _decision_node)
_WORKFLOW.add_node("finalize", _finalize_node)
_WORKFLOW.set_entry_point("prepare")
_WORKFLOW.add_edge("prepare", "decide")
_WORKFLOW.add_edge("decide", "finalize")
_WORKFLOW.add_edge("finalize", END)
_LLM_GRAPH = _WORKFLOW.compile()


def route_with_llm(
    last_three_turns, 
    user_message: str, 
    ml_context: Dict,
    rag_context: str = "",
    persona_instructions: str = "",
    persona_conditions: Dict = {},
    persona_formatting: Dict = {},
    form_data: Dict = None,
    completed_forms: list = None,
    session_data: Dict = None,
    active_flow: str = ""  # 🔴 NEW: Pass current active flow
) -> Dict:
    """
    Execute the LLM fallback workflow and return the structured decision dict.
    
    Args:
        form_data: User information collected from forms (name, email, etc.)
        completed_forms: List of completed form names for this session
        session_data: Full session data from Redis
        active_flow: Current active form flow (e.g., 'lead_questions')
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Format form_data into readable context
    form_data = form_data or {}
    if form_data:
        user_context_parts = []
        for key, value in form_data.items():
            formatted_key = key.replace("_", " ").title()
            user_context_parts.append(f"- {formatted_key}: {value}")
        user_context = "\\n".join(user_context_parts)
        logger.info(f"📋 USER CONTEXT FOR LLM: {user_context}")
    else:
        user_context = "No user information available yet."
        logger.info("⚠️ No form_data available - user_context is empty")
    
    # 🔴 Log active_flow for debugging
    if active_flow:
        logger.info(f"📍 ACTIVE FLOW FOR LLM: {active_flow}")
    
    result_state = _LLM_GRAPH.invoke(
        {
            "history": last_three_turns,
            "user_message": user_message,
            "ml_context": ml_context,
            "rag_context": rag_context,
            "persona_instructions": persona_instructions,
            "persona_conditions": persona_conditions,
            "persona_formatting": persona_formatting,
            "user_context": user_context,
            "session_data": session_data or {},
            "completed_forms": completed_forms or [],
            "form_data": form_data or {},
            "active_flow": active_flow or "",  # 🔴 NEW
        }
    )
    return result_state.get("decision", {})

