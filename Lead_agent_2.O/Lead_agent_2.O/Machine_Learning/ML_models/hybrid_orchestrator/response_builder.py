"""
Centralized response building for enterprise AI agent.

This module provides consistent response construction with routing metadata,
ensuring all responses follow the same structure and include performance metrics.
"""
import logging
from typing import Dict, List, Optional, Any

from models.hybrid_schemas import (
    ConversationResponse,
    RoutingMetadata,
    RoutingStepMetadata,
    ContextState,
    DecisionResult,
)
from ml_services.greeting_service import generate_greeting_response
from ml_services.langgraph_gateway import start_form_session, end_form_session
from hybrid_orchestrator.session_manager import append_turn, upsert_session, append_conversation, get_session
from hybrid_orchestrator.cta_injector import inject_ctas_if_appropriate

logger = logging.getLogger(__name__)


def build_routing_metadata(
    faq_result: Optional[DecisionResult] = None,
    ml_result: Optional[DecisionResult] = None,
    llm_result: Optional[DecisionResult] = None,
    selected_route: str = "",
    route_reason: str = "",
    rag_metadata: Optional[Dict] = None,
) -> RoutingMetadata:
    """
    Build comprehensive routing metadata from decision results.
    
    Args:
        faq_result: FAQ decision result (if attempted)
        ml_result: ML decision result (if attempted)
        llm_result: LLM decision result (if attempted)
        selected_route: The route that was ultimately selected
        route_reason: Explanation of why this route was selected
        rag_metadata: RAG search metadata (if attempted)
    
    Returns:
        RoutingMetadata with complete routing decision tree
    """
    steps: List[RoutingStepMetadata] = []
    timing_dict: Dict[str, float] = {}
    
    # Add FAQ step if attempted
    if faq_result:
        steps.append(RoutingStepMetadata(
            route_type="faq",
            attempted=True,
            confidence=faq_result.confidence,
            selected=faq_result.selected,
            reason=faq_result.reason or (
                f"FAQ confidence {faq_result.confidence:.3f} >= 0.8" if faq_result.selected
                else f"FAQ confidence {faq_result.confidence:.3f} < 0.8"
            ),
            timing_ms=faq_result.timing_ms,
        ))
        if faq_result.timing_ms:
            timing_dict["faq_ms"] = faq_result.timing_ms
    
    # Add ML step if attempted
    if ml_result:
        steps.append(RoutingStepMetadata(
            route_type="ml",
            attempted=True,
            confidence=ml_result.confidence,
            selected=ml_result.selected,
            reason=ml_result.reason or (
                f"ML intent '{ml_result.intent}' with confidence {ml_result.confidence:.3f} >= 0.8"
                if ml_result.selected
                else f"ML confidence {ml_result.confidence:.3f} < 0.8"
            ),
            timing_ms=ml_result.timing_ms,
        ))
        if ml_result.timing_ms:
            timing_dict["ml_ms"] = ml_result.timing_ms
    
    # Add LLM step if attempted
    if llm_result:
        steps.append(RoutingStepMetadata(
            route_type="llm",
            attempted=True,
            confidence=None,
            selected=llm_result.selected,
            reason=llm_result.reason or f"LLM action: {llm_result.response_data.get('action') if llm_result.response_data else 'unknown'}",
            timing_ms=llm_result.timing_ms,
        ))
        if llm_result.timing_ms:
            timing_dict["llm_ms"] = llm_result.timing_ms
    
    # Extract RAG metadata
    rag_attempted = False
    rag_results_count = None
    rag_domain = None
    rag_success = None
    
    if rag_metadata:
        rag_attempted = rag_metadata.get("attempted", False)
        rag_results_count = rag_metadata.get("results_count")
        rag_domain = rag_metadata.get("domain")
        rag_success = rag_metadata.get("success")
    
    return RoutingMetadata(
        routing_steps=steps,
        selected_route=selected_route,
        route_reason=route_reason,
        faq_attempted=faq_result is not None,
        faq_confidence=faq_result.confidence if faq_result else None,
        ml_attempted=ml_result is not None,
        ml_intent=ml_result.intent if ml_result else None,
        ml_confidence=ml_result.confidence if ml_result else None,
        llm_attempted=llm_result is not None,
        llm_action=llm_result.response_data.get("action") if (llm_result and llm_result.response_data) else None,
        llm_reason=llm_result.reason if llm_result else None,
        rag_attempted=rag_attempted,
        rag_results_count=rag_results_count,
        rag_domain=rag_domain,
        rag_success=rag_success,
        timing_ms=timing_dict if timing_dict else None,
    )



def build_faq_response(
    context: ContextState,
    faq_result: DecisionResult,
    routing_metadata: RoutingMetadata
) -> ConversationResponse:
    """
    Build FAQ response with metadata.
    
    Args:
        context: Conversation context
        faq_result: FAQ decision result
        routing_metadata: Complete routing metadata
    
    Returns:
        ConversationResponse for FAQ route
    """
    response_data = faq_result.response_data or {}
    answer = response_data.get("answer", "")
    
    # Append to conversation history
    append_turn(context.session_id, "assistant", answer)
    
    # Save to MongoDB history format
    domain = context.metadata.get("domain", "domain_3") if context.metadata else "domain_3"
    append_conversation(context.session_id, context.message, answer, domain)
    
    # Inject CTAs for FAQ responses
    cta_options = inject_ctas_if_appropriate(
        route="faq_semantic",
        active_flow=context.active_flow,
    )
    
    return ConversationResponse(
        session_id=context.session_id,
        response=answer,
        route="faq_semantic",
        intent=None,
        confidence=faq_result.confidence,
        matched_question=response_data.get("matched_question"),
        is_complete=True,
        routing_metadata=routing_metadata,
        options=cta_options,  # CTAs go in options field for frontend rendering
    )


def build_ml_greeting_response(
    context: ContextState,
    ml_result: DecisionResult,
    routing_metadata: RoutingMetadata
) -> ConversationResponse:
    """
    Build ML greeting response.
    
    Args:
        context: Conversation context
        ml_result: ML decision result
        routing_metadata: Complete routing metadata
    
    Returns:
        ConversationResponse for greeting route
    """
    from ml_services.greeting_service import generate_greeting_response, select_greeting
    
    domain = context.metadata.get("domain") or "tech_company"
    user_name = context.metadata.get("user_name")
    
    if ml_result.intent == "greeting" and ml_result.confidence == 1.0:
        # Pre-verified direct greeting bypasses the classifier completely
        try:
            response_message = select_greeting(domain, user_name)
        except Exception as e:
            logger.error(f"Error selecting greeting: {e}")
            response_message = "Hello! How can I help you today?"
            
        intent = "greeting"
        confidence = 1.0
    else:
        # Fallback to generating greeting via classifier
        result = generate_greeting_response(
            context.message,
            domain=domain,
            user_name=user_name,
        )
        
        # Ensure we have a valid response message
        response_message = (
            result.get("response_message")
            or result.get("greeting_message")
            or result.get("gratitude_message")
            or "Hello! How can I help you today?"
        )
        
        if not result.get("response_message"):
            logger.warning(
                "Greeting service returned None response_message for query '%s'. "
                "Intent: %s, Confidence: %s. Using fallback.",
                context.message,
                result.get("intent"),
                result.get("confidence"),
            )
        
        intent = result.get("intent")
        confidence = result.get("confidence")
    
    append_turn(context.session_id, "assistant", response_message)
    
    # Save to MongoDB history format
    domain_db = context.metadata.get("domain", "domain_3") if context.metadata else "domain_3"
    append_conversation(context.session_id, context.message, response_message, domain_db)
    
    # Inject CTAs for greeting responses
    cta_options = inject_ctas_if_appropriate(
        route="greeting_ml",
        active_flow=context.active_flow,
    )
    
    return ConversationResponse(
        session_id=context.session_id,
        response=response_message,
        route="greeting_ml",
        intent=intent,
        confidence=confidence,
        is_complete=True,
        routing_metadata=routing_metadata,
        options=cta_options,  # CTAs go in options field for frontend rendering
    )


def build_ml_form_response(
    context: ContextState,
    ml_result: DecisionResult,
    routing_metadata: RoutingMetadata,
    question_set: str
) -> ConversationResponse:
    """
    Build ML form response (start form flow).
    
    Args:
        context: Conversation context
        ml_result: ML decision result
        routing_metadata: Complete routing metadata
        question_set: The question set to start
    
    Returns:
        ConversationResponse for form route
    """
    # Extract domain for persona and RAG context
    domain = "default"
    if context.metadata:
        domain = context.metadata.get("domain") or context.metadata.get("extra", {}).get("domain", "default")
    
    # Use the current message as context for form initialization
    # This is the message that triggered the form (e.g., "Tushar" or "My name is John")
    previous_message = context.message
    
    # 🔴 CRITICAL: End any existing form session before starting a new one
    # This prevents "zombie" forms from staying active and intercepting messages
    if context.active_flow:
        logger.info(
            "🔄 Ending previous form '%s' before starting new form '%s' for session %s",
            context.active_flow, question_set, context.session_id
        )
        end_form_session(context.session_id)
    
    # Start form session with domain and previous context
    lang_result = start_form_session(
        question_set, 
        session_id=context.session_id,
        domain=domain,
        previous_message=previous_message
    )
    
    # Update session with form state
    upsert_session(
        context.session_id,
        {
            "active_flow": question_set,
            "graph_state_id": lang_result["session_id"],
        },
    )
    
    append_turn(context.session_id, "assistant", lang_result["agent_message"])
    
    # Save to MongoDB history format
    domain = context.metadata.get("domain", "domain_3") if context.metadata else "domain_3"
    append_conversation(context.session_id, context.message, lang_result["agent_message"], domain)
    
    # Extract question options and type if available
    options = None
    question_type = None
    state = lang_result.get("state") or {}
    current_key = state.get("current_question_key")
    if current_key:
        question_data = (state.get("question_set_data") or {}).get(current_key, {})
        question_type = question_data.get("type")  # text, options, or file
        if question_type == "options":
            options = question_data.get("options", [])
    
    return ConversationResponse(
        session_id=context.session_id,
        response=lang_result["agent_message"],
        route="form_ml",
        intent=ml_result.intent,
        confidence=ml_result.confidence,
        next_question=lang_result["agent_message"],
        options=options,
        question_type=question_type,  # Pass question type to frontend
        is_complete=lang_result["all_questions_completed"],
        collected_data=lang_result.get("collected_data", {}),
        # 🆕 Populate requested fields
        current_step=current_key,
        lead_data=lang_result.get("collected_data", {}),
        routing_metadata=routing_metadata,
    )




def build_llm_response(
    context: ContextState,
    llm_result: DecisionResult,
    ml_intent: Optional[str],
    ml_confidence: Optional[float],
    routing_metadata: RoutingMetadata,
    rag_chunks: Optional[List[Dict[str, Any]]] = None,
) -> ConversationResponse:
    """
    Build LLM response based on LLM decision.
    
    Args:
        context: Conversation context
        llm_result: LLM decision result
        ml_intent: ML intent (for context)
        ml_confidence: ML confidence (for context)
        routing_metadata: Complete routing metadata
        rag_chunks: RAG chunks retrieved for this response
    
    Returns:
        ConversationResponse for LLM route
    """
    response_data = llm_result.response_data or {}
    action = response_data.get("action", "CLARIFY")
    
    # 🟢 Handle HANDOFF_TO_HUMAN action - trigger human agent escalation
    if action == "HANDOFF_TO_HUMAN":
        # Get the assistant's response for the handoff request
        handoff_message = response_data.get("assistant_response") or \
            "I understand you'd like to speak with a human agent. Would you like me to connect you with one of our support representatives?"
        
        append_turn(context.session_id, "assistant", handoff_message)
        
        # Save to MongoDB history format
        domain = context.metadata.get("domain", "domain_3") if context.metadata else "domain_3"
        append_conversation(context.session_id, context.message, handoff_message, domain)
        
        routing_metadata.selected_route = "handoff_request"
        routing_metadata.route_reason = f"LLM initiated human handoff: {llm_result.reason}"
        
        logger.info(
            f"👤 HANDOFF requested for session {context.session_id}: {llm_result.reason}"
        )
        
        return ConversationResponse(
            session_id=context.session_id,
            response=handoff_message,
            route="handoff_request",
            intent="human_handoff",
            confidence=ml_confidence,
            is_complete=False,
            routing_metadata=routing_metadata,
            # 🟢 Signal widget to show handoff confirmation
            handoff_pending=True,
            handoff_mode=None,
            websocket_url=None,
        )
    

    
    # Handle form start actions
        question_set = (
            response_data.get("question_set")
            or "lead_questions"
        )
        
        # Extract domain for persona and RAG context
        domain = "default"
        if context.metadata:
            domain = context.metadata.get("domain") or context.metadata.get("extra", {}).get("domain", "default")
        
        # 🎯 Get user's FIRST message from session as original inquiry
        # This is their tech question like "Python", "Java", "IT services" etc.
        original_user_inquiry =  None
        try:
            session = get_session(context.session_id)
            if session and "conversation" in session:
                conversation = session["conversation"]
                # Find first substantial user message (not email, phone, or short answers)
                form_keywords = {'email', 'phone', 'number', 'name', 'yes', 'no', 'sure', 'ok'}
                for turn in conversation:
                    if turn.get("role") == "user":
                        msg = turn.get("message", "")
                        # Skip if it's an email, phone, or very short form answer
                        if '@' not in msg and not msg.isdigit() and len(msg) > 3:
                            msg_lower = msg.lower()
                            # Skip if it's a simple form answer
                            if not all(keyword in msg_lower for keyword in form_keywords):
                                original_user_inquiry = msg
                                logger.info(f"💾 Extracted original inquiry from session: '{original_user_inquiry[:50]}...'")
                                break
        except Exception as e:
            logger.warning(f"Could not extract original inquiry from session: {e}")
        
        # Use the current message as context for form initialization
        # This is the message that triggered the form (e.g., user's email)
        previous_message = original_user_inquiry or context.message
        
        # 🔴 CRITICAL: End any existing form session before starting a new one
        # This prevents "zombie" forms from staying active
        if context.active_flow:
            logger.info(
                "🔄 [LLM] Ending previous form '%s' before starting new form '%s' for session %s",
                context.active_flow, question_set, context.session_id
            )
            end_form_session(context.session_id)
        
        lang_result = start_form_session(
            question_set, 
            session_id=context.session_id,
            domain=domain,
            previous_message=previous_message
        )
        upsert_session(
            context.session_id,
            {
                "active_flow": question_set,
                "graph_state_id": lang_result["session_id"],
            },
        )
        append_turn(context.session_id, "assistant", lang_result["agent_message"])
        
        # Save to MongoDB history format
        domain = context.metadata.get("domain", "domain_3") if context.metadata else "domain_3"
        append_conversation(context.session_id, context.message, lang_result["agent_message"], domain)
        
        # Extract options and question_type
        options = None
        question_type = None
        state = lang_result.get("state") or {}
        current_key = state.get("current_question_key")
        if current_key:
            question_data = (state.get("question_set_data") or {}).get(current_key, {})
            question_type = question_data.get("type")  # text, options, or file
            if question_type == "options":
                options = question_data.get("options", [])
        
        routing_metadata.selected_route = "form_llm"
        routing_metadata.route_reason = f"LLM directed to start {question_set} form: {llm_result.reason}"
        
        return ConversationResponse(
            session_id=context.session_id,
            response=lang_result["agent_message"],
            route="form_llm",
            intent=question_set,
            confidence=ml_confidence,
            next_question=lang_result["agent_message"],
            options=options,
            question_type=question_type,  # Pass question type to frontend
            is_complete=lang_result["all_questions_completed"],
            collected_data=lang_result.get("collected_data", {}),
            routing_metadata=routing_metadata,
        )
    
    # Handle direct response or clarify
    # Different providers may return responses in slightly different formats
    import os
    import re
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    assistant_reply = response_data.get("assistant_response")
    
    # Provider-specific handling if needed
    if not assistant_reply:
        if action == "CLARIFY":
            assistant_reply = "I want to make sure I understand. Could you clarify that for me?"
        else:
            assistant_reply = "Here's how I can help with that."
        logger.warning(
            f"No assistant_response in LLM response_data for session {context.session_id}. "
            f"Provider: {provider}, Action: {action}. Using fallback message."
        )
    else:
        # 🔴 ENTERPRISE FIX: Sanitize LLM response to remove data requests when we already have info
        completed_forms = context.session.get("completed_forms", [])
        form_data = context.form_data or {}
        
        if completed_forms or form_data:
            # Patterns that indicate LLM is asking for info we already have
            INFO_REQUEST_PATTERNS = [
                r"could you( please)? share your email",
                r"may i have your email",
                r"can i get your email",
                r"what('s| is) your email",
                r"share your email",
                r"provide your email",
                r"share your phone",
                r"provide your phone",
                r"what('s| is) your phone",
                r"could you share your (contact|phone|email)",
                r"so i can reach out",
                r"so i can( get in)? touch",
                # Additional patterns for email requests
                r"share your email again",
                r"keep everything organized",
                r"confirm your email",
                r"verify your email",
                r"your email( address)? (to|for)",
                r"drop your email",
                r"send me your email",
                r"need your email",
                r"what email",
                r"which email",
            ]
            
            original_reply = assistant_reply
            for pattern in INFO_REQUEST_PATTERNS:
                if re.search(pattern, assistant_reply, re.IGNORECASE):
                    logger.warning(
                        f"🚫 RESPONSE SANITIZER: Removing info request from LLM response. "
                        f"Pattern matched: '{pattern}'. Form data available: {list(form_data.keys())}. "
                        f"Completed forms: {completed_forms}"
                    )
                    # Remove the sentence containing the pattern
                    assistant_reply = re.sub(
                        r'[^.!?]*' + pattern + r'[^.!?]*[.!?]?',
                        '',
                        assistant_reply,
                        flags=re.IGNORECASE
                    ).strip()
            
            if assistant_reply != original_reply:
                # Clean up any trailing punctuation or weird spacing
                assistant_reply = re.sub(r'\s+', ' ', assistant_reply).strip()
                assistant_reply = re.sub(r'^[,.\s]+', '', assistant_reply).strip()
                if not assistant_reply:
                    assistant_reply = "I'm happy to help! Is there anything else you'd like to know?"
                logger.info(f"✅ Sanitized response: {assistant_reply[:100]}...")
        
        logger.info(
            f"✅ LLM ({provider}) generated response for session {context.session_id}: "
            f"{assistant_reply[:100]}..."
        )
    
    append_turn(context.session_id, "assistant", assistant_reply)
    
    # Save to MongoDB history format
    domain = context.metadata.get("domain", "domain_3") if context.metadata else "domain_3"
    append_conversation(context.session_id, context.message, assistant_reply, domain)
    
    route_label = "llm_router" if action == "CLARIFY" else "llm_direct"
    routing_metadata.selected_route = route_label
    routing_metadata.route_reason = f"LLM ({provider}) generated direct response: {llm_result.reason}"
    
    # Inject CTAs for LLM direct/clarify responses
    cta_options = inject_ctas_if_appropriate(
        route=route_label,
        active_flow=context.active_flow,
    )
    
    return ConversationResponse(
        session_id=context.session_id,
        response=assistant_reply,
        route=route_label,
        intent=ml_intent,
        confidence=ml_confidence,
        routing_metadata=routing_metadata,
        rag_chunks=rag_chunks if rag_chunks else None,
        options=cta_options,  # CTAs go in options field for frontend rendering
    )


def build_form_continue_response(
    context: ContextState,
    form_result: DecisionResult
) -> ConversationResponse:
    """
    Build form continuation response.
    
    Args:
        context: Conversation context
        form_result: Form continuation decision result
    
    Returns:
        ConversationResponse for form continuation
    """
    response_data = form_result.response_data or {}
    
    # Build routing metadata for form continuation
    routing_metadata = RoutingMetadata(
        routing_steps=[],
        selected_route="form_continue",
        route_reason=form_result.reason or f"Continuing active form flow: {context.active_flow}",
        faq_attempted=False,
        ml_attempted=False,
        llm_attempted=False,
        timing_ms={"form_continue_ms": form_result.timing_ms} if form_result.timing_ms else None,
    )
    
    # Extract options and question_type if available
    options = None
    question_type = None
    state = response_data.get("state") or {}
    current_key = state.get("current_question_key")
    if current_key:
        question_data = (state.get("question_set_data") or {}).get(current_key, {})
        question_type = question_data.get("type")  # text, options, or file
        if question_type == "options":
            options = question_data.get("options", [])
    
    # Save to MongoDB history format with collected form data
    agent_message = response_data.get("agent_message", "")
    if context.message and agent_message:
        domain = context.metadata.get("domain", "domain_3") if context.metadata else "domain_3"
        # Update session with collected_data before saving to MongoDB
        collected_data = response_data.get("collected_data", {})
        if collected_data:
            upsert_session(context.session_id, {"form_data": collected_data})
        append_conversation(context.session_id, context.message, agent_message, domain)
    
    return ConversationResponse(
        session_id=context.session_id,
        response=response_data.get("agent_message", ""),
        route="form_continue",
        intent=context.active_flow,
        confidence=None,
        next_question=response_data.get("agent_message"),
        options=options,
        question_type=question_type,  # Pass question type to frontend
        is_complete=response_data.get("all_questions_completed", False),
        collected_data=response_data.get("collected_data", {}),
        # 🆕 Populate requested fields
        current_step=current_key,
        lead_data=response_data.get("collected_data", {}),
        routing_metadata=routing_metadata,
    )





