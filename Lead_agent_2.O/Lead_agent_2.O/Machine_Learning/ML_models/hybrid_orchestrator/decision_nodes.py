"""
Modular decision nodes for enterprise AI agent routing.

Each node handles a specific routing decision with standardized input/output,
timing measurement, and error handling.

🔒 ANTI-GRAVITY ARCHITECTURE:
All LLM decisions are now passed through Intent Guard and Consent Governor
before execution. LLMs suggest, code decides.
"""
import logging
import time
import re
from typing import Dict, Optional, Tuple

from config.settings import FAQ_CONFIDENCE_THRESHOLD, ML_CONFIDENCE_THRESHOLD
from hybrid_orchestrator.confidence_router import route_decision
from hybrid_orchestrator.logging_utils import log_task
from hybrid_orchestrator.session_manager import append_turn, upsert_session
from models.hybrid_schemas import ContextState, DecisionResult
from ml_services.classifier_gateway import classify_query
from ml_services.faq_service import ask_faq
from ml_services.langgraph_gateway import end_form_session, send_form_message, start_form_session

# 🔒 GOVERNANCE IMPORTS - Anti-Gravity Architecture
try:
    from governance import (
        IntentGuard,
        ConsentGovernor,
        AuditLogger,
        DecisionEnvelope,
        ActionType,
    )
    GOVERNANCE_ENABLED = True
except ImportError:
    GOVERNANCE_ENABLED = False
    logging.warning("Governance module not available - running in legacy mode")

logger = logging.getLogger(__name__)


class FAQDecisionNode:
    """
    Decision node for FAQ semantic search.
    
    Handles FAQ lookup with caching and returns decision result.
    """
    
    @staticmethod
    def execute(context: ContextState, threshold: Optional[float] = None) -> DecisionResult:
        """
        Execute FAQ lookup decision.
        
        Args:
            context: Conversation context
            threshold: Confidence threshold (defaults to FAQ_CONFIDENCE_THRESHOLD)
        
        Returns:
            DecisionResult with FAQ lookup outcome
        """
        start_time = time.time()
        threshold = threshold or FAQ_CONFIDENCE_THRESHOLD or 0.8
        
        try:
            # Extract domain from context metadata (it's a Dict, not an object!)
            # Priority: tenant_id > extra.domain > default
            domain = "default"
            if context.metadata:
                # Metadata is a Dict[str, Any], so use .get() for access
                if context.metadata.get("domain"):
                    domain = context.metadata["domain"]
                elif context.metadata.get("extra") and isinstance(context.metadata["extra"], dict):
                    domain = context.metadata["extra"].get("domain", domain)
            
            log_task(
                "decision.faq.attempt",
                "start",
                session_id=context.session_id,
                message=context.message,
                threshold=threshold,
                domain=domain,
            )
            
            # Perform FAQ lookup with domain
            faq_result = ask_faq(context.message, domain=domain, threshold=threshold)
            faq_confidence = faq_result.get("confidence") or 0.0
            timing_ms = (time.time() - start_time) * 1000
            
            # Check if FAQ returned high confidence answer
            if faq_result.get("answer") and faq_confidence >= threshold:
                log_task(
                    "decision.faq.selected",
                    "success",
                    session_id=context.session_id,
                    confidence=faq_confidence,
                    matched_question=faq_result.get("matched_question"),
                    timing_ms=timing_ms,
                )
                
                return DecisionResult(
                    selected=True,
                    route="faq_semantic",
                    confidence=faq_confidence,
                    response_data={
                        "answer": faq_result.get("answer"),
                        "matched_question": faq_result.get("matched_question"),
                        "source": faq_result.get("source", "vector_db"),
                    },
                    timing_ms=timing_ms,
                    reason=f"FAQ semantic search returned high confidence ({faq_confidence:.3f} >= {threshold})",
                )
            else:
                log_task(
                    "decision.faq.rejected",
                    "low_confidence",
                    session_id=context.session_id,
                    confidence=faq_confidence,
                    threshold=threshold,
                    timing_ms=timing_ms,
                )
                
                return DecisionResult(
                    selected=False,
                    route=None,
                    confidence=faq_confidence,
                    timing_ms=timing_ms,
                    reason=f"FAQ confidence {faq_confidence:.3f} < {threshold}",
                )
        
        except Exception as e:
            timing_ms = (time.time() - start_time) * 1000
            logger.error(
                "FAQ decision node error for session %s: %s",
                context.session_id,
                str(e),
                exc_info=True
            )
            log_task(
                "decision.faq.error",
                "exception",
                session_id=context.session_id,
                error=str(e),
                timing_ms=timing_ms,
            )
            
            return DecisionResult(
                selected=False,
                route=None,
                timing_ms=timing_ms,
                reason=f"FAQ lookup failed: {str(e)}",
                error=str(e),
            )


class MLDecisionNode:
    """
    Decision node for ML intent classification.
    
    Handles ML classification and routes to greeting/form handlers.
    """
    
    @staticmethod
    def execute(context: ContextState, threshold: Optional[float] = None) -> Tuple[DecisionResult, Dict]:
        """
        Execute ML classification decision.
        
        Args:
            context: Conversation context
            threshold: Confidence threshold (defaults to ML_CONFIDENCE_THRESHOLD)
        
        Returns:
            Tuple of (DecisionResult, ml_result_dict)
        """
        start_time = time.time()
        threshold = threshold or ML_CONFIDENCE_THRESHOLD or 0.8
        
        try:
            log_task(
                "decision.ml.attempt",
                "start",
                session_id=context.session_id,
                message=context.message,
                threshold=threshold,
            )
            
            # Extract domain from context metadata
            domain = "default"
            if context.metadata:
                if context.metadata.get("domain"):
                    domain = context.metadata["domain"]
                elif context.metadata.get("extra") and isinstance(context.metadata["extra"], dict):
                    domain = context.metadata["extra"].get("domain", domain)
            
            # Perform ML classification
            ml_result = classify_query(context.message, domain=domain, return_all_probs=True)
            ml_confidence = ml_result.get("confidence") or 0.0
            ml_intent = ml_result.get("intent")
            timing_ms = (time.time() - start_time) * 1000
            
            # Check if ML confidence meets threshold
            if ml_confidence >= threshold:
                decision = route_decision(ml_intent, ml_confidence)
                
                log_task(
                    "decision.ml.selected",
                    "high_confidence",
                    session_id=context.session_id,
                    intent=ml_intent,
                    confidence=ml_confidence,
                    route=decision.route,
                    timing_ms=timing_ms,
                )
                
                return DecisionResult(
                    selected=True,
                    route=decision.route,
                    confidence=ml_confidence,
                    intent=ml_intent,
                    timing_ms=timing_ms,
                    reason=f"ML classifier identified '{ml_intent}' with high confidence ({ml_confidence:.3f} >= {threshold})",
                ), ml_result
            else:
                log_task(
                    "decision.ml.rejected",
                    "low_confidence",
                    session_id=context.session_id,
                    intent=ml_intent,
                    confidence=ml_confidence,
                    threshold=threshold,
                    timing_ms=timing_ms,
                )
                
                return DecisionResult(
                    selected=False,
                    route=None,
                    confidence=ml_confidence,
                    intent=ml_intent,
                    timing_ms=timing_ms,
                    reason=f"ML confidence {ml_confidence:.3f} < {threshold}",
                ), ml_result
        
        except Exception as e:
            timing_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            if hasattr(e, "detail"):
                error_msg = str(e.detail)
            
            logger.error(
                "ML decision node error for session %s: %s",
                context.session_id,
                error_msg,
                exc_info=True
            )
            log_task(
                "decision.ml.error",
                "exception",
                session_id=context.session_id,
                error=error_msg,
                timing_ms=timing_ms,
            )
            
            return DecisionResult(
                selected=False,
                route=None,
                timing_ms=timing_ms,
                reason=f"ML classification failed: {error_msg}",
                error=error_msg,
            ), {}


class LLMDecisionNode:
    """
    Decision node for LLM fallback routing.
    
    Handles complex queries that FAQ and ML couldn't resolve.
    Note: LLM does NOT handle greetings - that's ML's responsibility.
    """
    
    @staticmethod
    def execute(
        context: ContextState,
        ml_result: Dict,
        faq_confidence: Optional[float] = None,
        ml_confidence: Optional[float] = None,
        rag_context: str = "",
        persona_instructions: str = "",
        persona_conditions: Dict = {},
        persona_formatting: Dict = {}
    ) -> DecisionResult:
        """
        Execute LLM fallback decision.
        
        Args:
            context: Conversation context
            ml_result: ML classification result for context
            faq_confidence: FAQ confidence score (for logging)
            ml_confidence: ML confidence score (for logging)
            rag_context: Retrieved documents for RAG
            persona_instructions: Domain-specific persona instructions
            persona_conditions: Domain-specific persona conditions
            persona_formatting: Domain-specific persona formatting guidelines
        """
        start_time = time.time()
        
        try:
            log_task(
                "decision.llm.attempt",
                "start",
                session_id=context.session_id,
                message=context.message,
                faq_confidence=faq_confidence,
                ml_confidence=ml_confidence,
                has_rag_context=bool(rag_context),
                has_persona=bool(persona_instructions)
            )
            
            # Log RAG context availability
            if rag_context:
                logger.info(f"🧠 LLM Decision: RAG context provided ({len(rag_context)} chars) for knowledge-based response")
            else:
                logger.info("🧠 LLM Decision: No RAG context available, using general knowledge")
            
            # Lazy import LLM router to reduce startup time and memory footprint
            # LLM is only used as fallback, so we don't want to load it on module import
            from hybrid_orchestrator.lcel_router import route_with_llm
            
            # Get conversation history for LLM
            history = context.last_three_turns
            
            # Call LLM router with form_data for user context
            completed_forms = context.session.get("completed_forms", [])
            
            llm_choice = route_with_llm(
                history, 
                context.message, 
                ml_result,
                rag_context=rag_context,
                persona_instructions=persona_instructions,
                persona_conditions=persona_conditions,
                persona_formatting=persona_formatting,
                form_data=context.form_data,  # Pass user info from forms
                completed_forms=completed_forms,  # 🔴 NEW: Prevent duplicate forms
                session_data=context.session,  # 🔴 FIX: Pass session data
                active_flow=context.active_flow or ""  # 🔴 NEW: Pass current flow
            )
            suggested_action = (llm_choice.get("action") or "CLARIFY").upper()
            llm_reason = llm_choice.get("reason") or f"LLM selected action: {suggested_action}"
            timing_ms = (time.time() - start_time) * 1000
            
            # 🔒 ANTI-GRAVITY ARCHITECTURE: LLM suggests, code decides
            # Pass through Intent Guard and Consent Governor before execution
            if GOVERNANCE_ENABLED:
                # Step 1: Create decision envelope from LLM suggestion
                envelope = DecisionEnvelope.from_llm_response({
                    "action": suggested_action,
                    "message": llm_choice.get("assistant_response"),
                    "reason": llm_reason,
                })
                envelope.session_id = context.session_id
                
                # Step 2: Intent Guard - ML intent overrides LLM if confident
                envelope = IntentGuard.enforce(
                    envelope,
                    ml_intent=ml_result.get("intent"),
                    ml_confidence=ml_confidence
                )
                
                # Step 3: Consent Governor - Check consent requirements
                consent_state = context.session.get("consent_state", {})
                envelope = ConsentGovernor.check(envelope, consent_state)
                
                # Step 4: Get final action (may be overridden)
                action = envelope.get_final_action()
                
                # Step 5: Audit log the decision
                AuditLogger.log_decision(envelope, context.session_id)
                
                # Log if override occurred
                if envelope.was_overridden():
                    logger.warning(
                        "🔒 GOVERNANCE OVERRIDE for session %s: LLM suggested '%s', "
                        "final action '%s'. Reason: %s",
                        context.session_id,
                        suggested_action,
                        action,
                        envelope.override_reason
                    )
                    llm_reason = f"[OVERRIDDEN] {envelope.override_reason}. Original: {llm_reason}"
                
                # Handle consent request
                if action == "REQUEST_CONSENT":
                    return DecisionResult(
                        selected=True,
                        route="consent_required",
                        timing_ms=timing_ms,
                        reason=f"Consent required: {envelope.consent_prompts}",
                        response_data={
                            "action": "REQUEST_CONSENT",
                            "consent_prompts": envelope.consent_prompts,
                            "assistant_response": envelope.consent_prompts[0] if envelope.consent_prompts else "May I collect some information?",
                        },
                    )
            else:
                # Legacy mode - no governance
                action = suggested_action
            
            # Validate action - LLM should NOT return RESPOND_GREETING
            if action == "RESPOND_GREETING":
                logger.warning(
                    "LLM returned RESPOND_GREETING for session %s - this should be handled by ML. "
                    "Converting to RESPOND_MESSAGE.",
                    context.session_id
                )
                action = "RESPOND_MESSAGE"
                llm_reason = f"LLM incorrectly returned RESPOND_GREETING, converted to RESPOND_MESSAGE. Original: {llm_reason}"
            
            log_task(
                "decision.llm.selected",
                "fallback",
                session_id=context.session_id,
                action=action,
                reason=llm_reason,
                timing_ms=timing_ms,
                governance_enabled=GOVERNANCE_ENABLED,
            )
            
            # Note: Job inquiry detection is now handled by LLM with tool calling
            # The LLM extracts job_search_keywords and the tool is executed in lcel_router.py
            # No hardcoded pattern matching needed here anymore
            
            # Determine route based on actions
            if action == "START_LEAD_FORM":
                question_set = "lead_questions"
                route = "form_llm"
            elif action == "CLARIFY":
                route = "llm_router"
            else:
                route = "llm_direct"
            
            return DecisionResult(
                selected=True,
                route=route,
                confidence=None,
                timing_ms=timing_ms,
                reason=llm_reason,
                response_data={
                    "action": action,
                    "assistant_response": llm_choice.get("assistant_response"),
                    "question_set": llm_choice.get("question_set"),
                    "job_search_keywords": llm_choice.get("job_search_keywords"),
                    "job_search_results": llm_choice.get("job_search_results"),
                    "governance_override": envelope.was_overridden() if GOVERNANCE_ENABLED else False,
                },
            )
        
        except Exception as e:
            timing_ms = (time.time() - start_time) * 1000
            logger.error(
                "LLM decision node error for session %s: %s",
                context.session_id,
                str(e),
                exc_info=True
            )
            log_task(
                "decision.llm.error",
                "exception",
                session_id=context.session_id,
                error=str(e),
                timing_ms=timing_ms,
            )
            
            # Fallback to clarify on error
            return DecisionResult(
                selected=True,
                route="llm_router",
                timing_ms=timing_ms,
                reason=f"LLM fallback failed, using clarify: {str(e)}",
                error=str(e),
                response_data={
                    "action": "CLARIFY",
                    "assistant_response": "I'm having trouble understanding that. Could you rephrase?",
                },
            )


class FormContinuationNode:
    """
    Decision node for handling active form flows.
    
    Short-circuits routing when user is already in a form flow.
    """
    

    
    @staticmethod
    def _is_off_topic(message: str, active_flow: str) -> bool:
        """
        Detect if user message is off-topic during an active form.
        
        Args:
            message: User's message
            active_flow: Current form flow
            
        Returns:
            True if off-topic detected
        """
        message_lower = message.lower().strip()
        words = message_lower.split()
        
        # 1. Very short messages are usually form answers (unless affirmative/negative)
        if len(words) <= 2:
            return False
            
        # 2. Check for question indicators (off-topic questions)
        question_indicators = ["?", "what", "how", "why", "when", "where", "who", "can you", "do you", "tell me"]
        if any(ind in message_lower for ind in question_indicators):
            return True
            
        # 3. Check for company/brand name mentions that are usually FAQ queries
        COMPANY_KEYWORDS = {"EnterpriseAI", "company", "services", "about", "pricing", "contact", "admission process", "tuition"}
        if any(kw in message_lower for kw in COMPANY_KEYWORDS):
            return True
            
        return False
    
    @staticmethod
    def execute(context: ContextState) -> Optional[DecisionResult]:
        """
        Check if form continuation is needed.
        
        Args:
            context: Conversation context
        
        Returns:
            DecisionResult if form continuation needed, None otherwise
        """
        if not context.active_flow:
            return None
        

        
        # 🔴 ESCAPE HATCH 4: Off-topic detection (trigger confirmation)
        if FormContinuationNode._is_off_topic(context.message, context.active_flow):
            logger.info(
                "🤔 OFF-TOPIC DETECTED: Prompting user for form continuation for session %s",
                context.session_id
            )
            # Store the off-topic query in session so we can re-process it if user says "No"
            upsert_session(context.session_id, {"pending_off_topic_query": context.message})
            
            return DecisionResult(
                selected=True,
                route="form_escape_confirmation",
                confidence=1.0,
                intent="form_escape_confirmation",
                reason="Off-topic query detected during active form"
            )
        
        start_time = time.time()
        
        try:
            log_task(
                "decision.form.continue",
                "start",
                session_id=context.session_id,
                active_flow=context.active_flow,
            )
            
            # Continue the form flow
            append_turn(context.session_id, "user", context.message)
            try:
                lang_result = send_form_message(context.session_id, context.message)
            except KeyError as e:
                # 🛑 ERROR HANDLER: Session lost (e.g. server restart) but active_flow stuck
                logger.error(f"❌ Session lost during form flow: {e}")
                
                # Clear active flow so next request starts fresh
                upsert_session(context.session_id, {"active_flow": None, "graph_state_id": None})
                
                return DecisionResult(
                    selected=True,
                    route="form_error_reset",
                    intent="session_reset",
                    confidence=1.0,
                    response_data={
                        "agent_message": "My memory of our conversation was briefly interrupted. Could you please say that again?",
                        "all_questions_completed": False
                    },
                    reason="Session lost in gateway"
                )
            
            append_turn(context.session_id, "assistant", lang_result["agent_message"])
            
            # Handle form completion
            if lang_result.get("all_questions_completed"):
                end_form_session(context.session_id)
                
                # CRITICAL: Persist collected form data to session for future reference
                collected_data = lang_result.get("collected_data", {})
                question_set = context.active_flow  # e.g., "lead_questions"
                
                logger.info(
                    f"Form completed for session {context.session_id}. "
                    f"Persisting collected data: {list(collected_data.keys())}"
                )
                
                # 1. First persist the collected data
                upsert_session(
                    context.session_id,
                    {
                        "graph_state_id": None,
                        "form_data": collected_data,
                    },
                )
                
                # 2. Extract Lead ID from agent state if available
                agent_state = lang_result.get("state", {})
                lead_id = agent_state.get("lead_id")
                
                # 3. Then mark as completed (triggers status update + EMAIL notification)
                from hybrid_orchestrator.session_manager import mark_form_completed
                mark_form_completed(context.session_id, question_set, lead_id=lead_id)
            
            timing_ms = (time.time() - start_time) * 1000
            
            log_task(
                "decision.form.continue",
                "success",
                session_id=context.session_id,
                active_flow=context.active_flow,
                completed=lang_result["all_questions_completed"],
                timing_ms=timing_ms,
            )
            
            return DecisionResult(
                selected=True,
                route="form_continue",
                timing_ms=timing_ms,
                reason=f"Continuing active form flow: {context.active_flow}",
                response_data={
                    "agent_message": lang_result["agent_message"],
                    "all_questions_completed": lang_result["all_questions_completed"],
                    "collected_data": lang_result.get("collected_data", {}),
                    "state": lang_result.get("state", {}),
                },
            )
        
        except Exception as e:
            timing_ms = (time.time() - start_time) * 1000
            logger.error(
                "Form continuation node error for session %s: %s",
                context.session_id,
                str(e),
                exc_info=True
            )
            log_task(
                "decision.form.error",
                "exception",
                session_id=context.session_id,
                error=str(e),
                timing_ms=timing_ms,
            )
            
            # Return error result - router will handle fallback
            return DecisionResult(
                selected=False,
                route=None,
                timing_ms=timing_ms,
                reason=f"Form continuation failed: {str(e)}",
                error=str(e),
            )




