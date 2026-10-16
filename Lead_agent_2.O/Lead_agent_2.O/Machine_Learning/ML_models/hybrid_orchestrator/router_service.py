"""
Enterprise AI Agent - Main Orchestration Service

This module implements the core orchestration logic for the enterprise AI agent,
following a modular architecture with clear separation of concerns.

🔒 ANTI-GRAVITY ARCHITECTURE (December 2024):
    - Policy Router is the SINGLE AUTHORITY for routing decisions
    - FAQ/ML/RAG are SIGNALS, not decisions
    - Conversation mode gates routing
    - LLM is ADVISORY, code DECIDES

Architecture:
    The routing pipeline now follows policy-driven routing:
    1. Mode Gate (FORM_ACTIVE / HANDOFF bypass with escape handling)
    2. Collect cheap signals (FAQ, ML, RAG, message analysis)
    3. Policy Router selects single winner
    4. Governance enforces action permissions
    5. Execute winner and produce legacy response

Key Components:
    - Decision Nodes: Modular routing decisions (FAQ, ML, LLM, Form) [LEGACY]
    - Policy Router: Single authority for route selection [ANTI-GRAVITY]
    - Response Builder: Centralized response construction
    - Context Manager: Efficient conversation context handling
    - Session Manager: Session state management (Redis-backed)
"""
import importlib
import logging
import os
from pathlib import Path
import sys
import time
from functools import lru_cache
from typing import Dict, Optional, List, Any

from config.settings import FAQ_CONFIDENCE_THRESHOLD, ML_CONFIDENCE_THRESHOLD
from hybrid_orchestrator.confidence_router import route_decision
from hybrid_orchestrator.context_manager import build_context, validate_context
from hybrid_orchestrator.decision_nodes import (
    FAQDecisionNode,
    FormContinuationNode,
    LLMDecisionNode,
    MLDecisionNode,
)
from hybrid_orchestrator.logging_utils import log_task
from hybrid_orchestrator.response_builder import (

    build_faq_response,
    build_form_continue_response,
    build_llm_response,
    build_ml_form_response,
    build_ml_greeting_response,
    build_routing_metadata,
)
from hybrid_orchestrator.session_manager import (
    create_session, 
    get_session,
    upsert_session,
    get_mode,
    set_mode,
    get_last_route_family,
    set_last_route_family,
    dec_qa_turns,
    append_turn,
)
from models.hybrid_schemas import ConversationRequest, ConversationResponse, DecisionResult, RoutingMetadata

# ---------------------------------------------------------------------------
# 🧠 INTENT CLASSIFICATION HELPERS
# ---------------------------------------------------------------------------

HARD_LEAD_KEYWORDS = {
    "apply", "admission", "eligibility", "fee", "fees", "deadline", 
    "documents", "entrance exam", "scholarship", "intake", "apply online", "process"
}

def classify_intent_and_score(query: str, ml_result: DecisionResult = None, faq_result: DecisionResult = None, faq_threshold: float = 0.8) -> tuple[str, float]:
    """
    Determines the primary INTENT (FAQ vs LEAD) and normalization SCORE (0-100).
    Follows the design requiring explicit intent classification before scoring.
    """
    normalized_query = query.lower()
    
    # 1. Keyword Signal Check
    keyword_hits = [word for word in HARD_LEAD_KEYWORDS if word in normalized_query]
    hard_lead_signal = len(keyword_hits) >= 2
    
    # 2. Determine Intent
    intent = "FAQ" # Default assumption
    
    # Helper to clean predict intent from ML
    ml_intent = ml_result.intent if ml_result else ""
    ml_conf = ml_result.confidence if ml_result and ml_result.confidence is not None else 0.0
    faq_conf = faq_result.confidence if faq_result and faq_result.confidence is not None else 0.0

    # If ML explicitly says it's a lead intent
    if ml_intent in ["lead_generation", "apply", "form_ml", "lead_collection"]:
        intent = "LEAD"
    
    # Overwrite if Hard Signal found (Stronger than semantic drift)
    elif hard_lead_signal:
        intent = "LEAD"
        
    # Fallback: if FAQ confidence is very low but ML confidence is decent
    # This catches "admission?" where FAQ might struggle but ML knows the topic
    elif faq_conf < 0.6 and ml_conf > 0.6 and ml_intent == "general":
         # Be careful with 'general' - only promote if query has at least one keyword
         if len(keyword_hits) > 0:
             intent = "LEAD"

    # 3. Determine Score (0-100)
    score = 0.0
    
    if intent == "LEAD":
        # Base score from ML model
        score = ml_conf * 100
        
        # Boost score if keywords present (Reward specificity)
        if hard_lead_signal:
            score = max(score, 95.0) # Force high score for explicit keywords
        elif len(keyword_hits) == 1:
            score = min(score + 10, 100.0) # Slight boost
            
    else: # Intent is FAQ (or unknown fallback)
        score = faq_conf * 100
        
    return intent, min(score, 100.0)

# 🔒 ANTI-GRAVITY: Policy Router Imports
try:
    from hybrid_orchestrator.routing import (
        PolicyRouter,
        get_policy_router,
        RouteCandidate,
        PolicyDecision,
        analyze_message,
        detect_escape_intent,
        collect_faq_signal,
        collect_ml_signal,
        collect_rag_signal,
        collect_clarify_signal,
        collect_llm_signal,
        log_policy_decision,
        log_escape_handled,
    )
    POLICY_ROUTER_ENABLED = True
except ImportError as e:
    POLICY_ROUTER_ENABLED = False
    logging.warning(f"⚠️ Policy Router not available - using legacy routing: {e}")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 🔒 ANTI-GRAVITY: Feature Flag Configuration (Gradual Rollout)
# ---------------------------------------------------------------------------

# Environment-based flag: set ENABLE_POLICY_ROUTER=true to enable
ENABLE_POLICY_ROUTER = os.getenv("ENABLE_POLICY_ROUTER", "false").lower() == "true"

# Tenant-based rollout list (comma-separated in env)
# Example: POLICY_ROUTER_TENANTS=internal_test,staging_tenant
POLICY_ROUTER_TENANTS = set(
    t.strip() for t in os.getenv("POLICY_ROUTER_TENANTS", "").split(",") if t.strip()
)

# Shadow mode: run policy router but DON'T execute, just log decisions
POLICY_ROUTER_SHADOW_MODE = os.getenv("POLICY_ROUTER_SHADOW_MODE", "false").lower() == "true"


def _policy_router_enabled(session: Dict, metadata: Optional[Any] = None) -> bool:
    """
    Check if Policy Router should be used for this request.
    
    Priority:
    1. POLICY_ROUTER_ENABLED must be True (module loaded)
    2. ENABLE_POLICY_ROUTER env flag OR tenant in allowed list
    
    Args:
        session: Current session dict
        metadata: Request metadata
    
    Returns:
        True if Policy Router should be used
    """
    if not POLICY_ROUTER_ENABLED:
        return False
    
    # Global enable flag
    if ENABLE_POLICY_ROUTER:
        return True
    
    # Domain-based rollout via extra.domain
    if metadata and POLICY_ROUTER_TENANTS:
        domain_id = None
        if hasattr(metadata, 'extra') and isinstance(metadata.extra, dict):
            domain_id = metadata.extra.get('domain')
        
        if domain_id and domain_id in POLICY_ROUTER_TENANTS:
            logger.debug(f"🔓 Policy Router enabled for domain: {domain_id}")
            return True
    
    return False


# Intent to question set mapping for form routing
FORM_INTENT_TO_SET = {
    "lead_questions": "lead_questions",
    "lead_form": "lead_questions",
    "lead_collection": "lead_questions",
}


def _ensure_session(req: ConversationRequest) -> Dict:
    """
    Ensure session exists, creating if necessary.
    
    Args:
        req: Conversation request
    
    Returns:
        Session dictionary
    """
    session = get_session(req.session_id)
    if session:
        return session
    return create_session({"session_id": req.session_id})


import asyncio
from ml_services.persona_service import get_persona_service

# --- RAG Import Helpers ------------------------------------------------------

@lru_cache()
def _discover_rag_path() -> Optional[str]:
    """
    Locate the RAG project directory relative to this module or via env override.
    """
    env_rag_root = os.getenv("RAG_ROOT")
    if env_rag_root:
        candidate = Path(env_rag_root).expanduser().resolve()
        if candidate.exists():
            return str(candidate)

    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        candidate = parent / "RAG"
        if candidate.exists():
            return str(candidate)
    return None


def _ensure_rag_path() -> None:
    rag_path = _discover_rag_path()
    if not rag_path:
        raise ImportError(
            "Unable to locate the RAG project directory. Set RAG_ROOT or ensure "
            "a sibling 'RAG' folder exists relative to ML_models."
        )
    if rag_path not in sys.path:
        # CRITICAL: Insert at position 0 to prioritize RAG's 'app' over root's 'app'
        sys.path.insert(0, rag_path)
    
    # CRITICAL: Clear any cached 'app' module that might be from the wrong location (root)
    # This resolves the module name collision between root/app and RAG/app
    modules_to_clear = [key for key in list(sys.modules.keys()) if key == 'app' or key.startswith('app.')]
    for mod_name in modules_to_clear:
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, '__file__') and mod.__file__:
            # Only clear if it's NOT from the RAG directory
            if rag_path and rag_path not in mod.__file__:
                del sys.modules[mod_name]


@lru_cache()
def _get_rag_query_module():
    _ensure_rag_path()
    
    # CRITICAL: Clear any cached 'app' module that might be from the wrong location (root)
    # This resolves the module name collision between root/app and RAG/app
    rag_path = _discover_rag_path()
    modules_to_clear = [key for key in sys.modules.keys() if key == 'app' or key.startswith('app.')]
    for mod_name in modules_to_clear:
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, '__file__') and mod.__file__:
            # Only clear if it's NOT from the RAG directory
            if rag_path and rag_path not in mod.__file__:
                del sys.modules[mod_name]
    
    return importlib.import_module("app.services.document_query")

async def _search_rag(query: str, domain: str) -> tuple[str, dict, list]:
    """
    Execute RAG search asynchronously using Unified RAG Service.
    Searches both Document RAG and Web Scraper data.
    Returns: (context_string, metadata_dict, raw_chunks_list)
    """
    import time
    start_time = time.time()
    
    metadata = {
        "attempted": True,
        "success": False,
        "results_count": 0,
        "domain": domain,
        "error": None,
        "timing_ms": 0,
        "sources": []  # Track which sources returned results
    }
    
    logger.info(f"🔄 UNIFIED RAG SEARCH START: domain={domain}, query=\"{query[:50]}...\"")
    
    try:
        # Try Unified RAG Service first (includes both Document RAG and Web Scraper)
        try:
            _ensure_rag_path()  # Ensure RAG path is in sys.path before importing
            from app.services.unified_rag_service import get_unified_rag_service
            unified_service = get_unified_rag_service()
            
            # Run in thread pool since FAISS might be CPU bound
            results = await asyncio.to_thread(
                unified_service.search,
                query=query,
                domain=domain,
                top_k=5,
                threshold=0.3  # Lower threshold for better recall
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            metadata["timing_ms"] = elapsed_ms
            metadata["results_count"] = len(results)
            metadata["success"] = True
            
            # Track sources used
            sources_used = set()
            for r in results:
                sources_used.add(r.source_type)
            metadata["sources"] = list(sources_used)
            
            if not results:
                logger.warning(f"⚠️ UNIFIED RAG: No results found for domain={domain} in {elapsed_ms:.1f}ms")
                return "", metadata, []
            
            logger.info(
                f"✅ UNIFIED RAG SUCCESS: found {len(results)} results from {metadata['sources']} "
                f"for domain={domain} in {elapsed_ms:.1f}ms"
            )
            
            # Format results for LLM and prepare raw chunks
            context_parts = []
            raw_chunks = []
            
            for doc in results:
                # Include source type in context for LLM
                source_label = f"[{doc.source_type.upper()}]"
                if doc.source_url:
                    source_label += f" {doc.source_url}"
                
                context_parts.append(f"Source: {source_label}\nContent: {doc.content}")
                
                # Build raw chunk data for response
                raw_chunks.append({
                    "chunk_id": doc.chunk_id,
                    "doc_id": doc.doc_id,
                    "source_file": doc.source_url or "",
                    "content": doc.content,
                    "score": doc.score,
                    "source_type": doc.source_type,  # Include source type
                    "source_url": doc.source_url,     # Include source URL
                    "metadata": doc.metadata
                })
            
            return "\n\n".join(context_parts), metadata, raw_chunks
            
        except ImportError as ie:
            logger.warning(f"⚠️ Unified RAG not available, falling back to Document RAG: {ie}")
            # Fall back to legacy Document RAG
            query_module = _get_rag_query_module()
            query_service = query_module.get_document_query_service()
            
            results = await asyncio.to_thread(
                query_service.search,
                query=query,
                domain=domain,
                top_k=3,
                threshold=0.3
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            metadata["timing_ms"] = elapsed_ms
            metadata["results_count"] = len(results)
            metadata["success"] = True
            metadata["sources"] = ["document"]
            
            if not results:
                logger.warning(f"⚠️ RAG SEARCH: No results found for domain={domain} in {elapsed_ms:.1f}ms")
                return "", metadata, []
            
            logger.info(f"✅ RAG SEARCH SUCCESS: found {len(results)} results for domain={domain} in {elapsed_ms:.1f}ms")
            
            context_parts = []
            raw_chunks = []
            
            for doc in results:
                context_parts.append(f"Source: {doc.source_file}\nContent: {doc.content}")
                raw_chunks.append({
                    "chunk_id": doc.chunk_id,
                    "doc_id": doc.doc_id,
                    "source_file": doc.source_file,
                    "content": doc.content,
                    "score": doc.score,
                    "source_type": "document",
                    "source_url": doc.source_file,
                    "metadata": doc.metadata
                })
            
            return "\n\n".join(context_parts), metadata, raw_chunks
            
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.error(f"❌ RAG SEARCH FAILED: {e} ({elapsed_ms:.1f}ms)")
        metadata["error"] = str(e)
        metadata["timing_ms"] = elapsed_ms
        return "", metadata, []



# ---------------------------------------------------------------------------
# 🔒 ANTI-GRAVITY: Dual-Path Execution Entry Point
# ---------------------------------------------------------------------------

async def process_conversation(req: ConversationRequest) -> ConversationResponse:
    """
    Process conversation through enterprise AI agent pipeline.
    
    🔒 ANTI-GRAVITY: Dual-path execution with feature flag.
    - Default: Legacy threshold-based routing (unchanged behavior)
    - Flag enabled: Policy-driven routing with continuity preservation
    
    Args:
        req: Conversation request with message and session info
    
    Returns:
        ConversationResponse with routing metadata
    
    Raises:
        ValueError: If context validation fails
    """
    # Log conversation ingress
    log_task(
        "conversation.ingress",
        "start",
        session_id=req.session_id,
        message=req.message,
        metadata=req.metadata.dict() if req.metadata else {},
    )
    
    # Ensure session exists
    session = _ensure_session(req)

    # -----------------------------------------------------------------------
    # 🔴 LEAD SUBMISSION INTERCEPT (Migration from /submit-lead)
    # -----------------------------------------------------------------------
    if req.message == "Submit Lead Form" and req.metadata and req.metadata.extra:
        try:
            from ml_utils.mongodb_session_client import save_lead_to_mongo
            
            lead_data = req.metadata.extra
            # Ensure session_id is present
            if "session_id" not in lead_data:
                lead_data["session_id"] = req.session_id
                
            lead_id = save_lead_to_mongo(lead_data)
            
            if lead_id:
                return ConversationResponse(
                    session_id=req.session_id,
                    response="Thank you! Your details have been submitted successfully. Our team will contact you shortly.",
                    route="lead_submission",
                    is_complete=True,
                    collected_data={"lead_id": lead_id}
                )
            else:
                logger.error(f"❌ Failed to save lead for session {req.session_id}")
                return ConversationResponse(
                    session_id=req.session_id,
                    response="I apologize, but there was an issue submitting your details. Please try again later.",
                    route="lead_submission_error",
                    is_complete=True
                )
        except Exception as e:
            logger.error(f"❌ Error processing lead submission: {e}")
            return ConversationResponse(
                session_id=req.session_id,
                response="I apologize, but there was an error processing your request.",
                route="lead_submission_error",
                is_complete=True
            )



    
    # 🔒 ANTI-GRAVITY: Feature flag dispatch
    use_policy_router = _policy_router_enabled(session, req.metadata)
    
    if use_policy_router and not POLICY_ROUTER_SHADOW_MODE:
        # NEW PATH: Policy-driven routing
        logger.info(f"🔓 Using Policy Router for session {req.session_id}")
        return await _policy_process_conversation(req, session)
    
    # LEGACY PATH: Threshold-based routing (default)
    legacy_response = await _legacy_process_conversation(req, session)
    
    # 🔒 SHADOW MODE: Run policy router simulation for comparison logging
    if use_policy_router and POLICY_ROUTER_SHADOW_MODE:
        try:
            await _shadow_policy_comparison(req, session, legacy_response)
        except Exception as e:
            logger.warning(f"⚠️ Shadow mode comparison failed: {e}")
            
    # 🌐 TRANSLATION LAYER: Translate response if user locale is not English
    try:
        if req.metadata and req.metadata.locale:
            locale = req.metadata.locale
            if locale and not locale.lower().startswith('en'):
                from ml_services.translation_service import translate_text
                
                # Translate main response
                if legacy_response.response:
                    translated_text = await asyncio.to_thread(
                        translate_text, 
                        legacy_response.response, 
                        locale
                    )
                    legacy_response.response = translated_text
                    
                # Translate options if they are simple strings (CTAs) or dictionaries with labels
                if legacy_response.options and isinstance(legacy_response.options, list):
                    translated_options = []
                    for opt in legacy_response.options:
                        if isinstance(opt, str):
                            t_opt = await asyncio.to_thread(translate_text, opt, locale)
                            translated_options.append(t_opt)
                        elif isinstance(opt, dict) and "label" in opt:
                            # Handle CTA objects (e.g. {"label": "Schedule Visit", "action": "..."})
                            t_label = await asyncio.to_thread(translate_text, opt["label"], locale)
                            # Create a copy to avoid mutating the original if it's shared
                            new_opt = opt.copy()
                            new_opt["label"] = t_label
                            translated_options.append(new_opt)
                        else:
                            translated_options.append(opt)
                    legacy_response.options = translated_options
                        
                logger.info(f"🌍 Translated response to {locale}")
    except Exception as e:
        logger.error(f"❌ Translation failed: {e}")
    
    return legacy_response


async def _shadow_policy_comparison(
    req: ConversationRequest, 
    session: Dict, 
    legacy_response: ConversationResponse
) -> None:
    """
    Run Policy Router in shadow mode and log what it WOULD have decided.
    Does not affect the response - only for comparison logging.
    """
    if not POLICY_ROUTER_ENABLED:
        return
    
    try:
        # Simulate policy decision
        context = build_context(req, session)
        message_signals = analyze_message(req.message)
        mode = get_mode(session)
        last_family = get_last_route_family(session)
        
        # Log the shadow comparison
        logger.info(
            f"🔮 SHADOW: Legacy route='{legacy_response.route}', "
            f"mode={mode}, last_family={last_family}, "
            f"signals={message_signals.to_dict()}"
        )
        
        # TODO: Add full policy router simulation here once Phase 7 is complete
        
    except Exception as e:
        logger.debug(f"Shadow comparison skipped: {e}")


async def _legacy_process_conversation(
    req: ConversationRequest, 
    session: Dict
) -> ConversationResponse:
    """
    LEGACY: Original threshold-based routing logic.
    
    This function is preserved unchanged from the pre-Anti-Gravity implementation.
    It is the default path when Policy Router is disabled.
    
    Routing priority: FAQ (>=0.8) → ML (>=0.8) → LLM
    """
    # Determine domain for RAG and Persona
    # Priority: req.domain > extra.domain > default
    from config.settings import DEFAULT_DOMAIN
    domain = DEFAULT_DOMAIN
    if getattr(req, "domain", None):
        domain = req.domain
    elif req.metadata and req.metadata.extra.get("domain"):
        domain = req.metadata.extra.get("domain")
    
    # Override default/tech_company to DEFAULT_DOMAIN for this deployment
    if domain in ["default", "tech_company"]:
        logger.info(f"🔄 Overriding domain '{domain}' to '{DEFAULT_DOMAIN}' for Tech deployment")
        domain = DEFAULT_DOMAIN
        # Update request metadata so context and nodes see the correct domain
        if req.metadata:
            req.metadata.extra["domain"] = domain
        else:
            from models.hybrid_schemas import ConversationMetadata
            req.metadata = ConversationMetadata(extra={"domain": domain})
    
    # Build conversation context
    context = build_context(req, session)
    if not validate_context(context):
        raise ValueError(f"Invalid context for session {req.session_id}")
    
    # Ensure context metadata reflects the resolved domain
    if context.metadata is not None:
        context.metadata['domain'] = domain
    else:
        context.metadata = {'domain': domain}
    
    logger.info(f"🎯 Resolved RAG domain: {domain} (from {'extra.domain' if req.metadata and req.metadata.extra.get('domain') else 'default'})")
    
    # 🔴 EARLY GREETING INTERCEPT (Bypass LLM/ML for simple greetings)
    normalized_query = req.message.strip().lower()
    if normalized_query in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "hii", "heya", "howdy", "hii there", "hello there", "hi there"]:
        logger.info(f"🛑 DIRECT GREETING DETECTED: Bypassing ML/LLM calls for '{normalized_query}'")
        mock_ml_result = DecisionResult(
            route="greeting_ml",
            intent="greeting",
            confidence=1.0,
            selected=True,
            reason="Direct keyword match for greeting"
        )
        routing_metadata = build_routing_metadata(
            selected_route="direct_greeting",
            route_reason=f"Exact match for basic greeting: '{normalized_query}'"
        )
        return build_ml_greeting_response(context, mock_ml_result, routing_metadata)

    # 🔴 ACTION INTERCEPT (Handle CTA actions directly)
    if normalized_query in ["start_project", "free_consultation", "apply_internship", "build_software"]:
        logger.info(f"🛑 ACTION DETECTED: Triggering lead form for '{normalized_query}'")
        mock_ml_result = DecisionResult(
            route="form_ml",
            intent="lead_questions",
            confidence=1.0,
            selected=True,
            reason=f"Direct CTA action: {normalized_query}"
        )
        routing_metadata = build_routing_metadata(
            selected_route="direct_action_form",
            route_reason=f"CTA action trigger: {normalized_query}"
        )
        return build_ml_form_response(context, mock_ml_result, routing_metadata, "lead_questions")

    # STEP 0: Message signals (CHEAP - run first)
    message_signals = analyze_message(req.message)

    # Initialize decision results to prevent scope errors
    interruption_prefix = ""
    faq_chk_result = None
    ml_result = None
    faq_result = None

    # STEP 1: ESCAPE CHECK (before mode bypass)
    # Allow users to escape from FORM_ACTIVE mode
    mode = get_mode(session) # Ensure mode is defined here for the new logic
    if mode == "FORM_ACTIVE":
        if message_signals.wants_cancel:
            # User wants to cancel form - exit form mode
            logger.info(f"🚪 [LEGACY] Form escape: cancel detected")
            from hybrid_orchestrator.routing.routing_logger import log_escape_handled
            log_escape_handled(req.session_id, "cancel", mode)
            set_mode(req.session_id, "DEFAULT", trigger="user_cancel")
            # Continue to normal routing (below)
            mode = "DEFAULT"
        elif message_signals.wants_handoff:
            # User wants human during form - initiate handoff
            logger.info(f"🚪 [LEGACY] Form escape: handoff requested")
            from hybrid_orchestrator.routing.routing_logger import log_escape_handled
            log_escape_handled(req.session_id, "handoff", mode)
            set_mode(req.session_id, "HANDOFF_PENDING", trigger="user_handoff_request")
            mode = "HANDOFF_PENDING"

    # 🔴 FORM CONFIRMATION MODE HANDLING
    if mode == "FORM_CONFIRMATION":
        if message_signals.is_affirmative:
            # 🟢 ANSWER + CONTINUE LOGIC
            # User confirmed they want to continue, but we should answer their off-topic query first if possible
            original_query = session.get("pending_off_topic_query")
            if original_query:
                logger.info(f"🔄 [LEGACY] Fetching answer for off-topic query before continuing: {original_query}")
                try:
                    # Quick semantic search for the answer
                    # We swap the message temporarily to check FAQ/KB
                    original_msg = context.message
                    context.message = original_query
                    
                    # Execute FAQ check (lightweight)
                    # We need to instantiate locally as it's not run yet
                    faq_node = FAQDecisionNode() 
                    faq_chk_result_temp = faq_node.execute(context, threshold=0.8)
                    
                    if faq_chk_result_temp.selected:
                        interruption_prefix = faq_chk_result_temp.response_data.get('answer', '')
                        logger.info(f"✅ Found off-topic answer (FAQ): {interruption_prefix[:50]}...")
                    else:
                        # Fallback to RAG + LLM if FAQ fails
                        # The user asked something complex, let's try to answer it intelligently
                        logger.info(f"🔄 [LEGACY] FAQ failed. Attempting RAG + LLM for off-topic query: {original_query}")
                        
                        # 1. Run RAG search for the original query
                        rag_context_str, _, _ = await _search_rag(original_query, domain)
                        
                        # 2. Call LLM Router to generate answer
                        from hybrid_orchestrator.lcel_router import route_with_llm
                        
                        # We need a minimal ML result structure for the router
                        # It doesn't need to be accurate, just structural
                        dummy_ml_result = {"intent": "general_info", "confidence": 0.5}
                        
                        llm_response = await asyncio.to_thread(
                            route_with_llm,
                            history=context.last_three_turns,
                            user_message=original_query,
                            ml_result=dummy_ml_result,
                            rag_context=rag_context_str,
                            persona_instructions="You are a helpful university assistant. Answer the user's question concisely.",
                            persona_conditions={},
                            persona_formatting={},
                            form_data=context.form_data,
                            completed_forms=context.session.get("completed_forms", []),
                            session_data=session,
                            active_flow=context.active_flow or ""
                        )
                        
                        if llm_response.get("assistant_response"):
                            interruption_prefix = llm_response["assistant_response"]
                            logger.info(f"✅ Found off-topic answer (LLM): {interruption_prefix[:50]}...")
                    
                    # Restore context
                    context.message = original_msg
                    # Clear pending query
                    upsert_session(req.session_id, {"pending_off_topic_query": None})
                except Exception as e:
                    logger.error(f"⚠️ Failed to fetch off-topic answer: {e}")

            logger.info(f"✅ [LEGACY] Form Confirmation: User said YES. Re-asking current question.")
            set_mode(req.session_id, "FORM_ACTIVE", trigger="user_confirmed_continuation")
            mode = "FORM_ACTIVE"
            req.message = "continue" 
            context.message = "continue"
            # Fall through to FORM_ACTIVE gate below
        elif message_signals.is_negative:
            logger.info(f"🚪 [LEGACY] Form Confirmation: User said NO. Escaping form.")
            from ml_services.langgraph_gateway import end_form_session
            end_form_session(req.session_id)
            upsert_session(req.session_id, {"active_flow": None, "graph_state_id": None})
            context.active_flow = None
            set_mode(req.session_id, "DEFAULT", trigger="user_declined_continuation")
            mode = "DEFAULT"
            
            # 🔄 RE-PROCESS: Get the stored offtopic query and use it as the current message
            original_query = session.get("pending_off_topic_query")
            if original_query:
                logger.info(f"🔄 [LEGACY] Re-processing original off-topic query: {original_query}")
                req.message = original_query
                context.message = original_query
                # Clear the pending query from session
                upsert_session(req.session_id, {"pending_off_topic_query": None})
            
            # Fall through to normal routing below
        else:
            logger.info(f"🤔 [LEGACY] Form Confirmation: Ambiguous response. Re-prompting.")
            return ConversationResponse(
                session_id=req.session_id,
                response="I'm sorry, I didn't quite catch that. Would you like to continue with the form or switch to your question?",
                route="form_confirmation_clarify",
                options=[
                    {"label": "✅ Yes, continue", "action": "CONTINUE_FORM"},
                    {"label": "❌ No, switch topic", "action": "SWITCH_TOPIC"}
                ],
                is_complete=False
            )

    
    # Initialize Decision Results
    faq_result: Optional[DecisionResult] = None
    ml_result: Optional[DecisionResult] = None
    ml_result_dict: Dict = {}
    llm_result: Optional[DecisionResult] = None
    
    # 🏁 DUAL PROCESSING: FAQ INTERRUPTION PREFIX
    interruption_prefix = ""
    
    # STEP 0a: FAQ Interruption check (Anti-Gravity Refinement)
    # We check for FAQ early so we can prepend it to the form/LLM response
    interrupt_threshold = 0.85 
    faq_chk_node = FAQDecisionNode()
    faq_chk_result = faq_chk_node.execute(context, threshold=interrupt_threshold)
    
    if faq_chk_result.selected:
        logger.info(f"🛑 FAQ INTERRUPTION DETECTED: Prepending and continuing flow.")
        faq_result = faq_chk_result
        interruption_prefix = faq_chk_result.response_data.get('answer', '')

    # STEP 0b: Try ML classifier (threshold 0.8) for early lead detection
    ml_threshold = ML_CONFIDENCE_THRESHOLD or 0.8
    ml_node = MLDecisionNode()
    ml_result, ml_result_dict = ml_node.execute(context, threshold=ml_threshold)

    # 🔒 INTENT-BASED ROUTING LOGIC (New Design)
    # Classify Intent and Score early to prioritize LEAD over FAQ
    intent_type, intent_score = classify_intent_and_score(
        req.message, 
        ml_result, 
        faq_chk_result, 
        interrupt_threshold
    )
    
    logger.info(f"🧠 INTENT CLASSIFICATION: Type={intent_type}, Score={intent_score:.1f}")
    
    # CASE B: LEAD Intent (Early Check)
    if intent_type == "LEAD":
        # Check if score >= 95 (HARD TRIGGER)
        if intent_score >= 95:
            logger.info("🔥 LEAD Intent HIGH (Score >= 95). Triggering Form.")
            
            # Map to correct intent for form building
            # If original prediction wasn't form or failed, force it
            if not ml_result.intent or ml_result.intent not in FORM_INTENT_TO_SET:
                # Default to general lead form
                ml_result.intent = "lead_form" 
            
            if ml_result.confidence is None:
                ml_result.confidence = 0.95 # Assume high confidence for hard trigger
                
            decision = route_decision(ml_result.intent, ml_result.confidence)
            question_set = FORM_INTENT_TO_SET.get(decision.intent) or "lead_questions"
            
            form_ml_response = build_ml_form_response(
                context, ml_result, build_routing_metadata(
                    ml_result=ml_result,
                    selected_route="form_ml",
                    route_reason=f"High Intent Lead Trigger (Score {intent_score:.1f} >= 95)"
                ),
                question_set
            )
            if interruption_prefix:
                form_ml_response.response = f"{interruption_prefix}\n\n{form_ml_response.response}"
                form_ml_response.route = "faq_interruption"
            return form_ml_response
            
        # Check if score >= 70 (SOFT CTA) - this will fall through to LLM later
        elif intent_score >= 70:
            logger.info("💡 LEAD Intent MEDIUM (70 <= Score < 95). Serving Info + Soft CTA.")
            if context.metadata is None: context.metadata = {}
            context.metadata["force_soft_cta"] = True
            context.metadata["soft_cta_text"] = "Would you like help with the admission process?"
            context.metadata["soft_cta_action"] = "START_LEAD_FORM"

    # STEP 0c: FAQ Early Return Logic (only if not hard lead trigger)
    if faq_chk_result.selected:
        # 🔒 ANTI-GRAVITY: RESTRICTED LLM USAGE FOR FAQ
        # If an FAQ is hit and no lead form or job browse flow is active, return the static answer directly.
        if not context.active_flow:
            logger.info("🎯 FAQ hit and no active flow. Returning static answer directly.")
            # Use original faq_threshold from later logic or use interrupt_threshold
            routing_metadata = build_routing_metadata(
                faq_result=faq_result,
                selected_route="faq_semantic",
                route_reason=f"FAQ semantic match ({faq_result.confidence:.3f} >= {interrupt_threshold}) with no active flow",
            )
            from hybrid_orchestrator.response_builder import build_faq_response
            return build_faq_response(context, faq_result, routing_metadata)

    # STEP 0b: Check for active form or job browse flow (short-circuit)
    form_node = FormContinuationNode()
    form_result = form_node.execute(context)
    
    if form_result and form_result.selected:

        
        # 🔴 Check for form escape confirmation (off-topic detection)
        if form_result.route == "form_escape_confirmation":
            logger.info(f"🤔 [LEGACY] Form escape confirmation requested")
            set_mode(req.session_id, "FORM_CONFIRMATION", trigger="off_topic_detected")
            
            return ConversationResponse(
                session_id=req.session_id,
                response="I noticed you asked an unrelated question. Would you like to continue with the form first, or would you like to switch to that question now?",
                route="form_escape_confirmation",
                options=[
                    {"label": "Yes, continue form", "action": "CONTINUE_FORM"},
                    {"label": "No, switch topic", "action": "SWITCH_TOPIC"}
                ],
                is_complete=False
            )

        # Normal form continuation (only if NOT escaping to answer off-topic)
        if not (context.metadata and context.metadata.get("escaped_from_job_form")):
            form_response = build_form_continue_response(context, form_result)
            if interruption_prefix:
                form_response.response = f"{interruption_prefix}\n\n{form_response.response}"
                form_response.route = "faq_interruption"
            return form_response
    

        
    # START RAG TASK IN PARALLEL
    # We start this early so it runs while we check FAQ/ML
    rag_task = asyncio.create_task(_search_rag(req.message, domain))
    
    # Initialize decision results
    faq_result: Optional[DecisionResult] = None
    ml_result: Optional[DecisionResult] = None
    ml_result_dict: Dict = {}
    llm_result: Optional[DecisionResult] = None
    
    # STEP 1: Get FAQ signal (reusing early check if available)
    if not faq_result:
        faq_threshold = FAQ_CONFIDENCE_THRESHOLD or 0.8
        faq_node = FAQDecisionNode()
        faq_result = faq_node.execute(context, threshold=faq_threshold)
        if faq_result.selected:
             interruption_prefix = faq_result.response_data.get('answer', '')
    
    # STEP 2: Try ML classifier (already run early)
    # We preserve ml_result and ml_result_dict from the early check
    
    # -----------------------------------------------------------------------
    # END INTENT LOGIC - Fall through to original flows if no hard return
    # -----------------------------------------------------------------------
    
    if ml_result and ml_result.selected:
        # ML returned high confidence - route based on intent type
        decision = route_decision(ml_result.intent, ml_result.confidence)
        
        # Build routing metadata
        routing_metadata = build_routing_metadata(
            faq_result=faq_result,
            ml_result=ml_result,
            selected_route=decision.route,
            route_reason=ml_result.reason or f"ML classifier identified '{ml_result.intent}' with high confidence ({ml_result.confidence:.3f} >= {ml_threshold})",
        )
        
        if decision.route == "greeting_ml":
            logger.info("❌ RAG task cancelled: ML classified as greeting")
            rag_task.cancel() # Cancel RAG
            # Handle greeting/gratitude
            return build_ml_greeting_response(context, ml_result, routing_metadata)
        
        elif decision.route == "form_ml":
            # Handle form start
            question_set = FORM_INTENT_TO_SET.get(decision.intent)
            if not question_set:
                logger.warning(
                    "No question set mapped for intent '%s'. Falling back to LLM.",
                    decision.intent
                )
                ml_result.selected = False  # Mark as not selected so LLM handles it
            else:
                logger.info("❌ RAG task cancelled: ML starting form flow")
                rag_task.cancel() # Cancel RAG
                return build_ml_form_response(
                    context, ml_result, routing_metadata, question_set
                )
        

    
    # STEP 3: Fallback to LLM (FAQ and ML both failed or low confidence)
    
    # Await RAG results since we are falling back to LLM
    rag_context = ""
    rag_metadata = {}
    rag_chunks = []
    try:
        rag_context, rag_metadata, rag_chunks = await rag_task
        if rag_context:
            logger.info(f"✅ RAG context available for LLM: {rag_metadata.get('results_count', 0)} chunks from domain={rag_metadata.get('domain')}")
        else:
            logger.info(f"ℹ️ No RAG context available for LLM (domain={rag_metadata.get('domain')})")
    except asyncio.CancelledError:
        logger.debug("RAG task was cancelled (expected behavior)")
        rag_metadata = {"attempted": False, "success": False, "error": "Task cancelled"}
    except Exception as e:
        logger.error(f"❌ Error awaiting RAG task: {e}")
        rag_metadata = {"attempted": True, "success": False, "error": str(e)}

    
    # Get Persona
    persona_service = get_persona_service()
    persona = persona_service.get_persona(domain)
    persona_instructions = persona.instructions if persona else ""
    persona_conditions = persona.conditions if persona else {}
    persona_formatting = persona.formatting_style if persona else {}
    
    llm_node = LLMDecisionNode()
    llm_result = llm_node.execute(
        context,
        ml_result_dict,
        faq_confidence=faq_result.confidence if faq_result else None,
        ml_confidence=ml_result.confidence if ml_result else None,
        rag_context=rag_context,
        persona_instructions=persona_instructions,
        persona_conditions=persona_conditions,
        persona_formatting=persona_formatting
    )
    
    # Build routing metadata for LLM fallback
    faq_threshold = FAQ_CONFIDENCE_THRESHOLD or 0.8
    ml_threshold = ML_CONFIDENCE_THRESHOLD or 0.8
    
    # Extract confidence values for formatting (handle None)
    faq_conf = faq_result.confidence if faq_result and faq_result.confidence is not None else 0.0
    ml_conf = ml_result.confidence if ml_result and ml_result.confidence is not None else 0.0
    
    # Build route reason mentioning RAG usage
    route_reason = llm_result.reason or "LLM fallback"
    if rag_context and rag_metadata.get("results_count", 0) > 0:
        route_reason = f"LLM response using RAG knowledge base ({rag_metadata.get('results_count')} documents from {rag_metadata.get('domain')}): {route_reason}"
    else:
        route_reason += f" (FAQ confidence {faq_conf:.3f} < {faq_threshold}, ML confidence {ml_conf:.3f} < {ml_threshold})"
    
    routing_metadata = build_routing_metadata(
        faq_result=faq_result,
        ml_result=ml_result,
        llm_result=llm_result,
        selected_route=llm_result.route or "llm_fallback",
        route_reason=route_reason,
        rag_metadata=rag_metadata,
    )
    
    # Build and return LLM response
    legacy_response = build_llm_response(
        context,
        llm_result,
        ml_result.intent if ml_result else None,
        ml_result.confidence if ml_result else None,
        routing_metadata,
        rag_chunks=rag_chunks,
    )

    # 🏁 DUAL PROCESSING FINAL WRAPPER (LEGACY)
    if interruption_prefix and legacy_response:
        logger.info("🏁 Prepending FAQ answer to legacy LLM response.")
        if legacy_response.route != "faq_semantic":
            legacy_response.response = f"{interruption_prefix}\n\n{legacy_response.response}"
            legacy_response.route = "faq_interruption"
            
    return legacy_response


# ---------------------------------------------------------------------------
# 🔒 ANTI-GRAVITY: Policy-Driven Process Conversation
# ---------------------------------------------------------------------------

async def _policy_process_conversation(
    req: ConversationRequest, 
    session: Dict
) -> ConversationResponse:
    """
    🔒 ANTI-GRAVITY: Policy-driven routing with conversation mode awareness.
    
    Routing Pipeline:
    1. Escape check (before mode bypass)
    2. Mode gate (FORM_ACTIVE / HANDOFF_ACTIVE bypass)
    3. Collect cheap signals (FAQ, ML, RAG, message analysis)
    4. Policy Router selects single winner
    5. Execute winner (producing legacy response format)
    6. Update session mode + last_route_family
    7. Log policy decision
    
    CRITICAL: This function MUST produce identical response contracts to legacy.
    """
    # Build conversation context
    context = build_context(req, session)
    if not validate_context(context):
        raise ValueError(f"Invalid context for session {req.session_id}")
    
    # Determine domain for RAG and Persona
    # Priority: req.domain > extra.domain > default
    from config.settings import DEFAULT_DOMAIN
    domain = DEFAULT_DOMAIN
    if getattr(req, "domain", None):
        domain = req.domain
    elif req.metadata and req.metadata.extra.get("domain"):
        domain = req.metadata.extra.get("domain")
    
    # Override default/tech_company to DEFAULT_DOMAIN for this deployment
    if domain in ["default", "tech_company"]:
        logger.info(f"🔄 [POLICY] Overriding domain '{domain}' to '{DEFAULT_DOMAIN}' for Tech deployment")
        domain = DEFAULT_DOMAIN
        # Update request metadata
        if req.metadata:
            req.metadata.extra["domain"] = domain
        else:
            from models.hybrid_schemas import ConversationMetadata
            req.metadata = ConversationMetadata(extra={"domain": domain})
    
    logger.info(f"🎯 [POLICY] Resolved domain: {domain}")
    
    # 🔴 EARLY GREETING INTERCEPT (Bypass LLM/ML for simple greetings)
    normalized_query = req.message.strip().lower()
    if normalized_query in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "hii", "heya", "howdy", "hii there", "hello there", "hi there"]:
        logger.info(f"🛑 DIRECT GREETING DETECTED: Bypassing ML/LLM calls for '{normalized_query}'")
        mock_ml_result = DecisionResult(
            route="greeting_ml",
            intent="greeting",
            confidence=1.0,
            selected=True,
            reason="Direct keyword match for greeting"
        )
        routing_metadata = build_routing_metadata(
            selected_route="direct_greeting",
            route_reason=f"Exact match for basic greeting: '{normalized_query}'"
        )
        return build_ml_greeting_response(context, mock_ml_result, routing_metadata)

    # Get current conversation mode
    mode = get_mode(session)
    last_family = get_last_route_family(session)
    
    logger.info(f"🔒 [POLICY] Session {req.session_id}: mode={mode}, last_family={last_family}")
    
    # STEP 0: Message signals (CHEAP - run first)
    message_signals = analyze_message(req.message)
    
    # STEP 1: ESCAPE CHECK (before mode bypass)
    # Allow users to escape from FORM_ACTIVE mode
    if mode == "FORM_ACTIVE":
        if message_signals.wants_cancel:
            # User wants to cancel form - exit form mode
            logger.info(f"🚪 [POLICY] Form escape: cancel detected")
            log_escape_handled(req.session_id, "cancel", mode)
            set_mode(req.session_id, "DEFAULT", trigger="user_cancel")
            # Continue to normal routing (below)
            mode = "DEFAULT"
        elif message_signals.wants_handoff:
            # User wants human during form - initiate handoff
            logger.info(f"🚪 [POLICY] Form escape: handoff requested")
            log_escape_handled(req.session_id, "handoff", mode)
            set_mode(req.session_id, "HANDOFF_PENDING", trigger="user_handoff_request")
            mode = "HANDOFF_PENDING"
            
    if mode == "FORM_CONFIRMATION":
        if message_signals.is_affirmative:
            logger.info(f"✅ [POLICY] Form Confirmation: User said YES. Re-asking current question.")
            set_mode(req.session_id, "FORM_ACTIVE", trigger="user_confirmed_continuation")
            mode = "FORM_ACTIVE"
            # Instead of passing "yes" to form agent, we want to re-display the current question
            # We can do this by letting FormContinuationNode handle a message that doesn't change state
            # but triggers a re-send. Or better, just build a response here if possible.
            # For now, let's just clear the message so it might trigger a re-ask or be ignored
            # Actually, LangGraph agent might need a specific signal to "repeat".
            # Let's try setting message to something that won't be saved as data but triggers continuation.
            req.message = "continue" 
            context.message = "continue"
            # Fall through to FORM_ACTIVE gate below
        elif message_signals.is_negative:
            logger.info(f"🚪 [POLICY] Form Confirmation: User said NO. Escaping form.")
            # Clear form state and end session properly
            from ml_services.langgraph_gateway import end_form_session
            end_form_session(req.session_id)
            upsert_session(req.session_id, {"active_flow": None, "graph_state_id": None})
            context.active_flow = None
            set_mode(req.session_id, "DEFAULT", trigger="user_declined_continuation")
            mode = "DEFAULT"
            
            # 🔄 RE-PROCESS: Get the stored offtopic query and use it as the current message
            original_query = session.get("pending_off_topic_query")
            if original_query:
                logger.info(f"🔄 [POLICY] Re-processing original off-topic query: {original_query}")
                req.message = original_query
                context.message = original_query
                # Clear the pending query from session
                upsert_session(req.session_id, {"pending_off_topic_query": None})
            
            # Fall through to normal routing below
        else:
            logger.info(f"🤔 [POLICY] Form Confirmation: Ambiguous response. Re-prompting.")
            # Return a specialized response asking for clear Yes/No
            return ConversationResponse(
                session_id=req.session_id,
                response="I'm sorry, I didn't quite catch that. Would you like to continue with the form or switch to your question?",
                route="form_confirmation_clarify",
                options=[
                    {"label": "✅ Yes, continue", "action": "CONTINUE_FORM"},
                    {"label": "❌ No, switch topic", "action": "SWITCH_TOPIC"}
                ],
                is_complete=False
            )
    
    # STEP 2: MODE GATE (bypass routing for active flows)
    if mode == "FORM_ACTIVE":
        # Form continuation - delegate to FormContinuationNode
        form_node = FormContinuationNode()
        form_result = form_node.execute(context)
        
        if form_result and form_result.selected:
            # Handle resume/job escapes same as legacy
            if form_result.intent == "resume_upload_escape" and form_result.response_data:
                logger.info(f"📄 [POLICY] Resume escape routing")
                return build_llm_response(
                    context=context,
                    llm_result=form_result,
                    ml_intent=None,
                    ml_confidence=None,
                    routing_metadata=build_routing_metadata(
                        selected_route="resume_escape",
                        route_reason="Resume upload detected - routing to job application",
                    ),
                )
            

            
            # 🔴 Confirmation for off-topic queries
            if form_result.route == "form_escape_confirmation":
                logger.info(f"🤔 [POLICY] Form escape confirmation requested")
                # Set mode to confirmation so next turn handles Yes/No
                set_mode(req.session_id, "FORM_CONFIRMATION", trigger="off_topic_detected")
                
                # Build specialized confirmation response
                return ConversationResponse(
                    session_id=req.session_id,
                    response="I noticed you asked an unrelated question. Would you like to continue with the form first, or would you like to switch to that question now?",
                    route="form_escape_confirmation",
                    options=[
                        {"label": "✅ Yes, continue form", "action": "CONTINUE_FORM"},
                        {"label": "❌ No, switch topic", "action": "SWITCH_TOPIC"}
                    ],
                    is_complete=False
                )
            
            # Normal form continuation
            return build_form_continue_response(context, form_result)
    
    if mode == "HANDOFF_ACTIVE":
        # TODO: Implement handoff handler bypass
        # For now, fall through to normal routing
        logger.info(f"🔄 [POLICY] HANDOFF_ACTIVE - routing to handoff handler (TODO)")
    
    # STEP 3: Check for job browse continuation (legacy compat)

    
    # STEP 4: COLLECT CHEAP SIGNALS (parallel where possible)
    signal_collection_start = time.time()
    
    # Start RAG task in parallel
    rag_task = asyncio.create_task(_search_rag(req.message, domain))
    
    # FAQ signal
    faq_threshold = FAQ_CONFIDENCE_THRESHOLD or 0.8
    faq_node = FAQDecisionNode()
    faq_result = faq_node.execute(context, threshold=faq_threshold)
    faq_candidate = collect_faq_signal(
        {"confidence": faq_result.confidence, "answer": faq_result.response, "timing_ms": 0} if faq_result else None,
        threshold=faq_threshold
    )
    
    # ML signal
    ml_threshold = ML_CONFIDENCE_THRESHOLD or 0.8
    ml_node = MLDecisionNode()
    ml_result, ml_result_dict = ml_node.execute(context, threshold=ml_threshold)
    ml_candidate = collect_ml_signal(
        {"intent": ml_result.intent, "confidence": ml_result.confidence} if ml_result else None,
        threshold=ml_threshold
    )
    
    # Await RAG results
    rag_context = ""
    rag_metadata = {}
    rag_chunks = []
    rag_candidate = None
    rag_quality = None
    
    try:
        rag_context, rag_metadata, rag_chunks = await rag_task
        if rag_chunks:
            rag_candidate, rag_quality = collect_rag_signal(rag_context, rag_metadata, rag_chunks)
        else:
            rag_candidate = collect_clarify_signal(message_signals.is_clarification, is_ambiguous=True)
    except asyncio.CancelledError:
        rag_metadata = {"attempted": False}
    except Exception as e:
        logger.error(f"❌ [POLICY] RAG error: {e}")
        rag_metadata = {"attempted": True, "error": str(e)}
    
    signal_collection_ms = (time.time() - signal_collection_start) * 1000
    
    # Build candidate list
    candidates = []
    if faq_candidate:
        candidates.append(faq_candidate)
    if ml_candidate:
        candidates.append(ml_candidate)
    if rag_candidate:
        candidates.append(rag_candidate)
    
    # Always add CLARIFY and LLM as fallback candidates
    candidates.append(collect_clarify_signal(message_signals.is_clarification))
    candidates.append(collect_llm_signal(
        has_rag_context=bool(rag_context),
        rag_quality_passes=rag_quality.get("passes", False) if rag_quality else False
    ))
    
    # STEP 5: POLICY ROUTER DECISION
    policy_router = get_policy_router()
    decision = policy_router.select(
        candidates=candidates,
        session_mode=mode,
        last_route_family=last_family,
        signals=message_signals,
        rag_quality=rag_quality,
    )
    
    # Log the decision
    log_policy_decision(
        session_id=req.session_id,
        decision=decision,
        message_preview=req.message[:50] if req.message else "",
        extra={"signal_collection_ms": round(signal_collection_ms, 2)}
    )
    
    # STEP 6: UPDATE SESSION STATE
    if decision.mode_after != mode:
        set_mode(req.session_id, decision.mode_after, trigger=f"policy_decision:{decision.winner.family}")
    
    if decision.winner.family != last_family:
        set_last_route_family(req.session_id, decision.winner.family)
    
    # Decrement QA turns if in QA_CONTINUATION
    if mode == "QA_CONTINUATION":
        dec_qa_turns(req.session_id)
    
    # STEP 7: EXECUTE WINNER (producing legacy response format)
    return await _execute_policy_winner(
        decision=decision,
        context=context,
        req=req,
        session=session,
        domain=domain,
        faq_result=faq_result,
        ml_result=ml_result,
        ml_result_dict=ml_result_dict,
        rag_context=rag_context,
        rag_metadata=rag_metadata,
        rag_chunks=rag_chunks,
    )


async def _execute_policy_winner(
    decision: PolicyDecision,
    context,
    req: ConversationRequest,
    session: Dict,
    domain: str,
    faq_result,
    ml_result,
    ml_result_dict: Dict,
    rag_context: str,
    rag_metadata: Dict,
    rag_chunks: List,
) -> ConversationResponse:
    """
    Execute the winning route and produce legacy-compatible response.
    
    Maps internal winner_family to legacy route strings and response builders.
    This is the CRITICAL translation layer between Anti-Gravity and legacy.
    """
    winner = decision.winner
    
    # FINAL RESPONSE HOLDER
    final_response = None
    
    # FAQ winner
    if winner.family == "FAQ" and faq_result and faq_result.selected:
        faq_threshold = FAQ_CONFIDENCE_THRESHOLD or 0.8
        routing_metadata = build_routing_metadata(
            faq_result=faq_result,
            selected_route="faq_semantic",
            route_reason=f"[POLICY] FAQ selected (score={winner.final_score:.3f})",
        )
        final_response = build_faq_response(context, faq_result, routing_metadata)
    
    # ML_ACTION winner - greeting or form
    elif winner.family == "ML_ACTION" and ml_result:
        intent = ml_result.intent
        routing_decision = route_decision(intent, ml_result.confidence)
        
        ml_threshold = ML_CONFIDENCE_THRESHOLD or 0.8
        routing_metadata = build_routing_metadata(
            faq_result=faq_result,
            ml_result=ml_result,
            selected_route=routing_decision.route,
            route_reason=f"[POLICY] ML selected: {intent} (score={winner.final_score:.3f})",
        )
        
        if routing_decision.route == "greeting_ml":
            final_response = build_ml_greeting_response(context, ml_result, routing_metadata)
        
        elif routing_decision.route == "form_ml":
            question_set = FORM_INTENT_TO_SET.get(intent)
            if question_set:
                final_response = build_ml_form_response(context, ml_result, routing_metadata, question_set)
    
    # RAG_QA, LLM, CLARIFY, or FAQ-interruption (handled via LLM logic)
    if not final_response:
        persona_service = get_persona_service()
        persona = persona_service.get_persona(domain)
        persona_instructions = persona.instructions if persona else ""
        persona_conditions = persona.conditions if persona else {}
        persona_formatting = persona.formatting_style if persona else {}
        
        llm_node = LLMDecisionNode()
        llm_result = llm_node.execute(
            context,
            ml_result_dict,
            faq_confidence=faq_result.confidence if faq_result else None,
            ml_confidence=ml_result.confidence if ml_result else None,
            rag_context=rag_context,
            persona_instructions=persona_instructions,
            persona_conditions=persona_conditions,
            persona_formatting=persona_formatting
        )
        
        # Determine route and reason based on winner family
        route_name = "llm_fallback"
        route_reason = f"[POLICY] {winner.family} selected (score={winner.final_score:.3f})"
        
        if winner.family == "RAG_QA" and rag_context:
            route_name = "rag_qa"
            route_reason = f"[POLICY] RAG-grounded response ({rag_metadata.get('results_count', 0)} chunks)"
        elif winner.family == "CLARIFY":
            route_name = "clarify"
            route_reason = f"[POLICY] Clarification requested - {decision.reason_codes}"
        elif winner.family == "FAQ": # FAQ answer for Dual-Processing
            route_name = "faq_semantic"
            route_reason = f"[POLICY] FAQ selected (score={winner.final_score:.3f})"
            llm_result.assistant_response = faq_result.response
            llm_result.selected = True
        
        routing_metadata = build_routing_metadata(
            faq_result=faq_result,
            ml_result=ml_result,
            llm_result=llm_result,
            selected_route=route_name,
            route_reason=route_reason,
            rag_metadata=rag_metadata,
        )
        
        final_response = build_llm_response(
            context,
            llm_result,
            ml_result.intent if ml_result else None,
            ml_result.confidence if ml_result else None,
            routing_metadata,
            rag_chunks=rag_chunks,
        )

    # 🏁 DUAL PROCESSING FINAL WRAPPER
    if interruption_prefix and final_response:
        logger.info("🏁 Prepending FAQ answer to final decision response.")
        # Ensure we don't repeat if FAQ was already the main response
        if final_response.route != "faq_semantic":
            final_response.response = f"{interruption_prefix}\n\n{final_response.response}"
            final_response.route = "faq_interruption"
        
    return final_response
