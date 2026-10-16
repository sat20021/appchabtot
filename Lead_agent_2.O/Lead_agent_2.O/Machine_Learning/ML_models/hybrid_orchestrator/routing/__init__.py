"""
Anti-Gravity Policy Router - Routing Module

This module contains the policy-driven routing components:
- types.py: Core data structures (RouteCandidate, PolicyDecision, etc.)
- message_analyzer.py: Cheap signal extraction
- rag_quality_gate.py: RAG quality enforcement
- policy_router.py: Main policy routing logic
- signal_collectors.py: Adapters for existing nodes
- routing_logger.py: Structured logging

🔒 ANTI-GRAVITY ARCHITECTURE:
   - Signals (FAQ, ML, RAG) are collected first
   - PolicyRouter.select() arbitrates to choose winner
   - Governance enforces action permissions
   - Execution produces legacy response format
"""
from .types import (
    RouteCandidate,
    PolicyDecision,
    MessageSignals,
    RAGQualityMetrics,
    RouteFamily,
    ConversationMode,
    VALID_ROUTE_FAMILIES,
    VALID_CONVERSATION_MODES,
)

from .message_analyzer import (
    analyze_message,
    detect_escape_intent,
)

from .rag_quality_gate import (
    RAGQualityGate,
    get_rag_quality_gate,
)

from .policy_router import (
    PolicyRouter,
    get_policy_router,
)

from .signal_collectors import (
    collect_faq_signal,
    collect_ml_signal,
    collect_rag_signal,
    collect_handoff_signal,
    collect_clarify_signal,
    collect_llm_signal,
)

from .routing_logger import (
    log_policy_decision,
    log_mode_transition,
    log_governance_override,
    log_signal_collection,
    log_escape_handled,
)

__all__ = [
    # Types
    "RouteCandidate",
    "PolicyDecision",
    "MessageSignals",
    "RAGQualityMetrics",
    "RouteFamily",
    "ConversationMode",
    "VALID_ROUTE_FAMILIES",
    "VALID_CONVERSATION_MODES",
    # Message Analysis
    "analyze_message",
    "detect_escape_intent",
    # RAG Quality
    "RAGQualityGate",
    "get_rag_quality_gate",
    # Policy Router
    "PolicyRouter",
    "get_policy_router",
    # Signal Collectors
    "collect_faq_signal",
    "collect_ml_signal",
    "collect_rag_signal",
    "collect_handoff_signal",
    "collect_clarify_signal",
    "collect_llm_signal",
    # Logging
    "log_policy_decision",
    "log_mode_transition",
    "log_governance_override",
    "log_signal_collection",
    "log_escape_handled",
]
