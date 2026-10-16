"""
Anti-Gravity Policy Router - Routing Logger

Structured logging for all routing decisions.
Every turn MUST log the decision with candidates and reasons.

🔒 ANTI-GRAVITY PRINCIPLE:
   - All routing decisions must be explainable via logs
   - Logs are structured JSON for analysis
"""
import json
import logging
from typing import Dict, Any, Optional

from .types import PolicyDecision

logger = logging.getLogger(__name__)

# Separate logger for routing events (can be configured independently)
routing_logger = logging.getLogger("anti_gravity.routing")


def log_policy_decision(
    session_id: str,
    decision: PolicyDecision,
    message_preview: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a routing decision with full context.
    
    Args:
        session_id: Session identifier
        decision: PolicyDecision from PolicyRouter.select()
        message_preview: First 50 chars of user message (for debugging)
        extra: Additional context to log
    """
    log_data = {
        "event": "ROUTING_DECISION",
        "session_id": session_id,
        "message_preview": message_preview[:50] if message_preview else "",
        **decision.to_log_dict(),
        **(extra or {}),
    }
    
    # Log as structured JSON
    routing_logger.info(json.dumps(log_data))
    
    # Also log human-readable summary to main logger
    logger.info(
        f"🔀 ROUTE: {decision.winner.family} "
        f"(score={decision.winner.final_score:.3f}) "
        f"mode: {decision.mode_before} → {decision.mode_after} "
        f"session={session_id[:8]}..."
    )


def log_mode_transition(
    session_id: str,
    mode_before: str,
    mode_after: str,
    trigger: str,
) -> None:
    """
    Log a conversation mode transition.
    
    Args:
        session_id: Session identifier
        mode_before: Previous mode
        mode_after: New mode
        trigger: What triggered the transition
    """
    log_data = {
        "event": "MODE_TRANSITION",
        "session_id": session_id,
        "mode_before": mode_before,
        "mode_after": mode_after,
        "trigger": trigger,
    }
    
    routing_logger.info(json.dumps(log_data))
    logger.info(f"🔄 MODE: {mode_before} → {mode_after} (trigger: {trigger}) session={session_id[:8]}...")


def log_governance_override(
    session_id: str,
    proposed_action: str,
    final_action: str,
    reason: str,
) -> None:
    """
    Log a governance override (when governance changes the action).
    
    Args:
        session_id: Session identifier
        proposed_action: Action proposed by Policy Router
        final_action: Action after Governance enforcement
        reason: Reason for the override
    """
    log_data = {
        "event": "GOVERNANCE_OVERRIDE",
        "session_id": session_id,
        "proposed_action": proposed_action,
        "final_action": final_action,
        "reason": reason,
    }
    
    routing_logger.warning(json.dumps(log_data))
    logger.warning(
        f"🔒 GOVERNANCE OVERRIDE: {proposed_action} → {final_action} "
        f"reason={reason} session={session_id[:8]}..."
    )


def log_signal_collection(
    session_id: str,
    faq_confidence: Optional[float],
    ml_intent: Optional[str],
    ml_confidence: Optional[float],
    rag_passes: bool,
    rag_top_score: Optional[float],
    timing_ms: float,
) -> None:
    """
    Log signal collection results (before policy decision).
    
    Args:
        session_id: Session identifier
        faq_confidence: FAQ confidence score
        ml_intent: ML classified intent
        ml_confidence: ML confidence score
        rag_passes: Whether RAG quality gate passed
        rag_top_score: Top RAG score
        timing_ms: Total time for signal collection
    """
    log_data = {
        "event": "SIGNAL_COLLECTION",
        "session_id": session_id,
        "faq_confidence": round(faq_confidence, 4) if faq_confidence else None,
        "ml_intent": ml_intent,
        "ml_confidence": round(ml_confidence, 4) if ml_confidence else None,
        "rag_passes": rag_passes,
        "rag_top_score": round(rag_top_score, 4) if rag_top_score else None,
        "timing_ms": round(timing_ms, 2),
    }
    
    routing_logger.debug(json.dumps(log_data))


def log_escape_handled(
    session_id: str,
    escape_type: str,  # "cancel" | "handoff"
    from_mode: str,
) -> None:
    """
    Log when an escape intent is handled (cancel/handoff from form).
    
    Args:
        session_id: Session identifier
        escape_type: Type of escape (cancel or handoff)
        from_mode: Mode being escaped from
    """
    log_data = {
        "event": "ESCAPE_HANDLED",
        "session_id": session_id,
        "escape_type": escape_type,
        "from_mode": from_mode,
    }
    
    routing_logger.info(json.dumps(log_data))
    logger.info(f"🚪 ESCAPE: {escape_type} from {from_mode} session={session_id[:8]}...")
