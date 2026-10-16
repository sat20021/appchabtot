"""
Audit Logger - Decision Audit Trail

Logs every decision for compliance and debugging.
If audited, you can reconstruct EVERY decision.

This is critical for legal compliance and debugging hallucinations.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional, Any

from governance.decision_envelope import DecisionEnvelope


logger = logging.getLogger("governance.audit")


class AuditLogger:
    """
    Decision audit trail logger.
    
    Logs:
    - ML intent and confidence
    - LLM suggestion
    - Final action (after all overrides)
    - Consent state
    - Any overrides applied
    """
    
    @classmethod
    def log_decision(
        cls,
        envelope: DecisionEnvelope,
        session_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a decision for audit trail.
        
        Args:
            envelope: The decision envelope to log
            session_id: Session identifier
            additional_context: Any additional context to log
        """
        audit_record = {
            "event": "decision_audit",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id or envelope.session_id,
            
            # ML context
            "ml_intent": envelope.ml_intent,
            "ml_confidence": envelope.ml_confidence,
            
            # LLM suggestion
            "llm_suggested_action": envelope.suggested_action,
            "llm_reason": envelope.llm_reason,
            
            # Final decision
            "final_action": envelope.get_final_action(),
            
            # Enforcement
            "overrides_applied": envelope.overrides_applied,
            "override_reason": envelope.override_reason,
            
            # Consent
            "consent_prompts": envelope.consent_prompts,
            
            # Additional context
            "context": additional_context or {}
        }
        
        # Log as JSON for structured logging
        logger.info(json.dumps(audit_record))
    
    @classmethod
    def log_consent_change(
        cls,
        session_id: str,
        scope: str,
        granted: bool,
        source: str = "user_explicit"
    ) -> None:
        """
        Log a consent change event.
        
        Args:
            session_id: Session identifier
            scope: Consent scope that changed
            granted: Whether consent was granted or revoked
            source: How the consent change was triggered
        """
        audit_record = {
            "event": "consent_change",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "scope": scope,
            "granted": granted,
            "source": source
        }
        
        logger.info(json.dumps(audit_record))
    
    @classmethod
    def log_state_transition(
        cls,
        session_id: str,
        from_state: str,
        to_state: str,
        trigger: str,
        allowed: bool
    ) -> None:
        """
        Log a state transition attempt.
        
        Args:
            session_id: Session identifier
            from_state: Previous state
            to_state: Attempted new state
            trigger: What triggered the transition
            allowed: Whether transition was allowed
        """
        audit_record = {
            "event": "state_transition",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "from_state": from_state,
            "to_state": to_state,
            "trigger": trigger,
            "allowed": allowed
        }
        
        logger.info(json.dumps(audit_record))
    
    @classmethod
    def log_intent_override(
        cls,
        session_id: str,
        ml_intent: str,
        ml_confidence: float,
        suggested_action: str,
        final_action: str,
        reason: str
    ) -> None:
        """
        Log when Intent Guard overrides an LLM suggestion.
        
        This is critical for debugging hallucinations.
        """
        audit_record = {
            "event": "intent_override",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "ml_intent": ml_intent,
            "ml_confidence": ml_confidence,
            "suggested_action": suggested_action,
            "final_action": final_action,
            "reason": reason
        }
        
        # Log at WARNING level since this indicates LLM tried to do something wrong
        logger.warning(json.dumps(audit_record))
    
    @classmethod
    def log_form_start(
        cls,
        session_id: str,
        form_type: str,
        consent_granted: bool,
        ml_intent: str,
        ml_confidence: float
    ) -> None:
        """
        Log when a form is started.
        
        Useful for tracking form conversion and consent compliance.
        """
        audit_record = {
            "event": "form_start",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "form_type": form_type,
            "consent_granted": consent_granted,
            "ml_intent": ml_intent,
            "ml_confidence": ml_confidence
        }
        
        logger.info(json.dumps(audit_record))
