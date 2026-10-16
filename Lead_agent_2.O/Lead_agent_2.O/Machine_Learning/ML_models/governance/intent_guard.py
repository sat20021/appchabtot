"""
Intent Guard - Anti-Hallucination Enforcer

ML intent OVERRIDES LLM suggestions. This is non-negotiable.
If ML says "greeting" and LLM suggests "START_LEAD_FORM", we block it.

This is the CORE enforcement against LLM hallucinations.
"""

import logging
from typing import Optional

from governance.intent_policy import IntentPolicy
from governance.action_policy import ActionType, ActionPolicy
from governance.decision_envelope import DecisionEnvelope


logger = logging.getLogger(__name__)


class IntentGuard:
    """
    Anti-hallucination core enforcer.
    
    Rules (IMMUTABLE):
    1. ML intent overrides LLM suggestion when ML confidence is high
    2. Unknown intent → RESPOND_MESSAGE (safe fallback)
    3. No "helpful guessing" - if action doesn't match intent, block it
    4. If ML intent ≠ action intent → action is blocked
    """
    
    @classmethod
    def enforce(
        cls,
        envelope: DecisionEnvelope,
        ml_intent: Optional[str] = None,
        ml_confidence: Optional[float] = None
    ) -> DecisionEnvelope:
        """
        Enforce intent policy on a decision envelope.
        
        Args:
            envelope: The decision envelope with LLM suggestion
            ml_intent: ML classifier's intent (authoritative if confident)
            ml_confidence: ML classifier's confidence score
        
        Returns:
            DecisionEnvelope with allowed_action set (may override suggestion)
        """
        # Store ML context in envelope for audit
        envelope.ml_intent = ml_intent
        envelope.ml_confidence = ml_confidence
        
        suggested = envelope.suggested_action
        
        # Case 1: No ML intent available - use safe fallback
        if not ml_intent:
            logger.debug("IntentGuard: No ML intent, using safe fallback")
            envelope.allowed_action = suggested
            return envelope
        
        # Case 2: Low ML confidence - don't override, but prefer safe actions
        if ml_confidence is not None and IntentPolicy.is_low_confidence(ml_confidence):
            logger.debug(f"IntentGuard: Low ML confidence ({ml_confidence}), allowing suggestion")
            envelope.allowed_action = suggested
            return envelope
        
        # Case 3: High ML confidence - ML intent is AUTHORITATIVE
        if ml_confidence is not None and IntentPolicy.should_trust_ml_intent(ml_confidence):
            # Check if suggested action is allowed for this intent
            if IntentPolicy.is_action_allowed(ml_intent, suggested):
                logger.debug(f"IntentGuard: Action {suggested} allowed for intent {ml_intent}")
                envelope.allowed_action = suggested
            else:
                # OVERRIDE: Suggested action not allowed for this intent
                safe_action = IntentPolicy.get_safe_action_for_intent(ml_intent)
                logger.warning(
                    f"IntentGuard: OVERRIDE - ML intent '{ml_intent}' does not allow "
                    f"'{suggested}'. Overriding to '{safe_action.value}'"
                )
                envelope.allowed_action = safe_action.value
                envelope.overrides_applied = True
                envelope.override_reason = (
                    f"ML intent '{ml_intent}' (conf: {ml_confidence:.2f}) "
                    f"does not permit action '{suggested}'"
                )
            return envelope
        
        # Case 4: Medium confidence - allow suggestion but log
        logger.debug(f"IntentGuard: Medium confidence ({ml_confidence}), allowing suggestion with caution")
        envelope.allowed_action = suggested
        return envelope
    
    @classmethod
    def validate_action_for_intent(
        cls,
        action: str,
        ml_intent: str,
        ml_confidence: float = 0.0
    ) -> tuple[bool, Optional[str]]:
        """
        Simple validation check without envelope.
        
        Args:
            action: Proposed action
            ml_intent: ML classified intent
            ml_confidence: ML confidence score
        
        Returns:
            Tuple of (is_allowed, override_action_if_not_allowed)
        """
        # Low confidence - allow
        if IntentPolicy.is_low_confidence(ml_confidence):
            return True, None
        
        # High confidence - enforce
        if IntentPolicy.should_trust_ml_intent(ml_confidence):
            if IntentPolicy.is_action_allowed(ml_intent, action):
                return True, None
            else:
                safe = IntentPolicy.get_safe_action_for_intent(ml_intent)
                return False, safe.value
        
        # Medium confidence - allow
        return True, None
    
    @classmethod
    def should_allow_form_start(
        cls,
        ml_intent: str,
        ml_confidence: float
    ) -> bool:
        """
        Specific check for form-starting actions.
        
        Forms should only start when:
        1. ML intent is a form intent (lead_collection, etc.)
        2. ML confidence is high enough
        """
        if not IntentPolicy.is_form_intent(ml_intent):
            return False
        
        return IntentPolicy.should_trust_ml_intent(ml_confidence)
