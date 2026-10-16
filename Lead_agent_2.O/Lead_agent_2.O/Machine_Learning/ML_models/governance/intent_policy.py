"""
Intent Policy - Authoritative Intent Definitions

Defines the mapping between ML intents and allowed actions.
ML intent is AUTHORITATIVE - if ML says greeting, only greeting actions are allowed.

This is the SINGLE SOURCE OF TRUTH for intent → action mappings.
"""

from typing import Dict, List, Set, Optional
from governance.action_policy import ActionType


class IntentPolicy:
    """
    Policy governing intent-to-action mappings.
    
    ML intent OVERRIDES LLM suggestions. This is non-negotiable.
    """
    
    # Authoritative mapping: intent → allowed actions
    INTENT_TO_ALLOWED_ACTIONS: Dict[str, List[ActionType]] = {
        # Greeting intents - only respond
        "greeting": [ActionType.RESPOND_MESSAGE],
        "gratitude": [ActionType.RESPOND_MESSAGE],
        "farewell": [ActionType.RESPOND_MESSAGE],
        
        # Lead collection - can start form OR respond
        "lead_collection": [ActionType.START_LEAD_FORM, ActionType.RESPOND_MESSAGE],
        "lead_questions": [ActionType.START_LEAD_FORM, ActionType.RESPOND_MESSAGE],
        

        
        # General
        "general_query": [ActionType.RESPOND_MESSAGE, ActionType.CLARIFY],
        "help_request": [ActionType.RESPOND_MESSAGE, ActionType.HANDOFF_TO_HUMAN],
        
        # Fallback for unknown intents - ALWAYS safe
        "unknown": [ActionType.RESPOND_MESSAGE],
    }
    
    # Confidence threshold below which we treat as "unknown"
    LOW_CONFIDENCE_THRESHOLD: float = 0.5
    
    # Confidence threshold above which ML intent is authoritative
    HIGH_CONFIDENCE_THRESHOLD: float = 0.8
    
    @classmethod
    def get_allowed_actions(cls, intent: str) -> List[ActionType]:
        """Get list of allowed actions for an intent."""
        return cls.INTENT_TO_ALLOWED_ACTIONS.get(
            intent.lower(), 
            cls.INTENT_TO_ALLOWED_ACTIONS["unknown"]
        )
    
    @classmethod
    def is_action_allowed(cls, intent: str, action: str) -> bool:
        """
        Check if an action is allowed for a given intent.
        
        Args:
            intent: ML classified intent
            action: Proposed action (from LLM or elsewhere)
        
        Returns:
            True if action is allowed for this intent
        """
        try:
            action_type = ActionType(action)
            allowed = cls.get_allowed_actions(intent)
            return action_type in allowed
        except ValueError:
            return False
    
    @classmethod
    def is_form_intent(cls, intent: str) -> bool:
        """Check if this intent should trigger a form flow."""
        form_intents = {
            "lead_collection",
            "lead_questions", 
        }
        return intent.lower() in form_intents
    
    @classmethod
    def get_safe_action_for_intent(cls, intent: str) -> ActionType:
        """
        Get the safest allowed action for an intent.
        
        Always returns RESPOND_MESSAGE if available (least invasive).
        """
        allowed = cls.get_allowed_actions(intent)
        if ActionType.RESPOND_MESSAGE in allowed:
            return ActionType.RESPOND_MESSAGE
        return allowed[0] if allowed else ActionType.RESPOND_MESSAGE
    
    @classmethod
    def should_trust_ml_intent(cls, confidence: float) -> bool:
        """
        Determine if ML confidence is high enough to be authoritative.
        """
        return confidence >= cls.HIGH_CONFIDENCE_THRESHOLD
    
    @classmethod
    def is_low_confidence(cls, confidence: float) -> bool:
        """
        Determine if ML confidence is too low to trust.
        """
        return confidence < cls.LOW_CONFIDENCE_THRESHOLD
