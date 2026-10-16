"""
Action Policy - Canonical Action Registry

Defines all allowed actions in the system. No other file may define new actions.
This is the SINGLE SOURCE OF TRUTH for what actions exist.
"""

from enum import Enum
from typing import Set


class ActionType(str, Enum):
    """
    All possible actions in the system.
    
    LLMs may SUGGEST these, but only code can EXECUTE them.
    """
    # Response actions
    RESPOND_MESSAGE = "RESPOND_MESSAGE"
    CLARIFY = "CLARIFY"
    
    # Form actions (require consent)
    START_LEAD_FORM = "START_LEAD_FORM"
    CONTINUE_FORM = "CONTINUE_FORM"
    
    # Handoff actions
    HANDOFF_TO_HUMAN = "HANDOFF_TO_HUMAN"
    
    # Consent actions (system-generated, not LLM-generated)
    REQUEST_CONSENT = "REQUEST_CONSENT"
    
    # Internal actions
    NO_ACTION = "NO_ACTION"


class ActionPolicy:
    """
    Policy governing action usage and constraints.
    """
    
    # Actions that LLM is allowed to suggest
    LLM_SUGGESTABLE_ACTIONS: Set[ActionType] = {
        ActionType.RESPOND_MESSAGE,
        ActionType.CLARIFY,
        ActionType.START_LEAD_FORM,
        ActionType.HANDOFF_TO_HUMAN,
    }
    
    # Actions that require user consent before execution
    CONSENT_REQUIRED_ACTIONS: Set[ActionType] = {
        ActionType.START_LEAD_FORM,
    }
    
    # Safe fallback action when in doubt
    SAFE_FALLBACK = ActionType.RESPOND_MESSAGE
    
    @classmethod
    def is_llm_suggestable(cls, action: str) -> bool:
        """Check if LLM is allowed to suggest this action."""
        try:
            action_type = ActionType(action)
            return action_type in cls.LLM_SUGGESTABLE_ACTIONS
        except ValueError:
            return False
    
    @classmethod
    def requires_consent(cls, action: str) -> bool:
        """Check if this action requires user consent."""
        try:
            action_type = ActionType(action)
            return action_type in cls.CONSENT_REQUIRED_ACTIONS
        except ValueError:
            return False
    
    @classmethod
    def normalize_action(cls, action: str) -> ActionType:
        """
        Normalize action string to ActionType enum.
        Returns SAFE_FALLBACK for unknown actions.
        """
        try:
            return ActionType(action)
        except ValueError:
            return cls.SAFE_FALLBACK
