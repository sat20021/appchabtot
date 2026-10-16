"""
State Policy - Explicit State Machine Transitions

Defines all valid session states and transitions.
State changes MUST be explicit and validated - no side-channel mutations.

This is the SINGLE SOURCE OF TRUTH for state transitions.
"""

from enum import Enum
from typing import Dict, List, Set, Optional
from dataclasses import dataclass


class SessionState(str, Enum):
    """
    All possible session states.
    
    Transitions between states must be EXPLICIT and VALIDATED.
    No implicit state changes allowed.
    """
    # Initial/default state
    FREE_CHAT = "free_chat"
    
    # Consent flow states
    CONSENT_REQUESTED = "consent_requested"
    AWAITING_CONSENT_RESPONSE = "awaiting_consent_response"
    
    # Form states
    PII_FORM_ACTIVE = "pii_form_active"
    
    # Handoff states
    HUMAN_HANDOFF = "human_handoff"
    
    # Terminal states
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class StateTransition:
    """Record of a state transition."""
    from_state: SessionState
    to_state: SessionState
    trigger: str  # What caused the transition
    timestamp: str = ""


class StatePolicy:
    """
    Policy governing state transitions.
    
    Rules:
    - All transitions must be explicitly defined
    - Invalid transitions are BLOCKED (not auto-corrected)
    - State changes are logged for audit
    """
    
    # Valid state transitions: from_state → [allowed_to_states]
    VALID_TRANSITIONS: Dict[SessionState, List[SessionState]] = {
        SessionState.FREE_CHAT: [
            SessionState.CONSENT_REQUESTED,
            SessionState.HUMAN_HANDOFF,
        ],
        
        SessionState.CONSENT_REQUESTED: [
            SessionState.FREE_CHAT,  # User declined
            SessionState.PII_FORM_ACTIVE,  # PII consent granted
            SessionState.AWAITING_CONSENT_RESPONSE,
        ],
        
        SessionState.AWAITING_CONSENT_RESPONSE: [
            SessionState.FREE_CHAT,  # User declined
            SessionState.PII_FORM_ACTIVE,
        ],
        
        SessionState.PII_FORM_ACTIVE: [
            SessionState.FREE_CHAT,  # Form cancelled/completed
            SessionState.COMPLETED,
        ],
        

        
        SessionState.HUMAN_HANDOFF: [
            SessionState.FREE_CHAT,  # Handoff ended
            SessionState.COMPLETED,
        ],
        
        # Terminal states - no transitions out
        SessionState.COMPLETED: [],
        SessionState.ABANDONED: [],
    }
    
    # States where forms are active
    FORM_ACTIVE_STATES: Set[SessionState] = {
        SessionState.PII_FORM_ACTIVE,
    }
    
    # States that require special handling
    BLOCKING_STATES: Set[SessionState] = {
        SessionState.HUMAN_HANDOFF,
        SessionState.CONSENT_REQUESTED,
        SessionState.AWAITING_CONSENT_RESPONSE,
    }
    
    @classmethod
    def can_transition(
        cls, 
        from_state: SessionState, 
        to_state: SessionState
    ) -> bool:
        """
        Check if a state transition is valid.
        
        Args:
            from_state: Current session state
            to_state: Proposed new state
        
        Returns:
            True if transition is allowed
        """
        allowed = cls.VALID_TRANSITIONS.get(from_state, [])
        return to_state in allowed
    
    @classmethod
    def get_allowed_transitions(cls, from_state: SessionState) -> List[SessionState]:
        """Get list of valid states to transition to."""
        return cls.VALID_TRANSITIONS.get(from_state, [])
    
    @classmethod
    def is_form_active(cls, state: SessionState) -> bool:
        """Check if current state indicates an active form."""
        return state in cls.FORM_ACTIVE_STATES
    
    @classmethod
    def is_blocking_state(cls, state: SessionState) -> bool:
        """
        Check if current state blocks normal conversation routing.
        
        These states require special handling (consent flow, handoff, etc.)
        """
        return state in cls.BLOCKING_STATES
    
    @classmethod
    def is_terminal(cls, state: SessionState) -> bool:
        """Check if this is a terminal state (no transitions out)."""
        return len(cls.VALID_TRANSITIONS.get(state, [])) == 0
    
    @classmethod
    def validate_and_transition(
        cls,
        from_state: SessionState,
        to_state: SessionState,
        trigger: str = "unknown"
    ) -> tuple[bool, Optional[StateTransition]]:
        """
        Validate and create a state transition.
        
        Args:
            from_state: Current state
            to_state: Proposed state
            trigger: What caused this transition
        
        Returns:
            Tuple of (success, StateTransition if success else None)
        """
        from datetime import datetime
        
        if not cls.can_transition(from_state, to_state):
            return False, None
        
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            timestamp=datetime.utcnow().isoformat()
        )
        return True, transition
    
    @classmethod
    def get_state_for_action(cls, action: str) -> Optional[SessionState]:
        """
        Get the session state that should be set for an action.
        
        Returns None if action doesn't change state.
        """
        action_to_state = {
            "START_LEAD_FORM": SessionState.PII_FORM_ACTIVE,
            "HANDOFF_TO_HUMAN": SessionState.HUMAN_HANDOFF,
            "REQUEST_CONSENT": SessionState.CONSENT_REQUESTED,
        }
        return action_to_state.get(action)
