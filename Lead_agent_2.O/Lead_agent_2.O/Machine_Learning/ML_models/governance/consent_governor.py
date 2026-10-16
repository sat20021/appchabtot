"""
Consent Governor - DISABLED

Consent enforcement has been disabled. All actions are allowed without consent checks.
This file is kept for backward compatibility with existing imports.
"""

import logging
from typing import Dict, List

from governance.action_policy import ActionPolicy, ActionType
from governance.decision_envelope import DecisionEnvelope


logger = logging.getLogger(__name__)


class ConsentGovernor:
    """
    Consent enforcement governor (DISABLED).
    
    All methods now bypass consent checks and allow all actions.
    """
    
    @classmethod
    def check(
        cls,
        envelope: DecisionEnvelope,
        session_consent_state: Dict[str, bool]
    ) -> DecisionEnvelope:
        """
        Check if action is allowed - ALWAYS ALLOWS (consent disabled).
        
        Args:
            envelope: Decision envelope with allowed_action set
            session_consent_state: Dict of {scope_value: granted_bool} - ignored
        
        Returns:
            DecisionEnvelope unchanged (no consent checks)
        """
        logger.debug("ConsentGovernor: Consent enforcement is DISABLED - action allowed")
        return envelope
    
    @classmethod
    def can_execute(
        cls,
        action: str,
        session_consent_state: Dict[str, bool]
    ) -> tuple[bool, List[str]]:
        """
        Simple check if action can be executed - ALWAYS ALLOWS.
        
        Args:
            action: Proposed action
            session_consent_state: Current consent state - ignored
        
        Returns:
            Tuple of (True, []) - always allowed
        """
        return True, []
    
    @classmethod
    def grant_consent(
        cls,
        session_consent_state: Dict[str, bool],
        scope: str
    ) -> Dict[str, bool]:
        """
        Grant consent for a scope (no-op since consent is disabled).
        
        Args:
            session_consent_state: Current consent state (will be mutated)
            scope: Scope value to grant
        
        Returns:
            Updated consent state
        """
        session_consent_state[scope] = True
        return session_consent_state
    
    @classmethod
    def revoke_consent(
        cls,
        session_consent_state: Dict[str, bool],
        scope: str
    ) -> Dict[str, bool]:
        """
        Revoke consent for a scope (no-op since consent is disabled).
        
        Args:
            session_consent_state: Current consent state (will be mutated)
            scope: Scope value to revoke
        
        Returns:
            Updated consent state
        """
        session_consent_state[scope] = False
        return session_consent_state
    
    @classmethod
    def get_all_granted(cls, session_consent_state: Dict[str, bool]) -> List[str]:
        """Get list of all granted consent scopes."""
        return [scope for scope, granted in session_consent_state.items() if granted]
    
    @classmethod
    def is_marketing_consented(cls, session_consent_state: Dict[str, bool]) -> bool:
        """Check if marketing consent is granted."""
        return session_consent_state.get("marketing", False)
    
    @classmethod
    def create_empty_consent_state(cls) -> Dict[str, bool]:
        """Create a new empty consent state dict."""
        return {}
