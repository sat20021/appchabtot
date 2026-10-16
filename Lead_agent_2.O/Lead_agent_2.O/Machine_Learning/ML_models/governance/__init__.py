"""
Governance Module - Central Policy Enforcement Layer

This module implements the "Anti-Gravity" architecture:
- Authority in code, humility in prompts
- LLMs suggest, code decides
- No component bypasses these policies

All routers, agents, tools, and workflows MUST import these policies.

NOTE: Consent enforcement has been DISABLED. ConsentGovernor is kept for
backward compatibility but all consent checks are bypassed.
"""

from governance.action_policy import ActionType, ActionPolicy
from governance.intent_policy import IntentPolicy
from governance.state_policy import SessionState, StatePolicy
from governance.decision_envelope import DecisionEnvelope, ConsentRequest
from governance.intent_guard import IntentGuard
from governance.consent_governor import ConsentGovernor
from governance.audit_logger import AuditLogger

__all__ = [
    # Policies
    "ActionType",
    "ActionPolicy",
    "IntentPolicy",
    "SessionState",
    "StatePolicy",
    # Decision format
    "DecisionEnvelope",
    "ConsentRequest",
    # Enforcers
    "IntentGuard",
    "ConsentGovernor",
    "AuditLogger",
]
