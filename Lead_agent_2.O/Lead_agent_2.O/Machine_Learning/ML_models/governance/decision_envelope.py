"""
Decision Envelope - Canonical Decision Format

All decision-making components must output this exact structure.
LLMs produce SUGGESTIONS, enforcers determine FINAL actions.

This is the SINGLE SOURCE OF TRUTH for decision format.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class ConsentRequest:
    """Request for user consent."""
    scope: str  # ConsentScope value
    channels: List[str] = field(default_factory=list)
    fields: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "channels": self.channels,
            "fields": self.fields
        }


@dataclass
class DecisionEnvelope:
    """
    Canonical decision format.
    
    This is the ONLY format that all decision-making components use.
    
    Flow:
    1. LLM populates suggested_action and suggested_copy
    2. Intent Guard validates and may override
    3. Consent Governor validates and may request consent
    4. Final allowed_action is executed
    """
    
    # === LLM-provided fields (SUGGESTIONS only) ===
    suggested_action: str = "RESPOND_MESSAGE"
    suggested_copy: Optional[str] = None
    requested_consent: Optional[ConsentRequest] = None
    
    # LLM reasoning (for audit)
    llm_reason: Optional[str] = None
    
    # Job search context (if applicable)
    job_search_keywords: Optional[str] = None
    job_search_results: Optional[List[Dict]] = None
    
    # === Policy-enforced fields (CODE decides) ===
    allowed_action: Optional[str] = None
    final_copy: Optional[str] = None
    consent_prompts: List[str] = field(default_factory=list)
    
    # === Enforcement metadata ===
    overrides_applied: bool = False
    override_reason: Optional[str] = None
    
    # ML context (for Intent Guard)
    ml_intent: Optional[str] = None
    ml_confidence: Optional[float] = None
    
    # Session context
    session_id: Optional[str] = None
    session_state: Optional[str] = None
    
    # Audit timestamp
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/API responses."""
        return {
            "suggested_action": self.suggested_action,
            "suggested_copy": self.suggested_copy,
            "requested_consent": self.requested_consent.to_dict() if self.requested_consent else None,
            "llm_reason": self.llm_reason,
            "job_search_keywords": self.job_search_keywords,
            "job_search_results": self.job_search_results,
            "allowed_action": self.allowed_action,
            "final_copy": self.final_copy,
            "consent_prompts": self.consent_prompts,
            "overrides_applied": self.overrides_applied,
            "override_reason": self.override_reason,
            "ml_intent": self.ml_intent,
            "ml_confidence": self.ml_confidence,
            "session_id": self.session_id,
            "session_state": self.session_state,
            "timestamp": self.timestamp,
        }
    
    def get_final_action(self) -> str:
        """
        Get the final action to execute.
        
        Returns allowed_action if set by enforcers, otherwise suggested_action.
        """
        return self.allowed_action or self.suggested_action
    
    def get_final_copy(self) -> Optional[str]:
        """Get the final response copy."""
        return self.final_copy or self.suggested_copy
    
    def needs_consent(self) -> bool:
        """Check if consent is required before action can proceed."""
        return len(self.consent_prompts) > 0
    
    def was_overridden(self) -> bool:
        """Check if the LLM suggestion was overridden."""
        return self.overrides_applied
    
    @classmethod
    def from_llm_response(cls, response: Dict) -> "DecisionEnvelope":
        """
        Create envelope from LLM response dict.
        
        Expected format:
        {
            "action": "RESPOND_MESSAGE",
            "message": "...",
            "reason": "...",
            "job_search_keywords": "..." (optional)
        }
        """
        consent_req = None
        if response.get("requested_consent"):
            consent_data = response["requested_consent"]
            consent_req = ConsentRequest(
                scope=consent_data.get("scope", ""),
                channels=consent_data.get("channels", []),
                fields=consent_data.get("fields", [])
            )
        
        return cls(
            suggested_action=response.get("action", "RESPOND_MESSAGE"),
            suggested_copy=response.get("message"),
            llm_reason=response.get("reason"),
            job_search_keywords=response.get("job_search_keywords"),
            requested_consent=consent_req
        )
    
    @classmethod
    def create_safe_fallback(cls, copy: str = None) -> "DecisionEnvelope":
        """Create a safe fallback envelope (RESPOND_MESSAGE)."""
        return cls(
            suggested_action="RESPOND_MESSAGE",
            allowed_action="RESPOND_MESSAGE",
            suggested_copy=copy,
            final_copy=copy
        )
