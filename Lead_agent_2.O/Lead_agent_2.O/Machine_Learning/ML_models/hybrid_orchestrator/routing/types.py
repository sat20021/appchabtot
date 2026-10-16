"""
Anti-Gravity Policy Router - Core Types

This module defines the canonical data structures for policy-driven routing.
These types are internal to the Policy Router and do NOT affect external API contracts.

🔒 ANTI-GRAVITY PRINCIPLE:
   - RouteCandidate represents a SIGNAL, not a decision
   - PolicyDecision represents the ARBITRATED result
   - External route strings (faq, llm_router, form_continue) remain unchanged
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Type aliases for clarity
RouteFamily = str  # "FAQ" | "ML_ACTION" | "RAG_QA" | "LLM" | "CLARIFY" | "HANDOFF"
ConversationMode = str  # "DEFAULT" | "QA_CONTINUATION" | "FORM_ACTIVE" | "HANDOFF_PENDING" | "HANDOFF_ACTIVE"


# Valid values for validation
VALID_ROUTE_FAMILIES = {"FAQ", "ML_ACTION", "RAG_QA", "LLM", "CLARIFY", "HANDOFF"}
VALID_CONVERSATION_MODES = {"DEFAULT", "QA_CONTINUATION", "FORM_ACTIVE", "HANDOFF_PENDING", "HANDOFF_ACTIVE", "FORM_CONFIRMATION"}


@dataclass
class RouteCandidate:
    """
    A candidate signal from a routing source (FAQ, ML, RAG, LLM, etc.)
    
    This is NOT a decision - it's a signal that the Policy Router will evaluate.
    Multiple candidates are compared/scored to select a single winner.
    
    Attributes:
        family: Route family (FAQ, ML_ACTION, RAG_QA, LLM, CLARIFY, HANDOFF)
        raw_confidence: The native confidence score from the source (0.0-1.0)
        payload: Any route-specific data (answer text, intent, chunks, etc.)
        diagnostics: Debugging info (timing, source details, thresholds)
        requires_llm: If True, LLM must be called to execute this route (lazy)
        final_score: Computed score after policy adjustments (set by PolicyRouter)
        reason_codes: List of policy adjustments applied (set by PolicyRouter)
    """
    family: RouteFamily
    raw_confidence: float
    payload: Any
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    requires_llm: bool = False
    
    # Computed by PolicyRouter
    final_score: float = 0.0
    reason_codes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.family not in VALID_ROUTE_FAMILIES:
            raise ValueError(f"Invalid route family: {self.family}. Must be one of {VALID_ROUTE_FAMILIES}")
        self.final_score = self.raw_confidence  # Initialize from raw


@dataclass
class PolicyDecision:
    """
    The canonical result of policy-driven routing.
    
    Produced by PolicyRouter.select() after evaluating all candidates.
    This is the AUTHORITATIVE routing decision for a single user turn.
    
    Attributes:
        winner: The selected RouteCandidate
        ranked: All candidates with their final scores and reasons (for logging)
        mode_before: Conversation mode before this turn
        mode_after: Conversation mode after this turn (may change based on winner)
        reason_codes: High-level reasons for the decision
    """
    winner: RouteCandidate
    ranked: List[Dict[str, Any]]  # [{"family":..., "final_score":..., "reasons":[...]}]
    mode_before: ConversationMode
    mode_after: ConversationMode
    reason_codes: List[str] = field(default_factory=list)
    
    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict for structured logging."""
        return {
            "winner_family": self.winner.family,
            "winner_score": round(self.winner.final_score, 4),
            "winner_confidence": round(self.winner.raw_confidence, 4),
            "mode_before": self.mode_before,
            "mode_after": self.mode_after,
            "ranked": self.ranked,
            "reason_codes": self.reason_codes,
        }


@dataclass
class MessageSignals:
    """
    Cheap signals extracted from the user message.
    
    These are computed BEFORE any expensive operations (LLM, API calls).
    Used by PolicyRouter to adjust scoring and detect special intents.
    
    Attributes:
        is_short: Message has fewer than 5 words
        is_followup: Starts with follow-up patterns ("tell me more", "and", etc.)
        is_clarification: Looks like a clarification request
        is_affirmative: Affirmative short response (yes, ok)
        is_negative: Negative short response (no, nope)
        wants_handoff: Explicit request for human agent
        wants_cancel: Explicit cancel/stop intent
        raw_text: Original message text (for debugging)
    """
    is_short: bool = False
    is_followup: bool = False
    is_clarification: bool = False
    is_affirmative: bool = False
    is_negative: bool = False
    wants_handoff: bool = False
    wants_cancel: bool = False
    is_greeting: bool = False
    raw_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy passing."""
        return {
            "is_short": self.is_short,
            "is_followup": self.is_followup,
            "is_clarification": self.is_clarification,
            "is_affirmative": self.is_affirmative,
            "is_negative": self.is_negative,
            "wants_handoff": self.wants_handoff,
            "wants_cancel": self.wants_cancel,
            "is_greeting": self.is_greeting,
        }


@dataclass
class RAGQualityMetrics:
    """
    Quality metrics for RAG search results.
    
    Used by RAGQualityGate to determine if RAG results are trustworthy.
    
    Attributes:
        top_score: Score of the best matching chunk
        second_score: Score of the second-best chunk (for gap calculation)
        score_gap: Difference between top and second (higher = more confident)
        sources_count: Number of unique source documents
        chunks_count: Number of retrieved chunks
        doc_ids: List of source document IDs
        passes: Whether the quality gate passed
        is_ambiguous: High-scoring but small gap (need disambiguation)
    """
    top_score: float = 0.0
    second_score: float = 0.0
    score_gap: float = 0.0
    sources_count: int = 0
    chunks_count: int = 0
    doc_ids: List[str] = field(default_factory=list)
    
    # Gate results (set by RAGQualityGate)
    passes: bool = False
    is_ambiguous: bool = False
    
    @classmethod
    def from_chunks(cls, chunks: List[Dict]) -> "RAGQualityMetrics":
        """Create metrics from a list of RAG chunk results."""
        if not chunks:
            return cls()
        
        scores = [c.get("score", 0.0) for c in chunks]
        scores_sorted = sorted(scores, reverse=True)
        
        top = scores_sorted[0] if len(scores_sorted) > 0 else 0.0
        second = scores_sorted[1] if len(scores_sorted) > 1 else 0.0
        
        doc_ids = list(set(c.get("doc_id", "") for c in chunks if c.get("doc_id")))
        
        return cls(
            top_score=top,
            second_score=second,
            score_gap=top - second,
            sources_count=len(doc_ids),
            chunks_count=len(chunks),
            doc_ids=doc_ids,
        )
