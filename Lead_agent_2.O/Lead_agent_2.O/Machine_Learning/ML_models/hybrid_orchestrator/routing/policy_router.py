"""
Anti-Gravity Policy Router - Core Policy Logic

This is the SINGLE AUTHORITY for routing decisions.
All routing decisions MUST go through PolicyRouter.select().

🔒 ANTI-GRAVITY PRINCIPLES:
   - FAQ/ML/RAG are SIGNALS, not decisions
   - Policy applies scoring with bonuses and penalties
   - Continuity is preserved within QA_CONTINUATION mode
   - LLM is ADVISORY, code DECIDES
"""
import logging
from typing import Dict, List, Optional, Any

from .types import (
    RouteCandidate,
    PolicyDecision,
    MessageSignals,
    ConversationMode,
    RouteFamily,
    VALID_ROUTE_FAMILIES,
)

logger = logging.getLogger(__name__)

# Scoring adjustments
CONTINUITY_BONUS = 0.1  # Bonus for matching last_route_family in QA_CONTINUATION
AMBIGUITY_PENALTY = 0.15  # Penalty when message signals are ambiguous
CONTRADICTION_PENALTY = 0.2  # Penalty for contradicting previous answer class
FOLLOWUP_BONUS = 0.08  # Bonus when message is a follow-up


class PolicyRouter:
    """
    The single authority for policy-driven routing decisions.
    
    Takes candidates from signal collectors and applies policy-based scoring
    to select a single winner. The winner determines the route family, which
    is then mapped to a legacy route string for response construction.
    
    Scoring Formula:
        final_score = raw_confidence 
                    + continuity_bonus (if QA_CONTINUATION and matches last)
                    + followup_bonus (if message is a follow-up)
                    - ambiguity_penalty (if message or RAG is ambiguous)
                    - contradiction_penalty (if contradicts last turn)
    """
    
    def __init__(self):
        self.continuity_bonus = CONTINUITY_BONUS
        self.ambiguity_penalty = AMBIGUITY_PENALTY
        self.contradiction_penalty = CONTRADICTION_PENALTY
        self.followup_bonus = FOLLOWUP_BONUS
    
    def select(
        self,
        candidates: List[RouteCandidate],
        session_mode: ConversationMode,
        last_route_family: Optional[RouteFamily],
        signals: MessageSignals,
        rag_quality: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        """
        Select the winning route from candidates.
        
        Args:
            candidates: List of RouteCandidate from signal collectors
            session_mode: Current conversation mode (DEFAULT, QA_CONTINUATION, etc.)
            last_route_family: Route family from previous turn (if any)
            signals: MessageSignals extracted from user message
            rag_quality: Optional RAG quality evaluation result
        
        Returns:
            PolicyDecision with the winner and ranked list
        """
        if not candidates:
            # Fallback: Create a CLARIFY candidate if no candidates provided
            logger.warning("PolicyRouter: No candidates provided, falling back to CLARIFY")
            fallback = RouteCandidate(
                family="CLARIFY",
                raw_confidence=0.5,
                payload={"reason": "No candidates available"},
                diagnostics={"fallback": True},
            )
            return PolicyDecision(
                winner=fallback,
                ranked=[{"family": "CLARIFY", "final_score": 0.5, "reasons": ["NO_CANDIDATES"]}],
                mode_before=session_mode,
                mode_after=session_mode,
                reason_codes=["FALLBACK_NO_CANDIDATES"],
            )
        
        # Apply scoring to each candidate
        scored_candidates = []
        for candidate in candidates:
            scored = self._score_candidate(
                candidate,
                session_mode,
                last_route_family,
                signals,
                rag_quality,
            )
            scored_candidates.append(scored)
        
        # Sort by final score (descending)
        scored_candidates.sort(key=lambda c: c.final_score, reverse=True)
        
        # Select winner
        winner = scored_candidates[0]
        
        # Determine mode transition
        mode_after = self._determine_mode_after(winner, session_mode, signals)
        
        # Build ranked list for logging
        ranked = [
            {
                "family": c.family,
                "final_score": round(c.final_score, 4),
                "raw_confidence": round(c.raw_confidence, 4),
                "reasons": c.reason_codes,
            }
            for c in scored_candidates
        ]
        
        # Build reason codes
        reason_codes = list(winner.reason_codes)
        if session_mode == "QA_CONTINUATION" and winner.family == last_route_family:
            reason_codes.append("CONTINUITY_PRESERVED")
        
        decision = PolicyDecision(
            winner=winner,
            ranked=ranked,
            mode_before=session_mode,
            mode_after=mode_after,
            reason_codes=reason_codes,
        )
        
        logger.info(
            f"🎯 POLICY DECISION: winner={winner.family} (score={winner.final_score:.3f}), "
            f"mode: {session_mode} → {mode_after}, reasons={reason_codes}"
        )
        
        return decision
    
    def _score_candidate(
        self,
        candidate: RouteCandidate,
        session_mode: ConversationMode,
        last_route_family: Optional[RouteFamily],
        signals: MessageSignals,
        rag_quality: Optional[Dict[str, Any]],
    ) -> RouteCandidate:
        """
        Apply policy-based scoring to a candidate.
        
        Modifies candidate.final_score and candidate.reason_codes in place.
        """
        # Start with raw confidence
        score = candidate.raw_confidence
        reasons = []
        
        # RULE 1: Continuity bonus in QA_CONTINUATION mode
        if session_mode == "QA_CONTINUATION" and last_route_family:
            if candidate.family == last_route_family:
                score += self.continuity_bonus
                reasons.append(f"+{self.continuity_bonus} CONTINUITY")
            elif candidate.family in ("FAQ", "ML_ACTION"):
                # Prevent fast-path hijacking during QA continuation
                # FAQ and ML need higher confidence to override
                if candidate.raw_confidence < 0.9:
                    score -= self.continuity_bonus
                    reasons.append(f"-{self.continuity_bonus} ANTI_HIJACK")
        
        # RULE 2: Follow-up bonus
        if signals.is_followup:
            # Follow-ups should favor continuity
            if last_route_family and candidate.family == last_route_family:
                score += self.followup_bonus
                reasons.append(f"+{self.followup_bonus} FOLLOWUP")
            # Slight penalty to other candidates on follow-up
            elif last_route_family and candidate.family != last_route_family:
                score -= self.followup_bonus / 2
                reasons.append(f"-{self.followup_bonus/2} FOLLOWUP_MISMATCH")
        
        # RULE 3: Ambiguity penalty
        if signals.is_short or signals.is_clarification:
            # Short or unclear messages should penalize confident answers
            if candidate.family in ("FAQ", "RAG_QA") and candidate.raw_confidence < 0.85:
                score -= self.ambiguity_penalty
                reasons.append(f"-{self.ambiguity_penalty} AMBIGUOUS_MSG")
        
        # RULE 4: RAG ambiguity penalty
        if rag_quality and candidate.family == "RAG_QA":
            if rag_quality.get("is_ambiguous", False):
                score -= self.ambiguity_penalty
                reasons.append(f"-{self.ambiguity_penalty} RAG_AMBIGUOUS")
        
        # RULE 5: CLARIFY gets baseline score when others are weak
        if candidate.family == "CLARIFY":
            # CLARIFY is always available, gets moderate score
            score = max(score, 0.4)  # Floor of 0.4
            if score == 0.4:
                reasons.append("CLARIFY_FLOOR")
        
        # RULE 6: Handoff intent handling
        if signals.wants_handoff and candidate.family == "HANDOFF":
            score += 0.5  # Strong boost for explicit handoff request
            reasons.append("+0.5 EXPLICIT_HANDOFF")
        
        # Store results
        candidate.final_score = score
        candidate.reason_codes = reasons
        
        return candidate
    
    def _determine_mode_after(
        self,
        winner: RouteCandidate,
        session_mode: ConversationMode,
        signals: MessageSignals,
    ) -> ConversationMode:
        """
        Determine the conversation mode after this turn.
        """
        # Cancel/escape returns to DEFAULT
        if signals.wants_cancel:
            return "DEFAULT"
        
        # Handoff intent changes mode
        if winner.family == "HANDOFF":
            if signals.wants_handoff:
                return "HANDOFF_PENDING"
            return session_mode  # Keep current if no explicit request
        
        # Form actions change mode
        if winner.family == "ML_ACTION":
            intent = winner.payload.get("intent", "") if isinstance(winner.payload, dict) else ""
            if any(f in intent for f in ("lead", "form")):
                return "FORM_ACTIVE"
        
        # QA responses enter QA_CONTINUATION
        if winner.family in ("FAQ", "RAG_QA", "LLM"):
            return "QA_CONTINUATION"
        
        # CLARIFY stays in current mode or goes to DEFAULT
        if winner.family == "CLARIFY":
            if session_mode == "QA_CONTINUATION":
                return "QA_CONTINUATION"  # Clarify within QA
            return "DEFAULT"
        
        # Default: stay in current mode
        return session_mode


# Singleton instance
_router_instance = None


def get_policy_router() -> PolicyRouter:
    """Get the singleton PolicyRouter instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = PolicyRouter()
    return _router_instance
