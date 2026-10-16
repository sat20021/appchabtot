"""
Anti-Gravity Policy Router - Signal Collectors

Adapters that collect routing signals from existing decision nodes
WITHOUT executing routes. These produce RouteCandidate objects for
the Policy Router to evaluate.

🔒 ANTI-GRAVITY PRINCIPLES:
   - Existing FAQ/ML/RAG nodes remain unchanged
   - These adapters WRAP existing nodes
   - NO route execution happens here - only signal collection
   - LLM calls are LAZY (only if policy requires it)
"""
import logging
from typing import Dict, List, Optional, Any, Tuple

from .types import RouteCandidate, RAGQualityMetrics
from .rag_quality_gate import get_rag_quality_gate

logger = logging.getLogger(__name__)


def collect_faq_signal(
    faq_result: Optional[Dict[str, Any]],
    threshold: float = 0.8,
) -> Optional[RouteCandidate]:
    """
    Collect FAQ signal from an existing FAQ lookup result.
    
    Args:
        faq_result: Result from FAQDecisionNode.execute() or similar
                   Expected keys: selected, confidence, answer, timing_ms
        threshold: Confidence threshold for inclusion
    
    Returns:
        RouteCandidate for FAQ or None if below threshold
    """
    if not faq_result:
        return None
    
    confidence = faq_result.get("confidence", 0.0)
    
    # Always create candidate if we have a result (policy decides if it wins)
    candidate = RouteCandidate(
        family="FAQ",
        raw_confidence=confidence,
        payload={
            "answer": faq_result.get("answer", ""),
            "source": faq_result.get("source", ""),
            "matched_question": faq_result.get("matched_question", ""),
        },
        diagnostics={
            "timing_ms": faq_result.get("timing_ms", 0),
            "threshold": threshold,
            "above_threshold": confidence >= threshold,
        },
        requires_llm=False,
    )
    
    return candidate


def collect_ml_signal(
    ml_result: Optional[Dict[str, Any]],
    threshold: float = 0.8,
) -> Optional[RouteCandidate]:
    """
    Collect ML classification signal.
    
    Args:
        ml_result: Result from MLDecisionNode.execute() or similar
                   Expected keys: intent, confidence, timing_ms
        threshold: Confidence threshold for inclusion
    
    Returns:
        RouteCandidate for ML_ACTION or None
    """
    if not ml_result:
        return None
    
    confidence = ml_result.get("confidence", 0.0)
    intent = ml_result.get("intent", "")
    
    # Determine if this is an actionable intent
    # Greetings and forms are ML_ACTION
    actionable_intents = {
        "greeting", "gratitude",
        "lead_collection", "lead_questions", "lead_form",
    }
    
    is_actionable = intent in actionable_intents
    
    candidate = RouteCandidate(
        family="ML_ACTION",
        raw_confidence=confidence if is_actionable else confidence * 0.5,  # Downweight non-actionable
        payload={
            "intent": intent,
            "is_actionable": is_actionable,
            "original_confidence": confidence,
        },
        diagnostics={
            "timing_ms": ml_result.get("timing_ms", 0),
            "threshold": threshold,
            "above_threshold": confidence >= threshold,
            "is_greeting": intent in ("greeting", "gratitude"),
            "is_form": intent in ("lead_collection", "lead_questions", "lead_form"),
        },
        requires_llm=False,
    )
    
    return candidate


def collect_rag_signal(
    rag_context: str,
    rag_metadata: Dict[str, Any],
    chunks: List[Dict[str, Any]],
) -> Tuple[RouteCandidate, Dict[str, Any]]:
    """
    Collect RAG signal with quality gate evaluation.
    
    Args:
        rag_context: Formatted RAG context string
        rag_metadata: Metadata from RAG search
        chunks: Raw chunk data from RAG search
    
    Returns:
        Tuple of (RouteCandidate for RAG_QA or CLARIFY, quality_evaluation)
    """
    # Calculate quality metrics
    metrics = RAGQualityMetrics.from_chunks(chunks)
    
    # Run quality gate
    gate = get_rag_quality_gate()
    quality_eval = gate.evaluate({
        "top_score": metrics.top_score,
        "second_score": metrics.second_score,
        "score_gap": metrics.score_gap,
        "sources_count": metrics.sources_count,
    })
    
    if not quality_eval["passes"]:
        # RAG quality insufficient - return CLARIFY candidate
        candidate = RouteCandidate(
            family="CLARIFY",
            raw_confidence=0.5,
            payload={
                "reason": f"RAG quality insufficient: {quality_eval['reason']}",
                "had_rag_context": bool(rag_context),
            },
            diagnostics={
                "rag_quality": quality_eval,
                "sources_count": metrics.sources_count,
                "top_score": metrics.top_score,
            },
            requires_llm=True,  # CLARIFY typically needs LLM to phrase
        )
        return candidate, quality_eval
    
    # RAG quality passes - create RAG_QA candidate
    # Use top_score as confidence, adjusted by gap
    confidence = metrics.top_score
    if quality_eval["is_ambiguous"]:
        confidence *= 0.85  # Slight penalty for ambiguity
    
    candidate = RouteCandidate(
        family="RAG_QA",
        raw_confidence=confidence,
        payload={
            "context": rag_context,
            "chunks": chunks,
            "doc_ids": metrics.doc_ids,
            "is_ambiguous": quality_eval["is_ambiguous"],
        },
        diagnostics={
            "rag_quality": quality_eval,
            "sources_count": metrics.sources_count,
            "chunks_count": metrics.chunks_count,
            "timing_ms": rag_metadata.get("timing_ms", 0),
        },
        requires_llm=True,  # RAG_QA needs LLM to synthesize answer
    )
    
    return candidate, quality_eval


def collect_handoff_signal(
    wants_handoff: bool,
    session_has_pending_handoff: bool = False,
) -> Optional[RouteCandidate]:
    """
    Collect handoff signal based on message intent.
    
    Args:
        wants_handoff: True if message indicates handoff intent
        session_has_pending_handoff: True if handoff already pending
    
    Returns:
        RouteCandidate for HANDOFF or None
    """
    if not wants_handoff and not session_has_pending_handoff:
        return None
    
    confidence = 0.9 if wants_handoff else 0.6
    
    candidate = RouteCandidate(
        family="HANDOFF",
        raw_confidence=confidence,
        payload={
            "explicit_request": wants_handoff,
            "pending": session_has_pending_handoff,
        },
        diagnostics={
            "trigger": "explicit" if wants_handoff else "pending",
        },
        requires_llm=False,
    )
    
    return candidate


def collect_clarify_signal(
    is_clarification: bool,
    is_ambiguous: bool = False,
) -> RouteCandidate:
    """
    Create a CLARIFY candidate (always available as fallback).
    
    Args:
        is_clarification: True if message looks like a clarification request
        is_ambiguous: True if other signals are ambiguous
    
    Returns:
        RouteCandidate for CLARIFY
    """
    # Base confidence depends on signals
    if is_clarification:
        confidence = 0.6
    elif is_ambiguous:
        confidence = 0.5
    else:
        confidence = 0.3  # Always available but low priority
    
    candidate = RouteCandidate(
        family="CLARIFY",
        raw_confidence=confidence,
        payload={
            "is_clarification_request": is_clarification,
            "is_ambiguous": is_ambiguous,
        },
        diagnostics={},
        requires_llm=True,
    )
    
    return candidate


def collect_llm_signal(
    has_rag_context: bool,
    rag_quality_passes: bool,
) -> RouteCandidate:
    """
    Create LLM fallback candidate.
    
    LLM is the general purpose responder when other routes don't apply.
    
    Args:
        has_rag_context: True if RAG context is available
        rag_quality_passes: True if RAG quality gate passed
    
    Returns:
        RouteCandidate for LLM
    """
    # LLM confidence based on context availability
    if has_rag_context and rag_quality_passes:
        confidence = 0.7  # RAG-grounded LLM is more confident
    elif has_rag_context:
        confidence = 0.6  # Has context but quality is uncertain
    else:
        confidence = 0.5  # Pure LLM without grounding
    
    candidate = RouteCandidate(
        family="LLM",
        raw_confidence=confidence,
        payload={
            "has_rag_context": has_rag_context,
            "rag_quality_passes": rag_quality_passes,
        },
        diagnostics={},
        requires_llm=True,
    )
    
    return candidate
