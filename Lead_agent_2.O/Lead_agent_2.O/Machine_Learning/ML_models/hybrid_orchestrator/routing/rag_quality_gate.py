"""
Anti-Gravity Policy Router - RAG Quality Gate

This module enforces quality thresholds for RAG search results.
If RAG quality is insufficient, the Policy Router will route to CLARIFY.

🔒 ANTI-GRAVITY PRINCIPLE:
   - RAG cannot produce confident answers without quality guarantees
   - Low scores or ambiguous results must trigger clarification
   - Thresholds are configurable via environment or defaults
"""
import os
import logging
from typing import Dict, Any

from .types import RAGQualityMetrics

logger = logging.getLogger(__name__)

# Default thresholds - can be overridden via environment
DEFAULT_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.5"))
DEFAULT_MIN_GAP = float(os.getenv("RAG_MIN_GAP", "0.15"))


class RAGQualityGate:
    """
    Evaluates RAG search results against quality thresholds.
    
    Quality rules:
    1. MUST have at least one source document
    2. Top score MUST meet minimum threshold
    3. Score gap between top and second result indicates ambiguity
    
    Usage:
        gate = RAGQualityGate()
        result = gate.evaluate(rag_result)
        if not result["passes"]:
            # Route to CLARIFY instead of RAG_QA
    """
    
    def __init__(
        self, 
        min_score: float = DEFAULT_MIN_SCORE, 
        min_gap: float = DEFAULT_MIN_GAP
    ):
        """
        Initialize the RAG Quality Gate.
        
        Args:
            min_score: Minimum top score required (0.0-1.0)
            min_gap: Minimum gap between top 2 results to avoid ambiguity
        """
        self.min_score = min_score
        self.min_gap = min_gap
        logger.debug(f"RAGQualityGate initialized: min_score={min_score}, min_gap={min_gap}")
    
    def evaluate(self, rag: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate RAG search results against quality thresholds.
        
        Args:
            rag: RAG result dict with keys:
                - top_score: float (highest chunk score)
                - second_score: float (second highest, optional)
                - score_gap: float (difference between top 2, optional)
                - sources_count: int (number of source documents)
                - chunks: list (raw chunks, optional)
        
        Returns:
            Dict with evaluation results:
                - passes: bool - True if quality is sufficient
                - is_ambiguous: bool - True if multiple similar results
                - top: float - Top score
                - gap: float - Score gap
                - sources: int - Source count
                - reason: str - Human-readable reason
        """
        top = rag.get("top_score", 0.0)
        sources = rag.get("sources_count", 0)
        
        # Calculate gap if not provided
        if "score_gap" in rag:
            gap = rag["score_gap"]
        elif "second_score" in rag:
            gap = top - rag["second_score"]
        else:
            gap = top  # Only one result, gap is the full score
        
        # Rule 1: Must have at least one source
        if sources == 0:
            return {
                "passes": False,
                "is_ambiguous": False,
                "top": top,
                "gap": gap,
                "sources": sources,
                "reason": "NO_SOURCES",
            }
        
        # Rule 2: Top score must meet minimum
        if top < self.min_score:
            return {
                "passes": False,
                "is_ambiguous": False,
                "top": top,
                "gap": gap,
                "sources": sources,
                "reason": f"LOW_SCORE ({top:.3f} < {self.min_score})",
            }
        
        # Rule 3: Check for ambiguity (multiple similar results)
        is_ambiguous = (sources > 1) and (gap < self.min_gap)
        
        # Quality passes but may be ambiguous
        return {
            "passes": True,
            "is_ambiguous": is_ambiguous,
            "top": top,
            "gap": gap,
            "sources": sources,
            "reason": "AMBIGUOUS" if is_ambiguous else "OK",
        }
    
    def evaluate_from_chunks(self, chunks: list) -> Dict[str, Any]:
        """
        Evaluate RAG quality directly from a list of chunks.
        
        Args:
            chunks: List of chunk dicts, each with at least 'score' key
        
        Returns:
            Same as evaluate()
        """
        metrics = RAGQualityMetrics.from_chunks(chunks)
        return self.evaluate({
            "top_score": metrics.top_score,
            "second_score": metrics.second_score,
            "score_gap": metrics.score_gap,
            "sources_count": metrics.sources_count,
        })
    
    def get_candidate_family(self, evaluation: Dict[str, Any]) -> str:
        """
        Determine the appropriate route family based on evaluation.
        
        Args:
            evaluation: Result from evaluate()
        
        Returns:
            Route family: "RAG_QA" or "CLARIFY"
        """
        if not evaluation["passes"]:
            return "CLARIFY"
        
        # Ambiguous results can still use RAG_QA but with disambiguation prompt
        # The caller can check is_ambiguous to customize the response
        return "RAG_QA"


# Singleton instance with default configuration
_default_gate = None


def get_rag_quality_gate() -> RAGQualityGate:
    """Get the default RAG Quality Gate instance."""
    global _default_gate
    if _default_gate is None:
        _default_gate = RAGQualityGate()
    return _default_gate
