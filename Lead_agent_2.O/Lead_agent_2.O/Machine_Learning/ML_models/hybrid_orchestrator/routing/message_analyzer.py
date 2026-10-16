"""
Anti-Gravity Policy Router - Message Analyzer

Extracts cheap signals from user messages BEFORE any expensive operations.
These signals are used by the Policy Router for scoring adjustments.

🔒 ANTI-GRAVITY PRINCIPLE:
   - Signals are cheap - no LLM, no API calls
   - Used for continuity detection, escape handling, ambiguity adjustment
   - Must run synchronously before routing decisions
"""
import re
import logging
from typing import Dict

from .types import MessageSignals

logger = logging.getLogger(__name__)

# Pattern definitions
FOLLOWUP_PATTERNS = (
    "tell me more",
    "and ",
    "what about",
    "also",
    "ok",
    "okay",
    "yes",
    "sure",
    "go on",
    "continue",
    "what else",
    "anything else",
    "more details",
    "elaborate",
    "explain more",
)

CLARIFICATION_PATTERNS = (
    "what do you mean",
    "can you explain",
    "why",
    "how",
    "what is",
    "i don't understand",
    "unclear",
    "confused",
    "not sure",
)

HANDOFF_PATTERNS = (
    "human",
    "agent",
    "representative",
    "call me",
    "talk to someone",
    "real person",
    "speak to",
    "contact me",
    "customer service",
    "support agent",
)

CANCEL_PATTERNS = (
    "cancel",
    "stop",
    "end",
    "abort",
    "quit",
    "exit",
    "never mind",
    "forget it",
    "no thanks",
    "not interested",
    "go back",
)


def analyze_message(text: str) -> MessageSignals:
    """
    Extract cheap signals from a user message.
    
    This function is designed to be fast and deterministic.
    It uses simple pattern matching, not ML or LLMs.
    
    Args:
        text: Raw user message
    
    Returns:
        MessageSignals dataclass with extracted signals
    """
    t = (text or "").strip()
    t_lower = t.lower()
    # Robust cleaning: keep only alphanumeric and spaces
    # This removes emojis (❌, ✅) and punctuation (,) which breaks word matching
    clean_text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in t_lower)
    words = clean_text.split()
    word_count = len(words)
    
    # Short message detection
    is_short = word_count < 5
    
    # Follow-up detection - message starts with follow-up patterns
    is_followup = any(t_lower.startswith(p) for p in FOLLOWUP_PATTERNS)
    
    # Also consider very short affirmative responses as follow-ups
    if not is_followup and word_count <= 2:
        short_followups = {"yes", "yeah", "yep", "sure", "ok", "okay", "go", "continue", "more"}
        if any(w in short_followups for w in words):
            is_followup = True
    
    # Clarification request detection
    is_clarification = (
        ("?" in t and len(t) < 60) or 
        any(t_lower.startswith(p) for p in CLARIFICATION_PATTERNS)
    )
    
    # Affirmative/Negative detection
    affirmative_words = {"yes", "yeah", "yep", "sure", "ok", "okay", "correct", "y", "yup", "definitely"}
    negative_words = {"no", "nope", "nah", "incorrect", "n", "never", "not"}
    
    # Check first 3 words to be safe after cleaning (e.g. "yes, please" -> ["yes", "please"])
    is_affirmative = any(w in affirmative_words for w in words[:3])
    is_negative = any(w in negative_words for w in words[:3])
    
    # Handoff intent detection
    wants_handoff = any(p in t_lower for p in HANDOFF_PATTERNS)
    
    # Cancel/escape intent detection
    wants_cancel = any(p in t_lower for p in CANCEL_PATTERNS)
    
    signals = MessageSignals(
        is_short=is_short,
        is_followup=is_followup,
        is_clarification=is_clarification,
        is_affirmative=is_affirmative,
        is_negative=is_negative,
        wants_handoff=wants_handoff,
        wants_cancel=wants_cancel,
        raw_text=t,
    )
    
    # Debug logging for development
    if any([wants_handoff, wants_cancel, is_affirmative, is_negative]):
        logger.debug(
            f"🔍 MESSAGE SIGNALS: handoff={wants_handoff}, cancel={wants_cancel}, "
            f"aff={is_affirmative}, neg={is_negative}, words={words[:5]}"
        )
    
    return signals


def to_dict(text: str) -> Dict:
    """
    Analyze message and return as dictionary.
    
    Convenience function for easy integration.
    
    Args:
        text: Raw user message
    
    Returns:
        Dict with signal keys
    """
    return analyze_message(text).to_dict()


def detect_escape_intent(text: str) -> Dict[str, bool]:
    """
    Specifically detect escape intents for mode bypass handling.
    
    This is called BEFORE mode bypass to allow escaping from FORM_ACTIVE.
    
    Args:
        text: Raw user message
    
    Returns:
        Dict with: wants_handoff, wants_cancel, is_escape
    """
    signals = analyze_message(text)
    return {
        "wants_handoff": signals.wants_handoff,
        "wants_cancel": signals.wants_cancel,
        "is_escape": signals.wants_handoff or signals.wants_cancel,
    }
