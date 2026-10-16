"""
Lightweight helper to access intent classification logic without issuing HTTP calls.
Includes fallback mechanism for missing domains.
"""
import logging
from typing import Dict, Optional, List

from ml_services.classifier_service import (
    predict_batch_intents,
    predict_intent,
    verify_intent_local,
)
from ml_utils.helpers import discover_domains, initialize_embedder, load_domain

logger = logging.getLogger(__name__)

# Default domain when requested domain is not found
DEFAULT_DOMAIN = "default"

# Fallback intent prediction when no model is available
FALLBACK_INTENTS = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"],
    "lead_generation": ["buy", "order", "price", "cost", "purchase", "book", "apply"],
    "faq": ["what", "how", "when", "where", "why", "?"],
    "handoff": ["speak to human", "talk to person", "agent", "representative"],
}


def safe_classify(query: str, domain: str = None) -> Dict:
    """
    Safe classification with automatic fallback.
    Never raises - always returns a valid prediction.
    """
    try:
        return classify_query(query, domain=domain, return_all_probs=False)
    except Exception as e:
        logger.warning(f"Classification failed for domain '{domain}': {e}, using fallback")
        return fallback_classify(query, domain)


def fallback_classify(query: str, domain: str = None) -> Dict:
    """
    Rule-based fallback classifier when ML model is unavailable.
    """
    query_lower = query.lower()
    
    for intent, keywords in FALLBACK_INTENTS.items():
        if any(kw in query_lower for kw in keywords):
            return {
                "intent": intent,
                "confidence": 0.6,
                "domain": domain or DEFAULT_DOMAIN,
                "fallback": True
            }
    
    return {
        "intent": "general_info",
        "confidence": 0.4,
        "domain": domain or DEFAULT_DOMAIN,
        "fallback": True
    }


def ensure_classifier_ready(domain: Optional[str] = None) -> bool:
    """
    Make sure the embedder and requested domain model are loaded.
    Returns True if successful, False if using fallback.
    """
    try:
        initialize_embedder()
        available = discover_domains()
        
        if not available:
            logger.warning("No domains discovered, will use fallback classifier")
            return False
        
        target = domain or available[0]
        
        # Check if target domain exists, otherwise use first available or default
        if target not in available:
            logger.warning(f"Domain '{target}' not found. Available: {available}")
            if available:
                target = available[0]
                logger.info(f"Using fallback domain: {target}")
            else:
                return False
        
        load_domain(target)
        return True
    except Exception as e:
        logger.error(f"Failed to ensure classifier ready: {e}")
        return False


def classify_query(
    query: str, domain: Optional[str] = None, return_all_probs: bool = False
) -> Dict:
    """
    Predict intent for a single query using in-process classifier.
    Includes automatic fallback for missing domains.
    """
    try:
        # Try to ensure classifier is ready
        ready = ensure_classifier_ready(domain)
        
        if not ready:
            logger.warning(f"Classifier not ready for domain '{domain}', using fallback")
            return fallback_classify(query, domain)
        
        result = predict_intent(query, domain=domain, return_all_probs=return_all_probs)
        result["fallback"] = False
        return result
        
    except Exception as e:
        logger.error(f"Classification error for domain '{domain}': {e}")
        # Try with default domain as fallback
        try:
            if domain and domain != DEFAULT_DOMAIN:
                logger.info(f"Trying classification with default domain")
                ensure_classifier_ready(DEFAULT_DOMAIN)
                result = predict_intent(query, domain=DEFAULT_DOMAIN, return_all_probs=return_all_probs)
                result["fallback"] = True
                return result
        except Exception:
            pass
        
        # Final fallback to rule-based
        return fallback_classify(query, domain)


def classify_batch(
    queries: list, domain: Optional[str] = None, return_all_probs: bool = False
) -> Dict:
    """
    Predict intents for multiple queries.
    """
    ensure_classifier_ready(domain)
    return predict_batch_intents(
        queries, domain=domain, return_all_probs=return_all_probs
    )


def verify_intent(query: str, domain: Optional[str] = None) -> Dict:
    """
    Convenience wrapper over verify_intent_local.
    """
    ensure_classifier_ready(domain)
    return verify_intent_local(query, domain=domain)

