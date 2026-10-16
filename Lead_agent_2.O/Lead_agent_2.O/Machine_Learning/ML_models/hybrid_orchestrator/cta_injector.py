"""
CTA (Call-to-Action) Injector for Enterprise AI Agent.

This module determines when to inject CTA options into responses
and provides the standard CTA options for user engagement.

CTAs are shown for:
- FAQ answers
- Greeting responses  
- General LLM responses
- Off-topic questions during job browse

CTAs are hidden during:
- Form flows (lead form, job application)
- Job browse with job cards displayed
- Handoff flows
"""
import logging
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Routes where CTAs should NOT appear
NO_CTA_ROUTES = frozenset({
    "form_ml",
    "form_llm", 
    "form_continue",
    "handoff_request",
    "handoff_active",
    "resume_escape",
})

# Active flows where CTAs should NOT appear
NO_CTA_FLOWS = frozenset({
    "lead_questions",
})


def should_inject_ctas(
    route: str,
    active_flow: Optional[str] = None,
    options: Optional[List[Any]] = None,
    is_complete: bool = False,
) -> bool:
    """
    Determine if CTAs should be injected based on conversation context.
    
    Args:
        route: The route name (e.g., "faq_semantic", "form_continue")
        active_flow: Current active flow from session (e.g., "lead_questions")
        options: Current options in response (form options or job cards)
        is_complete: Whether the current flow is complete
        
    Returns:
        True if CTAs should be injected, False otherwise
    """
    # Check route blacklist
    if route in NO_CTA_ROUTES:
        logger.debug(f"🚫 CTA: Skipping - route '{route}' is blacklisted")
        return False
    
    # Check active flow blacklist (unless flow just completed)
    if active_flow in NO_CTA_FLOWS and not is_complete:
        logger.debug(f"🚫 CTA: Skipping - active_flow '{active_flow}' is blacklisted")
        return False
    
    # Don't inject if options already exist
    if options and isinstance(options, list) and len(options) > 0:
        logger.debug("🚫 CTA: Skipping - options already exist")
        return False
    
    logger.debug(f"✅ CTA: Injecting for route '{route}'")
    return True


def get_cta_options() -> List[Dict[str, str]]:
    """
    Return standard CTA options for user engagement.
    
    Returns:
        List of CTA dictionaries with label and action
    """
    return [
        {
            "label": "🚀 Start Your Project",
            "action": "START_PROJECT", 
            "type": "cta"
        },
        {
            "label": "📞 Get Free Consultation",
            "action": "FREE_CONSULTATION", 
            "type": "cta"
        },
        {
            "label": "🎓 Apply for Internship",
            "action": "APPLY_INTERNSHIP", 
            "type": "cta"
        },
        {
            "label": "💻 Build Your Software",
            "action": "BUILD_SOFTWARE", 
            "type": "cta"
        }
    ]


def inject_ctas_if_appropriate(
    route: str,
    active_flow: Optional[str] = None,
    options: Optional[List[Any]] = None,
    is_complete: bool = False,
) -> Optional[List[Dict[str, str]]]:
    """
    Helper function that returns CTAs if appropriate, None otherwise.
    
    Args:
        route: The route name
        active_flow: Current active flow
        options: Current options in response
        is_complete: Whether current flow is complete
        
    Returns:
        List of CTA options or None
    """
    if should_inject_ctas(route, active_flow, options, is_complete):
        return get_cta_options()
    return None
