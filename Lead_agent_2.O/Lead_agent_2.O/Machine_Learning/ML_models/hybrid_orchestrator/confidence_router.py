"""
Routing utility to decide how to handle a user utterance based on ML intent confidence.
"""
from dataclasses import dataclass
from typing import Dict, Optional

from config.settings import ML_CONFIDENCE_THRESHOLD


@dataclass
class RoutingDecision:
    route: str  # e.g., greeting_ml, form_ml, llm_router
    intent: Optional[str]
    confidence: float


GREETING_INTENTS = {"greeting", "gratitude"}
FORM_INTENTS = {
    "lead_collection",  # 🔴 NEW: Direct routing for lead data collection
    "lead_questions",
    "lead_form",
}

# 🔴 Intents that should be deferred to LLM for context-aware handling
LLM_DEFERRED_INTENTS = set()


def route_decision(intent: Optional[str], confidence: float) -> RoutingDecision:
    threshold = ML_CONFIDENCE_THRESHOLD or 0.8
    if intent and confidence >= threshold:
        if intent in GREETING_INTENTS:
            return RoutingDecision("greeting_ml", intent, confidence)
        if intent in FORM_INTENTS:
            return RoutingDecision("form_ml", intent, confidence)
        if intent in LLM_DEFERRED_INTENTS:
            # 🔴 Defer to LLM for context-aware handling (respects active_flow)
            return RoutingDecision("llm_router", intent, confidence)
    return RoutingDecision("llm_router", intent, confidence)

