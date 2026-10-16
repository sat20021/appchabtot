"""
Pydantic models shared by the hybrid conversation endpoint.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from uuid import uuid4

from pydantic import BaseModel, Field


class ConversationMetadata(BaseModel):
    channel: Optional[str] = Field(
        default=None, description="Source channel for analytics (web, sms, etc.)."
    )
    user_id: Optional[str] = Field(
        default=None, description="Upstream user identifier for personalization."
    )
    locale: Optional[str] = Field(
        default=None, description="ISO locale hint for downstream routing."
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary metadata payload."
    )


class ConversationRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's latest utterance.")
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        description="The upstream-provided session identifier. Auto-generated if omitted.",
    )
    domain: Optional[str] = Field(
        default=None,
        description="Domain context for routing and responses (e.g., 'algobridge'). Defaults to system default."
    )
    metadata: Optional[ConversationMetadata] = None


class RoutingStepMetadata(BaseModel):
    """Metadata for a single routing attempt."""
    route_type: str = Field(..., description="Type of route attempted (faq, ml, llm)")
    attempted: bool = Field(..., description="Whether this route was attempted")
    confidence: Optional[float] = Field(None, description="Confidence score if applicable")
    selected: bool = Field(..., description="Whether this route was selected")
    reason: Optional[str] = Field(None, description="Why this route was selected or rejected")
    timing_ms: Optional[float] = Field(None, description="Time taken in milliseconds")


class RoutingMetadata(BaseModel):
    """Complete routing decision metadata."""
    routing_steps: List[RoutingStepMetadata] = Field(
        default_factory=list,
        description="List of all routing attempts with their outcomes"
    )
    selected_route: str = Field(..., description="Which route was ultimately chosen")
    route_reason: str = Field(..., description="Why the selected route was chosen")
    faq_attempted: bool = Field(False, description="Whether FAQ lookup was attempted")
    faq_confidence: Optional[float] = Field(None, description="FAQ confidence if attempted")
    ml_attempted: bool = Field(False, description="Whether ML classification was attempted")
    ml_intent: Optional[str] = Field(None, description="ML intent if attempted")
    ml_confidence: Optional[float] = Field(None, description="ML confidence if attempted")
    llm_attempted: bool = Field(False, description="Whether LLM fallback was attempted")
    llm_action: Optional[str] = Field(None, description="LLM action if attempted")
    llm_reason: Optional[str] = Field(None, description="LLM reasoning if attempted")
    rag_attempted: bool = Field(False, description="Whether RAG search was attempted")
    rag_results_count: Optional[int] = Field(None, description="Number of RAG results found")
    rag_domain: Optional[str] = Field(None, description="Domain searched for RAG")
    rag_success: Optional[bool] = Field(None, description="Whether RAG search succeeded")
    timing_ms: Optional[Dict[str, float]] = Field(
        None,
        description="Timing breakdown: faq_ms, ml_ms, llm_ms"
    )



class ConversationResponse(BaseModel):
    session_id: str
    response: str
    route: str = Field(
        ..., description="How the orchestrator answered (greeting_ml, form_ml, llm_router, etc.)."
    )
    intent: Optional[str] = None
    confidence: Optional[float] = None
    matched_question: Optional[str] = None
    next_question: Optional[str] = None
    options: Optional[List[Any]] = Field(
        default=None, description="Available options: strings for simple prompts, or job card objects for carousel."
    )
    question_type: Optional[str] = Field(
        default=None, description="Type of the current question (text, options, file). Used by frontend to show appropriate input."
    )
    is_complete: bool = False
    collected_data: Optional[Dict[str, Any]] = None
    # 🆕 Fields for specific Lead Form requirements
    current_step: Optional[str] = None
    lead_data: Optional[Dict[str, Any]] = None

    context_tokens: Optional[int] = None
    routing_metadata: Optional[RoutingMetadata] = Field(
        None,
        description="Detailed routing decision metadata showing which paths were attempted and why"
    )
    rag_chunks: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="RAG document chunks retrieved and used for this response"
    )
    
    # Human Handoff Fields
    handoff_pending: bool = Field(
        default=False,
        description="If true, bot is asking user to confirm handoff to human agent"
    )
    handoff_mode: Optional[str] = Field(
        default=None,
        description="Current handoff state: 'waiting' (in queue), 'connected' (live chat), or None"
    )
    websocket_url: Optional[str] = Field(
        default=None,
        description="WebSocket URL for real-time chat when handoff is active"
    )
    
    debug: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional diagnostics (only returned in non-production environments).",
    )


@dataclass
class DecisionResult:
    """
    Standardized output from decision nodes.
    Used internally for passing decision data between nodes.
    """
    selected: bool = False
    route: Optional[str] = None
    confidence: Optional[float] = None
    intent: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None
    timing_ms: Optional[float] = None
    reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ContextState:
    """
    Conversation context state for efficient processing.
    Contains all necessary context for routing decisions.
    """
    session_id: str
    message: str
    session: Dict[str, Any]
    metadata: Dict[str, Any]
    last_three_turns: List[Dict[str, str]]
    active_flow: Optional[str] = None
    form_data: Optional[Dict[str, Any]] = None  # User info collected from forms (name, email, etc.)
