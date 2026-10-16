"""
Hybrid conversation endpoint.
"""
import logging
from fastapi import APIRouter, Depends

from models.hybrid_schemas import ConversationRequest, ConversationResponse

router = APIRouter(prefix="/conversation", tags=["Hybrid Conversation"])

logger = logging.getLogger(__name__)


@router.post("", response_model=ConversationResponse)
async def converse(request: ConversationRequest) -> ConversationResponse:
    """
    Process conversation through enterprise AI agent pipeline.
    
    Import is done here to avoid circular dependency at module load time.
    """
    # Import here to avoid circular dependency
    from hybrid_orchestrator.router_service import process_conversation
    
    logger.info("Hybrid conversation request: %s", request.json())
    response = await process_conversation(request)
    
    # Log the full response for debugging
    logger.info(
        f"📤 Returning response for session {response.session_id}: "
        f"route={response.route}, response_length={len(response.response)}, "
        f"is_complete={response.is_complete}"
    )
    
    return response

