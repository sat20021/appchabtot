"""
Persona Management API Routes
Provides endpoints for creating, updating, and retrieving domain-specific personas
"""
from fastapi import APIRouter, HTTPException
from ml_services.persona_service import get_persona_service, Persona

router = APIRouter(prefix="/api/v1/domains", tags=["Persona Management"])


@router.post("/{domain}/persona", response_model=dict)
async def create_persona(domain: str, persona: Persona):
    """
    Create or update a persona for a specific domain.
    
    Args:
        domain: The domain identifier
        persona: Persona configuration including name, role, tone, etc.
    
    Returns:
        Confirmation message with the updated persona
    """
    service = get_persona_service()
    service.update_persona(domain, persona)
    return {"message": f"Persona updated for domain: {domain}", "persona": persona}


@router.get("/{domain}/persona", response_model=Persona)
async def get_persona(domain: str):
    """
    Get the persona configuration for a specific domain.
    
    Args:
        domain: The domain identifier
    
    Returns:
        The persona configuration for the domain
    
    Raises:
        HTTPException: 404 if no persona found for the domain
    """
    service = get_persona_service()
    persona = service.get_persona(domain)
    if not persona:
        raise HTTPException(
            status_code=404, 
            detail=f"No persona found for domain: {domain}"
        )
    return persona
