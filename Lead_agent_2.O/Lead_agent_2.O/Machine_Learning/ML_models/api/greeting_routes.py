# api/greeting_routes.py
# Greeting Response Service API endpoints
# Refactored to use domain as URL path parameter
import logging
from typing import Optional, Dict, List

from fastapi import APIRouter, HTTPException, Body, Path

from models.schemas import GreetingRequest, GreetingResponse
from core.state import DOMAIN_GREETINGS, DOMAIN_GRATITUDE
from ml_services.greeting_service import (
    generate_greeting_response, 
    select_greeting, 
    save_domain_greeting,
    load_domain_greeting,
    list_greeting_domains
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/greetings", tags=["Greeting"])

# ============================================================================
# STATIC ROUTES FIRST (must come before parameterized routes)
# ============================================================================

@router.get("/domains", summary="List all greeting domains")
def get_greeting_domains():
    """List all available domains with greeting support."""
    try:
        domains = list_greeting_domains()
        return {
            "domains": domains,
            "count": len(domains)
        }
    except Exception as e:
        logger.error(f"Error listing domains: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# ============================================================================
# DOMAIN-SPECIFIC ENDPOINTS (parameterized routes must come AFTER static routes)
# ============================================================================

@router.post("/{domain}/respond", response_model=GreetingResponse)
async def generate_response_for_domain(
    domain: str = Path(..., description="Domain name for greeting context"),
    request: GreetingRequest = Body(...)
):
    """
    Generate greeting or gratitude response for a specific domain.
    
    Args:
        domain: Target domain (from URL path)
        request: GreetingRequest with query and optional user_name
        
    Returns:
        GreetingResponse with contextual response details
    """
    try:
        # URL domain takes precedence
        result = generate_greeting_response(
            request.query, 
            domain,  # Use URL domain 
            request.user_name
        )
        
        return GreetingResponse(**result)
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@router.post("/{domain}", summary="Add/Update Custom Greetings for Domain")
def set_domain_greetings(
    domain: str = Path(..., description="Domain name to set greetings for"),
    greetings: Dict = Body(..., description="Greeting configuration")
):
    """
    Set or update greeting messages for a specific domain.
    
    Expected format:
    {
        "general": ["Hello!", "Hi there!"],
        "personalized": ["Hello {name}!"],
        "time_based": {
            "morning": ["Good morning!"],
            "afternoon": ["Good afternoon!"],
            "evening": ["Good evening!"],
            "night": ["Good night!"]
        },
        "gratitude": ["You're welcome!"]
    }
    """
    try:
        # Validate greetings structure
        if not greetings:
            raise HTTPException(
                status_code=400,
                detail="Greetings cannot be empty"
            )
            
        # If the user pasted the entire config wrapped in the domain key, extract it
        if domain in greetings and isinstance(greetings[domain], dict) and len(greetings) == 1:
            greetings = greetings[domain]
        
        if not any(key in greetings for key in ["general", "personalized", "time_based", "gratitude"]):
            received_keys = list(greetings.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Greetings must contain at least one of: general, personalized, time_based, gratitude. Received keys: {received_keys}"
            )
        
        # Save to per-domain file
        success = save_domain_greeting(domain, greetings)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save greetings for domain: {domain}"
            )
        
        return {
            "message": f"Successfully set greetings for domain: {domain}",
            "domain": domain,
            "categories": list(greetings.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting greetings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set greetings: {str(e)}")


@router.get("/{domain}", summary="Get greetings for domain")
def get_domain_greetings(
    domain: str = Path(..., description="Domain name to get greetings for")
):
    """
    Get all greeting messages configured for a specific domain.
    """
    try:
        # Try to load from memory first, then from file
        if domain in DOMAIN_GREETINGS:
            greetings = DOMAIN_GREETINGS[domain].copy()
            if domain in DOMAIN_GRATITUDE and "gratitude" in DOMAIN_GRATITUDE[domain]:
                greetings["gratitude"] = DOMAIN_GRATITUDE[domain]["gratitude"]
            return {"domain": domain, "greetings": greetings}
        
        # Try to load from file
        config = load_domain_greeting(domain)
        if config:
            return {"domain": domain, "greetings": config}
        
        raise HTTPException(
            status_code=404,
            detail=f"No greetings found for domain: {domain}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting greetings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get greetings: {str(e)}")


@router.get("/{domain}/preview", summary="Preview greetings for domain")
def preview_domain_greetings(
    domain: str = Path(..., description="Domain to preview"),
    user_name: Optional[str] = None,
    count: int = 5
):
    """
    Preview sample greeting messages for a domain.
    
    Args:
        domain: Domain to preview
        user_name: Optional name for personalized preview
        count: Number of greeting samples to generate
    """
    try:
        samples = []
        for _ in range(count):
            greeting = select_greeting(domain, user_name)
            samples.append(greeting)
        
        return {
            "domain": domain,
            "user_name": user_name,
            "samples": samples,
            "personalized": user_name is not None
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error previewing greetings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{domain}", summary="Delete greetings for domain")
def delete_domain_greetings(
    domain: str = Path(..., description="Domain to delete greetings for")
):
    """
    Delete all greetings for a specific domain.
    """
    try:
        from pathlib import Path as PathLib
        from ml_utils.s3_handler import S3Handler
        
        # Remove from memory
        if domain in DOMAIN_GREETINGS:
            del DOMAIN_GREETINGS[domain]
        if domain in DOMAIN_GRATITUDE:
            del DOMAIN_GRATITUDE[domain]
        
        # Remove from Redis
        from ml_utils.redis_cache_handler import RedisCacheHandler
        redis_cache = RedisCacheHandler()
        if redis_cache.is_available():
            redis_cache.delete(f"greetings:{domain}")
        
        # Remove local file
        from ml_services.greeting_service import _get_domain_config_path
        config_path = _get_domain_config_path(domain)
        if config_path.exists():
            config_path.unlink()
            logger.info(f"Deleted local greeting config: {domain}")
        
        # Remove from S3
        try:
            s3 = S3Handler()
            if s3.s3_client:
                s3.delete_file(f"configs/greetings/{domain}.json")
                logger.info(f"Deleted S3 greeting config: {domain}")
        except Exception as e:
            logger.warning(f"Failed to delete from S3: {e}")
        
        return {"message": f"Deleted greetings for domain: {domain}"}
        
    except Exception as e:
        logger.error(f"Error deleting greetings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete greetings: {str(e)}")
