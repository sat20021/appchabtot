# api/classifier_routes.py
# Classifier Service API endpoints
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException

from models.schemas import (
    PredictRequest, PredictResponse, HealthResponse, ModelInfo, IntentProbability
)
from core.state import models, active_domain
from ml_utils.helpers import (
    discover_domains, load_domain, initialize_embedder,
    discover_domain_configs, load_domain_config, get_all_domain_configs
)
from ml_services.classifier_service import predict_intent, predict_batch_intents

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Classifier"])

@router.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint"""
    from core.state import models
    
    available_domains = discover_domains()
    total_intents = 0
    loaded_domains = list(models.keys())
    
    for domain in loaded_domains:
        if domain in models:
            total_intents += len(set(models[domain].idx2label_int.values()))
    
    return {
        "status": "healthy" if loaded_domains else "no_model",
        "active_domain": None,  # Deprecated - all domains available
        "available_domains": available_domains,
        "loaded_domains": loaded_domains,
        "num_intents": total_intents
    }

@router.get("/domains")
def list_domains():
    """List all available domains with their configurations"""
    from core.state import active_domain
    
    available_domains = discover_domains()
    domain_configs = get_all_domain_configs()
    
    # Build response with domain info and configs
    domains_with_configs = []
    for domain in available_domains:
        domain_info = {
            "domain_name": domain,
            "has_model": True,
            "config": domain_configs.get(domain, None)
        }
        domains_with_configs.append(domain_info)
    
    # Also include configs that don't have models yet
    for config_domain in domain_configs.keys():
        if config_domain not in available_domains:
            domains_with_configs.append({
                "domain_name": config_domain,
                "has_model": False,
                "config": domain_configs[config_domain]
            })
    
    return {
        "active_domain": active_domain,
        "total_domains": len(domains_with_configs),
        "domains": domains_with_configs
    }

@router.post("/set_active_domain")
def set_active_domain(domain: str):
    """
    DEPRECATED: This endpoint is kept for backward compatibility.
    All domains are now auto-loaded on demand when used.
    """
    available_domains = discover_domains()
    if domain not in available_domains:
        raise HTTPException(
            status_code=404, 
            detail=f"Domain '{domain}' not found. Available domains: {available_domains}"
        )
    
    try:
        load_domain(domain)
        from core.state import models
        
        return {
            "message": f"Domain '{domain}' is now loaded (note: all domains auto-load on demand)",
            "domain": domain,
            "intents": list(set(models[domain].idx2label_int.values()))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load domain: {str(e)}")

@router.post("/activate_all_domains")
def activate_all_domains():
    """
    Load and activate all available domains so they can be used simultaneously.
    This iterates through all discovered domains and loads them into memory.
    """
    import core.state as state_module
    
    available_domains = discover_domains()
    loaded = []
    errors = []
    
    logger.info(f"Activating all {len(available_domains)} domains...")
    
    for domain in available_domains:
        try:
            load_domain(domain)
            loaded.append(domain)
            logger.info(f"✅ Activated domain: {domain}")
        except Exception as e:
            error_msg = f"Failed to load {domain}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
    
    # Set the first loaded domain as active/default if none is set
    if loaded and not state_module.active_domain:
        state_module.active_domain = loaded[0]
    
    return {
        "message": f"Activated {len(loaded)}/{len(available_domains)} domains",
        "loaded_domains": loaded,
        "errors": errors,
        "active_default": state_module.active_domain
    }


@router.get("/model_info", response_model=ModelInfo)
def model_info(domain: Optional[str] = None):
    """Get information about a model. Auto-loads domain if not already loaded."""
    from core.state import models
    
    # Determine target domain
    if domain:
        target_domain = domain
    else:
        # Use first available domain if none specified
        available = discover_domains()
        if not available:
            raise HTTPException(status_code=503, detail="No domains available")
        target_domain = available[0]
    
    # Auto-load if needed
    if target_domain not in models:
        try:
            load_domain(target_domain)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Domain not found: {str(e)}")
    
    model = models[target_domain]
    
    return {
        "domain": target_domain,
        "embed_model": model.meta.get("embed_model", "unknown"),
        "intents": sorted(list(set(model.idx2label_int.values()))),
        "num_intents": len(set(model.idx2label_int.values())),
        "ml_model_type": model.clf.__class__.__name__ if model.clf else "unknown",
        "accuracy": model.meta.get("accuracy"),
        "training_date": model.meta.get("training_date")
    }

@router.post("/predict_intent", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict the intent of a user query.
    
    Domain is auto-loaded on demand if not already loaded.
    If no domain specified, uses the first available domain.
    
    Args:
        req: PredictRequest containing the query and optional metadata
        
    Returns:
        PredictResponse with intent, confidence, and optional probability distribution
    """
    from core.state import models, embedder
    
    # Determine target domain
    if req.domain:
        target_domain = req.domain
    else:
        # Use first available domain if none specified
        available = discover_domains()
        if not available:
            raise HTTPException(status_code=503, detail="No domains available")
        target_domain = available[0]
    
    # Auto-load domain if not already loaded
    if target_domain not in models:
        try:
            load_domain(target_domain)
            logger.info(f"Auto-loaded domain: {target_domain}")
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Failed to load domain '{target_domain}': {str(e)}")
    
    # Auto-initialize embedder if not loaded
    if embedder is None:
        try:
            initialize_embedder()
            logger.info("Auto-initialized embedder")
            # Re-import to get updated value
            from core.state import embedder as updated_embedder
            if updated_embedder is None:
                raise HTTPException(status_code=503, detail="Failed to initialize embedder")
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            raise HTTPException(status_code=503, detail=f"Failed to initialize embedder: {str(e)}")
    
    try:
        result = predict_intent(req.query, target_domain, req.return_all_probs)
        
        # Log prediction
        logger.info(f"[{target_domain}] Prediction: query='{req.query[:50]}...' intent={result['intent']} confidence={result['confidence']:.3f}")
        
        return PredictResponse(
            intent=result["intent"],
            confidence=result["confidence"],
            domain=result["domain"],
            all_probabilities=result.get("all_probabilities"),
            query=req.query,
            timestamp=datetime.utcnow().isoformat() + "Z",
            session_id=req.session_id,
            user_id=req.user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict_batch")
def predict_batch(
    queries: List[str], 
    domain: Optional[str] = None,
    return_all_probs: bool = False
):
    """
    Predict intents for multiple queries in batch
    
    Args:
        queries: List of query strings
        domain: Domain to use (defaults to active domain)
        return_all_probs: Whether to return all intent probabilities
        
    Returns:
        List of predictions
    """
    try:
        result = predict_batch_intents(queries, domain, return_all_probs)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@router.post("/reload_domain")
def reload_domain(domain: Optional[str] = None):
    """Reload a domain model (useful after retraining)"""
    from core.state import active_domain, models
    
    target_domain = domain or active_domain
    
    if not target_domain:
        raise HTTPException(status_code=400, detail="No domain specified and no active domain set")
    
    try:
        # Remove from cache if exists
        if target_domain in models:
            del models[target_domain]
        
        # Reload
        load_domain(target_domain)
        
        return {
            "message": f"Domain '{target_domain}' reloaded successfully",
            "domain": target_domain,
            "intents": list(set(models[target_domain].idx2label_int.values()))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload domain: {str(e)}")

@router.get("/domain_configs")
def get_domain_configs():
    """Get all domain configurations from config files"""
    try:
        all_configs = get_all_domain_configs()
        
        return {
            "total_configs": len(all_configs),
            "configs": all_configs
        }
    except Exception as e:
        logger.error(f"Error loading domain configs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load domain configs: {str(e)}")

@router.get("/domain_config/{domain}")
def get_specific_domain_config(domain: str):
    """Get configuration for a specific domain"""
    try:
        config = load_domain_config(domain)
        
        if config is None:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration not found for domain '{domain}'"
            )
        
        return {
            "domain": domain,
            "config": config
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading config for domain '{domain}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load domain config: {str(e)}")

@router.get("/available_configs")
def list_available_configs():
    """List all available domain config files"""
    try:
        config_domains = discover_domain_configs()
        trained_domains = discover_domains()
        
        configs_info = []
        for domain in config_domains:
            configs_info.append({
                "domain": domain,
                "has_trained_model": domain in trained_domains,
                "config_file": f"{domain}_config.json"
            })
        
        return {
            "total_configs": len(configs_info),
            "configs": configs_info
        }
    except Exception as e:
        logger.error(f"Error discovering configs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to discover configs: {str(e)}")

