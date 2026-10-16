# api/pipeline_routes.py
# Pipeline Manager API endpoints
# Refactored to use domain as URL path parameter
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Annotated

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Path as PathParam, Body

from models.schemas import (
    PipelineStatusResponse, IntentsConfig, PipelineStatus
)
from core.state import pipeline_state
from ml_utils.helpers import validate_intent_config, update_pipeline_state, MODELS_DIR
from ml_services.pipeline_service import run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# ============================================================================
# DOMAIN-SPECIFIC ENDPOINTS (New RESTful design)
# ============================================================================

@router.post("/{domain}/train")
async def train_domain_model(
    domain: Annotated[str, PathParam(..., description="Domain name to train model for")],
    config: Annotated[Dict, Body(...)],
    background_tasks: BackgroundTasks = None
):
    """
    Train a classification model for a specific domain.
    
    The domain in the URL path takes precedence over any domain in the config body.
    
    Expected body format:
    {
        "intents": {
            "greeting": {"templates": ["hi", "hello"]},
            "lead_form": {"templates": ["contact me", "get quote"]}
        },
        "roles": ["backend engineer"],  // optional
        "skills": ["python"],           // optional
        "samples_per_intent": 400,      // optional, default 400
        "train_split": 0.8,             // optional, default 0.8
        "add_noise": true               // optional, default true
    }
    """
    # Check if pipeline is already running
    if pipeline_state["status"] not in [PipelineStatus.IDLE, PipelineStatus.COMPLETED, PipelineStatus.FAILED]:
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline is already running with status: {pipeline_state['status']}"
        )
    
    try:
        # Build config with URL domain taking precedence
        config["domain"] = domain
        
        # Create IntentsConfig object
        intents_config = IntentsConfig(**config)
        
        # Validate configuration
        validate_intent_config(intents_config)
        
        # Reset pipeline state
        update_pipeline_state(
            status=PipelineStatus.IDLE,
            domain=domain,
            start_time=None,
            completion_time=None,
            current_step=None,
            error=None,
            dataset_path=None,
            ml_model_path=None,
            metrics=None
        )
        
        # Start pipeline in background
        background_tasks.add_task(run_pipeline, intents_config)
        
        return {
            "message": f"Pipeline started for domain: {domain}",
            "domain": domain,
            "intents": list(intents_config.intents.keys()),
            "samples_per_intent": intents_config.samples_per_intent
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting pipeline for {domain}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {str(e)}")


@router.post("/{domain}/upload")
async def upload_domain_config(
    domain: Annotated[str, PathParam(..., description="Domain name to upload config for")],
    file: Annotated[UploadFile, File(description="Intent configuration JSON file")],
    background_tasks: BackgroundTasks = None
):
    """
    Upload intent configuration JSON file for a specific domain and start the ML pipeline.
    
    The domain in the URL path takes precedence over any domain in the file.
    """
    try:
        # Read and parse JSON file
        content = await file.read()
        config_dict = json.loads(content)
        
        # URL domain takes precedence
        config_dict["domain"] = domain
        
        # Create config object and train
        config = IntentsConfig(**config_dict)
        
        # Validate configuration
        validate_intent_config(config)
        
        # Reset pipeline state
        update_pipeline_state(
            status=PipelineStatus.IDLE,
            domain=domain,
            start_time=None,
            completion_time=None,
            current_step=None,
            error=None,
            dataset_path=None,
            ml_model_path=None,
            metrics=None
        )
        
        # Start pipeline in background
        background_tasks.add_task(run_pipeline, config)
        
        return {
            "message": f"Pipeline started for domain: {domain}",
            "domain": domain,
            "intents": list(config.intents.keys()),
            "samples_per_intent": config.samples_per_intent
        }
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing uploaded file for {domain}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@router.get("/{domain}/status", response_model=PipelineStatusResponse)
def get_domain_pipeline_status(
    domain: str = PathParam(..., description="Domain to check status for")
):
    """
    Get pipeline status for a specific domain.
    
    Note: Currently there's a single global pipeline state. 
    This endpoint filters to show only if the current pipeline matches the domain.
    """
    if pipeline_state["domain"] == domain:
        return PipelineStatusResponse(**pipeline_state)
    
    # Check if domain has a trained model
    domain_path = MODELS_DIR / domain
    if domain_path.exists():
        meta_path = domain_path / "model_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                return PipelineStatusResponse(
                    status=PipelineStatus.COMPLETED,
                    domain=domain,
                    completion_time=meta.get("training_date"),
                    current_step="Model available",
                    ml_model_path=str(domain_path / "classifier.pkl"),
                    metrics={"accuracy": meta.get("accuracy", 0.0)}
                )
            except Exception:
                pass
    
    return PipelineStatusResponse(
        status=PipelineStatus.IDLE,
        domain=domain,
        current_step="No model trained for this domain"
    )


@router.get("/{domain}/model")
def get_domain_model_info(
    domain: str = PathParam(..., description="Domain to get model info for")
):
    """Get model information for a specific domain."""
    domain_path = MODELS_DIR / domain
    meta_path = domain_path / "model_meta.json"
    
    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for domain: {domain}"
        )
    
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        
        return {
            "domain": domain,
            "num_intents": meta.get("num_intents", 0),
            "intents": list(meta.get("idx2label", {}).values()),
            "accuracy": meta.get("accuracy", 0.0),
            "training_date": meta.get("training_date", "unknown"),
            "embed_model": meta.get("embed_model", "unknown"),
            "has_classifier": (domain_path / "classifier.pkl").exists()
        }
    except Exception as e:
        logger.error(f"Error loading model info for {domain}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{domain}/stop")
def stop_domain_pipeline(
    domain: str = PathParam(..., description="Domain to stop pipeline for")
):
    """Stop the pipeline if it's running for the specified domain."""
    from datetime import datetime
    
    if pipeline_state["domain"] != domain:
        return {"message": f"No pipeline is running for domain: {domain}"}
    
    if pipeline_state["status"] in [PipelineStatus.IDLE, PipelineStatus.COMPLETED, PipelineStatus.FAILED]:
        return {"message": f"No active pipeline for domain: {domain}"}
    
    update_pipeline_state(
        status=PipelineStatus.FAILED,
        completion_time=datetime.utcnow().isoformat() + "Z",
        error="Pipeline stopped by user"
    )
    
    return {"message": f"Pipeline stop requested for domain: {domain}"}


# ============================================================================
# DOMAIN LISTING ENDPOINT
# ============================================================================

@router.get("/domains")
def list_pipeline_domains():
    """List all available trained domains from local storage."""
    models_dir = MODELS_DIR
    models_dir.mkdir(exist_ok=True)
    
    domains_dict = {}
    
    # 1. Check local filesystem first
    if models_dir.exists():
        for domain_path in models_dir.iterdir():
            if domain_path.is_dir():
                meta_path = domain_path / "model_meta.json"
                if meta_path.exists():
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                        domains_dict[domain_path.name] = {
                            "domain": domain_path.name,
                            "num_intents": meta.get("num_intents", 0),
                            "accuracy": meta.get("accuracy", 0.0),
                            "training_date": meta.get("training_date", "unknown"),
                            "intents": list(meta.get("idx2label", {}).values()),
                            "source": "local"
                        }
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {domain_path.name}: {e}")
    
    # 2. Check remote/secondary storage for additional domains
    try:
        from ml_utils.s3_handler import S3Handler
        s3 = S3Handler()
        
        s3_files = s3.list_files("models/")
        s3_domains = set()
        
        for file_key in s3_files:
            parts = file_key.split('/')
            if len(parts) >= 2 and parts[0] == "models":
                s3_domains.add(parts[1])
        
        for domain_name in s3_domains:
            if domain_name in domains_dict:
                domains_dict[domain_name]["source"] = "local+storage"
                continue
            
            # Try to download metadata from storage
            s3_meta_key = f"models/{domain_name}/model_meta.json"
            local_meta_path = models_dir / domain_name / "model_meta.json"
            
            try:
                # Ensure directory exists
                local_meta_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Check if we should download
                should_download = True
                if local_meta_path.exists():
                    # Only download if newer? For now just skip if exists
                    should_download = s3.use_s3 # If not using s3, it's already local
                
                if should_download:
                    s3.download_file(s3_meta_key, str(local_meta_path))
                
                if local_meta_path.exists():
                    with open(local_meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        
                    domains_dict[domain_name] = {
                        "domain": domain_name,
                        "num_intents": meta.get("num_intents", 0),
                        "accuracy": meta.get("accuracy", 0.0),
                        "training_date": meta.get("training_date", "unknown"),
                        "intents": list(meta.get("idx2label", {}).values()),
                        "source": "storage"
                    }
            except Exception as e:
                logger.warning(f"Failed to load storage metadata for {domain_name}: {e}")
                
    except Exception as e:
        logger.error(f"Error checking storage for domains: {e}")
    
    return {"domains": list(domains_dict.values()), "count": len(domains_dict)}
