# services/classifier_service.py
# Classifier service for intent prediction
import logging
import numpy as np
from typing import Dict, Optional
from fastapi import HTTPException

from core.state import models, active_domain, embedder
from ml_utils.helpers import load_domain
from models.schemas import IntentProbability

logger = logging.getLogger(__name__)

def verify_intent_local(query: str, domain: Optional[str] = None) -> Dict:
    """
    Verify intent using the local classifier (no HTTP call needed)
    
    Args:
        query: User's query
        domain: Optional domain for classification
        
    Returns:
        Dictionary with intent, confidence, and domain
    """
    from core.state import active_domain, embedder, models
    
    target_domain = domain or active_domain
    
    if not target_domain:
        raise HTTPException(status_code=503, detail="No active domain set")
    
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not loaded")
    
    # Check Redis cache first
    try:
        from ml_utils.redis_client import get_cached_ml_prediction, cache_ml_prediction
        
        cached_prediction = get_cached_ml_prediction(query, target_domain)
        if cached_prediction:
            logger.debug(f"✅ ML cache HIT for: {query[:50]}... (domain: {target_domain})")
            return cached_prediction
    except Exception as e:
        logger.debug(f"ML cache check failed: {e}")
    
    # Load domain if not already loaded
    if target_domain not in models:
        try:
            load_domain(target_domain)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Domain not found: {str(e)}")
    
    model = models[target_domain]
    
    try:
        text = query.strip()
        
        if not text:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "domain": target_domain
            }
        
        # Generate embedding
        emb = embedder.encode([text], convert_to_numpy=True)
        
        # Get predictions
        probs = model.clf.predict_proba(emb)[0]
        top_idx = int(np.argmax(probs))
        intent = model.idx2label_int.get(top_idx, "unknown")
        confidence = float(probs[top_idx])
        
        result = {
            "intent": intent,
            "confidence": confidence,
            "domain": target_domain
        }
        
        # Cache the prediction
        try:
            from ml_utils.redis_client import cache_ml_prediction
            cache_ml_prediction(query, target_domain, result)
            logger.debug(f"💾 ML prediction cached for: {query[:50]}... (domain: {target_domain})")
        except Exception as e:
            logger.debug(f"ML prediction caching failed: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error verifying intent: {e}")
        raise HTTPException(status_code=500, detail=f"Error verifying intent: {str(e)}")

def predict_intent(query: str, domain: Optional[str] = None, return_all_probs: bool = False) -> Dict:
    """
    Predict intent for a single query
    
    Args:
        query: User query text
        domain: Domain to use (defaults to active domain)
        return_all_probs: Whether to return all intent probabilities
        
    Returns:
        Dictionary with prediction results
    """
    from core.state import active_domain, embedder, models
    from ml_utils.helpers import load_domain
    
    target_domain = domain or active_domain
    
    if not target_domain:
        raise HTTPException(status_code=503, detail="No active domain set")
    
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not loaded")
    
    # Load domain if not already loaded
    if target_domain not in models:
        try:
            load_domain(target_domain)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Domain not found: {str(e)}")
    
    model = models[target_domain]
    
    try:
        text = query.strip()
        
        # Handle empty query
        if not text:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "domain": target_domain,
                "all_probabilities": None
            }
        
        # Generate embedding
        emb = embedder.encode([text], convert_to_numpy=True)
        
        # Get predictions
        probs = model.clf.predict_proba(emb)[0]
        top_idx = int(np.argmax(probs))
        intent = model.idx2label_int.get(top_idx, "unknown")
        confidence = float(probs[top_idx])
        
        # Prepare all probabilities if requested
        all_probs = None
        if return_all_probs:
            all_probs = [
                IntentProbability(
                    intent=model.idx2label_int.get(i, f"unknown_{i}"), 
                    probability=float(p)
                )
                for i, p in enumerate(probs)
            ]
            # Sort by probability descending
            all_probs.sort(key=lambda x: x.probability, reverse=True)
        
        return {
            "intent": intent,
            "confidence": confidence,
            "domain": target_domain,
            "all_probabilities": all_probs
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

def predict_batch_intents(queries: list, domain: Optional[str] = None, return_all_probs: bool = False) -> Dict:
    """
    Predict intents for multiple queries in batch
    
    Args:
        queries: List of query strings
        domain: Domain to use (defaults to active domain)
        return_all_probs: Whether to return all intent probabilities
        
    Returns:
        Dictionary with batch prediction results
    """
    from core.state import active_domain, embedder, models
    from ml_utils.helpers import load_domain
    
    target_domain = domain or active_domain
    
    if not target_domain:
        raise HTTPException(status_code=503, detail="No active domain set")
    
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not loaded")
    
    if not queries:
        raise HTTPException(status_code=400, detail="Empty query list")
    
    if len(queries) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 queries")
    
    # Load domain if not already loaded
    if target_domain not in models:
        try:
            load_domain(target_domain)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Domain not found: {str(e)}")
    
    model = models[target_domain]
    
    try:
        # Process all queries
        cleaned_queries = [q.strip() for q in queries]
        
        # Generate embeddings for all queries at once
        embs = embedder.encode(cleaned_queries, convert_to_numpy=True)
        
        # Get predictions
        probs_batch = model.clf.predict_proba(embs)
        
        results = []
        for i, (text, probs) in enumerate(zip(cleaned_queries, probs_batch)):
            if not text:
                results.append({
                    "query": queries[i],
                    "intent": "unknown",
                    "confidence": 0.0,
                    "domain": target_domain
                })
                continue
            
            top_idx = int(np.argmax(probs))
            intent = model.idx2label_int.get(top_idx, "unknown")
            confidence = float(probs[top_idx])
            
            result = {
                "query": queries[i],
                "intent": intent,
                "confidence": confidence,
                "domain": target_domain
            }
            
            if return_all_probs:
                result["all_probabilities"] = [
                    {"intent": model.idx2label_int.get(j, f"unknown_{j}"), "probability": float(p)}
                    for j, p in enumerate(probs)
                ]
            
            results.append(result)
        
        return {"predictions": results, "count": len(results), "domain": target_domain}
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

