"""
FAQ semantic search helper that bridges the FAQ vector engine into the hybrid stack.
Enhanced with Redis caching for enterprise performance.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

try:
    from faq_app.config import Config as FAQConfig  # type: ignore
    from faq_app.services.engine import VectorEngine  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully when FAQ project missing
    FAQConfig = None
    VectorEngine = None

# Import Redis cache handler
try:
    from ml_utils.redis_cache_handler import RedisCacheHandler
    _redis_cache = RedisCacheHandler()
except ImportError:
    _redis_cache = None
    logging.warning("Redis cache handler not available, FAQ caching disabled")

logger = logging.getLogger(__name__)

_engine: Optional["VectorEngine"] = None
_engine_lock_imported = False

# Import Lock only if needed
try:
    from threading import Lock
    _engine_lock = Lock()
    _engine_lock_imported = True
except ImportError:
    _engine_lock = None


def _faq_available() -> bool:
    return VectorEngine is not None and FAQConfig is not None


def _normalize_question(question: str) -> str:
    return " ".join(question.lower().split())


def _get_engine():
    if not _faq_available():
        raise RuntimeError("FAQ search engine is not available")
    global _engine
    if _engine_lock_imported and _engine_lock:
        with _engine_lock:
            if _engine is None:
                _engine = VectorEngine()
    else:
        if _engine is None:
            _engine = VectorEngine()
    return _engine


def refresh_faq_index(domain: str = "default") -> Dict:
    """
    Rebuild the FAQ index so the latest pickle/index artifacts are hot-loaded.
    Automatically invalidates Redis cache for the domain.
    """
    if not _faq_available():
        logger.warning("FAQ project not detected; skipping index refresh.")
        return {"status": "skipped", "reason": "faq_unavailable"}

    try:
        engine = _get_engine()
        logger.info(f"Refreshing FAQ index for domain {domain}...")
        
        # CRITICAL: Invalidate Redis cache BEFORE rebuilding
        # This clears the OLD index/query cache so the new one isn't immediately deleted
        if _redis_cache and _redis_cache.is_available():
            invalidated = _redis_cache.invalidate_pattern(f"faq:{domain}:*")
            logger.info(f"✓ Invalidated {invalidated} FAQ cache entries for domain: {domain}")
        
        # Clear in-memory cache for this domain to force fresh build
        if domain in engine.indexes:
            del engine.indexes[domain]
            logger.info(f"✓ Cleared in-memory index for domain: {domain}")
        
        # Now rebuild and cache the NEW index
        result = engine.build_full_index(domain=domain)
        
        logger.info(f"FAQ index refresh complete for {domain}: {result}")
        return result
    except Exception as e:
        logger.error(f"Error refreshing FAQ index: {e}")
        return {"status": "error", "reason": str(e)}


def warm_up_faq_index(domain: str = "default") -> bool:
    """
    Warm up the FAQ index by initializing the engine and loading the index.
    """
    if not _faq_available():
        return False
        
    try:
        logger.info(f"🔥 Warming up FAQ index for domain: {domain}")
        # This triggers lazy loading of the engine and index
        _get_engine()
        # Perform a dummy search to ensure index is fully loaded in memory
        ask_faq("warmup", domain=domain)
        logger.info(f"🔥 FAQ index warmed up for {domain}")
        return True
    except Exception as e:
        logger.warning(f"FAQ warm-up failed for {domain}: {e}")
        return False


def ask_faq(question: str, domain: str = "default", threshold: Optional[float] = None) -> Dict:
    """
    Query the FAQ vector engine with Redis caching.
    
    Caching Strategy:
    - Redis cache (persistent, shared across instances)
    - TTL: 1 hour (3600s)
    - Auto-invalidated on index rebuild
    
    Args:
        question: User query to search
        domain: Domain to search in (default: "default")
        threshold: Confidence threshold (defaults to 0.8)
    
    Returns:
        Dict with answer, confidence, matched_question, source, reason
    """
    if not _faq_available():
        return {"answer": None, "confidence": 0.0, "reason": "faq_unavailable"}

    try:
        normalized = _normalize_question(question)
        # Default to 0.8 threshold for FAQ routing
        eff_threshold = round(
            threshold or getattr(FAQConfig, "CONFIDENCE_THRESHOLD", 0.8), 3
        )
        
        # Redis cache key
        cache_key = f"faq:{domain}:{normalized}:{eff_threshold}"

        # 1. Try Redis cache first (fastest)
        if _redis_cache and _redis_cache.is_available():
            cached = _redis_cache.get_json(cache_key)
            if cached:
                logger.debug(f"✓ FAQ Redis cache HIT: {cache_key[:50]}...")
                return {**cached, "source": "redis_cache"}
        
        # 2. Query FAQ vector engine (cache miss)
        logger.debug(f"✗ FAQ cache MISS: {cache_key[:50]}... querying engine")
        engine = _get_engine()
        result = engine.search(question, domain=domain, threshold=eff_threshold)
        
        enriched = {
            "answer": result.get("answer"),
            "confidence": result.get("confidence") or result.get("score"),
            "matched_question": result.get("matched_question"),
            "reason": result.get("reason"),
            "source": result.get("source") or "vector_db",
            "domain": result.get("domain")
        }
        
        # 3. Cache in Redis (1 hour TTL)
        if _redis_cache and _redis_cache.is_available():
            _redis_cache.set_json(cache_key, enriched, ttl=3600)
            logger.debug(f"✓ Cached FAQ result in Redis (TTL=1h)")
        
        return enriched
    except Exception as e:
        logger.error(f"Error querying FAQ engine: {e}")
        return {"answer": None, "confidence": 0.0, "reason": "internal_error"}


def faq_stats(domain: str = "default") -> Dict:
    """
    Provide current FAQ status for health metrics.
    Includes Redis cache statistics.
    """
    if not _faq_available():
        return {"available": False, "vectors_indexed": 0, "redis_cache": False}
    
    try:
        engine = _get_engine()
        vectors = 0
        if domain in engine.indexes:
            vectors = engine.indexes[domain]['index'].ntotal
        
        stats = {
            "available": True,
            "vectors_indexed": vectors,
            "domain": domain,
            "redis_cache": _redis_cache.is_available() if _redis_cache else False
        }
        
        # Add Redis cache stats if available
        if _redis_cache and _redis_cache.is_available():
            redis_stats = _redis_cache.get_cache_stats()
            stats["redis_total_keys"] = redis_stats.get("total_keys", 0)
            stats["redis_hit_rate"] = redis_stats.get("hit_rate", 0)
        
        return stats
    except Exception as e:
        return {"available": False, "error": str(e)}


def load_faq_domain(domain: str) -> bool:
    """Explicitly loads a domain into memory."""
    if not _faq_available():
        return False
    try:
        engine = _get_engine()
        return engine.load_domain(domain)
    except Exception as e:
        logger.error(f"Error loading FAQ domain {domain}: {e}")
        return False


def clear_faq_cache(domain: Optional[str] = None):
    """
    Clear FAQ cache (Redis or legacy).
    
    Args:
        domain: If specified, only clear cache for that domain
    """
    if _redis_cache and _redis_cache.is_available():
        if domain:
            pattern = f"faq:{domain}:*"
            invalidated = _redis_cache.invalidate_pattern(pattern)
            logger.info(f"Cleared {invalidated} FAQ cache entries for domain: {domain}")
        else:
            # Clear all FAQ cache entries
            invalidated = _redis_cache.invalidate_pattern("faq:*")
            logger.info(f"Cleared {invalidated} FAQ cache entries (all domains)")
    else:
        logger.warning("Redis cache not available, cannot clear")


def reset_faq_engine():
    """
    Helper for tests to reset singleton.
    """
    global _engine
    _engine = None

