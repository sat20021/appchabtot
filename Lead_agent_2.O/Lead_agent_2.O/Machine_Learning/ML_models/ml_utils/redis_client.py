"""
Redis utility helper with graceful in-memory fallback.
"""
import json
import logging
import os
import hashlib
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Dict, Optional, List

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - dependency missing only during install
    redis = None

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL")

# TTL Configuration (in seconds)
DEFAULT_TTL_SECONDS = 60 * 5      # 5 minutes (sessions)
FAQ_CACHE_TTL = 60 * 60           # 1 hour (FAQ results)
ML_PREDICTION_TTL = 60 * 30       # 30 minutes (ML predictions)


class _InMemoryStore:
    """Minimal key/value store with TTL support used as fallback when Redis is unavailable."""

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._expirations: Dict[str, Optional[datetime]] = {}
        self._lock = RLock()

    def set(self, key: str, value: Any, ex: Optional[int] = None, **kwargs):
        with self._lock:
            self._store[key] = value
            self._expirations[key] = (
                datetime.now(timezone.utc) + timedelta(seconds=ex) if ex else None
            )

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            exp = self._expirations.get(key)
            if exp and exp < datetime.now(timezone.utc):
                self.delete(key)
                return None
            return self._store.get(key)

    def keys(self, pattern: str = "*") -> List[str]:
        """Emulate Redis KEYS command"""
        import fnmatch
        with self._lock:
            valid_keys = []
            for key in list(self._store.keys()):
                exp = self._expirations.get(key)
                if exp and exp < datetime.now(timezone.utc):
                    continue
                if fnmatch.fnmatch(key, pattern):
                    valid_keys.append(key)
            return valid_keys

    def delete(self, key: str):
        with self._lock:
            self._store.pop(key, None)
            self._expirations.pop(key, None)

    def flushall(self):
        """Clear all keys from the in-memory store."""
        with self._lock:
            self._store.clear()
            self._expirations.clear()

    # --- Redis Sorted Set (ZSET) Emulation for Rate Limiting ---

    def zremrangebyscore(self, key: str, min_score: float, max_score: float):
        """Emulate Redis ZREMRANGEBYSCORE"""
        with self._lock:
            if key not in self._store or not isinstance(self._store[key], list):
                return 0
            
            # Filter out items within the score range [min, max]
            # In our case, the items are (score, value) tuples
            original_len = len(self._store[key])
            self._store[key] = [
                item for item in self._store[key] 
                if item[0] < min_score or item[0] > max_score
            ]
            return original_len - len(self._store[key])

    def zcard(self, key: str) -> int:
        """Emulate Redis ZCARD"""
        with self._lock:
            if key not in self._store or not isinstance(self._store[key], list):
                return 0
            return len(self._store[key])

    def zadd(self, key: str, mapping: Dict[str, float]):
        """Emulate Redis ZADD"""
        with self._lock:
            if key not in self._store or not isinstance(self._store[key], list):
                self._store[key] = []
            
            for val, score in mapping.items():
                self._store[key].append((score, val))
            
            # Keep it sorted by score
            self._store[key].sort(key=lambda x: x[0])

    def zrange(self, key: str, start: int, end: int, withscores: bool = False):
        """Emulate Redis ZRANGE"""
        with self._lock:
            if key not in self._store or not isinstance(self._store[key], list):
                return []
            
            # End is inclusive in Redis
            if end == -1:
                items = self._store[key][start:]
            else:
                items = self._store[key][start:end + 1]
            
            if withscores:
                # Redis returns (value, score)
                return [(item[1], item[0]) for item in items]
            return [item[1] for item in items]

    def expire(self, key: str, seconds: int):
        """Emulate Redis EXPIRE"""
        with self._lock:
            if key in self._store:
                self._expirations[key] = datetime.now(timezone.utc) + timedelta(seconds=seconds)
                return True
            return False


_store: Optional[Any] = None
_is_redis = False


def _init_store():
    global _is_redis
    if redis and REDIS_URL:
        try:
            client = redis.from_url(REDIS_URL, decode_responses=True)
            client.ping()
            _is_redis = True
            logger.info("=" * 60)
            logger.info("✅ REDIS CONNECTION SUCCESSFUL")
            logger.info(f"   Connected to: {REDIS_URL}")
            logger.info("   Performance caching: ENABLED")
            logger.info("=" * 60)
            return client
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("=" * 60)
            logger.warning("⚠️  REDIS CONNECTION FAILED")
            logger.warning(f"   Error: {exc}")
            logger.warning("   Falling back to IN-MEMORY cache")
            logger.warning("   Performance caching: ENABLED (in-memory)")
            logger.warning("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("ℹ️  REDIS NOT CONFIGURED")
        logger.info("   Using IN-MEMORY cache for sessions and performance")
        logger.info("   Performance caching: ENABLED (in-memory)")
        logger.info("=" * 60)
    _is_redis = False
    return _InMemoryStore()


def _get_store():
    global _store
    if _store is None:
        _store = _init_store()
    return _store


def is_redis_backed() -> bool:
    _get_store()
    return _is_redis


def set_json(key: str, data: Dict[str, Any], ttl: Optional[int] = None, nx: bool = False) -> bool:
    """
    Set JSON data with optional TTL and NX flag.
    
    Args:
        key: Redis key
        data: Data to store
        ttl: Time-to-live in seconds
        nx: If True, only set if key doesn't exist (for atomic locking)
    
    Returns:
        True if set successfully, False if nx=True and key exists
    """
    store = _get_store()
    payload = json.dumps(data)
    expiry = ttl or DEFAULT_TTL_SECONDS
    
    if nx:
        if _is_redis:
            # Redis native NX support
            result = store.set(key, payload, ex=expiry, nx=True)
            return result is not None
        else:
            # In-memory fallback with lock
            with store._lock:
                if key in store._store:
                    # Check if expired
                    exp = store._expirations.get(key)
                    if exp and exp < datetime.now(timezone.utc):
                        # Expired, allow overwrite
                        pass
                    else:
                        # Not expired, NX fails
                        return False
                store._store[key] = payload
                store._expirations[key] = datetime.now(timezone.utc) + timedelta(seconds=expiry)
                return True
    else:
        store.set(key, payload, ex=expiry)
        return True


def get_json(key: str) -> Optional[Dict[str, Any]]:
    store = _get_store()
    value = store.get(key)
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Failed to decode JSON for key '%s'", key)
        return None


def delete_key(key: str):
    store = _get_store()
    store.delete(key)


def flush_all():
    """Clear all cache entries. Used on server startup."""
    store = _get_store()
    if _is_redis:
        store.flushall()
        logger.info("🗑️  Redis cache cleared (flushall)")
    else:
        store.flushall()
        logger.info("🗑️  In-memory cache cleared")


# ===== FAQ CACHING =====

def cache_faq_result(query: str, results: list, ttl: int = FAQ_CACHE_TTL):
    """Cache FAQ search results."""
    query_hash = hashlib.md5(query.lower().encode('utf-8')).hexdigest()
    key = f"faq:query:{query_hash}"
    data = {
        "query": query,
        "results": results,
        "cached_at": datetime.now(timezone.utc).isoformat()
    }
    set_json(key, data, ttl=ttl)


def get_cached_faq(query: str) -> Optional[list]:
    """Retrieve cached FAQ results."""
    query_hash = hashlib.md5(query.lower().encode('utf-8')).hexdigest()
    key = f"faq:query:{query_hash}"
    cached = get_json(key)
    if cached:
        logger.debug(f"FAQ cache HIT for query: {query[:50]}...")
        return cached.get("results")
    logger.debug(f"FAQ cache MISS for query: {query[:50]}...")
    return None


# ===== ML PREDICTION CACHING =====

def cache_ml_prediction(query: str, domain: str, prediction: dict, ttl: int = ML_PREDICTION_TTL):
    """Cache ML intent prediction."""
    query_hash = hashlib.md5(query.lower().encode('utf-8')).hexdigest()
    key = f"ml:intent:{domain}:{query_hash}"
    data = {
        "query": query,
        "domain": domain,
        "prediction": prediction,
        "cached_at": datetime.now(timezone.utc).isoformat()
    }
    set_json(key, data, ttl=ttl)


def get_cached_ml_prediction(query: str, domain: str) -> Optional[dict]:
    """Retrieve cached ML prediction."""
    query_hash = hashlib.md5(query.lower().encode('utf-8')).hexdigest()
    key = f"ml:intent:{domain}:{query_hash}"
    cached = get_json(key)
    if cached:
        logger.debug(f"ML cache HIT for query: {query[:50]}... (domain: {domain})")
        return cached.get("prediction")
    logger.debug(f"ML cache MISS for query: {query[:50]}... (domain: {domain})")
    return None


def get_raw_store():
    """
    Access the underlying store (redis client or in-memory) for advanced use cases.
    """
    return _get_store()


def get_all_keys(pattern: str = "*") -> list:
    """
    Get all keys matching a pattern.
    
    Args:
        pattern: Redis pattern (e.g., "handoff_session:*")
        
    Returns:
        List of matching key names
    """
    store = _get_store()
    
    if _is_redis:
        # Use Redis KEYS command (for small datasets)
        # In production, consider SCAN for large datasets
        try:
            return store.keys(pattern)
        except Exception as e:
            logger.warning(f"Failed to get keys with pattern '{pattern}': {e}")
            return []
    else:
        # In-memory store: filter by pattern
        import fnmatch
        with store._lock:
            # Check expiration and filter
            valid_keys = []
            for key in list(store._store.keys()):
                exp = store._expirations.get(key)
                if exp and exp < datetime.now(timezone.utc):
                    continue
                if fnmatch.fnmatch(key, pattern):
                    valid_keys.append(key)
            return valid_keys

