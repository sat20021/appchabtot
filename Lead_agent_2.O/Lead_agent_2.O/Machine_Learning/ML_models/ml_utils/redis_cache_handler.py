import redis
import pickle
import json
import os
import logging
from typing import Any, Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


class RedisCacheHandler:
    """
    Enterprise-grade Redis caching layer for high-performance data access.
    
    Provides:
    - Sub-millisecond access to frequently used data
    - Automatic TTL-based expiration
    - JSON and binary (pickle) serialization
    - Namespace separation for different data types
    - Connection pooling for performance
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern for shared Redis connection pool"""
        if cls._instance is None:
            cls._instance = super(RedisCacheHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        use_cluster = os.getenv("REDIS_CLUSTER", "false").lower() == "true"
        
        try:
            if use_cluster:
                # Redis Cluster mode (high availability)
                try:
                    from redis.cluster import RedisCluster
                except ImportError:
                    logger.error("redis-py-cluster not installed. Install with: pip install redis[cluster]")
                    self.redis_client = None
                    self._available = False
                    self._initialized = True
                    return
                
                cluster_nodes_str = os.getenv("REDIS_CLUSTER_NODES", "localhost:7001,localhost:7002,localhost:7003")
                startup_nodes = []
                
                for node in cluster_nodes_str.split(","):
                    try:
                        host, port = node.strip().split(":")
                        startup_nodes.append({"host": host, "port": int(port)})
                    except Exception as e:
                        logger.warning(f"Invalid cluster node format '{node}': {e}")
                
                if not startup_nodes:
                    raise ValueError("No valid Redis cluster nodes configured")
                
                self.redis_client = RedisCluster(
                    startup_nodes=startup_nodes,
                    decode_responses=False,  # Binary mode for pickle support
                    skip_full_coverage_check=True,  # Allow partial cluster
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                
                # Test connection
                self.redis_client.ping()
                logger.info(f"✅ Redis Cluster connected: {len(startup_nodes)} nodes")
                self._available = True
                
            else:
                # Single instance mode (existing behavior)
                self.redis_client = redis.Redis.from_url(
                    redis_url,
                    decode_responses=False,  # Binary mode for pickle support
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Test connection
                self.redis_client.ping()
                logger.info(f"✅ Redis cache connected: {redis_url}")
                self._available = True
            
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable: {e}. Cache disabled, using fallback.")
            self.redis_client = None
            self._available = False
        
        self._initialized = True
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        return self._available and self.redis_client is not None
    
    # ==================== JSON Operations ====================
    
    def get_json(self, key: str) -> Optional[Dict]:
        """
        Get JSON data from Redis cache
        
        Args:
            key: Cache key (will be prefixed with 'json:')
            
        Returns:
            Parsed JSON dict or None if not found/error
        """
        if not self.is_available():
            return None
        
        try:
            data = self.redis_client.get(f"json:{key}")
            if data:
                logger.debug(f"✓ Redis cache HIT: json:{key}")
                return json.loads(data)
            logger.debug(f"✗ Redis cache MISS: json:{key}")
            return None
        except Exception as e:
            logger.warning(f"Redis get_json error for '{key}': {e}")
            return None
    
    def set_json(self, key: str, data: Dict, ttl: int = 3600) -> bool:
        """
        Set JSON data in Redis cache with TTL
        
        Args:
            key: Cache key (will be prefixed with 'json:')
            data: Dictionary to cache
            ttl: Time-to-live in seconds (default: 1 hour)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            self.redis_client.setex(
                f"json:{key}",
                ttl,
                json.dumps(data)
            )
            logger.debug(f"✓ Redis cache SET: json:{key} (TTL={ttl}s)")
            return True
        except Exception as e:
            logger.warning(f"Redis set_json error for '{key}': {e}")
            return False
    
    # ==================== Binary Operations ====================
    
    def get_binary(self, key: str) -> Optional[Any]:
        """
        Get pickled binary data from Redis cache
        
        Args:
            key: Cache key (will be prefixed with 'bin:')
            
        Returns:
            Unpickled object or None if not found/error
        """
        if not self.is_available():
            return None
        
        try:
            data = self.redis_client.get(f"bin:{key}")
            if data:
                logger.debug(f"✓ Redis cache HIT: bin:{key}")
                return pickle.loads(data)
            logger.debug(f"✗ Redis cache MISS: bin:{key}")
            return None
        except Exception as e:
            logger.warning(f"Redis get_binary error for '{key}': {e}")
            return None
    
    def set_binary(self, key: str, data: Any, ttl: int = 3600) -> bool:
        """
        Set pickled binary data in Redis cache with TTL
        
        Args:
            key: Cache key (will be prefixed with 'bin:')
            data: Any picklable Python object
            ttl: Time-to-live in seconds (default: 1 hour)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            serialized = pickle.dumps(data)
            # Don't cache if > 10MB (too large for Redis)
            if len(serialized) > 10 * 1024 * 1024:
                logger.warning(f"Binary data too large for Redis: {len(serialized)} bytes")
                return False
            
            self.redis_client.setex(
                f"bin:{key}",
                ttl,
                serialized
            )
            logger.debug(f"✓ Redis cache SET: bin:{key} (TTL={ttl}s, size={len(serialized)} bytes)")
            return True
        except Exception as e:
            logger.warning(f"Redis set_binary error for '{key}': {e}")
            return False
    
    # ==================== Cache Management ====================
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry (both JSON and binary)
        
        Args:
            key: Cache key (without prefix)
            
        Returns:
            True if at least one key was deleted
        """
        if not self.is_available():
            return False
        
        try:
            deleted = self.redis_client.delete(f"json:{key}", f"bin:{key}")
            logger.debug(f"✓ Redis cache INVALIDATE: {key} ({deleted} keys removed)")
            return deleted > 0
        except Exception as e:
            logger.warning(f"Redis invalidate error for '{key}': {e}")
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern
        
        Args:
            pattern: Redis key pattern (e.g., "faq:tech_company:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.is_available():
            return 0
        
        try:
            # Search for JSON, binary prefixed keys AND raw keys (for direct usage like VectorEngine)
            json_pattern = f"json:{pattern}"
            bin_pattern = f"bin:{pattern}"
            raw_pattern = pattern
            
            json_keys = self.redis_client.keys(json_pattern)
            bin_keys = self.redis_client.keys(bin_pattern)
            raw_keys = self.redis_client.keys(raw_pattern)
            
            # Use set to avoid duplicates if patterns overlap
            keys = list(set(list(json_keys) + list(bin_keys) + list(raw_keys)))
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"✓ Redis invalidated pattern '{pattern}': {deleted} keys (json:{len(json_keys)}, bin:{len(bin_keys)}, raw:{len(raw_keys)})")
                return deleted
            logger.debug(f"No keys found for pattern '{pattern}' (searched json:{json_pattern}, bin:{bin_pattern}, raw:{raw_pattern})")
            return 0
        except Exception as e:
            logger.warning(f"Redis invalidate_pattern error for '{pattern}': {e}")
            return 0
    
    def clear_all(self) -> bool:
        """Clear all cache entries (use with caution!)"""
        if not self.is_available():
            return False
        
        try:
            self.redis_client.flushdb()
            logger.warning("⚠️ Redis cache CLEARED (all keys deleted)")
            return True
        except Exception as e:
            logger.error(f"Redis clear_all error: {e}")
            return False
    
    # ==================== Warm Cache / Prefetching ====================
    
    def warm_cache_configs(self) -> Dict[str, bool]:
        """
        Prefetch all domain configs into Redis on startup
        
        Returns:
            Dict mapping domain names to success status
        """
        if not self.is_available():
            logger.info("Redis unavailable, skipping cache warming")
            return {}
        
        logger.info("🔥 Warming Redis cache with domain configs...")
        
        from ml_utils.s3_handler import S3Handler
        
        results = {}
        s3 = S3Handler()
        
        if not s3.s3_client:
            logger.warning("S3 unavailable, cannot warm cache")
            return {}
        
        try:
            # List all config files in S3
            config_files = s3.list_files("configs/")
            
            for s3_key in config_files:
                if not s3_key.endswith("_config.json"):
                    continue
                
                # Extract domain name
                filename = Path(s3_key).name
                domain = filename.replace("_config.json", "")
                
                try:
                    # Download config
                    local_path = Path(f"configs/{filename}")
                    local_path.parent.mkdir(exist_ok=True)
                    
                    if s3.download_file(s3_key, str(local_path)):
                        with open(local_path, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                        
                        # Cache in Redis
                        success = self.set_json(f"config:{domain}", config_data, ttl=3600)
                        results[domain] = success
                        
                        if success:
                            logger.info(f"  ✓ Cached config: {domain}")
                    
                except Exception as e:
                    logger.warning(f"  ✗ Failed to cache config {domain}: {e}")
                    results[domain] = False
            
            logger.info(f"🔥 Cache warming complete: {sum(results.values())}/{len(results)} configs cached")
            return results
            
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
            return {}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        if not self.is_available():
            return {"status": "unavailable"}
        
        try:
            info = self.redis_client.info("stats")
            keyspace = self.redis_client.info("keyspace")
            
            return {
                "status": "connected",
                "total_keys": self.redis_client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(info),
                "keyspace": keyspace
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate percentage"""
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        
        if total == 0:
            return 0.0
        
        return round((hits / total) * 100, 2)
