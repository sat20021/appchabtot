"""
Rate Limiting Middleware

Implements Redis-based rate limiting with sliding window algorithm.
"""

import time
import logging
from typing import Callable, Optional, Dict
from datetime import datetime
from functools import wraps

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ml_utils.redis_client import get_raw_store

logger = logging.getLogger(__name__)


class RateLimitConfig:
    """Rate limit configuration"""
    
    # Default limits (requests per minute)
    DEFAULT_LIMITS = {
        "authenticated": 100,
        "anonymous": 10
    }
    
    # Endpoint-specific limits
    ENDPOINT_LIMITS = {
        "/conversation": {
            "authenticated": 100,
            "anonymous": 1000
        },
        "/api/v1/rag/query": {
            "authenticated": 50,
            "anonymous": 5
        },
        "/api/v1/faq/search": {
            "authenticated": 200,
            "anonymous": 20
        },
        "/api/v1/rag/ingest": {
            "authenticated": 10,
            "anonymous": 100
        }
    }
    
    @classmethod
    def get_limit(cls, endpoint: str, is_authenticated: bool) -> int:
        """Get rate limit for an endpoint"""
        endpoint_config = cls.ENDPOINT_LIMITS.get(endpoint, cls.DEFAULT_LIMITS)
        key = "authenticated" if is_authenticated else "anonymous"
        return endpoint_config.get(key, cls.DEFAULT_LIMITS[key])


class RateLimiter:
    """
    Redis-based sliding window rate limiter
    
    Uses Redis sorted sets to track requests in a sliding time window.
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client or get_raw_store()
        self.window_size = 60  # 60 seconds (1 minute)
    
    def _get_key(self, identifier: str, endpoint: str) -> str:
        """Generate Redis key for rate limiting"""
        return f"rate_limit:{endpoint}:{identifier}"
    
    def check_rate_limit(
        self, 
        identifier: str, 
        endpoint: str, 
        limit: int
    ) -> tuple[bool, Dict[str, int]]:
        """
        Check if request is within rate limit
        
        Args:
            identifier: User ID or IP address
            endpoint: API endpoint path
            limit: Maximum requests per window
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
            rate_limit_info contains: limit, remaining, reset_time
        """
        if not self.redis:
            # If Redis is unavailable, allow the request
            logger.warning("Redis unavailable, rate limiting disabled")
            return True, {"limit": limit, "remaining": limit, "reset": 0}
        
        key = self._get_key(identifier, endpoint)
        now = time.time()
        window_start = now - self.window_size
        
        try:
            # Remove old entries outside the window
            self.redis.zremrangebyscore(key, 0, window_start)
            
            # Count requests in current window
            request_count = self.redis.zcard(key)
            
            # Calculate remaining requests
            remaining = max(0, limit - request_count)
            
            # Check if limit exceeded
            if request_count >= limit:
                # Get oldest request timestamp to calculate reset time
                oldest = self.redis.zrange(key, 0, 0, withscores=True)
                reset_time = int(oldest[0][1] + self.window_size) if oldest else int(now + self.window_size)
                
                return False, {
                    "limit": limit,
                    "remaining": 0,
                    "reset": reset_time
                }
            
            # Add current request
            self.redis.zadd(key, {str(now): now})
            
            # Set expiry on the key
            self.redis.expire(key, self.window_size)
            
            return True, {
                "limit": limit,
                "remaining": remaining - 1,  # -1 for current request
                "reset": int(now + self.window_size)
            }
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # On error, allow the request
            return True, {"limit": limit, "remaining": limit, "reset": 0}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to apply rate limiting to all requests
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.limiter = RateLimiter()
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and metrics
        if request.url.path in ["/health", "/health/ready", "/metrics"]:
            return await call_next(request)
        
        # Get user identifier
        user_id = getattr(request.state, "user_id", None)
        is_authenticated = user_id is not None
        
        # Use user_id if authenticated, otherwise use IP
        identifier = user_id if is_authenticated else (request.client.host if request.client else "unknown")
        
        # Get rate limit for this endpoint
        endpoint = request.url.path
        limit = RateLimitConfig.get_limit(endpoint, is_authenticated)
        
        # Check rate limit
        is_allowed, rate_info = self.limiter.check_rate_limit(
            identifier, endpoint, limit
        )
        
        # Add rate limit headers
        response = None
        if not is_allowed:
            # Rate limit exceeded
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please try again later.",
                    "limit": rate_info["limit"],
                    "reset": rate_info["reset"]
                },
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(rate_info["reset"]),
                    "Retry-After": str(rate_info["reset"] - int(time.time()))
                }
            )
            return response
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])
        
        return response


def rate_limit(endpoint_name: str):
    """
    Decorator for rate limiting specific endpoints
    
    Usage:
        @router.post("/conversation")
        @rate_limit("conversation")
        async def conversation_endpoint(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            limiter = RateLimiter()
            
            # Get user identifier
            user_id = getattr(request.state, "user_id", None)
            is_authenticated = user_id is not None
            identifier = user_id if is_authenticated else (request.client.host if request.client else "unknown")
            
            # Get limit
            limit = RateLimitConfig.get_limit(f"/{endpoint_name}", is_authenticated)
            
            # Check rate limit
            is_allowed, rate_info = limiter.check_rate_limit(
                identifier, endpoint_name, limit
            )
            
            if not is_allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(rate_info["limit"]),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(rate_info["reset"]),
                        "Retry-After": str(rate_info["reset"] - int(time.time()))
                    }
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator
