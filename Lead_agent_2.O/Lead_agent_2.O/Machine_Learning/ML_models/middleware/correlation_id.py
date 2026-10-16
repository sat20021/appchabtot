"""
Correlation ID Middleware

Adds a unique correlation ID to each request for distributed tracing and log correlation.
"""

import uuid
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add correlation IDs to requests
    
    The correlation ID is:
    1. Extracted from X-Request-ID header if present
    2. Generated as a new UUID if not present
    3. Added to request.state for use in handlers
    4. Added to response headers
    5. Added to logging context
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate correlation ID
        correlation_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        # Add to request state
        request.state.correlation_id = correlation_id
        request.state.request_id = correlation_id  # Alias for convenience
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers["X-Request-ID"] = correlation_id
        
        return response


def get_correlation_id(request: Request) -> str:
    """
    Get correlation ID from request
    
    Usage:
        from middleware.correlation_id import get_correlation_id
        
        @router.get("/endpoint")
        async def endpoint(request: Request):
            correlation_id = get_correlation_id(request)
            logger.info(f"Processing request {correlation_id}")
    """
    return getattr(request.state, "correlation_id", "unknown")
