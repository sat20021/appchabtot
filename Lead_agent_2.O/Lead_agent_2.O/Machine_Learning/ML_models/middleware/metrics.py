"""
Prometheus Metrics Middleware

Exposes application metrics for monitoring and alerting.
"""

import time
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

# ============================================================================
# Metrics Definitions
# ============================================================================

# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

http_requests_in_progress = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests in progress",
    ["method", "endpoint"]
)

# Authentication metrics
auth_attempts_total = Counter(
    "auth_attempts_total",
    "Total authentication attempts",
    ["status"]  # success, failure
)

auth_token_generations_total = Counter(
    "auth_token_generations_total",
    "Total JWT tokens generated",
    ["token_type"]  # access, refresh
)

# Rate limiting metrics
rate_limit_exceeded_total = Counter(
    "rate_limit_exceeded_total",
    "Total rate limit exceeded events",
    ["endpoint"]
)

# Database metrics
db_operations_total = Counter(
    "db_operations_total",
    "Total database operations",
    ["operation", "collection", "status"]
)

db_operation_duration_seconds = Histogram(
    "db_operation_duration_seconds",
    "Database operation duration in seconds",
    ["operation", "collection"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

# Cache metrics
cache_operations_total = Counter(
    "cache_operations_total",
    "Total cache operations",
    ["operation", "status"]  # operation: get, set, delete; status: hit, miss, error
)

# LLM metrics
llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM API requests",
    ["provider", "model", "status"]
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration in seconds",
    ["provider", "model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

llm_tokens_used_total = Counter(
    "llm_tokens_used_total",
    "Total tokens used",
    ["provider", "model", "token_type"]  # token_type: prompt, completion
)

# RAG metrics
rag_queries_total = Counter(
    "rag_queries_total",
    "Total RAG queries",
    ["domain", "status"]
)

rag_documents_ingested_total = Counter(
    "rag_documents_ingested_total",
    "Total documents ingested",
    ["domain", "status"]
)

rag_query_duration_seconds = Histogram(
    "rag_query_duration_seconds",
    "RAG query duration in seconds",
    ["domain"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# FAQ metrics
faq_searches_total = Counter(
    "faq_searches_total",
    "Total FAQ searches",
    ["domain", "status"]
)

faq_search_duration_seconds = Histogram(
    "faq_search_duration_seconds",
    "FAQ search duration in seconds",
    ["domain"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect HTTP request metrics
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Get endpoint pattern (remove path parameters)
        endpoint = request.url.path
        method = request.method
        
        # Track in-progress requests
        http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()
        
        # Track request duration
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            status_code = response.status_code
            
            # Record metrics
            duration = time.time() - start_time
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            return response
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=500
            ).inc()
            
            raise
            
        finally:
            # Decrement in-progress counter
            http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()


def metrics_endpoint():
    """
    Endpoint to expose Prometheus metrics
    
    Usage:
        from middleware.metrics import metrics_endpoint
        
        @app.get("/metrics")
        async def metrics():
            return Response(
                content=metrics_endpoint(),
                media_type=CONTENT_TYPE_LATEST
            )
    """
    return generate_latest()
