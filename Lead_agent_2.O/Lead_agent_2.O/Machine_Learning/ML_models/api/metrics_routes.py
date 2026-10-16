"""
Metrics API Endpoint

Exposes Prometheus metrics for scraping.
"""

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST

from middleware.metrics import metrics_endpoint

router = APIRouter(tags=["Monitoring"])


@router.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint
    
    Exposes application metrics in Prometheus format for scraping.
    
    Metrics include:
    - HTTP request counts and durations
    - Authentication attempts and token generations
    - Rate limiting events
    - Database operation counts and durations
    - Cache hit/miss rates
    - LLM request counts, durations, and token usage
    - RAG query counts and durations
    - FAQ search counts and durations
    
    This endpoint should be scraped by Prometheus at regular intervals (e.g., every 15s).
    """
    return Response(
        content=metrics_endpoint(),
        media_type=CONTENT_TYPE_LATEST
    )
