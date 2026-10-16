"""
Health Check Endpoints

Provides comprehensive health check endpoints for Kubernetes orchestration.
"""

import logging
import time
from typing import Dict, Optional
from datetime import datetime

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from core.state import mongo_client, mongo_db
from ml_utils.redis_client import get_raw_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


class HealthStatus(BaseModel):
    """Health check response"""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    version: str = "3.0.0"
    checks: Dict[str, Dict]


class DependencyCheck(BaseModel):
    """Individual dependency health check"""
    status: str  # "up", "down", "degraded"
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    details: Optional[Dict] = None


def check_mongodb() -> DependencyCheck:
    """Check MongoDB connection"""
    start = time.time()
    try:
        if not mongo_client or not mongo_db:
            return DependencyCheck(
                status="down",
                error="MongoDB not initialized"
            )
        
        # Ping MongoDB
        mongo_client.admin.command('ping')
        response_time = (time.time() - start) * 1000
        
        # Get database stats
        stats = mongo_db.command('dbStats')
        
        return DependencyCheck(
            status="up",
            response_time_ms=round(response_time, 2),
            details={
                "database": mongo_db.name,
                "collections": stats.get("collections", 0),
                "dataSize": stats.get("dataSize", 0)
            }
        )
    except Exception as e:
        response_time = (time.time() - start) * 1000
        logger.error(f"MongoDB health check failed: {e}")
        return DependencyCheck(
            status="down",
            response_time_ms=round(response_time, 2),
            error=str(e)
        )


def check_redis() -> DependencyCheck:
    """Check Redis connection"""
    start = time.time()
    try:
        redis_client = get_raw_store()
        if not redis_client:
            return DependencyCheck(
                status="down",
                error="Redis client not initialized"
            )
        
        # Try to ping Redis (if it's a real Redis client)
        if hasattr(redis_client, 'ping'):
            redis_client.ping()
            response_time = (time.time() - start) * 1000
            
            # Get Redis info
            info = redis_client.info()
            
            return DependencyCheck(
                status="up",
                response_time_ms=round(response_time, 2),
                details={
                    "version": info.get("redis_version", "unknown"),
                    "uptime_seconds": info.get("uptime_in_seconds", 0),
                    "connected_clients": info.get("connected_clients", 0)
                }
            )
        else:
            # In-memory fallback
            return DependencyCheck(
                status="degraded",
                response_time_ms=0,
                details={"type": "in-memory"}
            )
    except Exception as e:
        response_time = (time.time() - start) * 1000
        logger.error(f"Redis health check failed: {e}")
        return DependencyCheck(
            status="down",
            response_time_ms=round(response_time, 2),
            error=str(e)
        )


def check_s3() -> DependencyCheck:
    """Check S3 connection"""
    start = time.time()
    try:
        import boto3
        import os
        
        bucket_name = os.getenv("S3_BUCKET_NAME")
        if not bucket_name:
            return DependencyCheck(
                status="degraded",
                error="S3_BUCKET_NAME not configured"
            )
        
        s3_client = boto3.client('s3')
        
        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
        response_time = (time.time() - start) * 1000
        
        return DependencyCheck(
            status="up",
            response_time_ms=round(response_time, 2),
            details={
                "bucket": bucket_name,
                "region": os.getenv("AWS_REGION", "unknown")
            }
        )
    except Exception as e:
        response_time = (time.time() - start) * 1000
        logger.error(f"S3 health check failed: {e}")
        return DependencyCheck(
            status="down",
            response_time_ms=round(response_time, 2),
            error=str(e)
        )


@router.get("", response_model=HealthStatus)
async def health_check():
    """
    Basic health check endpoint
    
    Returns 200 if service is running
    """
    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow(),
        checks={}
    )


@router.get("/ready", response_model=HealthStatus)
async def readiness_check(response: Response):
    """
    Readiness check for Kubernetes/orchestration
    
    Checks all critical dependencies:
    - MongoDB
    - Redis
    - S3
    
    Returns 200 if all dependencies are healthy, 503 otherwise
    """
    checks = {
        "mongodb": check_mongodb().dict(),
        "redis": check_redis().dict(),
        "s3": check_s3().dict()
    }
    
    # Determine overall status
    all_up = all(check["status"] == "up" for check in checks.values())
    any_down = any(check["status"] == "down" for check in checks.values())
    
    if all_up:
        overall_status = "healthy"
        status_code = status.HTTP_200_OK
    elif any_down:
        overall_status = "unhealthy"
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        overall_status = "degraded"
        status_code = status.HTTP_200_OK
    
    response.status_code = status_code
    
    return HealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow(),
        checks=checks
    )


@router.get("/live")
async def liveness_check():
    """
    Liveness check for Kubernetes
    
    Simple check to verify the service is alive
    Returns 200 if the process is running
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/startup")
async def startup_check(response: Response):
    """
    Startup check for Kubernetes
    
    Checks if the service has completed initialization
    """
    # Check if critical services are initialized
    checks = {
        "mongodb": check_mongodb().dict()
    }
    
    mongodb_up = checks["mongodb"]["status"] == "up"
    
    if mongodb_up:
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        }
    else:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        }
