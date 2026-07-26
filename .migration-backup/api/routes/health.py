"""
Health Check Routes
System status and diagnostics.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from datetime import datetime
import sys

from api.dependencies import (
    get_database,
    get_redis,
    get_gas_tracker,
    get_coinbase_collector,
    get_uniswap_collector
)
from api.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    services: dict


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db=Depends(get_database),
    redis=Depends(get_redis)
):
    """
    Check system health and service status.

    Returns status of all critical services.
    """
    services = {}

    # Check database
    try:
        db_healthy = db.health_check()
        services["database"] = "healthy" if db_healthy else "unhealthy"
    except Exception as e:
        services["database"] = f"error: {str(e)}"

    # Check Redis
    try:
        redis.ping()
        services["redis"] = "healthy"
    except Exception as e:
        services["redis"] = f"error: {str(e)}"

    # Check collectors
    try:
        gas_tracker = get_gas_tracker()
        services["gas_tracker"] = "initialized"
    except Exception as e:
        services["gas_tracker"] = f"error: {str(e)}"

    try:
        coinbase = get_coinbase_collector()
        services["coinbase"] = "initialized"
    except Exception as e:
        services["coinbase"] = f"error: {str(e)}"

    try:
        uniswap = get_uniswap_collector()
        services["uniswap"] = "initialized" if uniswap.w3.is_connected() else "disconnected"
    except Exception as e:
        services["uniswap"] = f"error: {str(e)}"

    # Overall status
    all_healthy = all(
        status in ["healthy", "initialized"]
        for status in services.values()
    )

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.now(),
        version=settings.app_version,
        services=services
    )


@router.get("/status")
async def status():
    """
    Quick status check.

    Simple endpoint for load balancers.
    """
    return {"status": "ok"}


@router.get("/info")
async def info():
    """
    Get API information.

    Returns version, Python version, and configuration.
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "api_prefix": settings.api_prefix,
        "docs_url": "/docs",
        "cors_enabled": True,
        "cache_ttl": settings.cache_ttl
    }
