"""
Health Check Endpoints
"""
from fastapi import APIRouter, status
from datetime import datetime
import psutil
import sys
import os

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "trading-ai-api"
    }


@router.get("/detailed")
async def detailed_health():
    """Detailed health check with system metrics."""
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "trading-ai-api",
        "system": {
            "cpu_percent": cpu_percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            }
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable
        }
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    # Check if dependencies are ready
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "websocket": True  # Placeholder
    }

    all_ready = all(checks.values())

    return {
        "ready": all_ready,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }, status.HTTP_200_OK if all_ready else status.HTTP_503_SERVICE_UNAVAILABLE


@router.get("/live")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat()
    }


async def check_database() -> bool:
    """Check database connectivity."""
    try:
        # TODO: Implement actual database check
        return True
    except Exception:
        return False


async def check_redis() -> bool:
    """Check Redis connectivity."""
    try:
        from infrastructure.redis_cache import get_redis_cache
        cache = get_redis_cache()
        return cache.ping()
    except Exception:
        return False
