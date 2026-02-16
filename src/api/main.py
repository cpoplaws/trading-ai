"""
Trading AI REST API
FastAPI application exposing trading system functionality.
"""
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from contextlib import asynccontextmanager
import logging
import time
from typing import Optional

from routes import (
    market_data,
    risk_management,
    trading_signals,
    portfolio,
    agents,
    health
)

logger = logging.getLogger(__name__)


# API metadata
API_TITLE = "Trading AI API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
🤖 **Trading AI REST API**

Enterprise-grade API for algorithmic trading with:
- Real-time market data from multiple exchanges
- Risk management (VaR, CVaR, position sizing)
- Trading signals and strategy analytics
- RL agent decisions and performance
- Portfolio management and statistics

## Authentication
All endpoints require an API key in the header:
```
X-API-Key: your-api-key-here
```

## Rate Limits
- **Free tier**: 60 requests/minute
- **Pro tier**: 600 requests/minute
- **Enterprise**: Unlimited

## WebSocket
Real-time data available at: `ws://api.example.com/ws`
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")

    # Initialize services
    # db_manager.init_database()
    # aggregator.start()

    logger.info("API ready to accept requests")

    yield

    # Shutdown
    logger.info("Shutting down API")
    # aggregator.stop()
    # db_manager.close()


# Create FastAPI application
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# ============================================================
# Middleware
# ============================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add X-Process-Time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Rate limiting middleware (placeholder)
@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Simple rate limiting middleware."""
    # TODO: Implement rate limiting with Redis
    # For now, just pass through
    response = await call_next(request)
    return response


# ============================================================
# Exception Handlers
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": str(request.url)
        }
    )


# ============================================================
# Authentication
# ============================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate API key."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )

    # TODO: Validate against database
    # For now, accept any key starting with "sk_"
    if not api_key.startswith("sk_"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )

    return api_key


# ============================================================
# Include Routers
# ============================================================

# Health and status
app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

# Market data endpoints
app.include_router(
    market_data.router,
    prefix="/api/v1/market",
    tags=["Market Data"],
    dependencies=[Depends(get_api_key)]
)

# Risk management endpoints
app.include_router(
    risk_management.router,
    prefix="/api/v1/risk",
    tags=["Risk Management"],
    dependencies=[Depends(get_api_key)]
)

# Trading signals endpoints
app.include_router(
    trading_signals.router,
    prefix="/api/v1/signals",
    tags=["Trading Signals"],
    dependencies=[Depends(get_api_key)]
)

# Portfolio endpoints
app.include_router(
    portfolio.router,
    prefix="/api/v1/portfolio",
    tags=["Portfolio"],
    dependencies=[Depends(get_api_key)]
)

# Agent endpoints
app.include_router(
    agents.router,
    prefix="/api/v1/agents",
    tags=["RL Agents"],
    dependencies=[Depends(get_api_key)]
)


# ============================================================
# Root Endpoints
# ============================================================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "status": "operational"
    }


@app.get("/api/v1", tags=["Root"])
async def api_info():
    """API information."""
    return {
        "version": API_VERSION,
        "endpoints": {
            "market_data": "/api/v1/market",
            "risk_management": "/api/v1/risk",
            "trading_signals": "/api/v1/signals",
            "portfolio": "/api/v1/portfolio",
            "agents": "/api/v1/agents"
        },
        "authentication": "API key required (X-API-Key header)",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    }


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
