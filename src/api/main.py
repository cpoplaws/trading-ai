"""Quantlytics REST API backed by the Itera engine."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import logging
import time

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from .config import load_api_settings
from .rate_limiter import RateLimiter, build_rate_limiter
from .routes import agents, health, market_data, portfolio, risk_management, trading_signals

logger = logging.getLogger(__name__)
settings = load_api_settings()

API_TITLE = "Quantlytics API"
API_VERSION = "1.1.0"


def _validate_production_settings() -> None:
    if settings.environment == "development":
        return
    missing = []
    if not settings.api_keys:
        missing.append("QUANTLYTICS_API_KEYS")
    if not settings.cors_origins or settings.cors_origins == ["http://localhost:3000"]:
        logger.warning("CORS origins look development-flavored in env=%s", settings.environment)
    if missing:
        raise RuntimeError(f"Missing required env outside development: {', '.join(missing)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s (%s)", API_TITLE, API_VERSION, settings.environment)
    _validate_production_settings()
    app.state.rate_limiter = await build_rate_limiter()
    yield
    logger.info("Shutting down %s", API_TITLE)


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Trading API for Quantlytics (Itera engine)",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def process_time_and_rate_limit(request: Request, call_next):
    start_time = time.time()

    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is not None:
        client_id = request.client.host if request.client else "unknown"
        allowed = await limiter.allow(client_id, settings.rate_limit_per_minute, window_seconds=60)
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded", "retry_after_seconds": 60},
                headers={"Retry-After": "60"},
            )

    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start_time:.4f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code, "path": str(request.url)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500, "path": str(request.url)},
    )


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _is_valid_api_key(api_key: str) -> bool:
    if not settings.api_keys:
        return settings.environment == "development"

    return any(hmac.compare_digest(api_key, configured) for configured in settings.api_keys)


async def get_api_key(api_key: str = Security(api_key_header)) -> str:
    if not settings.api_keys and settings.environment == "development":
        return api_key or "dev"

    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required")

    if not _is_valid_api_key(api_key):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")

    return api_key


app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(
    market_data.router,
    prefix="/api/v1/market",
    tags=["Market Data"],
    dependencies=[Depends(get_api_key)],
)
app.include_router(
    risk_management.router,
    prefix="/api/v1/risk",
    tags=["Risk Management"],
    dependencies=[Depends(get_api_key)],
)
app.include_router(
    trading_signals.router,
    prefix="/api/v1/signals",
    tags=["Trading Signals"],
    dependencies=[Depends(get_api_key)],
)
app.include_router(
    portfolio.router,
    prefix="/api/v1/portfolio",
    tags=["Portfolio"],
    dependencies=[Depends(get_api_key)],
)
app.include_router(
    agents.router,
    prefix="/api/v1/agents",
    tags=["Itera Agents"],
    dependencies=[Depends(get_api_key)],
)


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "engine": "Itera",
        "docs": "/docs",
        "health": "/health",
        "status": "operational",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
