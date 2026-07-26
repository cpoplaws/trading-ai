"""
API Configuration
Environment variables and settings.
"""
import os
from typing import List
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Info
    app_name: str = "Trading AI API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    reload: bool = True

    # CORS
    cors_origins: List[str] = [
        "http://localhost:3000",  # React frontend
        "http://localhost:3001",  # Grafana
        "http://localhost:8080",  # Alternative frontend
    ]

    # Database
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "trading_db")
    postgres_user: str = os.getenv("POSTGRES_USER", "trading_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "trading_password")

    # Redis
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = 0

    # External APIs
    coinbase_api_key: str = os.getenv("COINBASE_API_KEY", "")
    coinbase_api_secret: str = os.getenv("COINBASE_API_SECRET", "")
    ethereum_rpc_url: str = os.getenv("ETHEREUM_RPC_URL", "https://eth.llamarpc.com")
    etherscan_api_key: str = os.getenv("ETHERSCAN_API_KEY", "")

    # Cache
    cache_ttl: int = 10  # seconds

    # Rate Limiting
    rate_limit_per_minute: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env


settings = Settings()
