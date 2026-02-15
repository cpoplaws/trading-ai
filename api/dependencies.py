"""
API Dependencies
Shared dependencies for routes.
"""
import redis
from fastapi import Depends, HTTPException, status
from typing import Optional
import logging

from api.config import settings
from src.data_collection.coinbase_collector import CoinbaseCollector
from src.data_collection.uniswap_collector import UniswapCollector
from src.data_collection.gas_tracker import GasTracker
from src.onchain.dex_analyzer import DEXAnalyzer
from src.database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


# Singleton instances
_coinbase_collector: Optional[CoinbaseCollector] = None
_uniswap_collector: Optional[UniswapCollector] = None
_gas_tracker: Optional[GasTracker] = None
_dex_analyzer: Optional[DEXAnalyzer] = None
_database: Optional[DatabaseManager] = None
_redis_client: Optional[redis.Redis] = None


def get_coinbase_collector() -> CoinbaseCollector:
    """Get Coinbase collector instance."""
    global _coinbase_collector
    if _coinbase_collector is None:
        _coinbase_collector = CoinbaseCollector(
            api_key=settings.coinbase_api_key,
            api_secret=settings.coinbase_api_secret
        )
    return _coinbase_collector


def get_uniswap_collector() -> UniswapCollector:
    """Get Uniswap collector instance."""
    global _uniswap_collector
    if _uniswap_collector is None:
        _uniswap_collector = UniswapCollector(
            rpc_url=settings.ethereum_rpc_url
        )
    return _uniswap_collector


def get_gas_tracker() -> GasTracker:
    """Get gas tracker instance."""
    global _gas_tracker
    if _gas_tracker is None:
        _gas_tracker = GasTracker(
            rpc_url=settings.ethereum_rpc_url,
            etherscan_api_key=settings.etherscan_api_key
        )
    return _gas_tracker


def get_dex_analyzer() -> DEXAnalyzer:
    """Get DEX analyzer instance."""
    global _dex_analyzer
    if _dex_analyzer is None:
        _dex_analyzer = DEXAnalyzer(
            min_profit_usd=10.0,
            max_gas_gwei=50.0
        )
    return _dex_analyzer


def get_database() -> DatabaseManager:
    """Get database manager instance."""
    global _database
    if _database is None:
        _database = DatabaseManager(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password
        )
    return _database


def get_redis() -> redis.Redis:
    """Get Redis client instance."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True
        )
    return _redis_client


def check_redis_connection(redis_client: redis.Redis = Depends(get_redis)):
    """Check Redis connection."""
    try:
        redis_client.ping()
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis service unavailable"
        )


def check_database_connection(db: DatabaseManager = Depends(get_database)):
    """Check database connection."""
    try:
        if not db.health_check():
            raise Exception("Database health check failed")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable"
        )
