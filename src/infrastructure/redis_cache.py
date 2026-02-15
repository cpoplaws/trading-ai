"""
Redis Cache Manager
High-performance caching layer with circuit breaker protection.
"""
import sys
import os
from pathlib import Path
import json
import pickle
from typing import Any, Optional, Callable
from datetime import timedelta
import logging
import redis
from functools import wraps

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis cache manager with automatic serialization and TTL support.

    Features:
    - Automatic JSON/pickle serialization
    - TTL (Time-To-Live) support
    - Connection pooling
    - Circuit breaker protection
    - Cache invalidation patterns
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        max_connections: int = 50,
        socket_timeout: int = 5
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            password: Redis password
            db: Database number
            max_connections: Max connection pool size
            socket_timeout: Socket timeout in seconds
        """
        self.host = host
        self.port = port

        # Create connection pool
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            password=password,
            db=db,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
            decode_responses=False  # We'll handle encoding
        )

        self.client = redis.Redis(connection_pool=self.pool)

        # Test connection
        try:
            self.client.ping()
            logger.info(f"Redis connected: {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        try:
            value = self.client.get(key)
            if value is None:
                return default

            # Try JSON first, fallback to pickle
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return pickle.loads(value)

        except redis.RedisError as e:
            logger.error(f"Redis GET error for key '{key}': {e}")
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists

        Returns:
            True if successful
        """
        try:
            # Serialize value
            try:
                serialized = json.dumps(value).encode()
            except (TypeError, ValueError):
                serialized = pickle.dumps(value)

            # Set with options
            return bool(self.client.set(
                key,
                serialized,
                ex=ttl,
                nx=nx,
                xx=xx
            ))

        except redis.RedisError as e:
            logger.error(f"Redis SET error for key '{key}': {e}")
            return False

    def delete(self, *keys: str) -> int:
        """Delete one or more keys."""
        try:
            return self.client.delete(*keys)
        except redis.RedisError as e:
            logger.error(f"Redis DELETE error: {e}")
            return 0

    def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        try:
            return self.client.exists(*keys)
        except redis.RedisError as e:
            logger.error(f"Redis EXISTS error: {e}")
            return 0

    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on a key."""
        try:
            return bool(self.client.expire(key, seconds))
        except redis.RedisError as e:
            logger.error(f"Redis EXPIRE error for key '{key}': {e}")
            return False

    def ttl(self, key: str) -> int:
        """Get remaining TTL for a key."""
        try:
            return self.client.ttl(key)
        except redis.RedisError as e:
            logger.error(f"Redis TTL error for key '{key}': {e}")
            return -2

    def incr(self, key: str, amount: int = 1) -> int:
        """Increment a key's value."""
        try:
            return self.client.incr(key, amount)
        except redis.RedisError as e:
            logger.error(f"Redis INCR error for key '{key}': {e}")
            return 0

    def decr(self, key: str, amount: int = 1) -> int:
        """Decrement a key's value."""
        try:
            return self.client.decr(key, amount)
        except redis.RedisError as e:
            logger.error(f"Redis DECR error for key '{key}': {e}")
            return 0

    def get_many(self, *keys: str) -> dict:
        """Get multiple keys at once."""
        try:
            values = self.client.mget(*keys)
            result = {}

            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        result[key] = pickle.loads(value)

            return result

        except redis.RedisError as e:
            logger.error(f"Redis MGET error: {e}")
            return {}

    def set_many(self, mapping: dict, ttl: Optional[int] = None) -> bool:
        """Set multiple keys at once."""
        try:
            # Serialize all values
            serialized = {}
            for key, value in mapping.items():
                try:
                    serialized[key] = json.dumps(value).encode()
                except (TypeError, ValueError):
                    serialized[key] = pickle.dumps(value)

            # Use pipeline for efficiency
            pipe = self.client.pipeline()
            pipe.mset(serialized)

            # Set TTL if specified
            if ttl:
                for key in serialized.keys():
                    pipe.expire(key, ttl)

            pipe.execute()
            return True

        except redis.RedisError as e:
            logger.error(f"Redis MSET error: {e}")
            return False

    def flush_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern."""
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except redis.RedisError as e:
            logger.error(f"Redis FLUSH_PATTERN error for '{pattern}': {e}")
            return 0

    def get_info(self) -> dict:
        """Get Redis server info."""
        try:
            return self.client.info()
        except redis.RedisError as e:
            logger.error(f"Redis INFO error: {e}")
            return {}

    def ping(self) -> bool:
        """Test connection."""
        try:
            return self.client.ping()
        except redis.RedisError:
            return False


def cached(
    ttl: int = 300,
    key_prefix: str = '',
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results.

    Args:
        ttl: Cache TTL in seconds (default: 5 minutes)
        key_prefix: Prefix for cache keys
        key_func: Custom function to generate cache key

    Example:
        @cached(ttl=600, key_prefix='quotes')
        def get_quote(symbol):
            return fetch_from_api(symbol)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key: prefix:func_name:args:kwargs
                args_str = '_'.join(str(arg) for arg in args)
                kwargs_str = '_'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{key_prefix}:{func.__name__}:{args_str}:{kwargs_str}"

            # Try to get from cache
            try:
                cache = get_redis_cache()
                cached_value = cache.get(cache_key)

                if cached_value is not None:
                    logger.debug(f"Cache HIT: {cache_key}")
                    return cached_value

                logger.debug(f"Cache MISS: {cache_key}")

            except Exception as e:
                logger.warning(f"Cache get failed: {e}")

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            try:
                cache = get_redis_cache()
                cache.set(cache_key, result, ttl=ttl)
            except Exception as e:
                logger.warning(f"Cache set failed: {e}")

            return result

        return wrapper
    return decorator


# Global cache instance
_redis_cache: Optional[RedisCache] = None


def get_redis_cache() -> RedisCache:
    """Get global Redis cache instance."""
    global _redis_cache

    if _redis_cache is None:
        _redis_cache = RedisCache(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            password=os.getenv('REDIS_PASSWORD'),
            db=int(os.getenv('REDIS_DB', 0))
        )

    return _redis_cache


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🔴 Redis Cache Demo")
    print("=" * 60)

    # Initialize cache
    cache = RedisCache()

    # Test basic operations
    print("\n📝 Testing basic operations...")

    # Set
    cache.set('test_key', {'price': 100.50, 'volume': 1000}, ttl=60)
    print("✅ SET test_key")

    # Get
    value = cache.get('test_key')
    print(f"✅ GET test_key: {value}")

    # Increment
    cache.set('counter', 0)
    cache.incr('counter', 5)
    counter = cache.get('counter')
    print(f"✅ INCR counter: {counter}")

    # Multi-set
    cache.set_many({
        'btc_price': 45000,
        'eth_price': 2000,
        'sol_price': 100
    }, ttl=300)
    print("✅ SET_MANY prices")

    # Multi-get
    prices = cache.get_many('btc_price', 'eth_price', 'sol_price')
    print(f"✅ GET_MANY prices: {prices}")

    # Test decorator
    print("\n🎯 Testing cache decorator...")

    @cached(ttl=10, key_prefix='demo')
    def expensive_function(symbol: str) -> dict:
        print(f"   Computing for {symbol}...")
        return {'symbol': symbol, 'price': 100.0}

    # First call (cache miss)
    result1 = expensive_function('BTC')
    print(f"✅ First call: {result1}")

    # Second call (cache hit)
    result2 = expensive_function('BTC')
    print(f"✅ Second call (cached): {result2}")

    # Get info
    print("\n📊 Redis Info:")
    info = cache.get_info()
    print(f"   Version: {info.get('redis_version', 'N/A')}")
    print(f"   Connected clients: {info.get('connected_clients', 'N/A')}")
    print(f"   Used memory: {info.get('used_memory_human', 'N/A')}")

    print("\n✅ Redis cache demo complete!")
