"""
Market Data Caching Layer

Specialized caching for market data with TTL strategies and invalidation patterns.
"""
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from infrastructure.redis_cache import RedisCache, get_redis_cache, cached

logger = logging.getLogger(__name__)


class MarketDataCache:
    """
    Specialized cache for market data with intelligent TTL and invalidation.

    Cache Categories:
    - Ticker data: 1-5 second TTL
    - Order book snapshots: 100-500ms TTL
    - Trade history: 10 second TTL
    - OHLCV candles: 1 minute - 1 hour TTL (depends on timeframe)
    - Strategy signals: 10-60 second TTL
    - Exchange rate limits: 1 minute TTL
    """

    def __init__(self, cache: Optional[RedisCache] = None):
        """
        Initialize market data cache.

        Args:
            cache: Redis cache instance (creates new if None)
        """
        self.cache = cache or get_redis_cache()

        # TTL configurations (in seconds)
        self.ttl_config = {
            'ticker': 2,          # 2 seconds
            'orderbook': 0.5,     # 500ms
            'trades': 10,         # 10 seconds
            'candle_1m': 60,      # 1 minute
            'candle_5m': 300,     # 5 minutes
            'candle_1h': 3600,    # 1 hour
            'signal': 30,         # 30 seconds
            'rate_limit': 60,     # 1 minute
            'api_quota': 300,     # 5 minutes
        }

    # Ticker Data
    def get_ticker(self, exchange: str, symbol: str) -> Optional[Dict]:
        """Get cached ticker data."""
        key = f"ticker:{exchange}:{symbol}"
        return self.cache.get(key)

    def set_ticker(self, exchange: str, symbol: str, data: Dict) -> bool:
        """Cache ticker data."""
        key = f"ticker:{exchange}:{symbol}"
        return self.cache.set(key, data, ttl=self.ttl_config['ticker'])

    def get_tickers(self, exchange: str, symbols: List[str]) -> Dict[str, Dict]:
        """Get multiple tickers at once."""
        keys = [f"ticker:{exchange}:{symbol}" for symbol in symbols]
        cached = self.cache.get_many(*keys)

        # Map back to symbols
        result = {}
        for symbol in symbols:
            key = f"ticker:{exchange}:{symbol}"
            if key in cached:
                result[symbol] = cached[key]

        return result

    def set_tickers(self, exchange: str, data: Dict[str, Dict]) -> bool:
        """Cache multiple tickers at once."""
        mapping = {
            f"ticker:{exchange}:{symbol}": ticker_data
            for symbol, ticker_data in data.items()
        }
        return self.cache.set_many(mapping, ttl=self.ttl_config['ticker'])

    # Order Book Data
    def get_orderbook(self, exchange: str, symbol: str) -> Optional[Dict]:
        """Get cached order book."""
        key = f"orderbook:{exchange}:{symbol}"
        return self.cache.get(key)

    def set_orderbook(self, exchange: str, symbol: str, data: Dict) -> bool:
        """Cache order book."""
        key = f"orderbook:{exchange}:{symbol}"
        ttl = int(self.ttl_config['orderbook'] * 1000)  # Convert to ms
        return self.cache.set(key, data, ttl=max(1, ttl // 1000))

    # Trade History
    def get_trades(self, exchange: str, symbol: str) -> Optional[List]:
        """Get cached recent trades."""
        key = f"trades:{exchange}:{symbol}"
        return self.cache.get(key)

    def set_trades(self, exchange: str, symbol: str, trades: List) -> bool:
        """Cache recent trades."""
        key = f"trades:{exchange}:{symbol}"
        return self.cache.set(key, trades, ttl=self.ttl_config['trades'])

    # Candle/OHLCV Data
    def get_candles(
        self,
        exchange: str,
        symbol: str,
        interval: str
    ) -> Optional[List]:
        """Get cached candle data."""
        key = f"candles:{exchange}:{symbol}:{interval}"
        return self.cache.get(key)

    def set_candles(
        self,
        exchange: str,
        symbol: str,
        interval: str,
        candles: List
    ) -> bool:
        """Cache candle data."""
        key = f"candles:{exchange}:{symbol}:{interval}"
        ttl_key = f"candle_{interval.lower()}"
        ttl = self.ttl_config.get(ttl_key, 60)
        return self.cache.set(key, candles, ttl=ttl)

    # Strategy Signals
    def get_signal(self, strategy: str, symbol: str) -> Optional[Dict]:
        """Get cached strategy signal."""
        key = f"signal:{strategy}:{symbol}"
        return self.cache.get(key)

    def set_signal(
        self,
        strategy: str,
        symbol: str,
        signal: Dict,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache strategy signal."""
        key = f"signal:{strategy}:{symbol}"
        ttl = ttl or self.ttl_config['signal']
        return self.cache.set(key, signal, ttl=ttl)

    def invalidate_signals(self, strategy: Optional[str] = None) -> int:
        """Invalidate strategy signals."""
        if strategy:
            pattern = f"signal:{strategy}:*"
        else:
            pattern = "signal:*"
        return self.cache.flush_pattern(pattern)

    # Rate Limiting
    def get_rate_limit(self, exchange: str, endpoint: str) -> Optional[Dict]:
        """Get cached rate limit info."""
        key = f"rate_limit:{exchange}:{endpoint}"
        return self.cache.get(key)

    def set_rate_limit(
        self,
        exchange: str,
        endpoint: str,
        limit_info: Dict
    ) -> bool:
        """Cache rate limit info."""
        key = f"rate_limit:{exchange}:{endpoint}"
        return self.cache.set(
            key,
            limit_info,
            ttl=self.ttl_config['rate_limit']
        )

    def increment_api_calls(
        self,
        exchange: str,
        endpoint: str,
        window: int = 60
    ) -> int:
        """Increment API call counter."""
        key = f"api_calls:{exchange}:{endpoint}"
        count = self.cache.incr(key)

        # Set expiration if this is first call
        if count == 1:
            self.cache.expire(key, window)

        return count

    def get_api_calls(self, exchange: str, endpoint: str) -> int:
        """Get current API call count."""
        key = f"api_calls:{exchange}:{endpoint}"
        count = self.cache.get(key)
        return int(count) if count else 0

    # Cache Statistics
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        info = self.cache.get_info()

        return {
            'connected': self.cache.ping(),
            'memory_used': info.get('used_memory_human', 'N/A'),
            'total_keys': info.get('db0', {}).get('keys', 0),
            'hit_rate': self._calculate_hit_rate(info),
            'connected_clients': info.get('connected_clients', 0),
            'uptime_seconds': info.get('uptime_in_seconds', 0)
        }

    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calculate cache hit rate."""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses

        if total == 0:
            return 0.0

        return (hits / total) * 100

    def health_check(self) -> bool:
        """Check cache health."""
        return self.cache.ping()

    # Invalidation
    def invalidate_exchange(self, exchange: str) -> int:
        """Invalidate all data for an exchange."""
        patterns = [
            f"ticker:{exchange}:*",
            f"orderbook:{exchange}:*",
            f"trades:{exchange}:*",
            f"candles:{exchange}:*"
        ]

        total = 0
        for pattern in patterns:
            total += self.cache.flush_pattern(pattern)

        return total

    def invalidate_symbol(self, symbol: str) -> int:
        """Invalidate all data for a symbol across exchanges."""
        patterns = [
            f"ticker:*:{symbol}",
            f"orderbook:*:{symbol}",
            f"trades:*:{symbol}",
            f"candles:*:{symbol}:*",
            f"signal:*:{symbol}"
        ]

        total = 0
        for pattern in patterns:
            total += self.cache.flush_pattern(pattern)

        return total


# Decorator for caching market data functions
def cache_market_data(
    data_type: str,
    ttl: Optional[int] = None
):
    """
    Decorator for caching market data function results.

    Args:
        data_type: Type of data (ticker, orderbook, trades, etc.)
        ttl: Optional custom TTL

    Example:
        @cache_market_data('ticker', ttl=5)
        def get_ticker(exchange, symbol):
            return fetch_from_api(exchange, symbol)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract exchange and symbol from args/kwargs
            exchange = kwargs.get('exchange', args[0] if len(args) > 0 else None)
            symbol = kwargs.get('symbol', args[1] if len(args) > 1 else None)

            if not exchange or not symbol:
                return func(*args, **kwargs)

            # Get cache instance
            cache = MarketDataCache()

            # Try to get from cache
            if data_type == 'ticker':
                cached_value = cache.get_ticker(exchange, symbol)
            elif data_type == 'orderbook':
                cached_value = cache.get_orderbook(exchange, symbol)
            elif data_type == 'trades':
                cached_value = cache.get_trades(exchange, symbol)
            else:
                cached_value = None

            if cached_value is not None:
                logger.debug(f"Cache HIT: {data_type}:{exchange}:{symbol}")
                return cached_value

            logger.debug(f"Cache MISS: {data_type}:{exchange}:{symbol}")

            # Execute function
            result = func(*args, **kwargs)

            # Store in cache
            if result is not None:
                if data_type == 'ticker':
                    cache.set_ticker(exchange, symbol, result)
                elif data_type == 'orderbook':
                    cache.set_orderbook(exchange, symbol, result)
                elif data_type == 'trades':
                    cache.set_trades(exchange, symbol, result)

            return result

        return wrapper
    return decorator


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("📈 Market Data Cache Demo")
    print("=" * 60)

    # Initialize cache
    cache = MarketDataCache()

    # Test ticker caching
    print("\n📊 Testing ticker cache...")
    ticker_data = {
        'price': 45000.50,
        'volume': 1500.25,
        'high': 45500.00,
        'low': 44500.00,
        'timestamp': datetime.now().isoformat()
    }

    cache.set_ticker('binance', 'BTC-USDT', ticker_data)
    print(f"✅ Cached ticker: binance:BTC-USDT")

    retrieved = cache.get_ticker('binance', 'BTC-USDT')
    print(f"✅ Retrieved: {retrieved}")

    # Test multiple tickers
    print("\n📊 Testing multi-ticker cache...")
    tickers = {
        'BTC-USDT': {'price': 45000},
        'ETH-USDT': {'price': 2500},
        'SOL-USDT': {'price': 100}
    }
    cache.set_tickers('binance', tickers)
    print("✅ Cached multiple tickers")

    retrieved_tickers = cache.get_tickers('binance', ['BTC-USDT', 'ETH-USDT'])
    print(f"✅ Retrieved: {retrieved_tickers}")

    # Test order book caching
    print("\n📖 Testing orderbook cache...")
    orderbook = {
        'bids': [[45000, 1.5], [44999, 2.0]],
        'asks': [[45001, 1.2], [45002, 3.5]]
    }
    cache.set_orderbook('binance', 'BTC-USDT', orderbook)
    print("✅ Cached order book")

    # Test signal caching
    print("\n🎯 Testing signal cache...")
    signal = {
        'action': 'BUY',
        'confidence': 0.85,
        'timestamp': datetime.now().isoformat()
    }
    cache.set_signal('dca_bot', 'BTC', signal)
    print("✅ Cached signal")

    retrieved_signal = cache.get_signal('dca_bot', 'BTC')
    print(f"✅ Retrieved signal: {retrieved_signal}")

    # Test rate limiting
    print("\n⏱️  Testing rate limit tracking...")
    for i in range(5):
        count = cache.increment_api_calls('binance', '/api/v3/ticker')
    print(f"✅ API calls made: {cache.get_api_calls('binance', '/api/v3/ticker')}")

    # Get statistics
    print("\n📊 Cache Statistics:")
    stats = cache.get_stats()
    print(f"   Connected: {stats['connected']}")
    print(f"   Memory used: {stats['memory_used']}")
    print(f"   Total keys: {stats['total_keys']}")
    print(f"   Hit rate: {stats['hit_rate']:.2f}%")

    # Test invalidation
    print("\n🗑️  Testing cache invalidation...")
    invalidated = cache.invalidate_exchange('binance')
    print(f"✅ Invalidated {invalidated} keys for binance")

    print("\n✅ Market data cache demo complete!")
