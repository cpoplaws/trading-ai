#!/usr/bin/env python3
"""
Autonomous Trading Agent with Redis Caching

This example demonstrates using Redis for high-performance caching:
- Market data caching with intelligent TTLs
- Strategy signal caching
- API rate limit tracking
- Performance monitoring
"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autonomous_agent.trading_agent import AutonomousTradingAgent, AgentConfig
from infrastructure.market_data_cache import MarketDataCache
from infrastructure.redis_cache import RedisCache, get_redis_cache
from typing import Dict
import logging
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CachedTradingAgent(AutonomousTradingAgent):
    """
    Enhanced trading agent with Redis caching for performance.
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.cache = MarketDataCache()
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls_saved = 0

    async def start(self):
        """Start agent with Redis caching enabled."""
        logger.info("Starting trading agent with Redis caching...")

        # Verify Redis connection
        if not self.cache.health_check():
            logger.error("Redis connection failed!")
            raise ConnectionError("Cannot connect to Redis")

        logger.info("✅ Redis connected and healthy")

        # Start normal agent operation
        await super().start()

    async def _get_market_data(self) -> Dict:
        """
        Get market data with caching (override parent method).

        Returns:
            Dictionary of current prices
        """
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        exchange = 'binance'

        market_data = {}

        for symbol in symbols:
            # Try cache first
            cached_ticker = self.cache.get_ticker(exchange, symbol)

            if cached_ticker:
                # Cache hit
                self.cache_hits += 1
                self.api_calls_saved += 1
                market_data[symbol] = cached_ticker['price']
                logger.debug(f"Cache HIT: {symbol} @ ${cached_ticker['price']}")
            else:
                # Cache miss - simulate API call
                self.cache_misses += 1
                price = await self._fetch_price_from_api(exchange, symbol)

                # Cache the result
                ticker_data = {
                    'symbol': symbol,
                    'price': price,
                    'timestamp': datetime.now().isoformat(),
                    'exchange': exchange
                }
                self.cache.set_ticker(exchange, symbol, ticker_data)

                market_data[symbol] = price
                logger.debug(f"Cache MISS: {symbol} @ ${price}")

        return market_data

    async def _fetch_price_from_api(self, exchange: str, symbol: str) -> float:
        """
        Simulate fetching price from API.

        In production, this would make actual API calls.
        Rate limiting is tracked automatically.
        """
        # Track API call
        self.cache.increment_api_calls(exchange, '/ticker')

        # Simulate API delay
        await asyncio.sleep(0.05)

        # Simulate price (in production, call real API)
        import random
        base_prices = {
            'BTC-USD': 45000,
            'ETH-USD': 2500,
            'SOL-USD': 100
        }
        base = base_prices.get(symbol, 100)
        return base * (1 + random.uniform(-0.02, 0.02))

    async def _generate_signals(self) -> list:
        """
        Generate trading signals with caching.

        Returns:
            List of trading signals
        """
        signals = []

        # Get market data (cached)
        market_data = await self._get_market_data()

        # Check each strategy
        for strategy_name in self.config.enabled_strategies:
            for symbol in market_data.keys():
                # Try to get cached signal
                cached_signal = self.cache.get_signal(strategy_name, symbol)

                if cached_signal:
                    # Use cached signal if recent enough
                    signals.append(cached_signal)
                    logger.debug(
                        f"Using cached signal: {strategy_name}:{symbol}"
                    )
                    continue

                # Generate new signal
                signal = await self._compute_signal(
                    strategy_name,
                    symbol,
                    market_data[symbol]
                )

                if signal:
                    # Cache the signal
                    self.cache.set_signal(
                        strategy_name,
                        symbol,
                        signal,
                        ttl=30  # 30 seconds
                    )
                    signals.append(signal)

        return signals

    async def _compute_signal(
        self,
        strategy: str,
        symbol: str,
        price: float
    ) -> Dict:
        """
        Compute trading signal for a strategy.

        In production, this would use real strategy logic.
        """
        # Simulate strategy computation
        await asyncio.sleep(0.02)

        import random
        if random.random() > 0.8:  # 20% chance of signal
            return {
                'strategy': strategy,
                'symbol': symbol,
                'action': random.choice(['BUY', 'SELL']),
                'price': price,
                'confidence': random.uniform(0.6, 0.9),
                'timestamp': datetime.now().isoformat()
            }

        return None

    def get_cache_stats(self) -> Dict:
        """Get caching statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0

        redis_stats = self.cache.get_stats()

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'api_calls_saved': self.api_calls_saved,
            'redis_memory': redis_stats['memory_used'],
            'redis_keys': redis_stats['total_keys'],
            'redis_connected': redis_stats['connected']
        }

    async def stop(self):
        """Stop agent and show cache statistics."""
        stats = self.get_cache_stats()

        logger.info("\n" + "=" * 60)
        logger.info("CACHE STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Cache Hits: {stats['cache_hits']}")
        logger.info(f"Cache Misses: {stats['cache_misses']}")
        logger.info(f"Hit Rate: {stats['hit_rate']:.2f}%")
        logger.info(f"API Calls Saved: {stats['api_calls_saved']}")
        logger.info(f"Redis Memory Used: {stats['redis_memory']}")
        logger.info(f"Redis Total Keys: {stats['redis_keys']}")

        await super().stop()


async def demo_cache_performance():
    """Demo showing cache performance improvement."""
    print("\n" + "=" * 70)
    print("CACHE PERFORMANCE DEMO")
    print("=" * 70)

    cache = MarketDataCache()

    # Warm up cache
    print("\n📝 Warming up cache...")
    test_data = {
        'BTC-USDT': {'price': 45000, 'volume': 1000},
        'ETH-USDT': {'price': 2500, 'volume': 500},
        'SOL-USDT': {'price': 100, 'volume': 10000}
    }
    cache.set_tickers('binance', test_data)
    print("✅ Cache warmed with test data")

    # Benchmark: Cache vs No Cache
    print("\n⏱️  Benchmark: 1000 reads")

    # With cache
    start = time.time()
    for _ in range(1000):
        cache.get_tickers('binance', ['BTC-USDT', 'ETH-USDT', 'SOL-USDT'])
    cached_time = time.time() - start

    print(f"   With cache: {cached_time:.4f}s ({1000/cached_time:.0f} ops/sec)")

    # Without cache (simulated)
    start = time.time()
    for _ in range(1000):
        # Simulate API call delay
        time.sleep(0.0001)  # 0.1ms per call
    uncached_time = time.time() - start

    print(f"   Without cache (simulated): {uncached_time:.4f}s")

    speedup = uncached_time / cached_time
    print(f"\n🚀 Speedup: {speedup:.1f}x faster with caching")


async def main():
    """Main entry point."""
    print("=" * 70)
    print("AUTONOMOUS TRADING AGENT WITH REDIS CACHING")
    print("=" * 70)

    # Check Redis connection
    print("\n🔴 Checking Redis connection...")
    try:
        redis = get_redis_cache()
        if not redis.ping():
            print("❌ Redis not available!")
            print("\n💡 To start Redis:")
            print("   docker run -d -p 6379:6379 redis:alpine")
            return
        print("✅ Redis connected")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("\n💡 To start Redis:")
        print("   docker run -d -p 6379:6379 redis:alpine")
        return

    # Run cache performance demo
    await demo_cache_performance()

    # Agent configuration
    config = AgentConfig(
        initial_capital=10000.0,
        paper_trading=True,
        check_interval_seconds=2,  # More frequent checks
        max_daily_loss=500.0,
        send_alerts=False,
        enabled_strategies=['dca_bot', 'market_making']
    )

    # Create agent with caching
    agent = CachedTradingAgent(config)

    try:
        print("\n🤖 Starting cached trading agent...")
        print("   Press Ctrl+C to stop\n")

        # Run for 30 seconds as demo
        await asyncio.wait_for(agent.start(), timeout=30.0)

    except asyncio.TimeoutError:
        print("\n⏱️  Demo timeout reached")
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    finally:
        await agent.stop()

    # Show final statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)

    print(f"\nPortfolio Value: ${agent.portfolio_value:,.2f}")
    print(f"Total P&L: ${agent.total_pnl:+,.2f}")
    print(f"Total Trades: {len(agent.trade_history)}")

    # Show cache performance improvement
    stats = agent.get_cache_stats()
    print("\n" + "-" * 70)
    print("CACHING IMPACT")
    print("-" * 70)
    print(f"Cache Hit Rate: {stats['hit_rate']:.2f}%")
    print(f"API Calls Saved: {stats['api_calls_saved']}")

    if stats['api_calls_saved'] > 0:
        # Estimate time saved (assuming 50ms per API call)
        time_saved_ms = stats['api_calls_saved'] * 50
        print(f"Estimated Time Saved: ~{time_saved_ms}ms")
        print(f"Performance Improvement: ~{stats['hit_rate']:.0f}% faster")

    print("\n✅ Agent stopped - cache statistics saved")


if __name__ == '__main__':
    asyncio.run(main())
