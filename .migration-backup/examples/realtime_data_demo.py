"""
Real-Time Market Data Demo
Demonstrates WebSocket connections, multi-exchange aggregation, and live trading signals.
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from collections import deque
import statistics

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.realtime import (
    BinanceWebSocket,
    BinanceConfig,
    BinanceStream,
    CoinbaseWebSocket,
    CoinbaseConfig,
    CoinbaseChannel,
    MarketDataAggregator,
    AggregatorConfig,
    AggregationStrategy,
    UnifiedMarketData
)


class PriceMonitor:
    """Monitor price movements and detect significant changes."""

    def __init__(self, symbol: str, window_size: int = 20):
        self.symbol = symbol
        self.window_size = window_size
        self.prices = deque(maxlen=window_size)
        self.volume = deque(maxlen=window_size)

    def update(self, price: float, volume: float = 0):
        """Update with new price."""
        self.prices.append(price)
        self.volume.append(volume)

    def get_stats(self) -> dict:
        """Get price statistics."""
        if len(self.prices) < 2:
            return {}

        prices_list = list(self.prices)
        return {
            'current': prices_list[-1],
            'mean': statistics.mean(prices_list),
            'median': statistics.median(prices_list),
            'stdev': statistics.stdev(prices_list),
            'min': min(prices_list),
            'max': max(prices_list),
            'change': prices_list[-1] - prices_list[0],
            'change_percent': ((prices_list[-1] - prices_list[0]) / prices_list[0] * 100),
            'total_volume': sum(self.volume)
        }

    def detect_spike(self, threshold_stdev: float = 2.0) -> bool:
        """Detect price spike (> N standard deviations)."""
        if len(self.prices) < self.window_size:
            return False

        stats = self.get_stats()
        current_deviation = abs(stats['current'] - stats['mean']) / stats['stdev']

        return current_deviation > threshold_stdev


class ArbitrageDetector:
    """Detect arbitrage opportunities across exchanges."""

    def __init__(self, min_spread_percent: float = 0.5):
        self.min_spread_percent = min_spread_percent
        self.opportunities = []

    def check_opportunity(self, data: UnifiedMarketData) -> bool:
        """Check if arbitrage opportunity exists."""
        if data.data_type != 'book_ticker':
            return False

        spread_percent = data.data.get('spread_percent', 0)

        if spread_percent >= self.min_spread_percent:
            opportunity = {
                'symbol': data.symbol,
                'buy_exchange': data.data['best_bid_exchange'],
                'sell_exchange': data.data['best_ask_exchange'],
                'buy_price': data.data['best_bid'],
                'sell_price': data.data['best_ask'],
                'spread': data.data['spread'],
                'spread_percent': spread_percent,
                'timestamp': data.timestamp
            }

            self.opportunities.append(opportunity)
            return True

        return False


async def demo_1_basic_binance():
    """Demo 1: Basic Binance WebSocket connection."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic Binance WebSocket")
    print("=" * 60)

    config = BinanceConfig(
        symbols=['BTCUSDT'],
        streams=[BinanceStream.TRADE, BinanceStream.TICKER]
    )

    client = BinanceWebSocket(config)

    trade_count = 0

    @client.on_trade
    def handle_trade(data):
        nonlocal trade_count
        trade_count += 1
        if trade_count <= 5:  # Show first 5 trades
            print(f"   💰 Trade: ${data.data['price']:,.2f} x {data.data['quantity']:.4f}")

    @client.on_ticker
    def handle_ticker(data):
        print(f"   📈 24h: ${data.data['last_price']:,.2f} "
              f"({data.data['price_change_percent']:+.2f}%) "
              f"Vol: {data.data['volume']:,.0f}")

    print("\n🔌 Connecting to Binance...")
    await client.connect()

    print("📡 Streaming for 10 seconds...\n")
    await asyncio.sleep(10)

    print(f"\n✅ Received {trade_count} trades")
    await client.disconnect()


async def demo_2_multi_exchange():
    """Demo 2: Multi-exchange comparison."""
    print("\n" + "=" * 60)
    print("Demo 2: Multi-Exchange Price Comparison")
    print("=" * 60)

    # Binance
    binance_config = BinanceConfig(
        symbols=['BTCUSDT', 'ETHUSDT'],
        streams=[BinanceStream.TICKER]
    )
    binance = BinanceWebSocket(binance_config)

    # Coinbase
    coinbase_config = CoinbaseConfig(
        product_ids=['BTC-USD', 'ETH-USD'],
        channels=[CoinbaseChannel.TICKER]
    )
    coinbase = CoinbaseWebSocket(coinbase_config)

    prices = {
        'BTC': {'binance': None, 'coinbase': None},
        'ETH': {'binance': None, 'coinbase': None}
    }

    @binance.on_ticker
    def handle_binance(data):
        symbol = 'BTC' if 'BTC' in data.symbol else 'ETH'
        prices[symbol]['binance'] = data.data['last_price']
        print_comparison(symbol)

    @coinbase.on_ticker
    def handle_coinbase(data):
        symbol = 'BTC' if 'BTC' in data.symbol else 'ETH'
        prices[symbol]['coinbase'] = data.data['price']
        print_comparison(symbol)

    def print_comparison(symbol):
        if prices[symbol]['binance'] and prices[symbol]['coinbase']:
            bin_price = prices[symbol]['binance']
            cb_price = prices[symbol]['coinbase']
            diff = cb_price - bin_price
            diff_pct = (diff / bin_price) * 100

            print(f"   {symbol}: Binance ${bin_price:,.2f} | "
                  f"Coinbase ${cb_price:,.2f} | "
                  f"Diff: ${diff:+,.2f} ({diff_pct:+.3f}%)")

    print("\n🔌 Connecting to exchanges...")
    await asyncio.gather(
        binance.connect(),
        coinbase.connect()
    )

    print("📡 Comparing prices for 15 seconds...\n")
    await asyncio.sleep(15)

    await asyncio.gather(
        binance.disconnect(),
        coinbase.disconnect()
    )


async def demo_3_aggregated_feed():
    """Demo 3: Unified aggregated data feed."""
    print("\n" + "=" * 60)
    print("Demo 3: Aggregated Market Data Feed")
    print("=" * 60)

    config = AggregatorConfig(
        exchanges=['binance', 'coinbase'],
        symbols={
            'BTC/USD': {
                'binance': 'BTCUSDT',
                'coinbase': 'BTC-USD'
            },
            'ETH/USD': {
                'binance': 'ETHUSDT',
                'coinbase': 'ETH-USD'
            }
        },
        aggregation_strategy=AggregationStrategy.BEST_BID_ASK,
        update_interval=2.0  # Update every 2 seconds
    )

    aggregator = MarketDataAggregator(config)

    update_count = 0

    def handle_book(data: UnifiedMarketData):
        nonlocal update_count
        update_count += 1

        print(f"   📖 {data.symbol}:")
        print(f"      Best Bid: ${data.data['best_bid']:,.2f} "
              f"@ {data.data['best_bid_exchange']}")
        print(f"      Best Ask: ${data.data['best_ask']:,.2f} "
              f"@ {data.data['best_ask_exchange']}")
        print(f"      Spread: ${data.data['spread']:,.2f} "
              f"({data.data['spread_percent']:.3f}%)")
        print()

    aggregator.subscribe('*', 'book_ticker', handle_book)

    print("\n🚀 Starting aggregator...")
    await aggregator.start()

    print("📡 Streaming aggregated data for 20 seconds...\n")
    await asyncio.sleep(20)

    print(f"\n📊 Stats: {update_count} aggregated updates published")
    await aggregator.stop()


async def demo_4_price_monitoring():
    """Demo 4: Price monitoring and spike detection."""
    print("\n" + "=" * 60)
    print("Demo 4: Price Monitoring & Spike Detection")
    print("=" * 60)

    config = BinanceConfig(
        symbols=['BTCUSDT'],
        streams=[BinanceStream.TRADE]
    )

    client = BinanceWebSocket(config)
    monitor = PriceMonitor('BTCUSDT', window_size=50)

    @client.on_trade
    def handle_trade(data):
        price = data.data['price']
        volume = data.data['quantity']

        monitor.update(price, volume)

        # Check for spike every 50 trades
        if len(monitor.prices) == monitor.window_size:
            stats = monitor.get_stats()

            if monitor.detect_spike(threshold_stdev=2.0):
                print(f"   ⚠️  SPIKE DETECTED!")
                print(f"      Current: ${stats['current']:,.2f}")
                print(f"      Mean: ${stats['mean']:,.2f}")
                print(f"      StdDev: ${stats['stdev']:,.2f}")
                print()

    print("\n🔌 Connecting to Binance...")
    await client.connect()

    print("📡 Monitoring for price spikes (30 seconds)...\n")
    await asyncio.sleep(30)

    # Print final stats
    stats = monitor.get_stats()
    print(f"\n📊 Final Statistics:")
    print(f"   Current: ${stats['current']:,.2f}")
    print(f"   Mean: ${stats['mean']:,.2f}")
    print(f"   Range: ${stats['min']:,.2f} - ${stats['max']:,.2f}")
    print(f"   Change: ${stats['change']:,.2f} ({stats['change_percent']:+.2f}%)")
    print(f"   Total Volume: {stats['total_volume']:,.4f}")

    await client.disconnect()


async def demo_5_arbitrage_detection():
    """Demo 5: Cross-exchange arbitrage detection."""
    print("\n" + "=" * 60)
    print("Demo 5: Arbitrage Opportunity Detection")
    print("=" * 60)

    config = AggregatorConfig(
        exchanges=['binance', 'coinbase'],
        symbols={
            'BTC/USD': {
                'binance': 'BTCUSDT',
                'coinbase': 'BTC-USD'
            }
        },
        aggregation_strategy=AggregationStrategy.BEST_BID_ASK,
        update_interval=1.0
    )

    aggregator = MarketDataAggregator(config)
    detector = ArbitrageDetector(min_spread_percent=0.3)

    def handle_book(data: UnifiedMarketData):
        if detector.check_opportunity(data):
            opp = detector.opportunities[-1]
            print(f"   🚨 ARBITRAGE OPPORTUNITY!")
            print(f"      Symbol: {opp['symbol']}")
            print(f"      Buy @ {opp['buy_exchange']}: ${opp['buy_price']:,.2f}")
            print(f"      Sell @ {opp['sell_exchange']}: ${opp['sell_price']:,.2f}")
            print(f"      Profit: ${opp['spread']:,.2f} ({opp['spread_percent']:.2f}%)")
            print()

    aggregator.subscribe('*', 'book_ticker', handle_book)

    print("\n🚀 Starting arbitrage detector...")
    await aggregator.start()

    print("📡 Scanning for opportunities (30 seconds)...\n")
    await asyncio.sleep(30)

    print(f"\n📊 Found {len(detector.opportunities)} opportunities")

    if detector.opportunities:
        # Show top 3
        top_opps = sorted(
            detector.opportunities,
            key=lambda x: x['spread_percent'],
            reverse=True
        )[:3]

        print("\n🏆 Top 3 Opportunities:")
        for i, opp in enumerate(top_opps, 1):
            print(f"   {i}. {opp['symbol']}: {opp['spread_percent']:.2f}% "
                  f"(${opp['spread']:,.2f})")

    await aggregator.stop()


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("🌐 Real-Time Market Data Infrastructure Demo")
    print("=" * 60)
    print("\nThis demo showcases:")
    print("  1. Basic WebSocket connections")
    print("  2. Multi-exchange price comparison")
    print("  3. Unified aggregated data feeds")
    print("  4. Price monitoring and spike detection")
    print("  5. Cross-exchange arbitrage detection")

    try:
        # Run demos sequentially
        await demo_1_basic_binance()
        await demo_2_multi_exchange()
        await demo_3_aggregated_feed()
        await demo_4_price_monitoring()
        await demo_5_arbitrage_detection()

        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise in demo
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run demos
    asyncio.run(main())
