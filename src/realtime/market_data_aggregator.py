"""
Market Data Aggregator
Unified interface for real-time market data from multiple exchanges.
"""
import asyncio
import logging
from typing import Dict, List, Callable, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics

from websocket_manager import MarketData
from binance_websocket import BinanceWebSocket, BinanceConfig, BinanceStream
from coinbase_websocket import CoinbaseWebSocket, CoinbaseConfig, CoinbaseChannel

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Strategies for aggregating data from multiple exchanges."""
    FIRST = "first"  # Use first received
    LAST = "last"  # Use last received
    AVERAGE = "average"  # Average prices
    MEDIAN = "median"  # Median prices
    WEIGHTED = "weighted"  # Volume-weighted
    BEST_BID_ASK = "best_bid_ask"  # Best bid/ask across exchanges


@dataclass
class UnifiedMarketData:
    """Unified market data from multiple exchanges."""
    symbol: str
    data_type: str
    timestamp: datetime
    exchanges: List[str]
    data: dict
    source_data: List[MarketData] = field(default_factory=list)


@dataclass
class AggregatorConfig:
    """Aggregator configuration."""
    exchanges: List[str]  # ['binance', 'coinbase']
    symbols: Dict[str, str]  # Map unified symbol to exchange-specific
    # e.g., {'BTC/USD': {'binance': 'BTCUSDT', 'coinbase': 'BTC-USD'}}
    aggregation_strategy: AggregationStrategy = AggregationStrategy.BEST_BID_ASK
    buffer_size: int = 100  # Number of recent data points to keep
    update_interval: float = 0.1  # Minimum seconds between aggregated updates


class MarketDataAggregator:
    """
    Aggregates real-time market data from multiple exchanges.

    Features:
    - Multi-exchange support (Binance, Coinbase, etc.)
    - Symbol normalization
    - Data aggregation strategies
    - Unified price feeds
    - Cross-exchange arbitrage detection
    - Volume aggregation
    - Best bid/ask tracking
    """

    def __init__(self, config: AggregatorConfig):
        """
        Initialize market data aggregator.

        Args:
            config: Aggregator configuration
        """
        self.config = config

        # Exchange clients
        self.clients: Dict[str, any] = {}

        # Data buffers by symbol
        self.buffers: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=config.buffer_size))
        )

        # Latest data by symbol and data type
        self.latest: Dict[str, Dict[str, UnifiedMarketData]] = defaultdict(dict)

        # Last update time by symbol
        self.last_update: Dict[str, float] = {}

        # Subscribers
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)

        # Statistics
        self.stats = {
            'messages_received': 0,
            'updates_published': 0,
            'exchanges_active': 0
        }

        logger.info(f"Market data aggregator initialized: {len(config.symbols)} symbols")

    async def start(self):
        """Start all exchange connections."""
        tasks = []

        for exchange in self.config.exchanges:
            if exchange == 'binance':
                task = self._start_binance()
                tasks.append(task)
            elif exchange == 'coinbase':
                task = self._start_coinbase()
                tasks.append(task)
            else:
                logger.warning(f"Unknown exchange: {exchange}")

        # Connect all exchanges
        await asyncio.gather(*tasks, return_exceptions=True)

        self.stats['exchanges_active'] = len(self.clients)
        logger.info(f"Started {self.stats['exchanges_active']} exchanges")

    async def stop(self):
        """Stop all exchange connections."""
        tasks = []
        for client in self.clients.values():
            tasks.append(client.disconnect())

        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Stopped all exchanges")

    async def _start_binance(self):
        """Start Binance WebSocket connection."""
        # Get Binance symbols
        binance_symbols = []
        for unified_symbol, exchange_map in self.config.symbols.items():
            if 'binance' in exchange_map:
                binance_symbols.append(exchange_map['binance'])

        if not binance_symbols:
            return

        # Configure Binance client
        config = BinanceConfig(
            symbols=binance_symbols,
            streams=[BinanceStream.TRADE, BinanceStream.TICKER, BinanceStream.BOOK_TICKER]
        )

        client = BinanceWebSocket(config)

        # Register handlers
        client.on_trade(lambda data: self._handle_data('binance', data))
        client.on_ticker(lambda data: self._handle_data('binance', data))
        client.on_book_ticker(lambda data: self._handle_data('binance', data))

        await client.connect()
        self.clients['binance'] = client

        logger.info(f"Binance started with {len(binance_symbols)} symbols")

    async def _start_coinbase(self):
        """Start Coinbase WebSocket connection."""
        # Get Coinbase symbols
        coinbase_products = []
        for unified_symbol, exchange_map in self.config.symbols.items():
            if 'coinbase' in exchange_map:
                coinbase_products.append(exchange_map['coinbase'])

        if not coinbase_products:
            return

        # Configure Coinbase client
        config = CoinbaseConfig(
            product_ids=coinbase_products,
            channels=[CoinbaseChannel.TICKER, CoinbaseChannel.MATCHES, CoinbaseChannel.LEVEL2]
        )

        client = CoinbaseWebSocket(config)

        # Register handlers
        client.on_ticker(lambda data: self._handle_data('coinbase', data))
        client.on_matches(lambda data: self._handle_data('coinbase', data))
        client.on_level2(lambda data: self._handle_data('coinbase', data))

        await client.connect()
        self.clients['coinbase'] = client

        logger.info(f"Coinbase started with {len(coinbase_products)} products")

    def _handle_data(self, exchange: str, data: MarketData):
        """Handle data from any exchange."""
        try:
            # Normalize symbol
            unified_symbol = self._normalize_symbol(exchange, data.symbol)
            if not unified_symbol:
                return

            # Update statistics
            self.stats['messages_received'] += 1

            # Add to buffer
            self.buffers[unified_symbol][data.data_type].append(data)

            # Check if we should aggregate and publish
            now = asyncio.get_event_loop().time()
            last_update = self.last_update.get(unified_symbol, 0)

            if now - last_update >= self.config.update_interval:
                self._aggregate_and_publish(unified_symbol, data.data_type)
                self.last_update[unified_symbol] = now

        except Exception as e:
            logger.error(f"Error handling data: {e}")

    def _normalize_symbol(self, exchange: str, exchange_symbol: str) -> Optional[str]:
        """Normalize exchange-specific symbol to unified symbol."""
        for unified_symbol, exchange_map in self.config.symbols.items():
            if exchange in exchange_map and exchange_map[exchange] == exchange_symbol:
                return unified_symbol
        return None

    def _aggregate_and_publish(self, symbol: str, data_type: str):
        """Aggregate data from multiple exchanges and publish."""
        try:
            # Get recent data from all exchanges
            recent_data = list(self.buffers[symbol][data_type])

            if not recent_data:
                return

            # Aggregate based on strategy
            aggregated = self._aggregate_data(recent_data, data_type)

            if aggregated:
                # Store latest
                self.latest[symbol][data_type] = aggregated

                # Publish to subscribers
                self._publish_update(symbol, data_type, aggregated)

                # Update statistics
                self.stats['updates_published'] += 1

        except Exception as e:
            logger.error(f"Error aggregating data for {symbol}: {e}")

    def _aggregate_data(
        self,
        data_points: List[MarketData],
        data_type: str
    ) -> Optional[UnifiedMarketData]:
        """Aggregate data points using configured strategy."""
        if not data_points:
            return None

        try:
            latest_point = data_points[-1]
            exchanges = list(set(d.exchange for d in data_points))

            if data_type == 'trade':
                return self._aggregate_trades(data_points, exchanges)
            elif data_type == 'ticker':
                return self._aggregate_tickers(data_points, exchanges)
            elif data_type == 'book_ticker':
                return self._aggregate_book_tickers(data_points, exchanges)
            else:
                # Default: use latest
                return UnifiedMarketData(
                    symbol=self._normalize_symbol(latest_point.exchange, latest_point.symbol),
                    data_type=data_type,
                    timestamp=latest_point.timestamp,
                    exchanges=exchanges,
                    data=latest_point.data,
                    source_data=data_points
                )

        except Exception as e:
            logger.error(f"Error aggregating {data_type}: {e}")
            return None

    def _aggregate_trades(
        self,
        trades: List[MarketData],
        exchanges: List[str]
    ) -> UnifiedMarketData:
        """Aggregate trade data."""
        latest = trades[-1]
        prices = [t.data['price'] for t in trades]
        volumes = [t.data.get('quantity', t.data.get('size', 0)) for t in trades]

        if self.config.aggregation_strategy == AggregationStrategy.WEIGHTED:
            # Volume-weighted average price
            total_volume = sum(volumes)
            if total_volume > 0:
                vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume
            else:
                vwap = prices[-1]
            price = vwap
        elif self.config.aggregation_strategy == AggregationStrategy.MEDIAN:
            price = statistics.median(prices)
        elif self.config.aggregation_strategy == AggregationStrategy.AVERAGE:
            price = statistics.mean(prices)
        else:
            # Latest
            price = prices[-1]

        unified_symbol = self._normalize_symbol(latest.exchange, latest.symbol)

        return UnifiedMarketData(
            symbol=unified_symbol,
            data_type='trade',
            timestamp=latest.timestamp,
            exchanges=exchanges,
            data={
                'price': price,
                'volume': sum(volumes),
                'num_trades': len(trades),
                'exchanges': exchanges
            },
            source_data=trades
        )

    def _aggregate_tickers(
        self,
        tickers: List[MarketData],
        exchanges: List[str]
    ) -> UnifiedMarketData:
        """Aggregate ticker data."""
        latest = tickers[-1]
        prices = [t.data.get('last_price', t.data.get('price', 0)) for t in tickers]

        if self.config.aggregation_strategy == AggregationStrategy.MEDIAN:
            price = statistics.median(prices)
        elif self.config.aggregation_strategy == AggregationStrategy.AVERAGE:
            price = statistics.mean(prices)
        else:
            price = prices[-1]

        volumes = [t.data.get('volume', t.data.get('volume_24h', 0)) for t in tickers]
        unified_symbol = self._normalize_symbol(latest.exchange, latest.symbol)

        return UnifiedMarketData(
            symbol=unified_symbol,
            data_type='ticker',
            timestamp=latest.timestamp,
            exchanges=exchanges,
            data={
                'price': price,
                'volume_24h': sum(volumes),
                'num_exchanges': len(exchanges),
                'price_range': [min(prices), max(prices)],
                'price_std': statistics.stdev(prices) if len(prices) > 1 else 0
            },
            source_data=tickers
        )

    def _aggregate_book_tickers(
        self,
        book_tickers: List[MarketData],
        exchanges: List[str]
    ) -> UnifiedMarketData:
        """Aggregate book ticker data (best bid/ask)."""
        latest = book_tickers[-1]

        # Get best bid (highest) and best ask (lowest) across exchanges
        bids = [(t.data['bid_price'], t.data['bid_quantity'], t.exchange)
                for t in book_tickers if 'bid_price' in t.data]
        asks = [(t.data['ask_price'], t.data['ask_quantity'], t.exchange)
                for t in book_tickers if 'ask_price' in t.data]

        best_bid = max(bids, key=lambda x: x[0]) if bids else (0, 0, '')
        best_ask = min(asks, key=lambda x: x[0]) if asks else (0, 0, '')

        unified_symbol = self._normalize_symbol(latest.exchange, latest.symbol)

        return UnifiedMarketData(
            symbol=unified_symbol,
            data_type='book_ticker',
            timestamp=latest.timestamp,
            exchanges=exchanges,
            data={
                'best_bid': best_bid[0],
                'best_bid_quantity': best_bid[1],
                'best_bid_exchange': best_bid[2],
                'best_ask': best_ask[0],
                'best_ask_quantity': best_ask[1],
                'best_ask_exchange': best_ask[2],
                'spread': best_ask[0] - best_bid[0] if best_ask[0] > 0 else 0,
                'spread_percent': ((best_ask[0] - best_bid[0]) / best_bid[0] * 100)
                                 if best_bid[0] > 0 else 0
            },
            source_data=book_tickers
        )

    def _publish_update(self, symbol: str, data_type: str, data: UnifiedMarketData):
        """Publish update to subscribers."""
        # Symbol-specific subscribers
        for handler in self.subscribers[f"{symbol}:{data_type}"]:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Subscriber error: {e}")

        # All symbols subscribers
        for handler in self.subscribers[f"*:{data_type}"]:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Subscriber error: {e}")

    def subscribe(self, symbol: str, data_type: str, handler: Callable):
        """
        Subscribe to aggregated market data.

        Args:
            symbol: Unified symbol (e.g., 'BTC/USD') or '*' for all
            data_type: Data type (e.g., 'trade', 'ticker', 'book_ticker')
            handler: Callback function
        """
        key = f"{symbol}:{data_type}"
        self.subscribers[key].append(handler)
        logger.info(f"Subscribed to {key}")

    def get_latest(self, symbol: str, data_type: str) -> Optional[UnifiedMarketData]:
        """Get latest aggregated data."""
        return self.latest.get(symbol, {}).get(data_type)

    def get_stats(self) -> dict:
        """Get aggregator statistics."""
        return {
            **self.stats,
            'symbols': len(self.config.symbols),
            'exchanges_configured': len(self.config.exchanges),
            'subscribers': sum(len(subs) for subs in self.subscribers.values())
        }


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("🌐 Market Data Aggregator Demo")
    print("=" * 60)

    async def demo():
        # Configure aggregator
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
            update_interval=1.0
        )

        aggregator = MarketDataAggregator(config)

        # Subscribe to aggregated data
        def handle_trade(data: UnifiedMarketData):
            print(f"💰 {data.symbol} TRADE: ${data.data['price']:,.2f} | "
                  f"Vol: {data.data['volume']:.4f} | "
                  f"Exchanges: {', '.join(data.exchanges)}")

        def handle_ticker(data: UnifiedMarketData):
            print(f"📈 {data.symbol} TICKER: ${data.data['price']:,.2f} | "
                  f"24h Vol: {data.data['volume_24h']:,.2f} | "
                  f"Exchanges: {data.data['num_exchanges']} | "
                  f"Spread: ${data.data['price_range'][1] - data.data['price_range'][0]:,.2f}")

        def handle_book(data: UnifiedMarketData):
            print(f"📖 {data.symbol} BOOK: "
                  f"Bid ${data.data['best_bid']:,.2f} ({data.data['best_bid_exchange']}) | "
                  f"Ask ${data.data['best_ask']:,.2f} ({data.data['best_ask_exchange']}) | "
                  f"Spread: {data.data['spread_percent']:.3f}%")

        aggregator.subscribe('*', 'trade', handle_trade)
        aggregator.subscribe('*', 'ticker', handle_ticker)
        aggregator.subscribe('*', 'book_ticker', handle_book)

        # Start aggregator
        print("\n🚀 Starting aggregator...")
        await aggregator.start()

        # Stream data for 30 seconds
        print("\n📡 Streaming aggregated data (30 seconds)...\n")
        await asyncio.sleep(30)

        # Get stats
        print("\n📊 Aggregator Stats:")
        stats = aggregator.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")

        # Stop aggregator
        print("\n🛑 Stopping aggregator...")
        await aggregator.stop()

        print("\n✅ Market data aggregator demo complete!")

    # Run demo
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
