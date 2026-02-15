"""
Binance WebSocket Client
Real-time market data from Binance exchange.
"""
import asyncio
import logging
from typing import Dict, List, Callable, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from websocket_manager import (
    WebSocketManager,
    WebSocketConfig,
    MarketData,
    ConnectionState
)

logger = logging.getLogger(__name__)


class BinanceStream(Enum):
    """Binance WebSocket stream types."""
    TRADE = "trade"
    TICKER = "ticker"
    KLINE = "kline"
    DEPTH = "depth"
    AGG_TRADE = "aggTrade"
    BOOK_TICKER = "bookTicker"
    MINI_TICKER = "miniTicker"


@dataclass
class BinanceConfig:
    """Binance WebSocket configuration."""
    symbols: List[str]
    streams: List[BinanceStream]
    testnet: bool = False
    combined: bool = True  # Use combined stream


class BinanceWebSocket:
    """
    Binance WebSocket client for real-time market data.

    Supported Streams:
    - Trade: Individual trade data
    - Ticker: 24hr rolling window ticker
    - Kline: Candlestick data
    - Depth: Order book updates
    - AggTrade: Aggregated trades
    - BookTicker: Best bid/ask updates

    Features:
    - Multiple symbol subscription
    - Combined stream support
    - Automatic reconnection
    - Message parsing and routing
    """

    def __init__(self, config: BinanceConfig):
        """
        Initialize Binance WebSocket client.

        Args:
            config: Binance configuration
        """
        self.config = config
        self.manager: Optional[WebSocketManager] = None

        # Message handlers by stream type
        self.handlers: Dict[str, List[Callable]] = {
            stream.value: [] for stream in BinanceStream
        }

        # Build WebSocket URL
        self.url = self._build_url()

        logger.info(f"Binance WebSocket initialized: {len(config.symbols)} symbols")

    def _build_url(self) -> str:
        """Build Binance WebSocket URL."""
        base_url = "wss://stream.binance.com:9443" if not self.config.testnet else \
                   "wss://testnet.binance.vision"

        if self.config.combined:
            # Combined stream for multiple subscriptions
            streams = []
            for symbol in self.config.symbols:
                symbol_lower = symbol.lower()
                for stream_type in self.config.streams:
                    if stream_type == BinanceStream.KLINE:
                        # Kline requires interval
                        streams.append(f"{symbol_lower}@kline_1m")
                    elif stream_type == BinanceStream.DEPTH:
                        # Depth with 100ms update speed
                        streams.append(f"{symbol_lower}@depth@100ms")
                    else:
                        streams.append(f"{symbol_lower}@{stream_type.value}")

            stream_names = "/".join(streams)
            return f"{base_url}/stream?streams={stream_names}"
        else:
            # Single stream
            symbol = self.config.symbols[0].lower()
            stream = self.config.streams[0].value
            return f"{base_url}/ws/{symbol}@{stream}"

    async def connect(self):
        """Connect to Binance WebSocket."""
        ws_config = WebSocketConfig(
            url=self.url,
            name="binance",
            heartbeat_interval=60,
            reconnect_delay=5,
            ping_interval=20
        )

        self.manager = WebSocketManager(ws_config)

        # Register message handler
        self.manager.on_message(self._handle_message)

        # Connect
        await self.manager.connect()

        logger.info("Connected to Binance WebSocket")

    async def disconnect(self):
        """Disconnect from Binance WebSocket."""
        if self.manager:
            await self.manager.disconnect()

    def on_trade(self, handler: Callable):
        """Register trade stream handler."""
        self.handlers[BinanceStream.TRADE.value].append(handler)
        return handler

    def on_ticker(self, handler: Callable):
        """Register ticker stream handler."""
        self.handlers[BinanceStream.TICKER.value].append(handler)
        return handler

    def on_kline(self, handler: Callable):
        """Register kline stream handler."""
        self.handlers[BinanceStream.KLINE.value].append(handler)
        return handler

    def on_depth(self, handler: Callable):
        """Register depth stream handler."""
        self.handlers[BinanceStream.DEPTH.value].append(handler)
        return handler

    def on_agg_trade(self, handler: Callable):
        """Register aggregated trade handler."""
        self.handlers[BinanceStream.AGG_TRADE.value].append(handler)
        return handler

    def on_book_ticker(self, handler: Callable):
        """Register book ticker handler."""
        self.handlers[BinanceStream.BOOK_TICKER.value].append(handler)
        return handler

    async def _handle_message(self, raw_message: dict):
        """Parse and route Binance messages."""
        try:
            # Combined stream format
            if 'stream' in raw_message:
                stream_name = raw_message['stream']
                data = raw_message['data']
            else:
                # Single stream format
                data = raw_message
                stream_name = data.get('e', '')

            # Parse message based on event type
            event_type = data.get('e')

            if not event_type:
                return

            # Route to appropriate handlers
            market_data = self._parse_message(event_type, data)

            if market_data and event_type in self.handlers:
                for handler in self.handlers[event_type]:
                    try:
                        result = handler(market_data)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Handler error for {event_type}: {e}")

        except Exception as e:
            logger.error(f"Error handling Binance message: {e}")

    def _parse_message(self, event_type: str, data: dict) -> Optional[MarketData]:
        """Parse Binance message into MarketData format."""
        try:
            symbol = data.get('s', '')
            timestamp = datetime.fromtimestamp(data.get('E', 0) / 1000)

            if event_type == 'trade':
                return MarketData(
                    exchange='binance',
                    symbol=symbol,
                    data_type='trade',
                    timestamp=timestamp,
                    data={
                        'price': float(data['p']),
                        'quantity': float(data['q']),
                        'trade_id': data['t'],
                        'is_buyer_maker': data['m'],
                        'trade_time': datetime.fromtimestamp(data['T'] / 1000)
                    },
                    raw_message=data
                )

            elif event_type == 'aggTrade':
                return MarketData(
                    exchange='binance',
                    symbol=symbol,
                    data_type='agg_trade',
                    timestamp=timestamp,
                    data={
                        'price': float(data['p']),
                        'quantity': float(data['q']),
                        'first_trade_id': data['f'],
                        'last_trade_id': data['l'],
                        'is_buyer_maker': data['m'],
                        'trade_time': datetime.fromtimestamp(data['T'] / 1000)
                    },
                    raw_message=data
                )

            elif event_type == '24hrTicker':
                return MarketData(
                    exchange='binance',
                    symbol=symbol,
                    data_type='ticker',
                    timestamp=timestamp,
                    data={
                        'price_change': float(data['p']),
                        'price_change_percent': float(data['P']),
                        'weighted_avg_price': float(data['w']),
                        'last_price': float(data['c']),
                        'last_quantity': float(data['Q']),
                        'open_price': float(data['o']),
                        'high_price': float(data['h']),
                        'low_price': float(data['l']),
                        'volume': float(data['v']),
                        'quote_volume': float(data['q']),
                        'open_time': datetime.fromtimestamp(data['O'] / 1000),
                        'close_time': datetime.fromtimestamp(data['C'] / 1000),
                        'num_trades': data['n']
                    },
                    raw_message=data
                )

            elif event_type == 'kline':
                k = data['k']
                return MarketData(
                    exchange='binance',
                    symbol=symbol,
                    data_type='kline',
                    timestamp=timestamp,
                    data={
                        'interval': k['i'],
                        'open_time': datetime.fromtimestamp(k['t'] / 1000),
                        'close_time': datetime.fromtimestamp(k['T'] / 1000),
                        'open': float(k['o']),
                        'high': float(k['h']),
                        'low': float(k['l']),
                        'close': float(k['c']),
                        'volume': float(k['v']),
                        'quote_volume': float(k['q']),
                        'num_trades': k['n'],
                        'is_closed': k['x'],
                        'taker_buy_base_volume': float(k['V']),
                        'taker_buy_quote_volume': float(k['Q'])
                    },
                    raw_message=data
                )

            elif event_type == 'depthUpdate':
                return MarketData(
                    exchange='binance',
                    symbol=symbol,
                    data_type='depth',
                    timestamp=timestamp,
                    data={
                        'first_update_id': data['U'],
                        'final_update_id': data['u'],
                        'bids': [[float(p), float(q)] for p, q in data['b']],
                        'asks': [[float(p), float(q)] for p, q in data['a']]
                    },
                    raw_message=data
                )

            elif event_type == 'bookTicker':
                return MarketData(
                    exchange='binance',
                    symbol=symbol,
                    data_type='book_ticker',
                    timestamp=timestamp,
                    data={
                        'bid_price': float(data['b']),
                        'bid_quantity': float(data['B']),
                        'ask_price': float(data['a']),
                        'ask_quantity': float(data['A']),
                        'spread': float(data['a']) - float(data['b']),
                        'spread_percent': ((float(data['a']) - float(data['b'])) /
                                         float(data['b']) * 100)
                    },
                    raw_message=data
                )

            else:
                logger.warning(f"Unknown event type: {event_type}")
                return None

        except Exception as e:
            logger.error(f"Error parsing {event_type} message: {e}")
            return None

    async def subscribe_symbols(self, symbols: List[str], streams: List[str]):
        """
        Subscribe to additional symbols and streams.

        Args:
            symbols: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT'])
            streams: List of stream types (e.g., ['trade', 'ticker'])
        """
        if not self.manager or self.manager.state != ConnectionState.CONNECTED:
            raise RuntimeError("Not connected to Binance WebSocket")

        # Build subscription message
        params = []
        for symbol in symbols:
            symbol_lower = symbol.lower()
            for stream in streams:
                params.append(f"{symbol_lower}@{stream}")

        message = {
            "method": "SUBSCRIBE",
            "params": params,
            "id": int(datetime.now().timestamp() * 1000)
        }

        await self.manager.send(message)
        logger.info(f"Subscribed to {len(params)} streams")

    async def unsubscribe_symbols(self, symbols: List[str], streams: List[str]):
        """
        Unsubscribe from symbols and streams.

        Args:
            symbols: List of trading pairs
            streams: List of stream types
        """
        if not self.manager or self.manager.state != ConnectionState.CONNECTED:
            raise RuntimeError("Not connected to Binance WebSocket")

        params = []
        for symbol in symbols:
            symbol_lower = symbol.lower()
            for stream in streams:
                params.append(f"{symbol_lower}@{stream}")

        message = {
            "method": "UNSUBSCRIBE",
            "params": params,
            "id": int(datetime.now().timestamp() * 1000)
        }

        await self.manager.send(message)
        logger.info(f"Unsubscribed from {len(params)} streams")

    def get_stats(self) -> dict:
        """Get connection statistics."""
        if self.manager:
            return self.manager.get_stats()
        return {}


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("📊 Binance WebSocket Demo")
    print("=" * 60)

    async def demo():
        # Configure Binance WebSocket
        config = BinanceConfig(
            symbols=['BTCUSDT', 'ETHUSDT'],
            streams=[BinanceStream.TRADE, BinanceStream.TICKER, BinanceStream.BOOK_TICKER]
        )

        client = BinanceWebSocket(config)

        # Trade handler
        @client.on_trade
        def handle_trade(data: MarketData):
            print(f"💰 TRADE {data.symbol}: ${data.data['price']:,.2f} x {data.data['quantity']:.4f}")

        # Ticker handler
        @client.on_ticker
        def handle_ticker(data: MarketData):
            print(f"📈 TICKER {data.symbol}: ${data.data['last_price']:,.2f} "
                  f"({data.data['price_change_percent']:+.2f}%) "
                  f"Volume: {data.data['volume']:,.2f}")

        # Book ticker handler
        @client.on_book_ticker
        def handle_book_ticker(data: MarketData):
            print(f"📖 BOOK {data.symbol}: "
                  f"Bid ${data.data['bid_price']:,.2f} x {data.data['bid_quantity']:.4f} | "
                  f"Ask ${data.data['ask_price']:,.2f} x {data.data['ask_quantity']:.4f} | "
                  f"Spread: {data.data['spread_percent']:.3f}%")

        # Connect
        print("\n🔌 Connecting to Binance...")
        await client.connect()

        # Stream data for 30 seconds
        print("\n📡 Streaming market data (30 seconds)...\n")
        await asyncio.sleep(30)

        # Get stats
        print("\n📊 Connection Stats:")
        stats = client.get_stats()
        print(f"   State: {stats['state']}")
        print(f"   Messages: {stats['message_count']}")
        print(f"   Errors: {stats['error_count']}")

        # Disconnect
        print("\n🔌 Disconnecting...")
        await client.disconnect()

        print("\n✅ Binance WebSocket demo complete!")

    # Run demo
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
