"""
Coinbase WebSocket Client
Real-time market data from Coinbase Pro/Advanced Trade.
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


class CoinbaseChannel(Enum):
    """Coinbase WebSocket channels."""
    TICKER = "ticker"
    LEVEL2 = "level2"  # Order book
    MATCHES = "matches"  # Trades
    HEARTBEAT = "heartbeat"
    STATUS = "status"
    FULL = "full"  # Full order book updates


@dataclass
class CoinbaseConfig:
    """Coinbase WebSocket configuration."""
    product_ids: List[str]  # e.g., ['BTC-USD', 'ETH-USD']
    channels: List[CoinbaseChannel]
    sandbox: bool = False


class CoinbaseWebSocket:
    """
    Coinbase WebSocket client for real-time market data.

    Supported Channels:
    - Ticker: Real-time price updates
    - Level2: Order book snapshots and updates
    - Matches: Completed trades
    - Heartbeat: Connection health
    - Status: System status

    Features:
    - Multiple product subscription
    - Channel-based streaming
    - Automatic reconnection
    - Message parsing
    """

    def __init__(self, config: CoinbaseConfig):
        """
        Initialize Coinbase WebSocket client.

        Args:
            config: Coinbase configuration
        """
        self.config = config
        self.manager: Optional[WebSocketManager] = None

        # Message handlers by channel
        self.handlers: Dict[str, List[Callable]] = {
            channel.value: [] for channel in CoinbaseChannel
        }

        # Build WebSocket URL
        self.url = self._build_url()

        logger.info(f"Coinbase WebSocket initialized: {len(config.product_ids)} products")

    def _build_url(self) -> str:
        """Build Coinbase WebSocket URL."""
        if self.config.sandbox:
            return "wss://ws-feed-public.sandbox.exchange.coinbase.com"
        return "wss://ws-feed.exchange.coinbase.com"

    async def connect(self):
        """Connect to Coinbase WebSocket."""
        ws_config = WebSocketConfig(
            url=self.url,
            name="coinbase",
            heartbeat_interval=60,
            reconnect_delay=5,
            ping_interval=30
        )

        self.manager = WebSocketManager(ws_config)

        # Register message handler
        self.manager.on_message(self._handle_message)

        # Connect
        await self.manager.connect()

        # Subscribe to channels
        await self._subscribe()

        logger.info("Connected to Coinbase WebSocket")

    async def disconnect(self):
        """Disconnect from Coinbase WebSocket."""
        if self.manager:
            # Unsubscribe before disconnecting
            await self._unsubscribe()
            await self.manager.disconnect()

    async def _subscribe(self):
        """Subscribe to configured channels."""
        subscribe_message = {
            "type": "subscribe",
            "product_ids": self.config.product_ids,
            "channels": [channel.value for channel in self.config.channels]
        }

        await self.manager.send(subscribe_message)
        logger.info(f"Subscribed to {len(self.config.channels)} channels")

    async def _unsubscribe(self):
        """Unsubscribe from all channels."""
        unsubscribe_message = {
            "type": "unsubscribe",
            "product_ids": self.config.product_ids,
            "channels": [channel.value for channel in self.config.channels]
        }

        try:
            await self.manager.send(unsubscribe_message)
            logger.info("Unsubscribed from channels")
        except Exception as e:
            logger.error(f"Error unsubscribing: {e}")

    def on_ticker(self, handler: Callable):
        """Register ticker handler."""
        self.handlers[CoinbaseChannel.TICKER.value].append(handler)
        return handler

    def on_level2(self, handler: Callable):
        """Register level2 order book handler."""
        self.handlers[CoinbaseChannel.LEVEL2.value].append(handler)
        return handler

    def on_matches(self, handler: Callable):
        """Register matches (trades) handler."""
        self.handlers[CoinbaseChannel.MATCHES.value].append(handler)
        return handler

    def on_heartbeat(self, handler: Callable):
        """Register heartbeat handler."""
        self.handlers[CoinbaseChannel.HEARTBEAT.value].append(handler)
        return handler

    async def _handle_message(self, raw_message: dict):
        """Parse and route Coinbase messages."""
        try:
            message_type = raw_message.get('type')

            if not message_type:
                return

            # Handle subscription confirmations
            if message_type in ['subscriptions', 'error']:
                logger.info(f"Subscription response: {raw_message}")
                return

            # Parse message
            market_data = self._parse_message(message_type, raw_message)

            if market_data:
                # Route to appropriate handlers
                channel = self._get_channel_from_type(message_type)
                if channel and channel in self.handlers:
                    for handler in self.handlers[channel]:
                        try:
                            result = handler(market_data)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            logger.error(f"Handler error for {channel}: {e}")

        except Exception as e:
            logger.error(f"Error handling Coinbase message: {e}")

    def _get_channel_from_type(self, message_type: str) -> Optional[str]:
        """Map message type to channel."""
        type_to_channel = {
            'ticker': CoinbaseChannel.TICKER.value,
            'snapshot': CoinbaseChannel.LEVEL2.value,
            'l2update': CoinbaseChannel.LEVEL2.value,
            'match': CoinbaseChannel.MATCHES.value,
            'last_match': CoinbaseChannel.MATCHES.value,
            'heartbeat': CoinbaseChannel.HEARTBEAT.value,
            'status': CoinbaseChannel.STATUS.value,
            'open': CoinbaseChannel.FULL.value,
            'done': CoinbaseChannel.FULL.value,
            'change': CoinbaseChannel.FULL.value,
            'activate': CoinbaseChannel.FULL.value,
        }
        return type_to_channel.get(message_type)

    def _parse_message(self, message_type: str, data: dict) -> Optional[MarketData]:
        """Parse Coinbase message into MarketData format."""
        try:
            product_id = data.get('product_id', '')

            # Parse timestamp
            time_str = data.get('time', '')
            if time_str:
                timestamp = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()

            if message_type == 'ticker':
                return MarketData(
                    exchange='coinbase',
                    symbol=product_id,
                    data_type='ticker',
                    timestamp=timestamp,
                    data={
                        'price': float(data.get('price', 0)),
                        'best_bid': float(data.get('best_bid', 0)),
                        'best_ask': float(data.get('best_ask', 0)),
                        'volume_24h': float(data.get('volume_24h', 0)),
                        'low_24h': float(data.get('low_24h', 0)),
                        'high_24h': float(data.get('high_24h', 0)),
                        'open_24h': float(data.get('open_24h', 0)),
                        'sequence': data.get('sequence'),
                        'trade_id': data.get('trade_id'),
                        'side': data.get('side'),
                        'last_size': float(data.get('last_size', 0))
                    },
                    raw_message=data
                )

            elif message_type in ['match', 'last_match']:
                return MarketData(
                    exchange='coinbase',
                    symbol=product_id,
                    data_type='trade',
                    timestamp=timestamp,
                    data={
                        'trade_id': data.get('trade_id'),
                        'price': float(data.get('price', 0)),
                        'size': float(data.get('size', 0)),
                        'side': data.get('side'),
                        'maker_order_id': data.get('maker_order_id'),
                        'taker_order_id': data.get('taker_order_id'),
                        'sequence': data.get('sequence')
                    },
                    raw_message=data
                )

            elif message_type == 'snapshot':
                return MarketData(
                    exchange='coinbase',
                    symbol=product_id,
                    data_type='orderbook_snapshot',
                    timestamp=timestamp,
                    data={
                        'bids': [[float(p), float(s)] for p, s in data.get('bids', [])],
                        'asks': [[float(p), float(s)] for p, s in data.get('asks', [])],
                    },
                    raw_message=data
                )

            elif message_type == 'l2update':
                return MarketData(
                    exchange='coinbase',
                    symbol=product_id,
                    data_type='orderbook_update',
                    timestamp=timestamp,
                    data={
                        'changes': [
                            {
                                'side': change[0],
                                'price': float(change[1]),
                                'size': float(change[2])
                            }
                            for change in data.get('changes', [])
                        ]
                    },
                    raw_message=data
                )

            elif message_type == 'heartbeat':
                return MarketData(
                    exchange='coinbase',
                    symbol=product_id,
                    data_type='heartbeat',
                    timestamp=timestamp,
                    data={
                        'sequence': data.get('sequence'),
                        'last_trade_id': data.get('last_trade_id')
                    },
                    raw_message=data
                )

            else:
                logger.debug(f"Unhandled message type: {message_type}")
                return None

        except Exception as e:
            logger.error(f"Error parsing {message_type} message: {e}")
            return None

    async def subscribe_products(self, product_ids: List[str], channels: List[str]):
        """
        Subscribe to additional products and channels.

        Args:
            product_ids: List of product IDs (e.g., ['BTC-USD', 'ETH-USD'])
            channels: List of channel names (e.g., ['ticker', 'matches'])
        """
        if not self.manager or self.manager.state != ConnectionState.CONNECTED:
            raise RuntimeError("Not connected to Coinbase WebSocket")

        message = {
            "type": "subscribe",
            "product_ids": product_ids,
            "channels": channels
        }

        await self.manager.send(message)
        logger.info(f"Subscribed to {len(product_ids)} products")

    async def unsubscribe_products(self, product_ids: List[str], channels: List[str]):
        """
        Unsubscribe from products and channels.

        Args:
            product_ids: List of product IDs
            channels: List of channel names
        """
        if not self.manager or self.manager.state != ConnectionState.CONNECTED:
            raise RuntimeError("Not connected to Coinbase WebSocket")

        message = {
            "type": "unsubscribe",
            "product_ids": product_ids,
            "channels": channels
        }

        await self.manager.send(message)
        logger.info(f"Unsubscribed from {len(product_ids)} products")

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

    print("💰 Coinbase WebSocket Demo")
    print("=" * 60)

    async def demo():
        # Configure Coinbase WebSocket
        config = CoinbaseConfig(
            product_ids=['BTC-USD', 'ETH-USD'],
            channels=[CoinbaseChannel.TICKER, CoinbaseChannel.MATCHES, CoinbaseChannel.LEVEL2]
        )

        client = CoinbaseWebSocket(config)

        # Ticker handler
        @client.on_ticker
        def handle_ticker(data: MarketData):
            print(f"📈 TICKER {data.symbol}: ${data.data['price']:,.2f} | "
                  f"Bid: ${data.data['best_bid']:,.2f} | Ask: ${data.data['best_ask']:,.2f} | "
                  f"24h Vol: {data.data['volume_24h']:,.2f}")

        # Trade handler
        @client.on_matches
        def handle_trade(data: MarketData):
            side_emoji = "🟢" if data.data['side'] == 'buy' else "🔴"
            print(f"{side_emoji} TRADE {data.symbol}: ${data.data['price']:,.2f} x {data.data['size']:.4f} "
                  f"({data.data['side'].upper()})")

        # Order book handler
        @client.on_level2
        def handle_level2(data: MarketData):
            if data.data_type == 'orderbook_snapshot':
                best_bid = data.data['bids'][0] if data.data['bids'] else [0, 0]
                best_ask = data.data['asks'][0] if data.data['asks'] else [0, 0]
                print(f"📖 ORDERBOOK SNAPSHOT {data.symbol}: "
                      f"Best Bid: ${best_bid[0]:,.2f} x {best_bid[1]:.4f} | "
                      f"Best Ask: ${best_ask[0]:,.2f} x {best_ask[1]:.4f}")

        # Heartbeat handler
        @client.on_heartbeat
        def handle_heartbeat(data: MarketData):
            print(f"💓 HEARTBEAT {data.symbol}: seq={data.data['sequence']}")

        # Connect
        print("\n🔌 Connecting to Coinbase...")
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

        print("\n✅ Coinbase WebSocket demo complete!")

    # Run demo
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
