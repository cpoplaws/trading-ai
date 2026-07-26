"""
Real-Time Market Data Infrastructure
Provides WebSocket connections and unified market data aggregation.
"""
from .websocket_manager import (
    WebSocketManager,
    WebSocketPool,
    WebSocketConfig,
    ConnectionState,
    MarketData
)

from .binance_websocket import (
    BinanceWebSocket,
    BinanceConfig,
    BinanceStream
)

from .coinbase_websocket import (
    CoinbaseWebSocket,
    CoinbaseConfig,
    CoinbaseChannel
)

from .market_data_aggregator import (
    MarketDataAggregator,
    AggregatorConfig,
    UnifiedMarketData,
    AggregationStrategy
)

__all__ = [
    # WebSocket Core
    'WebSocketManager',
    'WebSocketPool',
    'WebSocketConfig',
    'ConnectionState',
    'MarketData',

    # Binance
    'BinanceWebSocket',
    'BinanceConfig',
    'BinanceStream',

    # Coinbase
    'CoinbaseWebSocket',
    'CoinbaseConfig',
    'CoinbaseChannel',

    # Aggregator
    'MarketDataAggregator',
    'AggregatorConfig',
    'UnifiedMarketData',
    'AggregationStrategy',
]

__version__ = '1.0.0'
