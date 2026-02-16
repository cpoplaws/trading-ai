# WebSocket Real-Time Data Infrastructure

## Overview

The trading AI system includes a comprehensive WebSocket infrastructure for receiving real-time market data from multiple cryptocurrency exchanges. This enables the autonomous agent and strategies to make decisions based on live market conditions rather than simulated or delayed data.

## Architecture

### Components

1. **WebSocket Manager** (`websocket_manager.py`)
   - Core WebSocket connection management
   - Automatic reconnection with exponential backoff
   - Heartbeat/ping-pong monitoring
   - Message queuing and callback handling
   - Connection pooling for multiple exchanges
   - Error recovery and failover

2. **Binance WebSocket Client** (`binance_websocket.py`)
   - Real-time data from Binance Spot and Futures
   - Supports multiple stream types (trade, ticker, kline, depth)
   - Combined stream for efficient multi-symbol subscriptions
   - Testnet support for development

3. **Coinbase WebSocket Client** (`coinbase_websocket.py`)
   - Real-time data from Coinbase Pro/Advanced Trade
   - Supports multiple channels (ticker, matches, level2 orderbook)
   - Sandbox environment support
   - Full order book updates

4. **Market Data Aggregator** (`market_data_aggregator.py`)
   - Unifies data from multiple exchanges
   - Calculates VWAP across exchanges
   - Best bid/ask aggregation
   - Order book depth aggregation
   - Latency monitoring

## Features

### Automatic Reconnection
- Exponential backoff strategy
- Configurable retry attempts
- State tracking (disconnected, connecting, connected, reconnecting)
- Graceful handling of connection failures

### Data Streams

#### Binance Streams
- **Trade**: Individual executed trades
- **Ticker**: 24hr rolling window statistics
- **Kline/Candlestick**: OHLCV data with various intervals
- **Depth**: Order book snapshots and updates
- **AggTrade**: Aggregated trades for reduced bandwidth
- **BookTicker**: Best bid/ask prices in real-time

#### Coinbase Channels
- **Ticker**: Real-time price and volume updates
- **Matches**: Completed trade executions
- **Level2**: Full order book snapshots and updates
- **Heartbeat**: Connection health monitoring
- **Status**: Exchange system status
- **Full**: Complete order flow (orders placed, filled, canceled)

### Latency Monitoring
- Round-trip time measurement
- Message receive timestamps
- Connection quality metrics
- Performance statistics

### Error Handling
- Connection timeouts
- Message parsing errors
- Rate limiting
- Network failures
- Exchange-specific errors

## Usage

### Basic Binance WebSocket

```python
from realtime import BinanceWebSocket, BinanceConfig, BinanceStream

# Configure
config = BinanceConfig(
    symbols=['BTCUSDT', 'ETHUSDT'],
    streams=[BinanceStream.TRADE, BinanceStream.TICKER],
    testnet=False,  # Use production
    combined=True    # Use combined stream
)

# Create client
ws = BinanceWebSocket(config)

# Register handler
@ws.on_trade
async def handle_trade(data):
    print(f"Trade: {data['symbol']} @ ${data['price']}")

# Connect
await ws.connect()
```

### Basic Coinbase WebSocket

```python
from realtime import CoinbaseWebSocket, CoinbaseConfig, CoinbaseChannel

# Configure
config = CoinbaseConfig(
    product_ids=['BTC-USD', 'ETH-USD'],
    channels=[CoinbaseChannel.TICKER, CoinbaseChannel.MATCHES],
    sandbox=False
)

# Create client
ws = CoinbaseWebSocket(config)

# Register handler
@ws.on_ticker
async def handle_ticker(data):
    print(f"Ticker: {data['product_id']} @ ${data['price']}")

# Connect
await ws.connect()
```

### Market Data Aggregator

```python
from realtime import MarketDataAggregator, AggregatorConfig

# Configure
config = AggregatorConfig(
    exchanges=['binance', 'coinbase'],
    symbols=['BTC', 'ETH'],
    update_interval_ms=100,
    enable_vwap=True,
    enable_book_aggregation=True
)

# Create aggregator
aggregator = MarketDataAggregator(config)

# Add WebSocket sources
await aggregator.add_source('binance', binance_ws)
await aggregator.add_source('coinbase', coinbase_ws)

# Register handler for unified data
@aggregator.on_update
async def handle_update(data):
    print(f"{data.symbol}: "
          f"Best Bid ${data.best_bid} ({data.best_bid_exchange}), "
          f"Best Ask ${data.best_ask} ({data.best_ask_exchange}), "
          f"VWAP ${data.vwap}")

# Start aggregation
await aggregator.start()
```

### Integration with Autonomous Agent

```python
from autonomous_agent import AutonomousTradingAgent, AgentConfig
from realtime import BinanceWebSocket, BinanceConfig, BinanceStream

class RealTimeAgent(AutonomousTradingAgent):
    def __init__(self, config):
        super().__init__(config)
        self.websocket = None
        self.market_data = {}

    async def start(self):
        # Set up WebSocket
        ws_config = BinanceConfig(
            symbols=['BTCUSDT'],
            streams=[BinanceStream.TRADE]
        )
        self.websocket = BinanceWebSocket(ws_config)

        @self.websocket.on_trade
        async def handle_trade(data):
            self.market_data[data['symbol']] = float(data['price'])

        await self.websocket.connect()

        # Start agent
        await super().start()

    async def _get_market_data(self):
        # Return real-time data instead of simulated
        return self.market_data
```

## Configuration

### WebSocket Manager Config

```python
from realtime import WebSocketConfig

config = WebSocketConfig(
    url='wss://example.com/ws',
    name='my_connection',
    heartbeat_interval=30,      # Send heartbeat every 30s
    reconnect_delay=5,          # Wait 5s before reconnecting
    max_reconnect_attempts=10,  # Try 10 times before giving up
    timeout=10,                 # 10s connection timeout
    ping_interval=20,           # Send ping every 20s
    ping_timeout=10             # Expect pong within 10s
)
```

### Binance Config

```python
config = BinanceConfig(
    symbols=['BTCUSDT', 'ETHUSDT'],      # Trading pairs
    streams=[                             # Data types
        BinanceStream.TRADE,
        BinanceStream.TICKER,
        BinanceStream.KLINE
    ],
    testnet=False,  # Use production (False) or testnet (True)
    combined=True   # Use combined stream for efficiency
)
```

### Coinbase Config

```python
config = CoinbaseConfig(
    product_ids=['BTC-USD', 'ETH-USD'],  # Products
    channels=[                            # Channels
        CoinbaseChannel.TICKER,
        CoinbaseChannel.MATCHES,
        CoinbaseChannel.LEVEL2
    ],
    sandbox=False  # Use production (False) or sandbox (True)
)
```

## Data Structures

### Trade Data

```python
{
    'symbol': 'BTCUSDT',
    'price': 42000.50,
    'quantity': 0.025,
    'timestamp': 1704067200000,
    'is_buyer_maker': True
}
```

### Ticker Data

```python
{
    'symbol': 'BTCUSDT',
    'price': 42000.50,
    'price_change': 500.00,
    'price_change_percent': 1.20,
    'high': 42500.00,
    'low': 41000.00,
    'volume': 15000.50,
    'quote_volume': 630021000.00
}
```

### Order Book Data

```python
{
    'symbol': 'BTCUSDT',
    'bids': [[42000.00, 1.5], [41999.50, 2.0]],  # [price, quantity]
    'asks': [[42001.00, 1.2], [42001.50, 3.5]],
    'timestamp': 1704067200000
}
```

### Unified Market Data

```python
{
    'symbol': 'BTC',
    'timestamp': 1704067200000,
    'best_bid': 42000.00,
    'best_ask': 42001.00,
    'best_bid_exchange': 'binance',
    'best_ask_exchange': 'coinbase',
    'vwap': 42000.75,
    'spread_bps': 2.38,
    'total_volume': 25.5,
    'num_exchanges': 2
}
```

## Error Handling

### Connection Errors

```python
ws = BinanceWebSocket(config)

@ws.on_error
async def handle_error(error):
    logger.error(f"WebSocket error: {error}")
    # Implement custom error handling
    if "rate limit" in str(error).lower():
        await asyncio.sleep(60)  # Wait before reconnecting
```

### State Changes

```python
@ws.on_state_change
async def handle_state_change(old_state, new_state):
    logger.info(f"State changed: {old_state} -> {new_state}")

    if new_state == ConnectionState.CONNECTED:
        # Connection established
        pass
    elif new_state == ConnectionState.RECONNECTING:
        # Attempting to reconnect
        pass
    elif new_state == ConnectionState.FAILED:
        # Connection failed permanently
        pass
```

## Performance Optimization

### Combined Streams

Use combined streams to reduce the number of WebSocket connections:

```python
# Good: 1 connection for multiple streams
config = BinanceConfig(
    symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
    streams=[BinanceStream.TRADE, BinanceStream.TICKER],
    combined=True  # Use combined stream
)

# Less efficient: Separate connection per stream
```

### Message Filtering

Filter messages early to reduce processing overhead:

```python
@ws.on_trade
async def handle_trade(data):
    # Filter out small trades
    if data['quantity'] < 0.01:
        return

    # Process significant trades only
    await process_trade(data)
```

### Buffering

Buffer messages for batch processing:

```python
message_buffer = []

@ws.on_trade
async def handle_trade(data):
    message_buffer.append(data)

    # Process in batches
    if len(message_buffer) >= 100:
        await process_batch(message_buffer)
        message_buffer.clear()
```

## Monitoring

### Connection Health

```python
# Check connection state
state = ws.manager.state
logger.info(f"Connection state: {state}")

# Get latency
latency_ms = ws.get_latency()
logger.info(f"Latency: {latency_ms:.2f}ms")

# Get message count
count = ws.manager.message_count
logger.info(f"Messages received: {count}")

# Get error count
errors = ws.manager.error_count
logger.info(f"Errors: {errors}")
```

### Statistics

```python
stats = aggregator.get_stats()
print(f"Total messages: {stats['total_messages']}")
print(f"Updates/sec: {stats['updates_per_second']}")
print(f"Avg latency: {stats['avg_latency_ms']}ms")
print(f"Active connections: {stats['active_connections']}")
```

## Testing

### Using Testnet/Sandbox

Always test with testnet/sandbox environments first:

```python
# Binance testnet
config = BinanceConfig(
    symbols=['BTCUSDT'],
    streams=[BinanceStream.TRADE],
    testnet=True  # Use testnet
)

# Coinbase sandbox
config = CoinbaseConfig(
    product_ids=['BTC-USD'],
    channels=[CoinbaseChannel.TICKER],
    sandbox=True  # Use sandbox
)
```

### Example Scripts

Run the demo scripts:

```bash
# Basic WebSocket demo
python examples/websocket_realtime_demo.py --duration 60

# Agent with real-time data
python examples/agent_with_realtime_data.py

# Binance only
python examples/websocket_realtime_demo.py --binance-only

# Coinbase only
python examples/websocket_realtime_demo.py --coinbase-only
```

## Best Practices

1. **Always use testnet/sandbox for development**
   - Never test with production API keys
   - Testnet data may differ from production

2. **Handle reconnections gracefully**
   - Expect periodic disconnections
   - Don't assume continuous connectivity
   - Store state to resume after reconnection

3. **Monitor connection health**
   - Track message rates
   - Monitor latency
   - Alert on connection issues

4. **Rate limiting**
   - Respect exchange rate limits
   - Use combined streams when possible
   - Implement backoff strategies

5. **Error handling**
   - Log all errors
   - Implement fallback strategies
   - Don't crash on parsing errors

6. **Resource management**
   - Close connections properly
   - Cancel async tasks
   - Clean up resources on shutdown

7. **Security**
   - Don't log sensitive data
   - Use environment variables for credentials
   - Validate all incoming data

## Troubleshooting

### Connection keeps disconnecting

- Check internet connection
- Verify API credentials (if required)
- Check exchange status page
- Review rate limits
- Increase reconnect_delay

### High latency

- Check network connection
- Try different exchange endpoint
- Use geographically closer servers
- Reduce message volume

### Missing messages

- Enable message acknowledgment
- Check buffer sizes
- Monitor for backpressure
- Verify stream subscriptions

### Memory leaks

- Clear old data from caches
- Limit buffer sizes
- Use bounded collections
- Monitor memory usage

## Production Deployment

### Checklist

- [ ] Switch from testnet to production endpoints
- [ ] Use production API keys (securely stored)
- [ ] Enable monitoring and alerting
- [ ] Set up logging aggregation
- [ ] Configure automatic restarts
- [ ] Test failover scenarios
- [ ] Document runbooks
- [ ] Set up health checks
- [ ] Configure rate limiting
- [ ] Enable security features

### Monitoring Endpoints

```python
# Health check endpoint for load balancer
async def health_check():
    if ws.manager.state == ConnectionState.CONNECTED:
        return {"status": "healthy", "latency": ws.get_latency()}
    return {"status": "unhealthy"}

# Metrics endpoint for Prometheus
async def metrics():
    return {
        "websocket_messages_total": ws.manager.message_count,
        "websocket_errors_total": ws.manager.error_count,
        "websocket_latency_ms": ws.get_latency(),
        "websocket_state": ws.manager.state.value
    }
```

## Summary

The WebSocket infrastructure provides:
- Robust real-time market data feeds
- Automatic reconnection and error recovery
- Multi-exchange support with unified interface
- Low latency and high throughput
- Production-ready reliability
- Comprehensive monitoring and debugging

This enables the autonomous trading agent to:
- React to market changes in real-time
- Make informed trading decisions
- Operate with live market data
- Maintain high performance
- Handle exchange outages gracefully
