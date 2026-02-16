#!/usr/bin/env python3
"""
Autonomous Trading Agent with Real-Time WebSocket Data

This example shows how to connect the autonomous trading agent
to live market data via WebSockets instead of using simulated data.
"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autonomous_agent.trading_agent import AutonomousTradingAgent, AgentConfig
from realtime import (
    BinanceWebSocket,
    BinanceConfig,
    BinanceStream,
    MarketDataAggregator,
    AggregatorConfig
)
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeAgent(AutonomousTradingAgent):
    """
    Enhanced autonomous agent using real-time WebSocket market data.
    """

    def __init__(self, config: AgentConfig, use_testnet: bool = True):
        super().__init__(config)
        self.use_testnet = use_testnet
        self.websocket = None
        self.market_data = {}
        self.last_prices = {}

    async def start(self):
        """Start agent with real-time data feeds."""
        logger.info("Starting real-time trading agent...")

        # Set up WebSocket connections
        await self._setup_websockets()

        # Start normal agent operation
        await super().start()

    async def _setup_websockets(self):
        """Set up WebSocket connections to exchanges."""
        logger.info("Setting up WebSocket connections...")

        # Configure Binance WebSocket
        binance_config = BinanceConfig(
            symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
            streams=[
                BinanceStream.TRADE,
                BinanceStream.TICKER,
                BinanceStream.BOOK_TICKER
            ],
            testnet=self.use_testnet,
            combined=True
        )

        self.websocket = BinanceWebSocket(binance_config)

        # Register handlers for market data
        @self.websocket.on_trade
        async def handle_trade(data):
            await self._process_trade_data(data)

        @self.websocket.on_ticker
        async def handle_ticker(data):
            await self._process_ticker_data(data)

        @self.websocket.on_book_ticker
        async def handle_book_ticker(data):
            await self._process_book_data(data)

        # Connect
        await self.websocket.connect()
        logger.info("WebSocket connected and streaming market data")

    async def _process_trade_data(self, data: Dict):
        """Process trade data from WebSocket."""
        symbol = data['symbol']
        price = float(data['price'])
        quantity = float(data['quantity'])

        # Update market data cache
        if symbol not in self.market_data:
            self.market_data[symbol] = {
                'last_price': price,
                'volume': 0.0,
                'trades': []
            }

        self.market_data[symbol]['last_price'] = price
        self.market_data[symbol]['volume'] += quantity
        self.market_data[symbol]['trades'].append({
            'price': price,
            'quantity': quantity,
            'timestamp': data['timestamp']
        })

        # Keep only last 100 trades
        if len(self.market_data[symbol]['trades']) > 100:
            self.market_data[symbol]['trades'] = \
                self.market_data[symbol]['trades'][-100:]

        # Update last prices for agent
        self.last_prices[symbol] = price

    async def _process_ticker_data(self, data: Dict):
        """Process ticker data from WebSocket."""
        symbol = data['symbol']

        if symbol not in self.market_data:
            self.market_data[symbol] = {}

        self.market_data[symbol].update({
            'price_change': float(data.get('price_change', 0)),
            'price_change_percent': float(data.get('price_change_percent', 0)),
            'high_24h': float(data.get('high', 0)),
            'low_24h': float(data.get('low', 0)),
            'volume_24h': float(data.get('volume', 0))
        })

    async def _process_book_data(self, data: Dict):
        """Process order book data from WebSocket."""
        symbol = data['symbol']

        if symbol not in self.market_data:
            self.market_data[symbol] = {}

        self.market_data[symbol].update({
            'best_bid': float(data['bid']),
            'best_ask': float(data['ask']),
            'spread': float(data['ask']) - float(data['bid'])
        })

    async def _get_market_data(self) -> Dict:
        """
        Get current market data (override parent method).

        Returns:
            Dictionary of current prices and market data
        """
        # Return real-time data instead of simulated
        return self.last_prices

    def get_market_stats(self, symbol: str) -> Dict:
        """
        Get detailed market statistics for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Market statistics
        """
        if symbol not in self.market_data:
            return {}

        data = self.market_data[symbol]
        return {
            'last_price': data.get('last_price', 0),
            'volume_24h': data.get('volume_24h', 0),
            'price_change_24h': data.get('price_change_percent', 0),
            'high_24h': data.get('high_24h', 0),
            'low_24h': data.get('low_24h', 0),
            'best_bid': data.get('best_bid', 0),
            'best_ask': data.get('best_ask', 0),
            'spread': data.get('spread', 0),
            'recent_trades': len(data.get('trades', []))
        }

    async def stop(self):
        """Stop agent and close WebSocket connections."""
        logger.info("Stopping real-time agent...")

        # Disconnect WebSocket
        if self.websocket:
            await self.websocket.disconnect()
            logger.info("WebSocket disconnected")

        # Stop agent
        await super().stop()


async def main():
    """Main entry point."""
    print("=" * 70)
    print("AUTONOMOUS TRADING AGENT WITH REAL-TIME DATA")
    print("=" * 70)

    # Agent configuration
    config = AgentConfig(
        initial_capital=10000.0,
        paper_trading=True,
        check_interval_seconds=5,
        max_daily_loss=500.0,
        send_alerts=False,
        enabled_strategies=['dca_bot', 'market_making']
    )

    # Create agent with real-time data
    agent = RealTimeAgent(config, use_testnet=True)

    try:
        print("\n🤖 Starting agent with live WebSocket data...")
        print("   Press Ctrl+C to stop\n")

        # Run for 120 seconds as demo
        await asyncio.wait_for(agent.start(), timeout=120.0)

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

    # Show market data stats
    print("\n" + "-" * 70)
    print("MARKET DATA RECEIVED")
    print("-" * 70)

    for symbol in agent.market_data.keys():
        stats = agent.get_market_stats(symbol)
        if stats:
            print(f"\n{symbol}:")
            print(f"  Last Price: ${stats['last_price']:,.2f}")
            print(f"  24h Volume: ${stats.get('volume_24h', 0):,.0f}")
            print(f"  24h Change: {stats.get('price_change_24h', 0):+.2f}%")
            if stats.get('best_bid'):
                print(f"  Best Bid: ${stats['best_bid']:,.2f}")
                print(f"  Best Ask: ${stats['best_ask']:,.2f}")
                spread_bps = (stats['spread'] / stats['best_bid']) * 10000
                print(f"  Spread: {spread_bps:.2f} bps")
            print(f"  Recent Trades Cached: {stats['recent_trades']}")

    print("\n✅ Agent stopped - all real-time data feeds closed")


if __name__ == '__main__':
    asyncio.run(main())
