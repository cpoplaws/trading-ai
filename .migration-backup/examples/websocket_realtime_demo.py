#!/usr/bin/env python3
"""
WebSocket Real-Time Data Demo

This example demonstrates the real-time WebSocket infrastructure:
- Connect to multiple exchanges simultaneously
- Subscribe to various data streams
- Aggregate data from different sources
- Monitor connection health
- Handle reconnections automatically
"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from realtime import (
    BinanceWebSocket,
    BinanceConfig,
    BinanceStream,
    CoinbaseWebSocket,
    CoinbaseConfig,
    CoinbaseChannel,
    MarketDataAggregator,
    AggregatorConfig
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealTimeDataDemo:
    """Demo showing real-time WebSocket data feeds."""

    def __init__(self):
        self.binance_ws = None
        self.coinbase_ws = None
        self.aggregator = None
        self.message_count = 0

    async def setup_binance(self):
        """Set up Binance WebSocket connection."""
        logger.info("Setting up Binance WebSocket...")

        config = BinanceConfig(
            symbols=['BTCUSDT', 'ETHUSDT'],
            streams=[
                BinanceStream.TRADE,
                BinanceStream.TICKER,
                BinanceStream.BOOK_TICKER
            ],
            testnet=True,  # Use testnet for demo
            combined=True
        )

        self.binance_ws = BinanceWebSocket(config)

        # Register handler for trade data
        @self.binance_ws.on_trade
        async def handle_trade(data):
            self.message_count += 1
            logger.info(
                f"Binance Trade: {data['symbol']} - "
                f"Price: ${data['price']:,.2f}, "
                f"Qty: {data['quantity']:.4f}"
            )

        # Register handler for ticker data
        @self.binance_ws.on_ticker
        async def handle_ticker(data):
            logger.info(
                f"Binance Ticker: {data['symbol']} - "
                f"Price: ${data['price']:,.2f}, "
                f"24h Change: {data['price_change_percent']:+.2f}%"
            )

        # Register handler for book ticker (best bid/ask)
        @self.binance_ws.on_book_ticker
        async def handle_book_ticker(data):
            spread = data['ask'] - data['bid']
            spread_bps = (spread / data['bid']) * 10000
            logger.info(
                f"Binance Book: {data['symbol']} - "
                f"Bid: ${data['bid']:,.2f}, Ask: ${data['ask']:,.2f}, "
                f"Spread: {spread_bps:.2f} bps"
            )

        await self.binance_ws.connect()
        logger.info("Binance WebSocket connected")

    async def setup_coinbase(self):
        """Set up Coinbase WebSocket connection."""
        logger.info("Setting up Coinbase WebSocket...")

        config = CoinbaseConfig(
            product_ids=['BTC-USD', 'ETH-USD'],
            channels=[
                CoinbaseChannel.TICKER,
                CoinbaseChannel.MATCHES,  # Trades
                CoinbaseChannel.HEARTBEAT
            ],
            sandbox=True  # Use sandbox for demo
        )

        self.coinbase_ws = CoinbaseWebSocket(config)

        # Register handler for ticker data
        @self.coinbase_ws.on_ticker
        async def handle_ticker(data):
            self.message_count += 1
            logger.info(
                f"Coinbase Ticker: {data['product_id']} - "
                f"Price: ${float(data['price']):,.2f}"
            )

        # Register handler for trades
        @self.coinbase_ws.on_matches
        async def handle_match(data):
            logger.info(
                f"Coinbase Trade: {data['product_id']} - "
                f"Price: ${float(data['price']):,.2f}, "
                f"Size: {float(data['size']):.4f}, "
                f"Side: {data['side']}"
            )

        # Register handler for heartbeat
        @self.coinbase_ws.on_heartbeat
        async def handle_heartbeat(data):
            logger.debug(f"Coinbase heartbeat: {data['last_trade_id']}")

        await self.coinbase_ws.connect()
        logger.info("Coinbase WebSocket connected")

    async def setup_aggregator(self):
        """Set up market data aggregator."""
        logger.info("Setting up market data aggregator...")

        config = AggregatorConfig(
            exchanges=['binance', 'coinbase'],
            symbols=['BTC', 'ETH'],
            update_interval_ms=100,
            enable_vwap=True,
            enable_book_aggregation=True
        )

        self.aggregator = MarketDataAggregator(config)

        # Add WebSocket clients to aggregator
        await self.aggregator.add_source('binance', self.binance_ws)
        await self.aggregator.add_source('coinbase', self.coinbase_ws)

        # Register handler for aggregated data
        @self.aggregator.on_update
        async def handle_aggregated_data(data):
            logger.info(
                f"Aggregated: {data.symbol} - "
                f"Best Bid: ${data.best_bid:,.2f} ({data.best_bid_exchange}), "
                f"Best Ask: ${data.best_ask:,.2f} ({data.best_ask_exchange}), "
                f"VWAP: ${data.vwap:,.2f}"
            )

        await self.aggregator.start()
        logger.info("Market data aggregator started")

    async def monitor_connections(self):
        """Monitor WebSocket connection health."""
        while True:
            await asyncio.sleep(10)

            # Check Binance connection
            if self.binance_ws:
                state = self.binance_ws.manager.state
                latency = self.binance_ws.get_latency()
                logger.info(
                    f"Binance: State={state.value}, "
                    f"Messages={self.binance_ws.manager.message_count}, "
                    f"Latency={latency:.2f}ms"
                )

            # Check Coinbase connection
            if self.coinbase_ws:
                state = self.coinbase_ws.manager.state
                latency = self.coinbase_ws.get_latency()
                logger.info(
                    f"Coinbase: State={state.value}, "
                    f"Messages={self.coinbase_ws.manager.message_count}, "
                    f"Latency={latency:.2f}ms"
                )

            # Show aggregated stats
            if self.aggregator:
                stats = self.aggregator.get_stats()
                logger.info(
                    f"Aggregator: Total messages={stats['total_messages']}, "
                    f"Updates/sec={stats['updates_per_second']:.1f}"
                )

            logger.info(f"Demo message count: {self.message_count}")

    async def run(self, duration_seconds: int = 60):
        """
        Run the demo.

        Args:
            duration_seconds: How long to run the demo
        """
        try:
            logger.info("=" * 70)
            logger.info("WEBSOCKET REAL-TIME DATA DEMO")
            logger.info("=" * 70)

            # Set up connections
            await self.setup_binance()
            await self.setup_coinbase()
            await self.setup_aggregator()

            logger.info("\nAll connections established!")
            logger.info(f"Running for {duration_seconds} seconds...")
            logger.info("Press Ctrl+C to stop early\n")

            # Start monitoring task
            monitor_task = asyncio.create_task(self.monitor_connections())

            # Run for specified duration
            await asyncio.sleep(duration_seconds)

            # Cleanup
            logger.info("\nDemo complete! Cleaning up...")
            monitor_task.cancel()

            if self.aggregator:
                await self.aggregator.stop()
            if self.binance_ws:
                await self.binance_ws.disconnect()
            if self.coinbase_ws:
                await self.coinbase_ws.disconnect()

            logger.info(f"Total messages received: {self.message_count}")

        except KeyboardInterrupt:
            logger.info("\nStopped by user")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
        finally:
            logger.info("Demo finished")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='WebSocket Real-Time Data Demo'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--binance-only',
        action='store_true',
        help='Connect to Binance only'
    )
    parser.add_argument(
        '--coinbase-only',
        action='store_true',
        help='Connect to Coinbase only'
    )

    args = parser.parse_args()

    demo = RealTimeDataDemo()

    # Run demo
    if args.binance_only:
        await demo.setup_binance()
        await demo.monitor_connections()
    elif args.coinbase_only:
        await demo.setup_coinbase()
        await demo.monitor_connections()
    else:
        await demo.run(duration_seconds=args.duration)


if __name__ == '__main__':
    asyncio.run(main())
