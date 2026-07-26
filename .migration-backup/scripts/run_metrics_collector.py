#!/usr/bin/env python3
"""
Metrics Collector Service
Continuously collects trading metrics and exports to Prometheus.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import logging
from datetime import datetime

from infrastructure.monitoring.exporters.prometheus_exporter import TradingMetricsExporter
from src.data_collection.coinbase_collector import CoinbaseCollector
from src.data_collection.uniswap_collector import UniswapCollector, WETH_ADDRESS, USDC_ADDRESS
from src.data_collection.gas_tracker import GasTracker
from src.onchain.dex_analyzer import DEXAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects metrics from various sources and exports to Prometheus.
    """

    def __init__(
        self,
        collection_interval: int = 30,  # seconds
        prometheus_port: int = 8001
    ):
        """
        Initialize metrics collector.

        Args:
            collection_interval: How often to collect metrics (seconds)
            prometheus_port: Port for Prometheus exporter
        """
        self.collection_interval = collection_interval

        # Initialize exporter
        self.exporter = TradingMetricsExporter(port=prometheus_port)

        # Initialize collectors
        self.coinbase = CoinbaseCollector()
        self.uniswap = UniswapCollector()
        self.gas_tracker = GasTracker()
        self.analyzer = DEXAnalyzer(min_profit_usd=10.0)

        logger.info(f"Metrics collector initialized (interval={collection_interval}s)")

    def collect_gas_metrics(self):
        """Collect and export gas price metrics."""
        try:
            gas = self.gas_tracker.get_current_gas_price()

            if gas:
                self.exporter.update_gas_prices({
                    'slow': gas.slow,
                    'standard': gas.standard,
                    'fast': gas.fast,
                    'instant': gas.instant,
                    'base_fee': gas.base_fee
                })
                logger.debug(f"Gas prices updated: {gas.standard:.1f} Gwei")
            else:
                self.exporter.record_error('gas_tracker')

        except Exception as e:
            logger.error(f"Failed to collect gas metrics: {e}")
            self.exporter.record_error('gas_tracker')

    def collect_price_metrics(self):
        """Collect and export price metrics from Coinbase."""
        try:
            symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']

            for symbol in symbols:
                try:
                    ticker = self.coinbase.get_ticker(symbol)

                    if ticker and 'price' in ticker:
                        token = symbol.replace('-USD', '')
                        price = float(ticker['price'])

                        self.exporter.update_token_price(
                            token=token,
                            exchange='coinbase',
                            price_usd=price
                        )
                        logger.debug(f"{token} price: ${price:,.2f}")

                        self.exporter.record_api_request('coinbase', f'/products/{symbol}/ticker')

                except Exception as e:
                    logger.error(f"Failed to get {symbol} price: {e}")
                    self.exporter.record_error('coinbase')

        except Exception as e:
            logger.error(f"Failed to collect price metrics: {e}")

    def collect_dex_metrics(self):
        """Collect and export DEX pool metrics."""
        try:
            # Get WETH/USDC pool (most liquid)
            pool = self.uniswap.get_pool_info(WETH_ADDRESS, USDC_ADDRESS)

            if pool:
                self.exporter.update_dex_pool(
                    pool_address=pool.address,
                    token0=pool.token0_symbol,
                    token1=pool.token1_symbol,
                    reserve0=pool.reserve0,
                    reserve1=pool.reserve1,
                    price=pool.price,
                    liquidity_usd=pool.liquidity_usd
                )
                logger.debug(f"DEX pool updated: {pool.token0_symbol}/{pool.token1_symbol}")

                # Also update DEX price
                self.exporter.update_token_price(
                    token='ETH',
                    exchange='uniswap',
                    price_usd=pool.price
                )

            else:
                self.exporter.record_error('uniswap')

        except Exception as e:
            logger.error(f"Failed to collect DEX metrics: {e}")
            self.exporter.record_error('uniswap')

    def collect_arbitrage_metrics(self):
        """Collect and export arbitrage opportunity metrics."""
        try:
            # Clear old opportunities
            self.exporter.clear_arbitrage_opportunities()

            # Get current prices
            coinbase_ticker = self.coinbase.get_ticker('ETH-USD')
            uniswap_pool = self.uniswap.get_pool_info(WETH_ADDRESS, USDC_ADDRESS)
            gas_price = self.gas_tracker.get_current_gas_price()

            if not all([coinbase_ticker, uniswap_pool, gas_price]):
                return

            cb_price = float(coinbase_ticker['price'])
            uni_price = uniswap_pool.price

            # Estimate gas cost
            gas_cost = self.gas_tracker.estimate_transaction_cost(
                'uniswap_v2_swap',
                gas_price_gwei=gas_price.standard,
                eth_price_usd=cb_price
            )

            # Find CEX-DEX arbitrage
            opp = self.analyzer.find_cex_dex_arbitrage(
                cex_price=cb_price,
                dex_price=uni_price,
                token='ETH',
                trade_size_usd=5000,
                gas_cost_usd=gas_cost.cost_usd
            )

            if opp and opp.is_profitable():
                self.exporter.update_arbitrage_opportunity(
                    opp_type='cex_dex',
                    token=opp.token,
                    buy_exchange=opp.buy_exchange,
                    sell_exchange=opp.sell_exchange,
                    profit_usd=opp.net_profit,
                    roi_percent=opp.roi_percent
                )
                self.exporter.increment_arbitrage_count('cex_dex')

                logger.info(f"🚨 Arbitrage opportunity: ${opp.net_profit:.2f} profit "
                           f"({opp.roi_percent:.2f}% ROI)")

        except Exception as e:
            logger.error(f"Failed to collect arbitrage metrics: {e}")

    def run(self):
        """Run the metrics collection loop."""
        logger.info("Starting metrics collector...")

        # Start Prometheus exporter
        self.exporter.start()

        logger.info(f"✓ Collecting metrics every {self.collection_interval}s")
        logger.info(f"✓ Prometheus metrics: http://localhost:{self.exporter.port}/metrics")

        iteration = 0

        try:
            while True:
                iteration += 1
                start_time = time.time()

                logger.info(f"\n{'='*60}")
                logger.info(f"Collection iteration #{iteration}")
                logger.info(f"{'='*60}")

                # Collect all metrics
                self.collect_gas_metrics()
                self.collect_price_metrics()

                # DEX metrics (less frequent - every 5 iterations)
                if iteration % 5 == 0:
                    self.collect_dex_metrics()

                # Arbitrage opportunities (less frequent - every 3 iterations)
                if iteration % 3 == 0:
                    self.collect_arbitrage_metrics()

                elapsed = time.time() - start_time
                logger.info(f"Collection completed in {elapsed:.2f}s")

                # Sleep until next collection
                sleep_time = max(0, self.collection_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("\n\nMetrics collector stopped by user")
        except Exception as e:
            logger.error(f"Metrics collector crashed: {e}", exc_info=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Trading Metrics Collector')
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Collection interval in seconds (default: 30)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8001,
        help='Prometheus exporter port (default: 8001)'
    )

    args = parser.parse_args()

    print("📊 Trading Metrics Collector")
    print("=" * 60)
    print(f"Interval: {args.interval}s")
    print(f"Prometheus: http://localhost:{args.port}/metrics")
    print("=" * 60)

    collector = MetricsCollector(
        collection_interval=args.interval,
        prometheus_port=args.port
    )

    collector.run()
