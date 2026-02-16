#!/usr/bin/env python3
"""
Autonomous Trading Agent with Live Binance Trading

This example shows how to connect the autonomous agent to live Binance trading.

⚠️ WARNING: This uses REAL MONEY if testnet=False
Start with testnet=True and small amounts!
"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autonomous_agent.trading_agent import AutonomousTradingAgent, AgentConfig
from exchanges.binance_trading_client import BinanceTradingClient, OrderSide
from typing import Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceIntegratedAgent(AutonomousTradingAgent):
    """
    Enhanced autonomous agent integrated with Binance exchange.
    """

    def __init__(self, config: AgentConfig, use_testnet: bool = True):
        super().__init__(config)
        self.use_testnet = use_testnet
        self.binance = None
        self.initial_balances = {}

    async def start(self):
        """Start agent with Binance connection."""
        logger.info("Starting Binance-integrated trading agent...")

        # Connect to Binance
        await self._connect_binance()

        # Start normal agent operation
        await super().start()

    async def _connect_binance(self):
        """Connect to Binance exchange."""
        logger.info("Connecting to Binance...")

        # Initialize Binance client
        self.binance = BinanceTradingClient(testnet=self.use_testnet)

        # Test connectivity
        if not self.binance.test_connectivity():
            raise ConnectionError("Failed to connect to Binance")

        logger.info(f"✅ Connected to Binance ({'TESTNET' if self.use_testnet else 'MAINNET'})")

        # Get initial balances
        balances = self.binance.get_balances()
        for balance in balances:
            self.initial_balances[balance['asset']] = balance['total']
            logger.info(f"Initial {balance['asset']}: {balance['total']:.8f}")

        # Verify we have trading balance
        usdt_balance = self.binance.get_balance('USDT')
        if usdt_balance['free'] < 10:
            logger.warning(f"Low USDT balance: ${usdt_balance['free']:.2f}")
            if not self.use_testnet:
                raise ValueError("Insufficient USDT balance for live trading")

    async def _get_market_data(self) -> Dict:
        """
        Get real-time market data from Binance.

        Returns:
            Dictionary of current prices
        """
        symbols = ['BTCUSDT', 'ETHUSDT']
        market_data = {}

        for symbol in symbols:
            try:
                price = self.binance.get_ticker_price(symbol)
                market_data[symbol] = price
                logger.debug(f"{symbol}: ${price:,.2f}")
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")

        return market_data

    async def _execute_trade(self, signal: Dict) -> Dict:
        """
        Execute trade on Binance.

        Args:
            signal: Trading signal

        Returns:
            Trade result
        """
        symbol = signal['symbol']
        action = signal['action']
        size = signal.get('size', 0.0)

        # Convert symbol format (BTC -> BTCUSDT)
        if not symbol.endswith('USDT'):
            symbol = f"{symbol}USDT"

        # Validate signal
        if action not in ['BUY', 'SELL']:
            logger.error(f"Invalid action: {action}")
            return {'success': False, 'error': 'Invalid action'}

        # Get current price
        try:
            current_price = self.binance.get_ticker_price(symbol)
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return {'success': False, 'error': str(e)}

        # Calculate quantity if size is in USD
        if size < 0.001:  # Size is likely in BTC
            quantity = size
        else:  # Size is in USD, convert to BTC
            quantity = size / current_price

        # Round to valid precision
        quantity = round(quantity, 6)

        # Validate minimum order size
        if quantity * current_price < 10:
            logger.warning(f"Order size too small: ${quantity * current_price:.2f}")
            return {'success': False, 'error': 'Order size too small ($10 min)'}

        logger.info(f"Executing {action} {quantity:.6f} {symbol} @ ${current_price:,.2f}")

        try:
            # Place order
            side = OrderSide.BUY if action == 'BUY' else OrderSide.SELL
            order = self.binance.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity
            )

            # Update portfolio
            self._update_portfolio_after_trade(symbol, action, quantity, current_price)

            return {
                'success': True,
                'order_id': order['orderId'],
                'symbol': symbol,
                'side': action,
                'quantity': quantity,
                'price': current_price,
                'value': quantity * current_price,
                'status': order['status']
            }

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {'success': False, 'error': str(e)}

    def _update_portfolio_after_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float
    ):
        """Update portfolio state after trade execution."""
        value = quantity * price

        if action == 'BUY':
            # Add position
            if symbol in self.positions:
                # Average in
                old_qty = self.positions[symbol]['quantity']
                old_price = self.positions[symbol]['avg_price']
                new_qty = old_qty + quantity
                new_price = ((old_qty * old_price) + (quantity * price)) / new_qty

                self.positions[symbol]['quantity'] = new_qty
                self.positions[symbol]['avg_price'] = new_price
            else:
                # New position
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'current_price': price
                }

            # Reduce cash
            self.portfolio_value -= value

        else:  # SELL
            # Reduce position
            if symbol in self.positions:
                old_qty = self.positions[symbol]['quantity']
                new_qty = old_qty - quantity

                if new_qty <= 0:
                    # Close position
                    del self.positions[symbol]
                else:
                    self.positions[symbol]['quantity'] = new_qty

            # Add cash
            self.portfolio_value += value

    async def get_live_performance(self) -> Dict:
        """Get live performance metrics from Binance."""
        try:
            # Get current balances
            current_balances = {}
            balances = self.binance.get_balances()

            for balance in balances:
                current_balances[balance['asset']] = balance['total']

            # Calculate P&L
            total_pnl = 0.0
            for asset, initial in self.initial_balances.items():
                current = current_balances.get(asset, 0.0)
                change = current - initial

                if asset == 'USDT':
                    total_pnl += change
                else:
                    # Convert to USDT value
                    try:
                        price = self.binance.get_ticker_price(f"{asset}USDT")
                        total_pnl += change * price
                    except:
                        pass

            # Get order history
            orders = []
            for symbol in ['BTCUSDT', 'ETHUSDT']:
                try:
                    symbol_orders = self.binance.get_my_trades(symbol, limit=10)
                    orders.extend(symbol_orders)
                except:
                    pass

            return {
                'total_pnl': total_pnl,
                'current_balances': current_balances,
                'initial_balances': self.initial_balances,
                'total_orders': len(orders),
                'portfolio_value': self.portfolio_value
            }

        except Exception as e:
            logger.error(f"Failed to get live performance: {e}")
            return {}

    async def stop(self):
        """Stop agent and show final stats."""
        logger.info("Stopping Binance-integrated agent...")

        # Get final performance
        if self.binance:
            performance = await self.get_live_performance()
            logger.info(f"Final P&L: ${performance.get('total_pnl', 0):.2f}")

        # Cancel any open orders
        if self.binance:
            for symbol in ['BTCUSDT', 'ETHUSDT']:
                try:
                    self.binance.cancel_all_orders(symbol)
                except:
                    pass

        await super().stop()


async def main():
    """Main entry point."""
    print("=" * 70)
    print("AUTONOMOUS TRADING AGENT WITH LIVE BINANCE")
    print("=" * 70)

    # Configuration
    USE_TESTNET = True  # ALWAYS start with True!
    INITIAL_CAPITAL = 100.0  # Start small

    print(f"\n⚠️  Trading Mode: {'TESTNET (Safe)' if USE_TESTNET else 'MAINNET (REAL MONEY)'}")
    print(f"💰 Initial Capital: ${INITIAL_CAPITAL}")

    if not USE_TESTNET:
        print("\n" + "!" * 70)
        print("WARNING: MAINNET MODE - YOU WILL USE REAL MONEY!")
        print("!" * 70)
        response = input("\nType 'I UNDERSTAND THE RISKS' to continue: ")
        if response != "I UNDERSTAND THE RISKS":
            print("Exiting for safety.")
            return

    # Agent configuration
    config = AgentConfig(
        initial_capital=INITIAL_CAPITAL,
        paper_trading=False,  # Use real (or testnet) orders
        check_interval_seconds=60,
        max_daily_loss=INITIAL_CAPITAL * 0.05,  # 5% max daily loss
        max_position_size=0.2,  # 20% max per position
        enabled_strategies=['dca_bot'],
        send_alerts=True
    )

    # Create agent
    agent = BinanceIntegratedAgent(config, use_testnet=USE_TESTNET)

    try:
        print(f"\n🤖 Starting agent...")
        print("   Press Ctrl+C to stop\n")

        # Run for 5 minutes as demo
        await asyncio.wait_for(agent.start(), timeout=300.0)

    except asyncio.TimeoutError:
        print("\n⏱️  Demo timeout reached")
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        await agent.stop()

    # Show final statistics
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)

    performance = await agent.get_live_performance()

    print(f"\nTotal P&L: ${performance.get('total_pnl', 0):+,.2f}")
    print(f"Total Orders: {performance.get('total_orders', 0)}")

    print("\nFinal Balances:")
    for asset, amount in performance.get('current_balances', {}).items():
        initial = agent.initial_balances.get(asset, 0)
        change = amount - initial
        print(f"  {asset}: {amount:.8f} ({change:+.8f})")

    print("\n✅ Agent stopped safely")


if __name__ == '__main__':
    asyncio.run(main())
