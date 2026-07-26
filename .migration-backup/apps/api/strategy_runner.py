"""
Strategy Execution Engine
Runs enabled strategies and executes trades on Alpaca
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class StrategyRunner:
    """Main strategy execution engine"""

    def __init__(self, trading_client, data_client, strategy_states: Dict[str, bool]):
        self.trading_client = trading_client
        self.data_client = data_client
        self.strategy_states = strategy_states
        self.strategies = {}
        self.running = False

        # Performance tracking
        self.performance = {}
        for strategy_id in strategy_states.keys():
            self.performance[strategy_id] = {
                "pnl": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0
            }

        logger.info("StrategyRunner initialized")

    def register_strategy(self, strategy_id: str, strategy_instance):
        """Register a strategy instance"""
        self.strategies[strategy_id] = strategy_instance
        logger.info(f"Registered strategy: {strategy_id}")

    async def start(self):
        """Start the execution loop"""
        self.running = True
        logger.info("🚀 StrategyRunner started - executing strategies every 60 seconds")

        while self.running:
            try:
                await self._execute_cycle()
            except Exception as e:
                logger.error(f"Error in execution cycle: {e}", exc_info=True)

            # Wait 60 seconds before next cycle
            await asyncio.sleep(60)

    async def stop(self):
        """Stop the execution loop"""
        self.running = False
        logger.info("StrategyRunner stopped")

    async def _execute_cycle(self):
        """Execute one cycle of all enabled strategies"""
        logger.info("--- Execution Cycle Started ---")

        if not self.trading_client:
            logger.warning("Trading client not available - skipping execution")
            return

        # Get enabled strategies
        enabled_strategies = [
            (sid, self.strategies[sid])
            for sid, enabled in self.strategy_states.items()
            if enabled and sid in self.strategies
        ]

        if not enabled_strategies:
            logger.info("No enabled strategies to execute")
            return

        logger.info(f"Executing {len(enabled_strategies)} enabled strategies")

        # Execute each enabled strategy
        for strategy_id, strategy in enabled_strategies:
            try:
                await self._execute_strategy(strategy_id, strategy)
            except Exception as e:
                logger.error(f"Error executing strategy {strategy_id}: {e}", exc_info=True)

        logger.info("--- Execution Cycle Complete ---")

    async def _execute_strategy(self, strategy_id: str, strategy):
        """Execute a single strategy"""
        logger.info(f"Executing strategy: {strategy_id}")

        try:
            # Get market data
            market_data = await self._get_market_data(strategy.symbols)

            # Generate signal
            signal = strategy.generate_signal(market_data)

            if signal == Signal.HOLD:
                logger.info(f"{strategy_id}: HOLD signal - no action")
                return

            # Execute trade
            if signal in [Signal.BUY, Signal.SELL]:
                await self._execute_trade(strategy_id, strategy, signal, market_data)

        except Exception as e:
            logger.error(f"Error in strategy {strategy_id}: {e}", exc_info=True)

    async def _get_market_data(self, symbols: List[str]) -> Dict:
        """Get current market data for symbols"""
        try:
            # Get latest bars for symbols
            # For now, return mock data (will be replaced with real Alpaca data)
            market_data = {}
            for symbol in symbols:
                market_data[symbol] = {
                    "price": 100.0,  # Placeholder
                    "volume": 1000000,
                    "timestamp": datetime.now()
                }
            return market_data
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}

    async def _execute_trade(self, strategy_id: str, strategy, signal: Signal, market_data: Dict):
        """Execute a trade for a strategy"""
        try:
            symbol = strategy.symbols[0]  # Primary symbol
            price = market_data[symbol]["price"]

            # Calculate position size (simple for now)
            account = self.trading_client.get_account()
            buying_power = float(account.buying_power)

            # Risk management: max 20% of buying power per trade
            max_position_size = buying_power * 0.20
            quantity = int(max_position_size / price)

            if quantity < 1:
                logger.warning(f"{strategy_id}: Insufficient buying power for trade")
                return

            # Submit order
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if signal == Signal.BUY else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )

            # Submit order to Alpaca
            order = self.trading_client.submit_order(order_data)

            logger.info(f"✅ {strategy_id}: {signal.value} {quantity} {symbol} @ ${price:.2f}")
            logger.info(f"Order ID: {order.id}")

            # Update performance metrics
            self.performance[strategy_id]["trades"] += 1

            # Broadcast trade event (will be implemented with WebSocket)
            trade_event = {
                "strategy_id": strategy_id,
                "signal": signal.value,
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "timestamp": datetime.now().isoformat(),
                "order_id": str(order.id)
            }
            logger.info(f"Trade event: {trade_event}")

        except Exception as e:
            logger.error(f"Error executing trade for {strategy_id}: {e}", exc_info=True)

    def get_performance(self, strategy_id: str) -> Dict:
        """Get performance metrics for a strategy"""
        return self.performance.get(strategy_id, {})

    def get_all_performance(self) -> Dict:
        """Get performance metrics for all strategies"""
        return self.performance
