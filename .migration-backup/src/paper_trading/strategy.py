"""
Paper Trading Strategy Framework
Allows backtesting and running strategies with paper trading.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from .engine import PaperTradingEngine, Exchange, OrderSide
from .portfolio import PaperPortfolio

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trading signal."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class StrategySignal:
    """Strategy trading signal."""
    timestamp: datetime
    signal: Signal
    price: float
    quantity: float
    confidence: float = 1.0
    metadata: Dict = None


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.

    Implement the `generate_signal` method to create custom strategies.
    """

    def __init__(self, name: str):
        """
        Initialize strategy.

        Args:
            name: Strategy name
        """
        self.name = name
        self.candles: List[Candle] = []

    @abstractmethod
    def generate_signal(self, candles: List[Candle], current_position: float) -> StrategySignal:
        """
        Generate trading signal based on market data.

        Args:
            candles: Historical candle data
            current_position: Current token position

        Returns:
            Trading signal
        """
        pass


class SimpleMovingAverageStrategy(BaseStrategy):
    """
    Simple Moving Average crossover strategy.

    - Buy when price crosses above MA
    - Sell when price crosses below MA
    """

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 30,
        position_size: float = 1.0
    ):
        """
        Initialize MA strategy.

        Args:
            short_window: Short MA period
            long_window: Long MA period
            position_size: Position size per trade
        """
        super().__init__(f"SMA_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size

    def generate_signal(self, candles: List[Candle], current_position: float) -> StrategySignal:
        """Generate signal based on MA crossover."""
        if len(candles) < self.long_window:
            return StrategySignal(
                timestamp=candles[-1].timestamp,
                signal=Signal.HOLD,
                price=candles[-1].close,
                quantity=0.0
            )

        # Calculate moving averages
        closes = [c.close for c in candles]
        short_ma = sum(closes[-self.short_window:]) / self.short_window
        long_ma = sum(closes[-self.long_window:]) / self.long_window

        current_price = candles[-1].close
        timestamp = candles[-1].timestamp

        # Generate signal
        if short_ma > long_ma and current_position == 0:
            # Bullish crossover - buy
            return StrategySignal(
                timestamp=timestamp,
                signal=Signal.BUY,
                price=current_price,
                quantity=self.position_size,
                confidence=abs(short_ma - long_ma) / long_ma,
                metadata={'short_ma': short_ma, 'long_ma': long_ma}
            )
        elif short_ma < long_ma and current_position > 0:
            # Bearish crossover - sell
            return StrategySignal(
                timestamp=timestamp,
                signal=Signal.SELL,
                price=current_price,
                quantity=current_position,
                confidence=abs(short_ma - long_ma) / long_ma,
                metadata={'short_ma': short_ma, 'long_ma': long_ma}
            )
        else:
            # Hold
            return StrategySignal(
                timestamp=timestamp,
                signal=Signal.HOLD,
                price=current_price,
                quantity=0.0,
                metadata={'short_ma': short_ma, 'long_ma': long_ma}
            )


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy based on price changes.

    - Buy on strong upward momentum
    - Sell on strong downward momentum or profit target
    """

    def __init__(
        self,
        lookback_period: int = 14,
        momentum_threshold: float = 0.02,  # 2% move
        position_size: float = 1.0,
        profit_target: float = 0.05  # 5% profit target
    ):
        """
        Initialize momentum strategy.

        Args:
            lookback_period: Period to measure momentum
            momentum_threshold: Threshold for entry (percentage)
            position_size: Position size per trade
            profit_target: Profit target for exit (percentage)
        """
        super().__init__(f"Momentum_{lookback_period}")
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
        self.position_size = position_size
        self.profit_target = profit_target
        self.entry_price = None

    def generate_signal(self, candles: List[Candle], current_position: float) -> StrategySignal:
        """Generate signal based on momentum."""
        if len(candles) < self.lookback_period + 1:
            return StrategySignal(
                timestamp=candles[-1].timestamp,
                signal=Signal.HOLD,
                price=candles[-1].close,
                quantity=0.0
            )

        current_price = candles[-1].close
        past_price = candles[-self.lookback_period].close
        momentum = (current_price - past_price) / past_price

        timestamp = candles[-1].timestamp

        # Check profit target if in position
        if current_position > 0 and self.entry_price:
            profit = (current_price - self.entry_price) / self.entry_price
            if profit >= self.profit_target:
                logger.info(f"Profit target reached: {profit*100:.2f}%")
                self.entry_price = None
                return StrategySignal(
                    timestamp=timestamp,
                    signal=Signal.SELL,
                    price=current_price,
                    quantity=current_position,
                    confidence=1.0,
                    metadata={'momentum': momentum, 'profit': profit}
                )

        # Generate entry/exit signals
        if momentum > self.momentum_threshold and current_position == 0:
            # Strong upward momentum - buy
            self.entry_price = current_price
            return StrategySignal(
                timestamp=timestamp,
                signal=Signal.BUY,
                price=current_price,
                quantity=self.position_size,
                confidence=min(momentum / self.momentum_threshold, 1.0),
                metadata={'momentum': momentum}
            )
        elif momentum < -self.momentum_threshold and current_position > 0:
            # Strong downward momentum - sell
            self.entry_price = None
            return StrategySignal(
                timestamp=timestamp,
                signal=Signal.SELL,
                price=current_price,
                quantity=current_position,
                confidence=min(abs(momentum) / self.momentum_threshold, 1.0),
                metadata={'momentum': momentum}
            )
        else:
            return StrategySignal(
                timestamp=timestamp,
                signal=Signal.HOLD,
                price=current_price,
                quantity=0.0,
                metadata={'momentum': momentum}
            )


class StrategyRunner:
    """
    Runs trading strategies with paper trading engine.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        portfolio: PaperPortfolio,
        engine: PaperTradingEngine,
        exchange: Exchange = Exchange.COINBASE,
        symbol: str = "ETH-USD"
    ):
        """
        Initialize strategy runner.

        Args:
            strategy: Trading strategy
            portfolio: Paper trading portfolio
            engine: Paper trading engine
            exchange: Exchange to trade on
            symbol: Trading pair
        """
        self.strategy = strategy
        self.portfolio = portfolio
        self.engine = engine
        self.exchange = exchange
        self.symbol = symbol

        # Extract token symbol
        self.token = symbol.split('-')[0]

        # Tracking
        self.signals: List[StrategySignal] = []
        self.trades_executed = 0

        logger.info(f"Strategy runner initialized: {strategy.name} on {exchange.value}")

    def execute_signal(self, signal: StrategySignal):
        """
        Execute a trading signal.

        Args:
            signal: Strategy signal
        """
        if signal.signal == Signal.HOLD:
            return

        # Check if we have sufficient balance
        if signal.signal == Signal.BUY:
            usd_needed = signal.quantity * signal.price * 1.01  # +1% buffer for fees/gas
            available_usd = self.portfolio.get_balance('USD')

            if available_usd < usd_needed:
                logger.warning(f"Insufficient USD: need ${usd_needed:.2f}, have ${available_usd:.2f}")
                return

        elif signal.signal == Signal.SELL:
            available_token = self.portfolio.get_balance(self.token)

            if available_token < signal.quantity:
                logger.warning(f"Insufficient {self.token}: need {signal.quantity}, have {available_token}")
                return

        # Execute order
        side = OrderSide.BUY if signal.signal == Signal.BUY else OrderSide.SELL

        order = self.engine.execute_market_order(
            exchange=self.exchange,
            symbol=self.symbol,
            side=side,
            quantity=signal.quantity,
            current_price=signal.price
        )

        # Update portfolio
        self.portfolio.process_order(order, signal.price)

        self.trades_executed += 1
        logger.info(f"Executed {signal.signal.value}: {signal.quantity} {self.token} @ ${signal.price:.2f}")

    def run_step(self, candles: List[Candle]):
        """
        Run one strategy step.

        Args:
            candles: Historical candle data up to current time
        """
        current_position = self.portfolio.get_balance(self.token)
        signal = self.strategy.generate_signal(candles, current_position)

        self.signals.append(signal)
        self.execute_signal(signal)


class Backtester:
    """
    Backtest strategies on historical data.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        exchange: Exchange = Exchange.COINBASE,
        symbol: str = "ETH-USD"
    ):
        """
        Initialize backtester.

        Args:
            strategy: Trading strategy to test
            initial_capital: Starting capital
            exchange: Exchange to simulate
            symbol: Trading pair
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.exchange = exchange
        self.symbol = symbol

        logger.info(f"Backtester initialized: {strategy.name} with ${initial_capital:,.2f}")

    def run(self, candles: List[Candle]) -> Dict:
        """
        Run backtest on historical candles.

        Args:
            candles: Historical OHLCV data

        Returns:
            Backtest results with metrics
        """
        logger.info(f"Starting backtest with {len(candles)} candles")

        # Initialize components
        portfolio = PaperPortfolio(initial_usd=self.initial_capital)
        engine = PaperTradingEngine()
        runner = StrategyRunner(self.strategy, portfolio, engine, self.exchange, self.symbol)

        # Run strategy on each candle
        for i in range(len(candles)):
            # Feed candles up to current point
            current_candles = candles[:i+1]
            runner.run_step(current_candles)

        # Calculate final results
        token = self.symbol.split('-')[0]
        final_prices = {token: candles[-1].close}
        pnl = portfolio.get_pnl(final_prices)

        results = {
            'strategy': self.strategy.name,
            'exchange': self.exchange.value,
            'symbol': self.symbol,
            'candles': len(candles),
            'period': {
                'start': candles[0].timestamp,
                'end': candles[-1].timestamp
            },
            'trades_executed': runner.trades_executed,
            'signals_generated': len([s for s in runner.signals if s.signal != Signal.HOLD]),
            'initial_capital': self.initial_capital,
            'final_value': pnl['current_value'],
            'total_pnl': pnl['total_pnl'],
            'total_pnl_percent': pnl['total_pnl_percent'],
            'total_fees': pnl['total_fees'],
            'total_gas': pnl['total_gas'],
            'net_pnl': pnl['net_pnl'],
            'win_rate': pnl['win_rate'],
            'max_drawdown': pnl['max_drawdown']
        }

        logger.info(f"Backtest complete: {runner.trades_executed} trades, "
                   f"P&L: ${pnl['total_pnl']:.2f} ({pnl['total_pnl_percent']:.2f}%)")

        return results


if __name__ == '__main__':
    import logging
    import sys
    import os

    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    logging.basicConfig(level=logging.INFO)

    print("📈 Paper Trading Strategy Demo")
    print("=" * 60)

    # Generate sample candles (simulated price data)
    print("\n1. Generating sample price data...")
    base_price = 2000.0
    candles = []

    for i in range(50):
        # Simulate price movement
        if i < 20:
            # Uptrend
            price = base_price + (i * 10)
        elif i < 35:
            # Downtrend
            price = base_price + (20 * 10) - ((i - 20) * 15)
        else:
            # Recovery
            price = base_price + (35 - 20) * (-15) + ((i - 35) * 20)

        candle = Candle(
            timestamp=datetime.now() - timedelta(hours=50-i),
            open=price,
            high=price * 1.01,
            low=price * 0.99,
            close=price,
            volume=1000000.0
        )
        candles.append(candle)

    print(f"Generated {len(candles)} candles")
    print(f"Price range: ${candles[0].close:.2f} - ${candles[-1].close:.2f}")

    # Test SMA strategy
    print("\n2. Testing SMA Strategy (10/30)")
    print("-" * 60)

    sma_strategy = SimpleMovingAverageStrategy(
        short_window=10,
        long_window=30,
        position_size=1.0
    )

    backtester = Backtester(
        strategy=sma_strategy,
        initial_capital=10000.0,
        exchange=Exchange.COINBASE,
        symbol="ETH-USD"
    )

    results = backtester.run(candles)

    print(f"\nResults:")
    print(f"  Trades: {results['trades_executed']}")
    print(f"  Final Value: ${results['final_value']:,.2f}")
    print(f"  Total P&L: ${results['total_pnl']:,.2f} ({results['total_pnl_percent']:.2f}%)")
    print(f"  Fees: ${results['total_fees']:.2f}")
    print(f"  Net P&L: ${results['net_pnl']:,.2f}")

    # Test Momentum strategy
    print("\n3. Testing Momentum Strategy")
    print("-" * 60)

    momentum_strategy = MomentumStrategy(
        lookback_period=10,
        momentum_threshold=0.02,
        position_size=1.0,
        profit_target=0.05
    )

    backtester2 = Backtester(
        strategy=momentum_strategy,
        initial_capital=10000.0,
        exchange=Exchange.UNISWAP,
        symbol="ETH-USD"
    )

    results2 = backtester2.run(candles)

    print(f"\nResults:")
    print(f"  Trades: {results2['trades_executed']}")
    print(f"  Final Value: ${results2['final_value']:,.2f}")
    print(f"  Total P&L: ${results2['total_pnl']:,.2f} ({results2['total_pnl_percent']:.2f}%)")
    print(f"  Fees: ${results2['total_fees']:.2f}")
    print(f"  Gas: ${results2['total_gas']:.2f}")
    print(f"  Net P&L: ${results2['net_pnl']:,.2f}")

    print("\n✅ Strategy backtesting demo complete!")
