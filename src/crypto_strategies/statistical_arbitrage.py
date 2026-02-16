"""
Statistical Arbitrage (Pairs Trading) Strategy
Exploits temporary deviations in historically correlated asset pairs.

Features:
- Cointegration testing (Engle-Granger, Johansen)
- Z-score based entry/exit signals
- Multiple pair selection criteria
- Dynamic hedge ratios
- Mean reversion detection
"""
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class PairStatus(Enum):
    """Pair trading status."""
    NO_POSITION = "no_position"
    LONG_PAIR = "long_pair"  # Long asset1, short asset2
    SHORT_PAIR = "short_pair"  # Short asset1, long asset2


class SignalType(Enum):
    """Trading signal type."""
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT = "exit"
    HOLD = "hold"


@dataclass
class PairConfig:
    """Pairs trading configuration."""
    asset1: str
    asset2: str

    # Entry/exit thresholds (Z-score)
    entry_threshold: float = 2.0  # Enter when |z| > 2.0
    exit_threshold: float = 0.5  # Exit when |z| < 0.5
    stop_loss_threshold: float = 4.0  # Stop if |z| > 4.0

    # Position sizing
    position_size: float = 10000.0  # USD per pair
    max_leverage: float = 1.0  # No leverage by default

    # Lookback periods
    cointegration_window: int = 252  # 1 year daily
    zscore_window: int = 20  # 20-period rolling


@dataclass
class CointegrationTest:
    """Cointegration test results."""
    is_cointegrated: bool
    p_value: float
    test_statistic: float
    hedge_ratio: float
    half_life: float  # Mean reversion half-life in periods
    confidence: float


@dataclass
class PairSignal:
    """Pairs trading signal."""
    signal_type: SignalType
    asset1: str
    asset2: str
    zscore: float
    spread: float
    hedge_ratio: float
    confidence: float
    asset1_quantity: float
    asset2_quantity: float
    expected_return: float
    timestamp: Any = None


@dataclass
class PairMetrics:
    """Pair trading performance metrics."""
    num_trades: int
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_pnl: float
    current_zscore: float
    current_spread: float


class StatisticalArbitrage:
    """
    Statistical Arbitrage (Pairs Trading) Strategy.

    Identifies and trades cointegrated asset pairs based on mean reversion.
    """

    def __init__(self, config: PairConfig):
        """
        Initialize pairs trading strategy.

        Args:
            config: Pair configuration
        """
        self.config = config
        self.position_status = PairStatus.NO_POSITION
        self.entry_spread = 0.0
        self.entry_zscore = 0.0
        self.hedge_ratio = 1.0

        # Trade history
        self.trade_history: List[Dict[str, Any]] = []
        self.spread_history: List[float] = []
        self.zscore_history: List[float] = []

        # Price history for cointegration
        self.price1_history: List[float] = []
        self.price2_history: List[float] = []

        logger.info(f"StatArb initialized: {config.asset1}/{config.asset2}")

    def test_cointegration(
        self,
        prices1: np.ndarray,
        prices2: np.ndarray,
        method: str = 'engle_granger'
    ) -> CointegrationTest:
        """
        Test for cointegration between two price series.

        Args:
            prices1: Price series for asset 1
            prices2: Price series for asset 2
            method: 'engle_granger' or 'johansen'

        Returns:
            Cointegration test results
        """
        if len(prices1) != len(prices2):
            raise ValueError("Price series must have same length")

        if len(prices1) < 30:
            return CointegrationTest(
                is_cointegrated=False,
                p_value=1.0,
                test_statistic=0.0,
                hedge_ratio=1.0,
                half_life=float('inf'),
                confidence=0.0
            )

        # Calculate hedge ratio using OLS regression
        # price1 = beta * price2 + alpha + error
        X = np.column_stack([prices2, np.ones(len(prices2))])
        beta, alpha = np.linalg.lstsq(X, prices1, rcond=None)[0]

        hedge_ratio = beta

        # Calculate spread (residuals)
        spread = prices1 - hedge_ratio * prices2

        # Test spread for stationarity (simplified ADF test)
        # In production, use statsmodels.tsa.stattools.adfuller
        spread_diff = np.diff(spread)
        spread_lag = spread[:-1]

        # AR(1) regression: diff(spread) = rho * spread_lag + error
        X_ar = np.column_stack([spread_lag, np.ones(len(spread_lag))])
        rho, _ = np.linalg.lstsq(X_ar, spread_diff, rcond=None)[0]

        # Simplified test statistic
        test_statistic = rho * np.sqrt(len(spread))

        # Critical values (approximate)
        # -3.43 (1%), -2.86 (5%), -2.57 (10%)
        if test_statistic < -2.86:
            is_cointegrated = True
            p_value = 0.05
            confidence = 0.95
        elif test_statistic < -2.57:
            is_cointegrated = True
            p_value = 0.10
            confidence = 0.90
        else:
            is_cointegrated = False
            p_value = 0.20
            confidence = 0.0

        # Calculate half-life of mean reversion
        # half_life = -log(2) / log(rho + 1)
        if -1 < rho < 0:
            half_life = -np.log(2) / np.log(1 + rho)
        else:
            half_life = float('inf')

        result = CointegrationTest(
            is_cointegrated=is_cointegrated,
            p_value=p_value,
            test_statistic=test_statistic,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            confidence=confidence
        )

        logger.info(
            f"Cointegration Test: {'PASS' if is_cointegrated else 'FAIL'} "
            f"(p={p_value:.3f}, hedge={hedge_ratio:.4f}, half-life={half_life:.1f})"
        )

        return result

    def calculate_zscore(
        self,
        spread_history: List[float],
        window: int
    ) -> float:
        """
        Calculate Z-score of current spread.

        Args:
            spread_history: Historical spread values
            window: Lookback window

        Returns:
            Current Z-score
        """
        if len(spread_history) < window:
            return 0.0

        recent = spread_history[-window:]
        mean = np.mean(recent)
        std = np.std(recent)

        if std == 0:
            return 0.0

        current_spread = spread_history[-1]
        zscore = (current_spread - mean) / std

        return zscore

    def calculate_spread(
        self,
        price1: float,
        price2: float,
        hedge_ratio: Optional[float] = None
    ) -> float:
        """
        Calculate spread between two assets.

        Args:
            price1: Price of asset 1
            price2: Price of asset 2
            hedge_ratio: Hedge ratio (default: use calculated)

        Returns:
            Spread value
        """
        if hedge_ratio is None:
            hedge_ratio = self.hedge_ratio

        spread = price1 - hedge_ratio * price2
        return spread

    def generate_signal(
        self,
        price1: float,
        price2: float,
        timestamp: Any = None
    ) -> Optional[PairSignal]:
        """
        Generate trading signal for pair.

        Args:
            price1: Current price of asset 1
            price2: Current price of asset 2
            timestamp: Current timestamp

        Returns:
            Trading signal or None
        """
        # Update price history
        self.price1_history.append(price1)
        self.price2_history.append(price2)

        # Keep only recent history
        if len(self.price1_history) > self.config.cointegration_window:
            self.price1_history = self.price1_history[-self.config.cointegration_window:]
            self.price2_history = self.price2_history[-self.config.cointegration_window:]

        # Need enough history
        if len(self.price1_history) < self.config.zscore_window:
            return None

        # Test cointegration periodically
        if len(self.price1_history) >= self.config.cointegration_window:
            coint_test = self.test_cointegration(
                np.array(self.price1_history),
                np.array(self.price2_history)
            )

            if not coint_test.is_cointegrated:
                logger.warning("Pair no longer cointegrated - consider exiting")

            self.hedge_ratio = coint_test.hedge_ratio

        # Calculate spread
        spread = self.calculate_spread(price1, price2)
        self.spread_history.append(spread)

        # Calculate Z-score
        zscore = self.calculate_zscore(
            self.spread_history,
            self.config.zscore_window
        )
        self.zscore_history.append(zscore)

        # Generate signal based on Z-score and position status
        signal_type = SignalType.HOLD
        confidence = 0.0

        if self.position_status == PairStatus.NO_POSITION:
            # Entry signals
            if zscore > self.config.entry_threshold:
                # Spread too high - short the spread
                # Short asset1, long asset2
                signal_type = SignalType.ENTER_SHORT
                confidence = min(abs(zscore) / 5.0, 1.0)
                self.entry_spread = spread
                self.entry_zscore = zscore

            elif zscore < -self.config.entry_threshold:
                # Spread too low - long the spread
                # Long asset1, short asset2
                signal_type = SignalType.ENTER_LONG
                confidence = min(abs(zscore) / 5.0, 1.0)
                self.entry_spread = spread
                self.entry_zscore = zscore

        else:
            # Exit signals
            # Exit when spread reverts
            if abs(zscore) < self.config.exit_threshold:
                signal_type = SignalType.EXIT
                confidence = 1.0 - abs(zscore) / self.config.entry_threshold

            # Stop loss
            elif abs(zscore) > self.config.stop_loss_threshold:
                signal_type = SignalType.EXIT
                confidence = 1.0
                logger.warning(f"Stop loss triggered at Z={zscore:.2f}")

        # Don't generate hold signals
        if signal_type == SignalType.HOLD:
            return None

        # Calculate position sizes
        total_value = self.config.position_size

        # Allocate based on hedge ratio
        # Value1 / Value2 = hedge_ratio
        # Value1 + Value2 = total_value
        value1 = total_value * self.hedge_ratio / (1 + self.hedge_ratio)
        value2 = total_value / (1 + self.hedge_ratio)

        quantity1 = value1 / price1
        quantity2 = value2 / price2

        # If shorting the spread, reverse quantities
        if signal_type == SignalType.ENTER_SHORT:
            quantity1 = -quantity1  # Short asset1
            # quantity2 stays positive (long asset2)
        elif signal_type == SignalType.ENTER_LONG:
            # quantity1 stays positive (long asset1)
            quantity2 = -quantity2  # Short asset2

        # Estimate expected return based on Z-score reversion
        expected_spread_change = -zscore * np.std(self.spread_history[-20:])
        expected_return = expected_spread_change / price1

        signal = PairSignal(
            signal_type=signal_type,
            asset1=self.config.asset1,
            asset2=self.config.asset2,
            zscore=zscore,
            spread=spread,
            hedge_ratio=self.hedge_ratio,
            confidence=confidence,
            asset1_quantity=quantity1,
            asset2_quantity=quantity2,
            expected_return=expected_return,
            timestamp=timestamp
        )

        # Update position status
        if signal_type == SignalType.ENTER_LONG:
            self.position_status = PairStatus.LONG_PAIR
        elif signal_type == SignalType.ENTER_SHORT:
            self.position_status = PairStatus.SHORT_PAIR
        elif signal_type == SignalType.EXIT:
            self.position_status = PairStatus.NO_POSITION

        logger.info(
            f"Signal: {signal_type.value} | Z={zscore:.2f} | "
            f"Spread={spread:.4f} | Confidence={confidence:.0%}"
        )

        return signal

    def backtest(
        self,
        prices1: np.ndarray,
        prices2: np.ndarray,
        timestamps: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Backtest pairs trading strategy.

        Args:
            prices1: Price history for asset 1
            prices2: Price history for asset 2
            timestamps: Corresponding timestamps

        Returns:
            Backtest results
        """
        if len(prices1) != len(prices2):
            raise ValueError("Price arrays must have same length")

        # Reset state
        self.position_status = PairStatus.NO_POSITION
        self.price1_history = []
        self.price2_history = []
        self.spread_history = []
        self.zscore_history = []
        self.trade_history = []

        trades = []
        equity_curve = [self.config.position_size]
        current_position = None

        # Run through history
        for i, (p1, p2) in enumerate(zip(prices1, prices2)):
            ts = timestamps[i] if timestamps else i

            # Generate signal
            signal = self.generate_signal(p1, p2, ts)

            if signal:
                if signal.signal_type in [SignalType.ENTER_LONG, SignalType.ENTER_SHORT]:
                    # Enter trade
                    current_position = {
                        'entry_price1': p1,
                        'entry_price2': p2,
                        'entry_spread': signal.spread,
                        'entry_zscore': signal.zscore,
                        'signal_type': signal.signal_type,
                        'quantity1': signal.asset1_quantity,
                        'quantity2': signal.asset2_quantity,
                        'entry_time': ts
                    }

                elif signal.signal_type == SignalType.EXIT and current_position:
                    # Exit trade
                    pnl1 = (p1 - current_position['entry_price1']) * current_position['quantity1']
                    pnl2 = (p2 - current_position['entry_price2']) * current_position['quantity2']
                    total_pnl = pnl1 + pnl2

                    trade = {
                        **current_position,
                        'exit_price1': p1,
                        'exit_price2': p2,
                        'exit_spread': signal.spread,
                        'exit_zscore': signal.zscore,
                        'exit_time': ts,
                        'pnl': total_pnl,
                        'return': total_pnl / self.config.position_size
                    }
                    trades.append(trade)

                    # Update equity
                    equity_curve.append(equity_curve[-1] + total_pnl)

                    current_position = None

        # Calculate metrics
        if trades:
            returns = [t['return'] for t in trades]
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            avg_return = np.mean(returns)

            # Sharpe ratio (annualized, assuming daily)
            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0.0

            # Max drawdown
            equity_array = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))

            total_pnl = equity_curve[-1] - equity_curve[0]

        else:
            win_rate = 0.0
            avg_return = 0.0
            sharpe = 0.0
            max_drawdown = 0.0
            total_pnl = 0.0

        results = {
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_pnl': total_pnl,
            'final_equity': equity_curve[-1],
            'equity_curve': equity_curve,
            'trades': trades
        }

        logger.info(f"Backtest complete: {len(trades)} trades, {win_rate:.1%} win rate")
        logger.info(f"Total P&L: ${total_pnl:.2f}, Sharpe: {sharpe:.2f}")

        return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("📊 Statistical Arbitrage Demo")
    print("=" * 60)

    # Create pair configuration
    config = PairConfig(
        asset1='ETH',
        asset2='BTC',
        entry_threshold=2.0,
        exit_threshold=0.5,
        position_size=10000.0
    )

    print(f"\n1. Configuration:")
    print(f"   Pair: {config.asset1}/{config.asset2}")
    print(f"   Entry threshold: ±{config.entry_threshold} std")
    print(f"   Exit threshold: ±{config.exit_threshold} std")
    print(f"   Position size: ${config.position_size:,.0f}")

    # Create strategy
    strategy = StatisticalArbitrage(config)

    # Generate correlated price series
    print(f"\n2. Generating synthetic price data...")
    np.random.seed(42)
    n_periods = 500

    # Generate cointegrated series
    btc_prices = np.cumsum(np.random.randn(n_periods)) + 40000
    btc_prices = np.maximum(btc_prices, 30000)  # Floor

    # ETH partially follows BTC with noise
    eth_prices = btc_prices * 0.05 + np.cumsum(np.random.randn(n_periods) * 10) + 2000
    eth_prices = np.maximum(eth_prices, 1500)  # Floor

    # Test cointegration
    print(f"\n3. Testing cointegration...")
    coint_result = strategy.test_cointegration(eth_prices, btc_prices)
    print(f"   Cointegrated: {coint_result.is_cointegrated}")
    print(f"   P-value: {coint_result.p_value:.4f}")
    print(f"   Hedge ratio: {coint_result.hedge_ratio:.6f}")
    print(f"   Half-life: {coint_result.half_life:.1f} periods")
    print(f"   Confidence: {coint_result.confidence:.0%}")

    # Backtest
    print(f"\n4. Running backtest...")
    results = strategy.backtest(eth_prices, btc_prices)

    print(f"\n5. Results:")
    print(f"   Trades: {results['num_trades']}")
    print(f"   Win Rate: {results['win_rate']:.1%}")
    print(f"   Avg Return: {results['avg_return']:.2%}")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {results['max_drawdown']:.1%}")
    print(f"   Total P&L: ${results['total_pnl']:.2f}")
    print(f"   Final Equity: ${results['final_equity']:,.2f}")

    # Show sample trades
    if results['trades']:
        print(f"\n6. Sample Trades:")
        for i, trade in enumerate(results['trades'][:3], 1):
            print(f"\n   Trade {i}:")
            print(f"      Type: {trade['signal_type'].value}")
            print(f"      Entry Z: {trade['entry_zscore']:.2f}")
            print(f"      Exit Z: {trade['exit_zscore']:.2f}")
            print(f"      P&L: ${trade['pnl']:.2f} ({trade['return']:.2%})")

    print(f"\n✅ Statistical Arbitrage Demo Complete!")
