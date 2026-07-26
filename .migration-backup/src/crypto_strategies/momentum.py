"""
Momentum Trading Strategy
Follows strong trends and momentum in the market.

Features:
- Multiple momentum indicators (MACD, ADX, Rate of Change)
- Trend strength measurement
- Multi-timeframe momentum confirmation
- Dynamic position sizing based on momentum strength
- Trailing stops for profit protection
"""
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend direction."""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"


class MomentumSignal(Enum):
    """Momentum signal types."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class MomentumConfig:
    """Momentum strategy configuration."""
    symbol: str

    # Moving averages
    fast_ma_period: int = 12
    slow_ma_period: int = 26
    signal_line_period: int = 9

    # ADX (Average Directional Index)
    adx_period: int = 14
    adx_threshold: float = 25.0  # Trending market threshold

    # Rate of Change
    roc_period: int = 12
    roc_threshold: float = 5.0  # 5% threshold

    # Trend confirmation
    trend_ma_period: int = 50  # Long-term trend
    require_trend_confirmation: bool = True

    # Position sizing
    base_position_size: float = 1000.0
    max_position_size: float = 5000.0
    scale_with_momentum: bool = True

    # Risk management
    use_trailing_stop: bool = True
    trailing_stop_percent: float = 5.0
    stop_loss_percent: float = 7.0
    take_profit_multiplier: float = 2.0  # Risk:Reward = 1:2


@dataclass
class MomentumIndicators:
    """Momentum indicator values."""
    macd: float
    macd_signal: float
    macd_histogram: float
    adx: float
    roc: float
    trend_strength: float
    moving_average_slope: float


@dataclass
class MomentumTrade:
    """Momentum trade signal."""
    signal: MomentumSignal
    trend_direction: TrendDirection
    momentum_score: float
    indicators: MomentumIndicators
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    trailing_stop: Optional[float] = None
    timestamp: Any = None


class MomentumStrategy:
    """
    Momentum Trading Strategy.

    Identifies and rides strong trends using multiple momentum indicators.
    """

    def __init__(self, config: MomentumConfig):
        """
        Initialize momentum strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.price_history: List[float] = []
        self.high_history: List[float] = []
        self.low_history: List[float] = []
        self.volume_history: List[float] = []

        self.trade_history: List[Dict[str, Any]] = []
        self.current_position: Optional[Dict[str, Any]] = None

        logger.info(f"Momentum Strategy initialized for {config.symbol}")

    def calculate_ema(
        self,
        prices: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices

        ema = np.zeros(len(prices))
        ema[0] = prices[0]

        multiplier = 2.0 / (period + 1)

        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]

        return ema

    def calculate_macd(
        self,
        prices: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Returns:
            (macd_line, signal_line, histogram)
        """
        if len(prices) < self.config.slow_ma_period:
            return 0.0, 0.0, 0.0

        # Calculate EMAs
        fast_ema = self.calculate_ema(prices, self.config.fast_ma_period)
        slow_ema = self.calculate_ema(prices, self.config.slow_ma_period)

        # MACD line
        macd_line = fast_ema - slow_ema

        # Signal line (EMA of MACD)
        if len(macd_line) >= self.config.signal_line_period:
            signal_line = self.calculate_ema(macd_line, self.config.signal_line_period)
        else:
            signal_line = macd_line

        # Histogram
        histogram = macd_line - signal_line

        return macd_line[-1], signal_line[-1], histogram[-1]

    def calculate_adx(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int
    ) -> float:
        """
        Calculate Average Directional Index (ADX).
        Measures trend strength (0-100).
        """
        if len(high) < period + 1:
            return 0.0

        # Calculate True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                abs(high[1:] - close[:-1]),
                abs(low[1:] - close[:-1])
            )
        )

        # Calculate directional movement
        plus_dm = np.maximum(high[1:] - high[:-1], 0)
        minus_dm = np.maximum(low[:-1] - low[1:], 0)

        # Smooth
        atr = np.mean(tr[-period:])
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0

        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0

        # ADX is smoothed DX
        # Simplified: use recent DX as proxy
        adx = dx

        return adx

    def calculate_roc(
        self,
        prices: np.ndarray,
        period: int
    ) -> float:
        """
        Calculate Rate of Change.
        ROC = (Price - Price[n]) / Price[n] * 100
        """
        if len(prices) < period + 1:
            return 0.0

        current_price = prices[-1]
        old_price = prices[-(period + 1)]

        if old_price == 0:
            return 0.0

        roc = ((current_price - old_price) / old_price) * 100

        return roc

    def calculate_trend_strength(
        self,
        prices: np.ndarray
    ) -> Tuple[float, TrendDirection]:
        """
        Calculate overall trend strength and direction.

        Returns:
            (strength: 0-1, direction)
        """
        if len(prices) < self.config.trend_ma_period:
            return 0.0, TrendDirection.NEUTRAL

        # Long-term moving average
        ma = np.mean(prices[-self.config.trend_ma_period:])
        current_price = prices[-1]

        # Distance from MA as percentage
        distance = (current_price - ma) / ma

        # Slope of MA
        if len(prices) >= self.config.trend_ma_period + 10:
            ma_old = np.mean(prices[-(self.config.trend_ma_period + 10):-10])
            slope = (ma - ma_old) / ma_old
        else:
            slope = 0.0

        # Combined strength
        strength = abs(distance) * 5 + abs(slope) * 10
        strength = min(strength, 1.0)

        # Determine direction
        if distance > 0.05 and slope > 0.01:
            direction = TrendDirection.STRONG_UP
        elif distance > 0.02 or slope > 0.005:
            direction = TrendDirection.UP
        elif distance < -0.05 and slope < -0.01:
            direction = TrendDirection.STRONG_DOWN
        elif distance < -0.02 or slope < -0.005:
            direction = TrendDirection.DOWN
        else:
            direction = TrendDirection.NEUTRAL

        return strength, direction

    def calculate_momentum_score(
        self,
        indicators: MomentumIndicators
    ) -> float:
        """
        Calculate composite momentum score (0-1).

        Args:
            indicators: Momentum indicators

        Returns:
            Momentum score
        """
        scores = []

        # MACD contribution
        if indicators.macd_histogram > 0:
            scores.append(min(abs(indicators.macd_histogram) / 100, 1.0))
        else:
            scores.append(-min(abs(indicators.macd_histogram) / 100, 1.0))

        # ADX contribution (trend strength)
        adx_score = min(indicators.adx / 50, 1.0)
        scores.append(adx_score if indicators.macd > 0 else -adx_score)

        # ROC contribution
        roc_score = min(abs(indicators.roc) / 20, 1.0)
        scores.append(roc_score if indicators.roc > 0 else -roc_score)

        # Trend strength contribution
        trend_score = indicators.trend_strength
        if indicators.moving_average_slope < 0:
            trend_score = -trend_score
        scores.append(trend_score)

        # Average
        momentum_score = np.mean(scores)

        return momentum_score

    def analyze_momentum(
        self,
        current_price: float,
        current_high: Optional[float] = None,
        current_low: Optional[float] = None
    ) -> MomentumIndicators:
        """
        Analyze momentum indicators.

        Args:
            current_price: Current price
            current_high: Current high (optional)
            current_low: Current low (optional)

        Returns:
            Momentum indicators
        """
        prices = np.array(self.price_history + [current_price])

        # Default high/low to price if not provided
        if current_high is None:
            current_high = current_price
        if current_low is None:
            current_low = current_price

        highs = np.array(self.high_history + [current_high])
        lows = np.array(self.low_history + [current_low])

        # MACD
        macd, macd_signal, histogram = self.calculate_macd(prices)

        # ADX
        adx = self.calculate_adx(highs, lows, prices, self.config.adx_period)

        # ROC
        roc = self.calculate_roc(prices, self.config.roc_period)

        # Trend
        trend_strength, _ = self.calculate_trend_strength(prices)

        # MA slope
        if len(prices) >= 20:
            recent_ma = np.mean(prices[-20:])
            old_ma = np.mean(prices[-30:-10]) if len(prices) >= 30 else recent_ma
            ma_slope = (recent_ma - old_ma) / old_ma if old_ma > 0 else 0.0
        else:
            ma_slope = 0.0

        return MomentumIndicators(
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=histogram,
            adx=adx,
            roc=roc,
            trend_strength=trend_strength,
            moving_average_slope=ma_slope
        )

    def generate_signal(
        self,
        current_price: float,
        current_high: Optional[float] = None,
        current_low: Optional[float] = None,
        timestamp: Any = None
    ) -> Optional[MomentumTrade]:
        """
        Generate momentum trading signal.

        Args:
            current_price: Current price
            current_high: Current high
            current_low: Current low
            timestamp: Current timestamp

        Returns:
            Trading signal or None
        """
        # Update histories
        self.price_history.append(current_price)
        self.high_history.append(current_high or current_price)
        self.low_history.append(current_low or current_price)

        # Keep recent history
        max_history = max(self.config.slow_ma_period, self.config.trend_ma_period) + 50
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.high_history = self.high_history[-max_history:]
            self.low_history = self.low_history[-max_history:]

        # Need minimum history
        if len(self.price_history) < self.config.slow_ma_period:
            return None

        # Analyze momentum
        indicators = self.analyze_momentum(current_price, current_high, current_low)

        # Calculate momentum score
        momentum_score = self.calculate_momentum_score(indicators)

        # Determine trend
        _, trend_direction = self.calculate_trend_strength(np.array(self.price_history))

        # Generate signal
        signal_type = MomentumSignal.HOLD

        # Check for trending market (ADX)
        if indicators.adx < self.config.adx_threshold:
            # Not trending enough
            return None

        # Bullish momentum
        if (indicators.macd > indicators.macd_signal and
            indicators.macd_histogram > 0 and
            indicators.roc > self.config.roc_threshold):

            # Check trend confirmation if required
            if self.config.require_trend_confirmation:
                if trend_direction in [TrendDirection.UP, TrendDirection.STRONG_UP]:
                    signal_type = MomentumSignal.STRONG_BUY if momentum_score > 0.7 else MomentumSignal.BUY
            else:
                signal_type = MomentumSignal.STRONG_BUY if momentum_score > 0.7 else MomentumSignal.BUY

        # Bearish momentum
        elif (indicators.macd < indicators.macd_signal and
              indicators.macd_histogram < 0 and
              indicators.roc < -self.config.roc_threshold):

            if self.config.require_trend_confirmation:
                if trend_direction in [TrendDirection.DOWN, TrendDirection.STRONG_DOWN]:
                    signal_type = MomentumSignal.STRONG_SELL if momentum_score < -0.7 else MomentumSignal.SELL
            else:
                signal_type = MomentumSignal.STRONG_SELL if momentum_score < -0.7 else MomentumSignal.SELL

        if signal_type == MomentumSignal.HOLD:
            return None

        # Calculate position size
        base_size = self.config.base_position_size

        if self.config.scale_with_momentum:
            size = base_size * (0.5 + abs(momentum_score) * 0.5)
        else:
            size = base_size

        size = min(size, self.config.max_position_size)

        # Calculate stops and targets
        if signal_type in [MomentumSignal.BUY, MomentumSignal.STRONG_BUY]:
            stop_loss = current_price * (1 - self.config.stop_loss_percent / 100)
            take_profit = current_price * (1 + self.config.stop_loss_percent * self.config.take_profit_multiplier / 100)
            trailing_stop = current_price * (1 - self.config.trailing_stop_percent / 100) if self.config.use_trailing_stop else None
        else:
            stop_loss = current_price * (1 + self.config.stop_loss_percent / 100)
            take_profit = current_price * (1 - self.config.stop_loss_percent * self.config.take_profit_multiplier / 100)
            trailing_stop = current_price * (1 + self.config.trailing_stop_percent / 100) if self.config.use_trailing_stop else None

        trade = MomentumTrade(
            signal=signal_type,
            trend_direction=trend_direction,
            momentum_score=momentum_score,
            indicators=indicators,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=size,
            trailing_stop=trailing_stop,
            timestamp=timestamp
        )

        logger.info(
            f"Signal: {signal_type.value} | Momentum: {momentum_score:.2f} | "
            f"MACD: {indicators.macd:.2f} | ADX: {indicators.adx:.1f}"
        )

        return trade

    def backtest(
        self,
        prices: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        timestamps: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Backtest momentum strategy.

        Args:
            prices: Price history
            highs: High prices (optional)
            lows: Low prices (optional)
            timestamps: Timestamps (optional)

        Returns:
            Backtest results
        """
        # Reset state
        self.price_history = []
        self.high_history = []
        self.low_history = []
        self.trade_history = []

        if highs is None:
            highs = prices
        if lows is None:
            lows = prices

        trades = []
        equity = [10000.0]
        position = None

        # Simulate
        for i, price in enumerate(prices):
            high = highs[i]
            low = lows[i]
            ts = timestamps[i] if timestamps else i

            # Check position
            if position:
                # Update trailing stop
                if position['is_long']:
                    if self.config.use_trailing_stop:
                        new_trailing = price * (1 - self.config.trailing_stop_percent / 100)
                        position['trailing_stop'] = max(position['trailing_stop'], new_trailing)

                    # Check exits
                    if low <= position['stop_loss'] or low <= position['trailing_stop']:
                        exit_price = max(position['stop_loss'], position['trailing_stop'])
                        pnl = (exit_price - position['entry_price']) * position['quantity']
                    elif high >= position['take_profit']:
                        exit_price = position['take_profit']
                        pnl = (exit_price - position['entry_price']) * position['quantity']
                    else:
                        continue

                else:  # Short
                    if self.config.use_trailing_stop:
                        new_trailing = price * (1 + self.config.trailing_stop_percent / 100)
                        position['trailing_stop'] = min(position['trailing_stop'], new_trailing)

                    if high >= position['stop_loss'] or high >= position['trailing_stop']:
                        exit_price = min(position['stop_loss'], position['trailing_stop'])
                        pnl = (position['entry_price'] - exit_price) * abs(position['quantity'])
                    elif low <= position['take_profit']:
                        exit_price = position['take_profit']
                        pnl = (position['entry_price'] - exit_price) * abs(position['quantity'])
                    else:
                        continue

                # Close trade
                trade = {
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return': pnl / position['size'],
                    'holding_periods': i - position['entry_index']
                }
                trades.append(trade)
                equity.append(equity[-1] + pnl)
                position = None

            # Generate new signal
            if position is None:
                signal = self.generate_signal(price, high, low, ts)

                if signal:
                    is_long = signal.signal in [MomentumSignal.BUY, MomentumSignal.STRONG_BUY]
                    quantity = signal.position_size / price

                    position = {
                        'entry_price': price,
                        'quantity': quantity if is_long else -quantity,
                        'size': signal.position_size,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'trailing_stop': signal.trailing_stop or (signal.stop_loss if is_long else price * 2),
                        'is_long': is_long,
                        'entry_index': i
                    }

        # Metrics
        if trades:
            returns = [t['return'] for t in trades]
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            avg_return = np.mean(returns)
            sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
            total_return = (equity[-1] - equity[0]) / equity[0]
        else:
            win_rate = avg_return = sharpe = total_return = 0.0

        results = {
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'final_equity': equity[-1],
            'equity_curve': equity,
            'trades': trades
        }

        logger.info(f"Backtest: {len(trades)} trades, {win_rate:.1%} win rate, {total_return:.1%} return")

        return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("🚀 Momentum Strategy Demo")
    print("=" * 60)

    # Configuration
    config = MomentumConfig(
        symbol='BTC',
        fast_ma_period=12,
        slow_ma_period=26,
        adx_threshold=25.0,
        base_position_size=1000.0,
        use_trailing_stop=True
    )

    print(f"\n1. Configuration:")
    print(f"   Symbol: {config.symbol}")
    print(f"   MACD: {config.fast_ma_period}/{config.slow_ma_period}")
    print(f"   ADX Threshold: {config.adx_threshold}")
    print(f"   Trailing Stop: {config.trailing_stop_percent}%")

    # Create strategy
    strategy = MomentumStrategy(config)

    # Generate trending price data
    print(f"\n2. Generating trending price data...")
    np.random.seed(42)
    n = 300
    prices = 40000 + np.cumsum(np.random.randn(n) * 200 + 50)  # Uptrend with noise
    prices = np.maximum(prices, 35000)

    # Backtest
    print(f"\n3. Running backtest...")
    results = strategy.backtest(prices)

    print(f"\n4. Results:")
    print(f"   Trades: {results['num_trades']}")
    print(f"   Win Rate: {results['win_rate']:.1%}")
    print(f"   Avg Return: {results['avg_return']:.2%}")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"   Total Return: {results['total_return']:.1%}")
    print(f"   Final Equity: ${results['final_equity']:,.2f}")

    print(f"\n✅ Momentum Strategy Demo Complete!")
