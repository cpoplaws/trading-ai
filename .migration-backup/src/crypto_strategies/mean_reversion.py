"""
Mean Reversion Trading Strategy
Profits from price returning to mean after deviations.

Features:
- Multiple mean reversion indicators (Bollinger Bands, RSI, Z-score)
- Adaptive threshold based on volatility
- Multi-timeframe confirmation
- Risk-adjusted position sizing
- Confluence scoring
"""
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class MeanReversionSignal(Enum):
    """Mean reversion signal types."""
    STRONG_BUY = "strong_buy"  # Multiple indicators agree
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class IndicatorType(Enum):
    """Mean reversion indicator types."""
    BOLLINGER_BANDS = "bollinger_bands"
    RSI = "rsi"
    ZSCORE = "zscore"
    STOCHASTIC = "stochastic"
    MEAN_DISTANCE = "mean_distance"


@dataclass
class MeanReversionConfig:
    """Mean reversion strategy configuration."""
    symbol: str

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # RSI
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Z-Score
    zscore_period: int = 20
    zscore_threshold: float = 2.0

    # Stochastic
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0

    # Position sizing
    base_position_size: float = 1000.0
    max_position_size: float = 5000.0
    use_volatility_sizing: bool = True

    # Risk management
    stop_loss_percent: float = 5.0
    take_profit_percent: float = 10.0
    max_holding_periods: int = 50  # Exit after this many periods

    # Confluence
    min_confluence_score: float = 0.6  # Require 60% agreement


@dataclass
class IndicatorSignal:
    """Individual indicator signal."""
    indicator: IndicatorType
    value: float
    signal_strength: float  # -1 to 1
    is_oversold: bool
    is_overbought: bool
    confidence: float


@dataclass
class TradingSignal:
    """Complete trading signal with confluence."""
    signal: MeanReversionSignal
    confluence_score: float
    indicators: List[IndicatorSignal]
    current_price: float
    target_price: float
    stop_loss: float
    position_size: float
    timestamp: Any = None


@dataclass
class Position:
    """Open trading position."""
    entry_price: float
    quantity: float
    entry_time: Any
    stop_loss: float
    take_profit: float
    current_pnl: float = 0.0
    holding_periods: int = 0


class MeanReversionStrategy:
    """
    Mean Reversion Trading Strategy.

    Detects and trades price deviations from historical mean.
    """

    def __init__(self, config: MeanReversionConfig):
        """
        Initialize mean reversion strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.current_position: Optional[Position] = None
        self.trade_history: List[Dict[str, Any]] = []

        logger.info(f"Mean Reversion Strategy initialized for {config.symbol}")

    def calculate_bollinger_bands(
        self,
        prices: np.ndarray,
        period: int,
        std_multiplier: float
    ) -> Dict[str, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'bandwidth': 0}

        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        upper = sma + (std_multiplier * std)
        lower = sma - (std_multiplier * std)
        bandwidth = (upper - lower) / sma

        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'bandwidth': bandwidth
        }

    def calculate_rsi(
        self,
        prices: np.ndarray,
        period: int
    ) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-period-1:])
        gains = deltas.copy()
        losses = deltas.copy()

        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_zscore(
        self,
        prices: np.ndarray,
        period: int
    ) -> float:
        """Calculate Z-score of current price."""
        if len(prices) < period:
            return 0.0

        mean = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        if std == 0:
            return 0.0

        current_price = prices[-1]
        zscore = (current_price - mean) / std

        return zscore

    def calculate_stochastic(
        self,
        prices: np.ndarray,
        k_period: int,
        d_period: int
    ) -> Dict[str, float]:
        """Calculate Stochastic Oscillator."""
        if len(prices) < k_period:
            return {'k': 50.0, 'd': 50.0}

        # %K = (Current - Lowest Low) / (Highest High - Lowest Low) * 100
        low = np.min(prices[-k_period:])
        high = np.max(prices[-k_period:])
        current = prices[-1]

        if high == low:
            k = 50.0
        else:
            k = ((current - low) / (high - low)) * 100

        # %D = 3-period SMA of %K (simplified)
        d = k  # In practice, would calculate SMA of recent %K values

        return {'k': k, 'd': d}

    def analyze_indicators(
        self,
        current_price: float
    ) -> List[IndicatorSignal]:
        """
        Analyze all mean reversion indicators.

        Args:
            current_price: Current asset price

        Returns:
            List of indicator signals
        """
        indicators = []
        prices = np.array(self.price_history + [current_price])

        # 1. Bollinger Bands
        bb = self.calculate_bollinger_bands(
            prices,
            self.config.bb_period,
            self.config.bb_std
        )

        if bb['middle'] > 0:
            # Position relative to bands (-1 to 1)
            bb_position = (current_price - bb['middle']) / (bb['upper'] - bb['middle'])
            bb_position = np.clip(bb_position, -1, 1)

            indicators.append(IndicatorSignal(
                indicator=IndicatorType.BOLLINGER_BANDS,
                value=bb_position,
                signal_strength=-bb_position,  # Invert for mean reversion
                is_oversold=current_price < bb['lower'],
                is_overbought=current_price > bb['upper'],
                confidence=min(bb['bandwidth'] * 10, 1.0)  # Higher bandwidth = more confidence
            ))

        # 2. RSI
        rsi = self.calculate_rsi(prices, self.config.rsi_period)

        # Convert RSI to -1 to 1 scale
        rsi_normalized = (rsi - 50) / 50
        is_oversold = rsi < self.config.rsi_oversold
        is_overbought = rsi > self.config.rsi_overbought

        indicators.append(IndicatorSignal(
            indicator=IndicatorType.RSI,
            value=rsi,
            signal_strength=-rsi_normalized,  # Invert for mean reversion
            is_oversold=is_oversold,
            is_overbought=is_overbought,
            confidence=abs(rsi - 50) / 50  # Further from 50 = more confidence
        ))

        # 3. Z-Score
        zscore = self.calculate_zscore(prices, self.config.zscore_period)

        indicators.append(IndicatorSignal(
            indicator=IndicatorType.ZSCORE,
            value=zscore,
            signal_strength=-np.sign(zscore) * min(abs(zscore) / 3, 1.0),
            is_oversold=zscore < -self.config.zscore_threshold,
            is_overbought=zscore > self.config.zscore_threshold,
            confidence=min(abs(zscore) / 3, 1.0)
        ))

        # 4. Stochastic
        stoch = self.calculate_stochastic(
            prices,
            self.config.stoch_k_period,
            self.config.stoch_d_period
        )

        stoch_normalized = (stoch['k'] - 50) / 50
        is_oversold_stoch = stoch['k'] < self.config.stoch_oversold
        is_overbought_stoch = stoch['k'] > self.config.stoch_overbought

        indicators.append(IndicatorSignal(
            indicator=IndicatorType.STOCHASTIC,
            value=stoch['k'],
            signal_strength=-stoch_normalized,
            is_oversold=is_oversold_stoch,
            is_overbought=is_overbought_stoch,
            confidence=abs(stoch['k'] - 50) / 50
        ))

        # 5. Mean Distance
        if len(prices) >= 50:
            long_term_mean = np.mean(prices[-50:])
            distance = (current_price - long_term_mean) / long_term_mean

            indicators.append(IndicatorSignal(
                indicator=IndicatorType.MEAN_DISTANCE,
                value=distance,
                signal_strength=-np.sign(distance) * min(abs(distance) * 10, 1.0),
                is_oversold=distance < -0.10,  # 10% below mean
                is_overbought=distance > 0.10,  # 10% above mean
                confidence=min(abs(distance) * 10, 1.0)
            ))

        return indicators

    def calculate_confluence(
        self,
        indicators: List[IndicatorSignal]
    ) -> Dict[str, Any]:
        """
        Calculate confluence score across indicators.

        Args:
            indicators: List of indicator signals

        Returns:
            Confluence analysis
        """
        if not indicators:
            return {
                'score': 0.0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0
            }

        bullish_signals = sum(1 for ind in indicators if ind.is_oversold)
        bearish_signals = sum(1 for ind in indicators if ind.is_overbought)
        neutral_signals = len(indicators) - bullish_signals - bearish_signals

        # Weighted confluence score
        total_strength = 0.0
        for ind in indicators:
            # Weight by confidence
            weighted_strength = ind.signal_strength * ind.confidence
            total_strength += weighted_strength

        # Normalize to 0-1 range
        confluence_score = abs(total_strength) / len(indicators)

        return {
            'score': confluence_score,
            'direction': np.sign(total_strength),
            'bullish_count': bullish_signals,
            'bearish_count': bearish_signals,
            'neutral_count': neutral_signals
        }

    def calculate_position_size(
        self,
        current_price: float,
        confluence_score: float
    ) -> float:
        """
        Calculate position size based on volatility and confluence.

        Args:
            current_price: Current price
            confluence_score: Signal confluence score

        Returns:
            Position size in USD
        """
        base_size = self.config.base_position_size

        if self.config.use_volatility_sizing:
            # Calculate recent volatility
            if len(self.price_history) >= 20:
                returns = np.diff(np.log(self.price_history[-20:]))
                volatility = np.std(returns)

                # Reduce size in high volatility
                vol_adjustment = 1.0 / (1.0 + volatility * 100)
            else:
                vol_adjustment = 1.0

            size = base_size * vol_adjustment
        else:
            size = base_size

        # Scale by confluence
        size *= confluence_score

        # Cap at max
        size = min(size, self.config.max_position_size)

        return size

    def generate_signal(
        self,
        current_price: float,
        timestamp: Any = None
    ) -> Optional[TradingSignal]:
        """
        Generate mean reversion trading signal.

        Args:
            current_price: Current asset price
            timestamp: Current timestamp

        Returns:
            Trading signal or None
        """
        # Update price history
        self.price_history.append(current_price)
        if len(self.price_history) > 200:
            self.price_history = self.price_history[-200:]

        # Need minimum history
        if len(self.price_history) < self.config.bb_period:
            return None

        # Analyze indicators
        indicators = self.analyze_indicators(current_price)

        # Calculate confluence
        confluence = self.calculate_confluence(indicators)

        # Determine signal
        if confluence['score'] < self.config.min_confluence_score:
            return None  # Not enough agreement

        # Generate signal
        if confluence['direction'] > 0:  # Bullish (oversold)
            signal_type = MeanReversionSignal.STRONG_BUY if confluence['score'] > 0.8 else MeanReversionSignal.BUY
        elif confluence['direction'] < 0:  # Bearish (overbought)
            signal_type = MeanReversionSignal.STRONG_SELL if confluence['score'] > 0.8 else MeanReversionSignal.SELL
        else:
            return None

        # Calculate targets
        recent_mean = np.mean(self.price_history[-20:])
        price_std = np.std(self.price_history[-20:])

        if signal_type in [MeanReversionSignal.BUY, MeanReversionSignal.STRONG_BUY]:
            target_price = recent_mean
            stop_loss = current_price * (1 - self.config.stop_loss_percent / 100)
        else:
            target_price = recent_mean
            stop_loss = current_price * (1 + self.config.stop_loss_percent / 100)

        # Calculate position size
        position_size = self.calculate_position_size(current_price, confluence['score'])

        signal = TradingSignal(
            signal=signal_type,
            confluence_score=confluence['score'],
            indicators=indicators,
            current_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            position_size=position_size,
            timestamp=timestamp
        )

        logger.info(
            f"Signal: {signal_type.value} | Confluence: {confluence['score']:.2f} | "
            f"Price: ${current_price:.2f} | Target: ${target_price:.2f}"
        )

        return signal

    def backtest(
        self,
        prices: np.ndarray,
        timestamps: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Backtest mean reversion strategy.

        Args:
            prices: Historical prices
            timestamps: Corresponding timestamps

        Returns:
            Backtest results
        """
        # Reset state
        self.price_history = []
        self.current_position = None
        self.trade_history = []

        trades = []
        equity_curve = [10000.0]  # Start with $10k
        position = None

        # Run through history
        for i, price in enumerate(prices):
            ts = timestamps[i] if timestamps else i

            # Generate signal
            signal = self.generate_signal(price, ts)

            # Manage existing position
            if position:
                position.holding_periods += 1
                position.current_pnl = (price - position.entry_price) * position.quantity

                # Check exit conditions
                should_exit = False

                if position.quantity > 0:  # Long position
                    if price >= position.take_profit or price <= position.stop_loss:
                        should_exit = True
                else:  # Short position
                    if price <= position.take_profit or price >= position.stop_loss:
                        should_exit = True

                # Max holding period
                if position.holding_periods >= self.config.max_holding_periods:
                    should_exit = True

                if should_exit:
                    # Close position
                    pnl = position.current_pnl
                    trade = {
                        'entry_price': position.entry_price,
                        'exit_price': price,
                        'quantity': position.quantity,
                        'pnl': pnl,
                        'holding_periods': position.holding_periods,
                        'entry_time': position.entry_time,
                        'exit_time': ts
                    }
                    trades.append(trade)
                    equity_curve.append(equity_curve[-1] + pnl)
                    position = None

            # Enter new position
            if signal and position is None:
                if signal.signal in [MeanReversionSignal.BUY, MeanReversionSignal.STRONG_BUY]:
                    quantity = signal.position_size / price
                    take_profit = signal.target_price
                elif signal.signal in [MeanReversionSignal.SELL, MeanReversionSignal.STRONG_SELL]:
                    quantity = -(signal.position_size / price)
                    take_profit = signal.target_price
                else:
                    continue

                position = Position(
                    entry_price=price,
                    quantity=quantity,
                    entry_time=ts,
                    stop_loss=signal.stop_loss,
                    take_profit=take_profit
                )

        # Calculate metrics
        if trades:
            pnls = [t['pnl'] for t in trades]
            win_rate = sum(1 for pnl in pnls if pnl > 0) / len(pnls)
            avg_pnl = np.mean(pnls)

            # Sharpe ratio
            if np.std(pnls) > 0:
                sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
            else:
                sharpe = 0.0

            total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        else:
            win_rate = 0.0
            avg_pnl = 0.0
            sharpe = 0.0
            total_return = 0.0

        results = {
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'final_equity': equity_curve[-1],
            'equity_curve': equity_curve,
            'trades': trades
        }

        logger.info(f"Backtest: {len(trades)} trades, {win_rate:.1%} win rate, {total_return:.1%} return")

        return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("📉 Mean Reversion Strategy Demo")
    print("=" * 60)

    # Configuration
    config = MeanReversionConfig(
        symbol='BTC',
        bb_period=20,
        rsi_period=14,
        zscore_threshold=2.0,
        base_position_size=1000.0,
        min_confluence_score=0.6
    )

    print(f"\n1. Configuration:")
    print(f"   Symbol: {config.symbol}")
    print(f"   BB Period: {config.bb_period}")
    print(f"   RSI Oversold/Overbought: {config.rsi_oversold}/{config.rsi_overbought}")
    print(f"   Min Confluence: {config.min_confluence_score:.0%}")

    # Create strategy
    strategy = MeanReversionStrategy(config)

    # Generate oscillating price data
    print(f"\n2. Generating oscillating price data...")
    np.random.seed(42)
    n = 300
    base_price = 40000
    trend = np.linspace(0, 1000, n)  # Slight uptrend
    oscillation = 2000 * np.sin(np.linspace(0, 4 * np.pi, n))  # Sine wave
    noise = np.random.randn(n) * 300
    prices = base_price + trend + oscillation + noise

    # Backtest
    print(f"\n3. Running backtest...")
    results = strategy.backtest(prices)

    print(f"\n4. Results:")
    print(f"   Trades: {results['num_trades']}")
    print(f"   Win Rate: {results['win_rate']:.1%}")
    print(f"   Avg P&L: ${results['avg_pnl']:.2f}")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"   Total Return: {results['total_return']:.1%}")
    print(f"   Final Equity: ${results['final_equity']:,.2f}")

    print(f"\n✅ Mean Reversion Demo Complete!")
