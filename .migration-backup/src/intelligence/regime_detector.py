"""
Advanced Market Regime Detection
Uses multiple indicators and machine learning to detect market regimes.

Market Regimes:
- Bull Market (sustained uptrend)
- Bear Market (sustained downtrend)
- High Volatility (choppy, uncertain)
- Low Volatility (range-bound, stable)
- Trending (directional movement)
- Mean Reverting (oscillating around mean)
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    UNKNOWN = "unknown"


@dataclass
class RegimeSignal:
    """Signal about current market regime."""
    regime: MarketRegime
    confidence: float  # 0.0 to 1.0
    indicators: Dict[str, float]
    timestamp: datetime
    description: str


class RegimeDetector:
    """
    Advanced market regime detection using multiple indicators.

    Methods:
    1. Trend Analysis: Moving averages, ADX
    2. Volatility Analysis: ATR, Bollinger Bands
    3. Mean Reversion: Z-score, RSI
    4. Volume Analysis: Volume trends
    5. Composite Score: Weighted combination
    """

    def __init__(
        self,
        lookback_period: int = 100,
        volatility_threshold: float = 0.02,
        trend_threshold: float = 0.015
    ):
        """
        Initialize regime detector.

        Args:
            lookback_period: Historical period for analysis
            volatility_threshold: Threshold for high/low volatility
            trend_threshold: Threshold for trending behavior
        """
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold

        logger.info(f"RegimeDetector initialized (lookback: {lookback_period})")

    def detect_regime(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None
    ) -> RegimeSignal:
        """
        Detect current market regime.

        Args:
            prices: Price series (recent to oldest)
            volume: Volume series (optional)

        Returns:
            RegimeSignal with detected regime and confidence
        """
        if len(prices) < self.lookback_period:
            logger.warning(f"Insufficient data: {len(prices)} < {self.lookback_period}")
            return RegimeSignal(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                indicators={},
                timestamp=datetime.now(),
                description="Insufficient data for regime detection"
            )

        # Calculate indicators
        indicators = self._calculate_indicators(prices, volume)

        # Detect regime using multiple methods
        trend_regime, trend_confidence = self._detect_trend(indicators)
        volatility_regime, vol_confidence = self._detect_volatility(indicators)
        mean_reversion_score = self._detect_mean_reversion(indicators)

        # Combine signals for final regime
        regime, confidence = self._combine_signals(
            trend_regime,
            trend_confidence,
            volatility_regime,
            vol_confidence,
            mean_reversion_score,
            indicators
        )

        # Generate description
        description = self._generate_description(regime, indicators)

        return RegimeSignal(
            regime=regime,
            confidence=confidence,
            indicators=indicators,
            timestamp=datetime.now(),
            description=description
        )

    def _calculate_indicators(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Calculate all indicators for regime detection."""
        indicators = {}

        # Price-based indicators
        returns = prices.pct_change().dropna()

        # 1. Moving averages
        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1]
        sma_100 = prices.rolling(100).mean().iloc[-1] if len(prices) >= 100 else sma_50
        current_price = prices.iloc[-1]

        indicators['sma_20'] = sma_20
        indicators['sma_50'] = sma_50
        indicators['sma_100'] = sma_100
        indicators['price_vs_sma20'] = (current_price - sma_20) / sma_20
        indicators['price_vs_sma50'] = (current_price - sma_50) / sma_50
        indicators['sma20_vs_sma50'] = (sma_20 - sma_50) / sma_50

        # 2. Trend strength (ADX approximation)
        high_low_range = (prices.rolling(14).max() - prices.rolling(14).min()).iloc[-1]
        atr = returns.abs().rolling(14).mean().iloc[-1]
        indicators['trend_strength'] = high_low_range / current_price if current_price > 0 else 0

        # 3. Volatility
        volatility = returns.std()
        indicators['volatility'] = volatility
        indicators['volatility_percentile'] = self._percentile_rank(
            returns.rolling(20).std().dropna(),
            volatility
        )

        # 4. ATR (Average True Range)
        indicators['atr'] = atr

        # 5. Bollinger Bands
        bb_middle = prices.rolling(20).mean().iloc[-1]
        bb_std = prices.rolling(20).std().iloc[-1]
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0

        indicators['bb_width'] = bb_width
        indicators['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5

        # 6. RSI (Relative Strength Index)
        rsi = self._calculate_rsi(returns)
        indicators['rsi'] = rsi

        # 7. Z-score (mean reversion indicator)
        z_score = (current_price - prices.rolling(20).mean().iloc[-1]) / prices.rolling(20).std().iloc[-1]
        indicators['z_score'] = z_score if not np.isnan(z_score) else 0

        # 8. Momentum
        momentum_10 = (current_price - prices.iloc[-10]) / prices.iloc[-10] if len(prices) >= 10 else 0
        momentum_20 = (current_price - prices.iloc[-20]) / prices.iloc[-20] if len(prices) >= 20 else 0
        indicators['momentum_10'] = momentum_10
        indicators['momentum_20'] = momentum_20

        # 9. Volume (if available)
        if volume is not None and len(volume) > 0:
            vol_sma = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            indicators['volume_ratio'] = current_volume / vol_sma if vol_sma > 0 else 1.0
        else:
            indicators['volume_ratio'] = 1.0

        return indicators

    def _detect_trend(self, indicators: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """Detect trend regime and confidence."""
        # Analyze moving average alignment
        sma_alignment = indicators['sma20_vs_sma50']
        price_position = indicators['price_vs_sma50']
        trend_strength = indicators['trend_strength']
        momentum = indicators['momentum_20']

        # Bull trend criteria
        bull_score = 0.0
        if sma_alignment > 0.01:  # SMA20 > SMA50
            bull_score += 0.3
        if price_position > 0.02:  # Price well above SMA50
            bull_score += 0.3
        if momentum > 0.05:  # Strong positive momentum
            bull_score += 0.2
        if trend_strength > self.trend_threshold:
            bull_score += 0.2

        # Bear trend criteria
        bear_score = 0.0
        if sma_alignment < -0.01:  # SMA20 < SMA50
            bear_score += 0.3
        if price_position < -0.02:  # Price well below SMA50
            bear_score += 0.3
        if momentum < -0.05:  # Strong negative momentum
            bear_score += 0.2
        if trend_strength > self.trend_threshold:
            bear_score += 0.2

        # Determine regime
        if bull_score > bear_score and bull_score > 0.5:
            return MarketRegime.BULL_TREND, bull_score
        elif bear_score > bull_score and bear_score > 0.5:
            return MarketRegime.BEAR_TREND, bear_score
        elif trend_strength > self.trend_threshold:
            return MarketRegime.TRENDING, max(bull_score, bear_score)
        else:
            return MarketRegime.MEAN_REVERTING, 0.6

    def _detect_volatility(self, indicators: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """Detect volatility regime and confidence."""
        volatility = indicators['volatility']
        vol_percentile = indicators['volatility_percentile']
        bb_width = indicators['bb_width']

        # High volatility criteria
        high_vol_score = 0.0
        if volatility > self.volatility_threshold:
            high_vol_score += 0.4
        if vol_percentile > 0.7:  # Volatility in top 30%
            high_vol_score += 0.3
        if bb_width > 0.1:  # Wide Bollinger Bands
            high_vol_score += 0.3

        # Low volatility criteria
        low_vol_score = 0.0
        if volatility < self.volatility_threshold / 2:
            low_vol_score += 0.4
        if vol_percentile < 0.3:  # Volatility in bottom 30%
            low_vol_score += 0.3
        if bb_width < 0.05:  # Narrow Bollinger Bands
            low_vol_score += 0.3

        # Determine regime
        if high_vol_score > low_vol_score and high_vol_score > 0.6:
            return MarketRegime.HIGH_VOLATILITY, high_vol_score
        elif low_vol_score > high_vol_score and low_vol_score > 0.6:
            return MarketRegime.LOW_VOLATILITY, low_vol_score
        else:
            # Moderate volatility
            return MarketRegime.TRENDING, 0.5

    def _detect_mean_reversion(self, indicators: Dict[str, float]) -> float:
        """Calculate mean reversion score (0 to 1)."""
        z_score = abs(indicators['z_score'])
        rsi = indicators['rsi']
        bb_position = indicators['bb_position']

        mean_reversion_score = 0.0

        # High Z-score suggests mean reversion opportunity
        if z_score > 2.0:
            mean_reversion_score += 0.4
        elif z_score > 1.5:
            mean_reversion_score += 0.2

        # RSI extremes suggest mean reversion
        if rsi > 70 or rsi < 30:
            mean_reversion_score += 0.3

        # Bollinger Band extremes
        if bb_position > 0.9 or bb_position < 0.1:
            mean_reversion_score += 0.3

        return min(mean_reversion_score, 1.0)

    def _combine_signals(
        self,
        trend_regime: MarketRegime,
        trend_confidence: float,
        volatility_regime: MarketRegime,
        vol_confidence: float,
        mean_reversion_score: float,
        indicators: Dict[str, float]
    ) -> Tuple[MarketRegime, float]:
        """Combine all signals for final regime determination."""
        # Weight different signals
        trend_weight = 0.5
        volatility_weight = 0.3
        mean_reversion_weight = 0.2

        # If strong mean reversion signal
        if mean_reversion_score > 0.7:
            return MarketRegime.MEAN_REVERTING, mean_reversion_score

        # If high volatility dominates
        if volatility_regime == MarketRegime.HIGH_VOLATILITY and vol_confidence > 0.7:
            return MarketRegime.HIGH_VOLATILITY, vol_confidence

        # If low volatility dominates
        if volatility_regime == MarketRegime.LOW_VOLATILITY and vol_confidence > 0.7:
            return MarketRegime.LOW_VOLATILITY, vol_confidence

        # Default to trend regime
        return trend_regime, trend_confidence

    def _calculate_rsi(self, returns: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        gains = returns.clip(lower=0).rolling(period).mean().iloc[-1]
        losses = (-returns.clip(upper=0)).rolling(period).mean().iloc[-1]

        if losses == 0:
            return 100.0
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        return rsi if not np.isnan(rsi) else 50.0

    def _percentile_rank(self, series: pd.Series, value: float) -> float:
        """Calculate percentile rank of value in series."""
        if len(series) == 0:
            return 0.5
        return (series < value).sum() / len(series)

    def _generate_description(self, regime: MarketRegime, indicators: Dict[str, float]) -> str:
        """Generate human-readable description of regime."""
        descriptions = {
            MarketRegime.BULL_TREND: (
                f"Strong uptrend detected. Price {indicators['price_vs_sma50']*100:.1f}% "
                f"above 50-day average. Momentum: {indicators['momentum_20']*100:.1f}%"
            ),
            MarketRegime.BEAR_TREND: (
                f"Strong downtrend detected. Price {indicators['price_vs_sma50']*100:.1f}% "
                f"below 50-day average. Momentum: {indicators['momentum_20']*100:.1f}%"
            ),
            MarketRegime.HIGH_VOLATILITY: (
                f"High volatility environment. Volatility at {indicators['volatility_percentile']*100:.0f}th "
                f"percentile. Bollinger Band width: {indicators['bb_width']*100:.1f}%"
            ),
            MarketRegime.LOW_VOLATILITY: (
                f"Low volatility, range-bound market. Volatility at {indicators['volatility_percentile']*100:.0f}th "
                f"percentile. Bollinger Band width: {indicators['bb_width']*100:.1f}%"
            ),
            MarketRegime.TRENDING: (
                f"Trending market with trend strength {indicators['trend_strength']*100:.1f}%. "
                f"Direction: {'Up' if indicators['momentum_20'] > 0 else 'Down'}"
            ),
            MarketRegime.MEAN_REVERTING: (
                f"Mean reverting conditions. Z-score: {indicators['z_score']:.2f}, "
                f"RSI: {indicators['rsi']:.0f}"
            ),
            MarketRegime.UNKNOWN: "Insufficient data for regime classification"
        }

        return descriptions.get(regime, "Unknown regime")

    def get_trading_recommendations(self, regime_signal: RegimeSignal) -> Dict[str, str]:
        """Get trading recommendations based on regime."""
        recommendations = {
            MarketRegime.BULL_TREND: {
                "strategy": "Momentum, Trend Following",
                "position": "Long bias",
                "risk": "Medium",
                "avoid": "Short positions, mean reversion"
            },
            MarketRegime.BEAR_TREND: {
                "strategy": "Short selling, Defensive",
                "position": "Short bias or cash",
                "risk": "Medium-High",
                "avoid": "Long positions, momentum longs"
            },
            MarketRegime.HIGH_VOLATILITY: {
                "strategy": "Options, Range trading",
                "position": "Reduced size",
                "risk": "High",
                "avoid": "Large positions, tight stops"
            },
            MarketRegime.LOW_VOLATILITY: {
                "strategy": "Carry trades, Income strategies",
                "position": "Normal to increased size",
                "risk": "Low",
                "avoid": "Breakout trades"
            },
            MarketRegime.TRENDING: {
                "strategy": "Trend following, Momentum",
                "position": "Follow the trend",
                "risk": "Medium",
                "avoid": "Counter-trend trades"
            },
            MarketRegime.MEAN_REVERTING: {
                "strategy": "Mean reversion, RSI",
                "position": "Contrarian",
                "risk": "Medium",
                "avoid": "Momentum trades"
            }
        }

        return recommendations.get(
            regime_signal.regime,
            {"strategy": "Unknown", "position": "Neutral", "risk": "Unknown", "avoid": "All"}
        )
