"""
Intelligence Service
Aggregates market intelligence from multiple sources for trading decisions
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class SignalStrength(Enum):
    """Signal strength levels"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class IntelligenceService:
    """
    Aggregates intelligence from multiple sources:
    - Market regime detection
    - Sentiment analysis
    - Macro indicators
    - Technical analysis
    """

    def __init__(self):
        self.last_update = None
        self.current_intelligence = None

        # Component weights
        self.weights = {
            'regime': 0.35,      # Market regime most important
            'sentiment': 0.25,   # Sentiment analysis
            'macro': 0.20,       # Macro indicators
            'technical': 0.20    # Technical signals
        }

        logger.info("IntelligenceService initialized")

    def analyze(self, market_data: Dict) -> Dict:
        """
        Analyze market and generate intelligence report

        Args:
            market_data: Dictionary with price history, volume, etc.

        Returns:
            Intelligence report with regime, sentiment, signals, etc.
        """
        # Extract data
        prices = market_data.get("prices", [])
        volumes = market_data.get("volumes", [])

        if len(prices) < 20:
            return self._insufficient_data_response()

        prices_array = np.array(prices)
        volumes_array = np.array(volumes) if volumes else np.ones(len(prices))

        # 1. Detect market regime
        regime_data = self._detect_regime(prices_array)

        # 2. Analyze sentiment (simplified)
        sentiment_data = self._analyze_sentiment(prices_array)

        # 3. Macro indicators (simplified)
        macro_data = self._analyze_macro(prices_array)

        # 4. Technical analysis
        technical_data = self._analyze_technical(prices_array)

        # 5. Aggregate all sources
        intelligence = self._aggregate_intelligence(
            regime_data,
            sentiment_data,
            macro_data,
            technical_data
        )

        # 6. Generate recommendations
        intelligence["recommendations"] = self._generate_recommendations(intelligence)

        # 7. Identify alerts
        intelligence["alerts"] = self._identify_alerts(intelligence)

        self.current_intelligence = intelligence
        self.last_update = datetime.now()

        return intelligence

    def _detect_regime(self, prices: np.ndarray) -> Dict:
        """Detect current market regime"""
        # Calculate indicators
        returns = np.diff(prices) / prices[:-1]

        # Moving averages
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
        current_price = prices[-1]

        # Trend
        price_vs_sma20 = (current_price - sma_20) / sma_20
        price_vs_sma50 = (current_price - sma_50) / sma_50
        sma20_vs_sma50 = (sma_20 - sma_50) / sma_50

        # Volatility
        volatility = np.std(returns[-20:])

        # Momentum
        momentum = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0

        # Determine regime
        regime = MarketRegime.UNKNOWN
        confidence = 0.5

        # Bull trend: price > SMA20 > SMA50, positive momentum
        if price_vs_sma20 > 0.02 and sma20_vs_sma50 > 0.01 and momentum > 0.05:
            regime = MarketRegime.BULL_TREND
            confidence = min(0.9, 0.6 + abs(momentum) * 2)

        # Bear trend: price < SMA20 < SMA50, negative momentum
        elif price_vs_sma20 < -0.02 and sma20_vs_sma50 < -0.01 and momentum < -0.05:
            regime = MarketRegime.BEAR_TREND
            confidence = min(0.9, 0.6 + abs(momentum) * 2)

        # High volatility
        elif volatility > 0.03:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(volatility * 20, 0.9)

        # Low volatility
        elif volatility < 0.01:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = 0.7

        # Sideways
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.6

        return {
            "regime": regime.value,
            "confidence": confidence,
            "price_vs_sma20": price_vs_sma20,
            "price_vs_sma50": price_vs_sma50,
            "momentum": momentum,
            "volatility": volatility,
            "trend_strength": abs(momentum)
        }

    def _analyze_sentiment(self, prices: np.ndarray) -> Dict:
        """
        Analyze market sentiment
        (Simplified version - uses price action as proxy)
        """
        # Recent price action as sentiment proxy
        returns = np.diff(prices) / prices[:-1]
        recent_returns = returns[-10:]

        # Positive vs negative days
        positive_days = np.sum(recent_returns > 0)
        total_days = len(recent_returns)
        sentiment_ratio = positive_days / total_days if total_days > 0 else 0.5

        # Average return
        avg_return = np.mean(recent_returns)

        # Sentiment score (-1 to +1)
        sentiment_score = (sentiment_ratio - 0.5) * 2  # Map 0-1 to -1 to +1
        sentiment_score += avg_return * 10  # Adjust by return magnitude

        sentiment_score = np.clip(sentiment_score, -1.0, 1.0)

        # Classify sentiment
        if sentiment_score > 0.3:
            sentiment = "bullish"
        elif sentiment_score < -0.3:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "score": float(sentiment_score),
            "confidence": 0.6,
            "positive_days": int(positive_days),
            "total_days": int(total_days)
        }

    def _analyze_macro(self, prices: np.ndarray) -> Dict:
        """
        Analyze macro indicators
        (Simplified - uses trend and momentum)
        """
        # Long-term trend
        if len(prices) >= 100:
            long_term_return = (prices[-1] - prices[-100]) / prices[-100]
            trend = "expansion" if long_term_return > 0.1 else "contraction" if long_term_return < -0.1 else "stable"
        else:
            long_term_return = 0
            trend = "stable"

        # Score based on long-term trend
        macro_score = np.clip(long_term_return * 5, -1.0, 1.0)

        return {
            "trend": trend,
            "score": float(macro_score),
            "long_term_return": float(long_term_return),
            "confidence": 0.5
        }

    def _analyze_technical(self, prices: np.ndarray) -> Dict:
        """Technical analysis indicators"""
        # RSI
        returns = np.diff(prices) / prices[:-1]
        gains = returns.copy()
        losses = returns.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # MACD (simplified)
        ema_12 = np.mean(prices[-12:])
        ema_26 = np.mean(prices[-26:]) if len(prices) >= 26 else ema_12
        macd = ema_12 - ema_26

        # Bollinger Bands
        sma_20 = np.mean(prices[-20:])
        std_20 = np.std(prices[-20:])
        bb_upper = sma_20 + (2 * std_20)
        bb_lower = sma_20 - (2 * std_20)
        bb_position = (prices[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5

        # Technical score
        technical_score = 0.0

        # RSI component
        if rsi < 30:
            technical_score += 0.4  # Oversold
        elif rsi > 70:
            technical_score -= 0.4  # Overbought
        else:
            technical_score += (50 - rsi) / 50 * 0.2

        # MACD component
        if macd > 0:
            technical_score += 0.3
        else:
            technical_score -= 0.3

        # Bollinger Band component
        if bb_position < 0.2:
            technical_score += 0.3
        elif bb_position > 0.8:
            technical_score -= 0.3

        technical_score = np.clip(technical_score, -1.0, 1.0)

        return {
            "rsi": float(rsi),
            "macd": float(macd),
            "bb_position": float(bb_position),
            "score": float(technical_score),
            "confidence": 0.7
        }

    def _aggregate_intelligence(
        self,
        regime_data: Dict,
        sentiment_data: Dict,
        macro_data: Dict,
        technical_data: Dict
    ) -> Dict:
        """Aggregate all intelligence sources"""

        # Map regime to score
        regime_scores = {
            "bull_trend": 0.8,
            "bear_trend": -0.8,
            "high_volatility": -0.3,
            "low_volatility": 0.2,
            "sideways": 0.0,
            "unknown": 0.0
        }

        regime_score = regime_scores.get(regime_data["regime"], 0.0)
        regime_score *= regime_data["confidence"]

        # Get other scores
        sentiment_score = sentiment_data["score"]
        macro_score = macro_data["score"]
        technical_score = technical_data["score"]

        # Weighted composite score
        composite_score = (
            regime_score * self.weights['regime'] +
            sentiment_score * self.weights['sentiment'] +
            macro_score * self.weights['macro'] +
            technical_score * self.weights['technical']
        )

        # Confidence (agreement between sources)
        scores = [regime_score, sentiment_score, macro_score, technical_score]
        avg_score = np.mean(scores)
        disagreement = np.std(scores)
        confidence = max(0.3, 1.0 - disagreement)

        # Signal strength
        if composite_score > 0.6:
            signal = SignalStrength.STRONG_BUY.value
        elif composite_score > 0.3:
            signal = SignalStrength.BUY.value
        elif composite_score < -0.6:
            signal = SignalStrength.STRONG_SELL.value
        elif composite_score < -0.3:
            signal = SignalStrength.SELL.value
        else:
            signal = SignalStrength.NEUTRAL.value

        return {
            "signal": signal,
            "composite_score": float(composite_score),
            "confidence": float(confidence),
            "regime": regime_data,
            "sentiment": sentiment_data,
            "macro": macro_data,
            "technical": technical_data,
            "timestamp": datetime.now().isoformat()
        }

    def _generate_recommendations(self, intelligence: Dict) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []

        regime = intelligence["regime"]["regime"]
        signal = intelligence["signal"]
        confidence = intelligence["confidence"]

        # Regime-based recommendations
        if regime == "bull_trend":
            recommendations.append("🐂 Bull market: Favor long positions and momentum strategies")
        elif regime == "bear_trend":
            recommendations.append("🐻 Bear market: Consider defensive positions or short strategies")
        elif regime == "high_volatility":
            recommendations.append("⚠️ High volatility: Reduce position sizes and use wider stops")
        elif regime == "low_volatility":
            recommendations.append("💤 Low volatility: Consider range-trading and mean reversion")
        elif regime == "sideways":
            recommendations.append("↔️ Sideways market: Mean reversion strategies may work well")

        # Signal-based recommendations
        if signal == "strong_buy" and confidence > 0.7:
            recommendations.append("💪 Strong buy signal with high confidence - good entry opportunity")
        elif signal == "strong_sell" and confidence > 0.7:
            recommendations.append("⚠️ Strong sell signal - consider reducing exposure")
        elif confidence < 0.5:
            recommendations.append("🤷 Low confidence - wait for clearer signals")

        # Technical recommendations
        rsi = intelligence["technical"]["rsi"]
        if rsi < 30:
            recommendations.append(f"📉 RSI at {rsi:.0f}: Oversold conditions")
        elif rsi > 70:
            recommendations.append(f"📈 RSI at {rsi:.0f}: Overbought conditions")

        return recommendations

    def _identify_alerts(self, intelligence: Dict) -> List[Dict]:
        """Identify risk alerts and opportunities"""
        alerts = []

        regime = intelligence["regime"]
        technical = intelligence["technical"]
        confidence = intelligence["confidence"]

        # High volatility alert
        if regime["regime"] == "high_volatility":
            alerts.append({
                "type": "warning",
                "message": f"High volatility detected ({regime['volatility']*100:.2f}%)",
                "severity": "high"
            })

        # Extreme RSI
        rsi = technical["rsi"]
        if rsi > 80:
            alerts.append({
                "type": "opportunity",
                "message": f"Extremely overbought (RSI: {rsi:.0f}) - potential reversal",
                "severity": "medium"
            })
        elif rsi < 20:
            alerts.append({
                "type": "opportunity",
                "message": f"Extremely oversold (RSI: {rsi:.0f}) - potential bounce",
                "severity": "medium"
            })

        # Strong directional signal
        if intelligence["signal"] in ["strong_buy", "strong_sell"] and confidence > 0.8:
            alerts.append({
                "type": "opportunity",
                "message": f"High-confidence {intelligence['signal'].replace('_', ' ')} signal",
                "severity": "high"
            })

        return alerts

    def _insufficient_data_response(self) -> Dict:
        """Return response when insufficient data"""
        return {
            "signal": "neutral",
            "composite_score": 0.0,
            "confidence": 0.0,
            "regime": {"regime": "unknown", "confidence": 0.0},
            "sentiment": {"sentiment": "neutral", "score": 0.0},
            "macro": {"trend": "unknown", "score": 0.0},
            "technical": {"rsi": 50.0, "score": 0.0},
            "recommendations": ["Insufficient data for analysis"],
            "alerts": [],
            "timestamp": datetime.now().isoformat()
        }

    def get_current_intelligence(self) -> Optional[Dict]:
        """Get most recent intelligence report"""
        return self.current_intelligence


# Global intelligence service instance
intelligence_service: Optional[IntelligenceService] = None


def get_intelligence_service() -> IntelligenceService:
    """Get or create global intelligence service instance"""
    global intelligence_service
    if intelligence_service is None:
        intelligence_service = IntelligenceService()
    return intelligence_service
