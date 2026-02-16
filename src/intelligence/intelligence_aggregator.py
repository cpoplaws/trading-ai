"""
Intelligence Aggregator
Combines multiple intelligence sources into actionable trading signals.

Sources:
- Market regime detection
- News sentiment
- Social media sentiment
- Macro economic data
- Technical indicators

Outputs:
- Composite intelligence score
- Trading signals
- Risk alerts
- Opportunity identification
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength levels."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class IntelligenceSource(Enum):
    """Intelligence data sources."""
    REGIME = "regime"
    NEWS = "news"
    SOCIAL = "social"
    MACRO = "macro"
    TECHNICAL = "technical"


@dataclass
class IntelligenceSignal:
    """Aggregated intelligence signal."""
    symbol: str
    signal: SignalStrength
    confidence: float  # 0.0 to 1.0
    score: float  # -1.0 (bearish) to +1.0 (bullish)

    # Component scores
    regime_score: float = 0.0
    sentiment_score: float = 0.0
    macro_score: float = 0.0
    technical_score: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    sources: Dict[str, Dict] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)


class IntelligenceAggregator:
    """
    Aggregates multiple intelligence sources for trading decisions.

    Combines:
    1. Market regime detection
    2. News sentiment
    3. Social media sentiment
    4. Macro economic indicators
    5. Technical analysis

    Outputs actionable signals with confidence scores.
    """

    def __init__(
        self,
        regime_weight: float = 0.3,
        sentiment_weight: float = 0.25,
        macro_weight: float = 0.25,
        technical_weight: float = 0.2
    ):
        """
        Initialize intelligence aggregator.

        Args:
            regime_weight: Weight for market regime (default 0.3)
            sentiment_weight: Weight for sentiment analysis (default 0.25)
            macro_weight: Weight for macro indicators (default 0.25)
            technical_weight: Weight for technical analysis (default 0.2)
        """
        self.weights = {
            'regime': regime_weight,
            'sentiment': sentiment_weight,
            'macro': macro_weight,
            'technical': technical_weight
        }

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

        logger.info(f"IntelligenceAggregator initialized with weights: {self.weights}")

    def aggregate(
        self,
        symbol: str,
        regime_data: Optional[Dict] = None,
        sentiment_data: Optional[Dict] = None,
        macro_data: Optional[Dict] = None,
        technical_data: Optional[Dict] = None
    ) -> IntelligenceSignal:
        """
        Aggregate intelligence from multiple sources.

        Args:
            symbol: Trading symbol
            regime_data: Market regime information
            sentiment_data: Sentiment analysis data
            macro_data: Macro economic data
            technical_data: Technical indicators

        Returns:
            IntelligenceSignal with aggregated intelligence
        """
        # Calculate component scores (-1 to +1)
        regime_score = self._score_regime(regime_data) if regime_data else 0.0
        sentiment_score = self._score_sentiment(sentiment_data) if sentiment_data else 0.0
        macro_score = self._score_macro(macro_data) if macro_data else 0.0
        technical_score = self._score_technical(technical_data) if technical_data else 0.0

        # Calculate weighted composite score
        composite_score = (
            regime_score * self.weights['regime'] +
            sentiment_score * self.weights['sentiment'] +
            macro_score * self.weights['macro'] +
            technical_score * self.weights['technical']
        )

        # Determine signal strength
        signal = self._score_to_signal(composite_score)

        # Calculate confidence based on agreement between sources
        confidence = self._calculate_confidence(
            regime_score, sentiment_score, macro_score, technical_score
        )

        # Identify alerts and opportunities
        alerts = self._identify_alerts(
            regime_data, sentiment_data, macro_data, technical_data
        )
        opportunities = self._identify_opportunities(
            composite_score, confidence, regime_data, sentiment_data
        )

        # Create intelligence signal
        intelligence = IntelligenceSignal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            score=composite_score,
            regime_score=regime_score,
            sentiment_score=sentiment_score,
            macro_score=macro_score,
            technical_score=technical_score,
            sources={
                'regime': regime_data or {},
                'sentiment': sentiment_data or {},
                'macro': macro_data or {},
                'technical': technical_data or {}
            },
            alerts=alerts,
            opportunities=opportunities
        )

        logger.info(
            f"Intelligence for {symbol}: {signal.value} "
            f"(score: {composite_score:.3f}, confidence: {confidence:.2f})"
        )

        return intelligence

    def _score_regime(self, regime_data: Dict) -> float:
        """Score market regime (-1 to +1)."""
        regime_type = regime_data.get('regime', '').lower()
        confidence = regime_data.get('confidence', 0.5)

        # Map regimes to scores
        regime_scores = {
            'bull_trend': 0.8,
            'trending': 0.6 if regime_data.get('momentum', 0) > 0 else -0.6,
            'mean_reverting': 0.0,  # Neutral for mean reversion
            'low_volatility': 0.3,  # Slightly bullish (stability)
            'high_volatility': -0.3,  # Slightly bearish (risk)
            'bear_trend': -0.8,
            'unknown': 0.0
        }

        base_score = regime_scores.get(regime_type, 0.0)
        return base_score * confidence

    def _score_sentiment(self, sentiment_data: Dict) -> float:
        """Score sentiment analysis (-1 to +1)."""
        news_sentiment = sentiment_data.get('news_sentiment', 0.0)
        social_sentiment = sentiment_data.get('social_sentiment', 0.0)
        volume = sentiment_data.get('mention_volume', 1.0)

        # Weight news more heavily, but consider social sentiment
        sentiment_score = (news_sentiment * 0.6 + social_sentiment * 0.4)

        # Amplify if high mention volume
        volume_multiplier = min(1.0 + (volume - 1.0) * 0.2, 1.5)
        return sentiment_score * volume_multiplier

    def _score_macro(self, macro_data: Dict) -> float:
        """Score macro economic indicators (-1 to +1)."""
        indicators = []

        # Interest rates (lower is bullish for equities)
        if 'interest_rate_trend' in macro_data:
            rate_trend = macro_data['interest_rate_trend']
            indicators.append(-rate_trend * 0.3)  # Inverse relationship

        # Inflation (moderate is good, extremes are bad)
        if 'inflation_rate' in macro_data:
            inflation = macro_data['inflation_rate']
            if 2.0 <= inflation <= 3.0:
                indicators.append(0.5)  # Goldilocks
            elif inflation < 1.0 or inflation > 5.0:
                indicators.append(-0.5)  # Too low or too high
            else:
                indicators.append(0.0)  # Neutral

        # GDP growth (higher is bullish)
        if 'gdp_growth' in macro_data:
            gdp = macro_data['gdp_growth']
            indicators.append(min(max(gdp / 5.0, -1.0), 1.0))

        # Unemployment (lower is bullish)
        if 'unemployment_trend' in macro_data:
            unemployment = macro_data['unemployment_trend']
            indicators.append(-unemployment * 0.5)

        # Market sentiment (VIX, etc.)
        if 'vix_level' in macro_data:
            vix = macro_data['vix_level']
            if vix < 15:
                indicators.append(0.4)  # Low fear = bullish
            elif vix > 30:
                indicators.append(-0.6)  # High fear = bearish
            else:
                indicators.append(0.0)

        return sum(indicators) / len(indicators) if indicators else 0.0

    def _score_technical(self, technical_data: Dict) -> float:
        """Score technical indicators (-1 to +1)."""
        scores = []

        # RSI
        if 'rsi' in technical_data:
            rsi = technical_data['rsi']
            if rsi < 30:
                scores.append(0.7)  # Oversold = bullish
            elif rsi > 70:
                scores.append(-0.7)  # Overbought = bearish
            elif 40 <= rsi <= 60:
                scores.append(0.2)  # Neutral-bullish
            else:
                scores.append(0.0)

        # MACD
        if 'macd_signal' in technical_data:
            macd = technical_data['macd_signal']
            scores.append(0.6 if macd == 'bullish' else -0.6)

        # Moving average crossover
        if 'ma_crossover' in technical_data:
            ma = technical_data['ma_crossover']
            if ma == 'golden_cross':
                scores.append(0.8)
            elif ma == 'death_cross':
                scores.append(-0.8)

        # Price vs moving averages
        if 'price_vs_ma50' in technical_data:
            pct = technical_data['price_vs_ma50']
            scores.append(min(max(pct * 2, -1.0), 1.0))

        # Volume
        if 'volume_trend' in technical_data:
            vol_trend = technical_data['volume_trend']
            # Increasing volume with price = bullish
            # Increasing volume against price = bearish
            price_trend = technical_data.get('price_trend', 0)
            scores.append(vol_trend * price_trend * 0.3)

        return sum(scores) / len(scores) if scores else 0.0

    def _score_to_signal(self, score: float) -> SignalStrength:
        """Convert composite score to signal strength."""
        if score > 0.6:
            return SignalStrength.STRONG_BUY
        elif score > 0.2:
            return SignalStrength.BUY
        elif score < -0.6:
            return SignalStrength.STRONG_SELL
        elif score < -0.2:
            return SignalStrength.SELL
        else:
            return SignalStrength.NEUTRAL

    def _calculate_confidence(
        self,
        regime_score: float,
        sentiment_score: float,
        macro_score: float,
        technical_score: float
    ) -> float:
        """Calculate confidence based on agreement between sources."""
        scores = [regime_score, sentiment_score, macro_score, technical_score]
        scores = [s for s in scores if s != 0.0]  # Remove missing sources

        if not scores:
            return 0.0

        # Calculate agreement (low std dev = high agreement)
        import numpy as np
        std_dev = np.std(scores)
        mean_abs = np.mean(np.abs(scores))

        # High confidence when:
        # 1. Low disagreement (low std dev)
        # 2. Strong signals (high mean absolute value)
        agreement = 1.0 - min(std_dev / 2.0, 1.0)
        strength = min(mean_abs, 1.0)

        confidence = (agreement * 0.6 + strength * 0.4)
        return min(max(confidence, 0.0), 1.0)

    def _identify_alerts(
        self,
        regime_data: Optional[Dict],
        sentiment_data: Optional[Dict],
        macro_data: Optional[Dict],
        technical_data: Optional[Dict]
    ) -> List[str]:
        """Identify risk alerts."""
        alerts = []

        # Regime alerts
        if regime_data:
            if regime_data.get('regime') == 'high_volatility':
                alerts.append("⚠️ High volatility detected - reduce position sizes")
            if regime_data.get('regime') == 'bear_trend':
                alerts.append("📉 Bear trend - consider defensive positioning")

        # Sentiment alerts
        if sentiment_data:
            if sentiment_data.get('news_sentiment', 0) < -0.7:
                alerts.append("📰 Strongly negative news sentiment")
            if sentiment_data.get('mention_volume', 0) > 3.0:
                alerts.append("📢 Unusually high mention volume - possible hype")

        # Macro alerts
        if macro_data:
            if macro_data.get('vix_level', 0) > 30:
                alerts.append("😰 VIX above 30 - market fear elevated")
            if macro_data.get('inflation_rate', 0) > 5.0:
                alerts.append("📈 High inflation - potential policy changes")

        # Technical alerts
        if technical_data:
            if technical_data.get('rsi', 50) > 80:
                alerts.append("🔴 RSI > 80 - extremely overbought")
            elif technical_data.get('rsi', 50) < 20:
                alerts.append("🟢 RSI < 20 - extremely oversold")

        return alerts

    def _identify_opportunities(
        self,
        composite_score: float,
        confidence: float,
        regime_data: Optional[Dict],
        sentiment_data: Optional[Dict]
    ) -> List[str]:
        """Identify trading opportunities."""
        opportunities = []

        # High-confidence signals
        if abs(composite_score) > 0.6 and confidence > 0.7:
            direction = "Long" if composite_score > 0 else "Short"
            opportunities.append(
                f"💡 High-confidence {direction} opportunity "
                f"(score: {composite_score:.2f}, confidence: {confidence:.2f})"
            )

        # Mean reversion opportunities
        if regime_data and regime_data.get('regime') == 'mean_reverting':
            if regime_data.get('z_score', 0) > 2:
                opportunities.append("↩️ Mean reversion opportunity - price extended")

        # Sentiment divergence
        if sentiment_data:
            news_sent = sentiment_data.get('news_sentiment', 0)
            social_sent = sentiment_data.get('social_sentiment', 0)
            if abs(news_sent - social_sent) > 0.5:
                opportunities.append("🔄 Sentiment divergence detected")

        # Trend following
        if regime_data and regime_data.get('regime') in ['bull_trend', 'bear_trend']:
            if regime_data.get('confidence', 0) > 0.7:
                opportunities.append(
                    f"📊 Strong {regime_data['regime'].replace('_', ' ')} - "
                    "consider trend following"
                )

        return opportunities
