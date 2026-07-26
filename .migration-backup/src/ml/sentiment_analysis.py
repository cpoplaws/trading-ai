"""
Sentiment Analysis Engine
Analyzes sentiment from social media, news, and on-chain data for trading signals.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re

logger = logging.getLogger(__name__)


class SentimentScore(Enum):
    """Sentiment classification."""
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"


class SourceType(Enum):
    """Data source types."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    TELEGRAM = "telegram"
    DISCORD = "discord"


@dataclass
class SentimentPost:
    """Social media post or news article."""
    post_id: str
    source: SourceType
    timestamp: datetime
    text: str
    author: str
    engagement: int  # likes, retweets, upvotes, etc.

    # Sentiment metrics
    sentiment_score: float = 0.0  # -1 (bearish) to +1 (bullish)
    confidence: float = 0.0

    # Metadata
    metadata: Dict = field(default_factory=dict)


@dataclass
class SentimentSignal:
    """Aggregated sentiment signal."""
    signal_id: str
    timestamp: datetime
    token: str

    # Sentiment metrics
    overall_score: float  # -1 to +1
    sentiment_class: SentimentScore
    confidence: float

    # Volume metrics
    total_posts: int
    bullish_posts: int
    bearish_posts: int
    neutral_posts: int

    # Engagement
    total_engagement: int
    avg_engagement: float

    # Source breakdown
    source_scores: Dict[str, float] = field(default_factory=dict)

    # Trading recommendation
    action: str = "hold"  # buy, sell, hold
    strength: float = 0.0  # 0-1

    # Historical comparison
    sentiment_change_24h: float = 0.0
    trend: str = "stable"  # rising, falling, stable


class SimpleSentimentAnalyzer:
    """
    Simple rule-based sentiment analyzer.

    In production, would use transformer models like BERT or FinBERT.
    """

    # Bullish keywords and their weights
    BULLISH_KEYWORDS = {
        'moon': 2.0, 'bullish': 1.5, 'buy': 1.5, 'pump': 1.5,
        'rocket': 2.0, 'long': 1.0, 'profit': 1.0, 'gain': 1.0,
        'breakout': 1.5, 'rally': 1.5, 'surge': 1.5, 'soar': 1.5,
        'bounce': 1.0, 'recovery': 1.0, 'green': 0.5, 'up': 0.5,
        'bullrun': 2.0, 'accumulate': 1.0, 'hodl': 1.0, 'ath': 1.5,
        'undervalued': 1.5, 'gem': 1.5, '📈': 1.5, '🚀': 2.0,
        '🌙': 2.0, '💎': 1.5, 'calls': 1.0, 'bullmarket': 1.5
    }

    # Bearish keywords and their weights
    BEARISH_KEYWORDS = {
        'crash': 2.0, 'dump': 2.0, 'sell': 1.5, 'bearish': 1.5,
        'short': 1.5, 'loss': 1.5, 'drop': 1.0, 'fall': 1.0,
        'collapse': 2.0, 'plunge': 1.5, 'tank': 1.5, 'red': 0.5,
        'down': 0.5, 'panic': 1.5, 'fear': 1.0, 'scam': 2.0,
        'rug': 2.0, 'overvalued': 1.5, 'bubble': 1.5, 'ponzi': 2.0,
        '📉': 1.5, 'rekt': 1.5, 'liquidated': 1.5, 'bearmarket': 1.5,
        'bottom': 1.0, 'dip': 0.5, 'correction': 1.0
    }

    def __init__(self):
        """Initialize sentiment analyzer."""
        logger.info("Sentiment analyzer initialized")

    def analyze_text(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            (sentiment_score, confidence) tuple
            sentiment_score: -1 (bearish) to +1 (bullish)
            confidence: 0 to 1
        """
        text_lower = text.lower()

        # Count keyword matches
        bullish_score = 0.0
        bearish_score = 0.0

        for keyword, weight in self.BULLISH_KEYWORDS.items():
            if keyword in text_lower:
                bullish_score += weight

        for keyword, weight in self.BEARISH_KEYWORDS.items():
            if keyword in text_lower:
                bearish_score += weight

        # Calculate net sentiment
        total_score = bullish_score + bearish_score

        if total_score == 0:
            return 0.0, 0.0  # Neutral, no confidence

        # Normalize to -1 to +1
        net_sentiment = (bullish_score - bearish_score) / (bullish_score + bearish_score)

        # Confidence based on signal strength
        confidence = min(1.0, total_score / 5.0)  # More keywords = higher confidence

        return net_sentiment, confidence

    def analyze_post(self, post: SentimentPost) -> SentimentPost:
        """
        Analyze sentiment of a post.

        Args:
            post: Post to analyze

        Returns:
            Post with sentiment scores added
        """
        sentiment, confidence = self.analyze_text(post.text)

        # Adjust confidence based on engagement
        # High engagement posts are more reliable
        if post.engagement > 1000:
            confidence = min(1.0, confidence * 1.2)
        elif post.engagement < 10:
            confidence *= 0.8

        post.sentiment_score = sentiment
        post.confidence = confidence

        return post


class SentimentAggregator:
    """
    Aggregates sentiment from multiple posts into trading signals.
    """

    def __init__(self, window_hours: int = 24):
        """
        Initialize sentiment aggregator.

        Args:
            window_hours: Time window for aggregation
        """
        self.window_hours = window_hours
        self.analyzer = SimpleSentimentAnalyzer()
        logger.info(f"Sentiment aggregator initialized (window={window_hours}h)")

    def aggregate_sentiment(
        self,
        posts: List[SentimentPost],
        token: str
    ) -> SentimentSignal:
        """
        Aggregate sentiment from multiple posts.

        Args:
            posts: List of posts to aggregate
            token: Token symbol

        Returns:
            Aggregated sentiment signal
        """
        if not posts:
            return self._create_neutral_signal(token)

        # Filter recent posts
        cutoff_time = datetime.now() - timedelta(hours=self.window_hours)
        recent_posts = [p for p in posts if p.timestamp >= cutoff_time]

        if not recent_posts:
            return self._create_neutral_signal(token)

        # Analyze all posts
        analyzed_posts = [self.analyzer.analyze_post(p) for p in recent_posts]

        # Calculate weighted average sentiment
        # Weight by confidence and engagement
        total_weight = 0.0
        weighted_sum = 0.0

        for post in analyzed_posts:
            weight = post.confidence * (1 + min(1.0, post.engagement / 1000))
            weighted_sum += post.sentiment_score * weight
            total_weight += weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Count sentiment distribution
        bullish = sum(1 for p in analyzed_posts if p.sentiment_score > 0.2)
        bearish = sum(1 for p in analyzed_posts if p.sentiment_score < -0.2)
        neutral = len(analyzed_posts) - bullish - bearish

        # Total engagement
        total_engagement = sum(p.engagement for p in analyzed_posts)
        avg_engagement = total_engagement / len(analyzed_posts)

        # Classify sentiment
        sentiment_class = self._classify_sentiment(overall_score)

        # Source breakdown
        source_scores = {}
        for source in SourceType:
            source_posts = [p for p in analyzed_posts if p.source == source]
            if source_posts:
                source_scores[source.value] = sum(p.sentiment_score for p in source_posts) / len(source_posts)

        # Calculate confidence
        confidence = min(1.0, len(analyzed_posts) / 50.0)  # More posts = higher confidence

        # Generate trading recommendation
        action, strength = self._generate_recommendation(overall_score, confidence)

        signal = SentimentSignal(
            signal_id=f"SENT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            token=token,
            overall_score=overall_score,
            sentiment_class=sentiment_class,
            confidence=confidence,
            total_posts=len(analyzed_posts),
            bullish_posts=bullish,
            bearish_posts=bearish,
            neutral_posts=neutral,
            total_engagement=total_engagement,
            avg_engagement=avg_engagement,
            source_scores=source_scores,
            action=action,
            strength=strength,
            sentiment_change_24h=0.0,  # Would calculate from historical data
            trend="stable"
        )

        logger.info(
            f"Sentiment signal for {token}: {sentiment_class.value.upper()} "
            f"({overall_score:+.2f}) | Action: {action.upper()} | "
            f"Posts: {len(analyzed_posts)}"
        )

        return signal

    def _classify_sentiment(self, score: float) -> SentimentScore:
        """Classify sentiment score into category."""
        if score >= 0.5:
            return SentimentScore.VERY_BULLISH
        elif score >= 0.2:
            return SentimentScore.BULLISH
        elif score <= -0.5:
            return SentimentScore.VERY_BEARISH
        elif score <= -0.2:
            return SentimentScore.BEARISH
        else:
            return SentimentScore.NEUTRAL

    def _generate_recommendation(self, score: float, confidence: float) -> Tuple[str, float]:
        """
        Generate trading recommendation from sentiment.

        Args:
            score: Sentiment score
            confidence: Confidence level

        Returns:
            (action, strength) tuple
        """
        # Only recommend if confident
        if confidence < 0.3:
            return "hold", 0.0

        strength = abs(score) * confidence

        if score > 0.3 and confidence > 0.5:
            return "buy", strength
        elif score < -0.3 and confidence > 0.5:
            return "sell", strength
        else:
            return "hold", 0.0

    def _create_neutral_signal(self, token: str) -> SentimentSignal:
        """Create neutral signal when no data available."""
        return SentimentSignal(
            signal_id=f"SENT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            token=token,
            overall_score=0.0,
            sentiment_class=SentimentScore.NEUTRAL,
            confidence=0.0,
            total_posts=0,
            bullish_posts=0,
            bearish_posts=0,
            neutral_posts=0,
            total_engagement=0,
            avg_engagement=0.0,
            action="hold",
            strength=0.0
        )

    def compare_tokens(self, signals: List[SentimentSignal]) -> str:
        """
        Compare sentiment across multiple tokens.

        Args:
            signals: List of sentiment signals

        Returns:
            Formatted comparison table
        """
        if not signals:
            return "No signals to compare"

        # Sort by sentiment score
        signals_sorted = sorted(signals, key=lambda x: x.overall_score, reverse=True)

        table = f"""
╔══════════════════════════════════════════════════════════════╗
║              SENTIMENT COMPARISON                             ║
╠══════════════════════════════════════════════════════════════╣
"""

        for signal in signals_sorted:
            sentiment_emoji = self._get_sentiment_emoji(signal.sentiment_class)
            table += f"""║ {signal.token:<10} {sentiment_emoji} {signal.sentiment_class.value.upper():<20}               ║
║   Score: {signal.overall_score:>+6.2f}  |  Posts: {signal.total_posts:>5}  |  Action: {signal.action.upper():<4}  ║
╠══════════════════════════════════════════════════════════════╣
"""

        table += "╚══════════════════════════════════════════════════════════════╝\n"

        return table

    def _get_sentiment_emoji(self, sentiment: SentimentScore) -> str:
        """Get emoji for sentiment."""
        emoji_map = {
            SentimentScore.VERY_BULLISH: "🚀",
            SentimentScore.BULLISH: "📈",
            SentimentScore.NEUTRAL: "➡️",
            SentimentScore.BEARISH: "📉",
            SentimentScore.VERY_BEARISH: "💀"
        }
        return emoji_map.get(sentiment, "❓")


if __name__ == '__main__':
    import logging
    import random

    logging.basicConfig(level=logging.INFO)

    print("💬 Sentiment Analysis Demo")
    print("=" * 60)

    # Sample social media posts
    print("\n1. Sample Social Media Posts...")
    print("-" * 60)

    sample_posts = [
        "ETH to the moon! 🚀🚀 Bullish breakout incoming!",
        "Just bought more ETH, this dip is a gift",
        "ETH looking very bearish, might crash soon",
        "Selling all my ETH, this looks like a dump",
        "ETH holding strong at support, accumulation phase",
        "HODL ETH 💎 Long term bullish, don't panic sell",
        "ETH overvalued, correction coming",
        "Massive ETH rally starting, don't miss out! 🌙",
        "ETH to $10k! Bullrun is here!",
        "Taking profits on ETH, market looks shaky",
    ]

    posts = []
    for i, text in enumerate(sample_posts):
        post = SentimentPost(
            post_id=f"POST-{i+1:03d}",
            source=random.choice(list(SourceType)),
            timestamp=datetime.now() - timedelta(hours=random.randint(0, 23)),
            text=text,
            author=f"user_{i+1}",
            engagement=random.randint(10, 5000)
        )
        posts.append(post)

    print(f"Generated {len(posts)} sample posts")

    # Analyze individual posts
    print("\n2. Individual Post Analysis...")
    print("-" * 60)

    analyzer = SimpleSentimentAnalyzer()

    for i, post in enumerate(posts[:3], 1):  # Show first 3
        analyzed = analyzer.analyze_post(post)
        print(f"\nPost {i}: \"{analyzed.text}\"")
        print(f"  Sentiment: {analyzed.sentiment_score:+.2f}")
        print(f"  Confidence: {analyzed.confidence:.2f}")
        print(f"  Source: {analyzed.source.value}")
        print(f"  Engagement: {analyzed.engagement}")

    # Aggregate sentiment
    print("\n3. Aggregated Sentiment Signal...")
    print("-" * 60)

    aggregator = SentimentAggregator(window_hours=24)
    signal = aggregator.aggregate_sentiment(posts, "ETH")

    print(f"\nToken: {signal.token}")
    print(f"Overall Score: {signal.overall_score:+.2f}")
    print(f"Sentiment: {signal.sentiment_class.value.upper()}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"\nPost Distribution:")
    print(f"  Bullish: {signal.bullish_posts}")
    print(f"  Bearish: {signal.bearish_posts}")
    print(f"  Neutral: {signal.neutral_posts}")
    print(f"  Total: {signal.total_posts}")
    print(f"\nEngagement:")
    print(f"  Total: {signal.total_engagement:,}")
    print(f"  Average: {signal.avg_engagement:.0f}")
    print(f"\nTrading Recommendation:")
    print(f"  Action: {signal.action.upper()}")
    print(f"  Strength: {signal.strength:.2f}")

    # Source breakdown
    if signal.source_scores:
        print(f"\nSentiment by Source:")
        for source, score in signal.source_scores.items():
            print(f"  {source}: {score:+.2f}")

    # Multi-token comparison
    print("\n4. Multi-Token Sentiment Comparison...")
    print("-" * 60)

    # Simulate signals for multiple tokens
    btc_posts = [
        SentimentPost(
            post_id=f"BTC-{i}",
            source=SourceType.TWITTER,
            timestamp=datetime.now(),
            text=text,
            author=f"btc_user_{i}",
            engagement=random.randint(100, 10000)
        )
        for i, text in enumerate([
            "Bitcoin breaking ATH soon! 🚀",
            "BTC is the future, buy now!",
            "Bitcoin looking strong, bullish",
            "BTC to 100k! Moon time 🌙",
        ])
    ]

    sol_posts = [
        SentimentPost(
            post_id=f"SOL-{i}",
            source=SourceType.REDDIT,
            timestamp=datetime.now(),
            text=text,
            author=f"sol_user_{i}",
            engagement=random.randint(100, 10000)
        )
        for i, text in enumerate([
            "Solana network congestion again, bearish",
            "SOL dump incoming, sell now",
            "Solana overvalued, bubble about to pop",
        ])
    ]

    btc_signal = aggregator.aggregate_sentiment(btc_posts, "BTC")
    sol_signal = aggregator.aggregate_sentiment(sol_posts, "SOL")
    eth_signal = signal

    comparison = aggregator.compare_tokens([btc_signal, eth_signal, sol_signal])
    print(comparison)

    print("✅ Sentiment analysis demo complete!")
