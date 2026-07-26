"""
Liquidation Hunter Strategy
Hunt for liquidation cascades in perpetual futures markets for high returns.

Strategy:
- Monitor liquidation levels on perpetual futures
- Detect potential liquidation cascades
- Enter positions before liquidation events
- Exit after liquidation spike for quick profits
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from decimal import Decimal
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


@dataclass
class LiquidationCluster:
    """Represents a cluster of liquidations at a price level."""
    price: float
    liquidation_amount_usd: float
    leverage_avg: float
    side: str  # 'long' or 'short'
    urgency_score: float  # 0-1, higher = more urgent
    estimated_cascade: float  # Potential cascade size in USD


@dataclass
class LiquidationSignal:
    """Trading signal based on liquidation analysis."""
    symbol: str
    action: str  # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0-1
    liquidation_target: float
    estimated_profit_percent: float
    risk_reward_ratio: float


class LiquidationHunter:
    """
    Liquidation hunting strategy for perpetual futures.

    How it works:
    1. Monitor open interest and funding rates
    2. Identify liquidation price clusters
    3. Detect potential liquidation cascades
    4. Enter positions to profit from cascades
    5. Quick exits (typically < 1 hour hold time)

    Risk: High volatility, requires fast execution
    Reward: 2-10% profit per trade
    """

    def __init__(
        self,
        min_liquidation_size: float = 1_000_000,  # $1M minimum
        min_confidence: float = 0.7,
        max_leverage: float = 5.0
    ):
        """
        Initialize liquidation hunter.

        Args:
            min_liquidation_size: Minimum liquidation cluster size to consider
            min_confidence: Minimum confidence score
            max_leverage: Maximum leverage to use
        """
        self.min_liquidation_size = min_liquidation_size
        self.min_confidence = min_confidence
        self.max_leverage = max_leverage

        # Cache for liquidation data
        self.liquidation_clusters: Dict[str, List[LiquidationCluster]] = {}
        self.recent_liquidations: List[Dict] = []

        logger.info(f"Liquidation hunter initialized (min size: ${min_liquidation_size:,.0f})")

    def analyze_liquidation_levels(
        self,
        symbol: str,
        current_price: float,
        open_interest: float,
        long_short_ratio: float,
        funding_rate: float
    ) -> List[LiquidationCluster]:
        """
        Analyze potential liquidation levels.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            current_price: Current market price
            open_interest: Total open interest in USD
            long_short_ratio: Long/Short ratio
            funding_rate: Current funding rate

        Returns:
            List of liquidation clusters
        """
        clusters = []

        # Estimate liquidation levels based on leverage
        typical_leverages = [5, 10, 20, 50, 100]

        # Calculate liquidation prices for longs (price drops)
        long_oi = open_interest * (long_short_ratio / (1 + long_short_ratio))
        for leverage in typical_leverages:
            # Liquidation price = entry * (1 - 1/leverage - fees)
            liq_price = current_price * (1 - 1/leverage - 0.001)

            # Estimate how much OI would be liquidated
            liq_amount = long_oi * (1 / leverage) * 0.5  # Rough estimate

            if liq_amount >= self.min_liquidation_size:
                urgency = self._calculate_urgency(
                    current_price, liq_price, funding_rate, 'long'
                )

                # Estimate cascade potential
                cascade_size = liq_amount * 1.5  # Cascades amplify 50%

                cluster = LiquidationCluster(
                    price=liq_price,
                    liquidation_amount_usd=liq_amount,
                    leverage_avg=leverage,
                    side='long',
                    urgency_score=urgency,
                    estimated_cascade=cascade_size
                )
                clusters.append(cluster)

        # Calculate liquidation prices for shorts (price rises)
        short_oi = open_interest - long_oi
        for leverage in typical_leverages:
            # Liquidation price = entry * (1 + 1/leverage + fees)
            liq_price = current_price * (1 + 1/leverage + 0.001)

            liq_amount = short_oi * (1 / leverage) * 0.5

            if liq_amount >= self.min_liquidation_size:
                urgency = self._calculate_urgency(
                    current_price, liq_price, funding_rate, 'short'
                )

                cascade_size = liq_amount * 1.5

                cluster = LiquidationCluster(
                    price=liq_price,
                    liquidation_amount_usd=liq_amount,
                    leverage_avg=leverage,
                    side='short',
                    urgency_score=urgency,
                    estimated_cascade=cascade_size
                )
                clusters.append(cluster)

        # Sort by urgency
        clusters.sort(key=lambda c: c.urgency_score, reverse=True)

        self.liquidation_clusters[symbol] = clusters

        logger.info(f"Found {len(clusters)} liquidation clusters for {symbol}")

        return clusters

    def _calculate_urgency(
        self,
        current_price: float,
        liquidation_price: float,
        funding_rate: float,
        side: str
    ) -> float:
        """
        Calculate urgency score for a liquidation cluster.

        Factors:
        - Distance to liquidation price
        - Funding rate pressure
        - Market momentum

        Returns:
            Urgency score (0-1, higher = more urgent)
        """
        # Distance factor (closer = more urgent)
        distance_percent = abs(liquidation_price - current_price) / current_price
        distance_score = max(0, 1 - (distance_percent * 10))  # 10% away = 0 urgency

        # Funding rate pressure
        if side == 'long':
            # Negative funding helps longs, positive hurts
            funding_score = max(0, 1 - funding_rate * 100)
        else:  # short
            # Positive funding helps shorts
            funding_score = max(0, 1 + funding_rate * 100)

        # Combined urgency
        urgency = (distance_score * 0.7) + (funding_score * 0.3)

        return max(0, min(1, urgency))

    def detect_liquidation_cascade(
        self,
        symbol: str,
        recent_liquidations: List[Dict],
        time_window_seconds: int = 60
    ) -> Optional[Dict]:
        """
        Detect if a liquidation cascade is occurring.

        Args:
            symbol: Trading pair
            recent_liquidations: Recent liquidation events
            time_window_seconds: Time window to analyze

        Returns:
            Cascade detection result
        """
        if not recent_liquidations:
            return None

        # Filter recent liquidations
        current_time = recent_liquidations[-1]['timestamp']
        window_liquidations = [
            liq for liq in recent_liquidations
            if (current_time - liq['timestamp']) <= time_window_seconds
        ]

        if len(window_liquidations) < 5:
            return None

        # Calculate cascade metrics
        total_liquidated = sum(liq['amount_usd'] for liq in window_liquidations)
        avg_liquidation = total_liquidated / len(window_liquidations)

        # Detect if liquidations are accelerating
        first_half = window_liquidations[:len(window_liquidations)//2]
        second_half = window_liquidations[len(window_liquidations)//2:]

        first_half_total = sum(liq['amount_usd'] for liq in first_half)
        second_half_total = sum(liq['amount_usd'] for liq in second_half)

        is_accelerating = second_half_total > first_half_total * 1.5

        # Determine side (long or short liquidations)
        long_liquidations = sum(1 for liq in window_liquidations if liq['side'] == 'long')
        short_liquidations = len(window_liquidations) - long_liquidations

        dominant_side = 'long' if long_liquidations > short_liquidations else 'short'

        if is_accelerating and total_liquidated > self.min_liquidation_size:
            return {
                'detected': True,
                'total_liquidated_usd': total_liquidated,
                'num_liquidations': len(window_liquidations),
                'dominant_side': dominant_side,
                'is_accelerating': is_accelerating,
                'cascade_severity': total_liquidated / self.min_liquidation_size
            }

        return None

    def generate_signal(
        self,
        symbol: str,
        current_price: float,
        clusters: List[LiquidationCluster]
    ) -> Optional[LiquidationSignal]:
        """
        Generate trading signal based on liquidation analysis.

        Args:
            symbol: Trading pair
            current_price: Current market price
            clusters: Liquidation clusters

        Returns:
            Trading signal if opportunity detected
        """
        if not clusters:
            return None

        # Find most urgent cluster
        best_cluster = max(clusters, key=lambda c: c.urgency_score)

        if best_cluster.urgency_score < self.min_confidence:
            return None

        # Generate signal based on liquidation direction
        if best_cluster.side == 'long':
            # Longs getting liquidated -> price going down -> short
            action = 'short'
            entry_price = current_price
            take_profit = best_cluster.price * 0.98  # 2% below liquidation
            stop_loss = current_price * 1.01  # 1% above entry

        else:  # short liquidations
            # Shorts getting liquidated -> price going up -> long
            action = 'long'
            entry_price = current_price
            take_profit = best_cluster.price * 1.02  # 2% above liquidation
            stop_loss = current_price * 0.99  # 1% below entry

        # Calculate profit and risk/reward
        profit_percent = abs(take_profit - entry_price) / entry_price * 100
        risk_percent = abs(stop_loss - entry_price) / entry_price * 100
        risk_reward = profit_percent / risk_percent if risk_percent > 0 else 0

        signal = LiquidationSignal(
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=best_cluster.urgency_score,
            liquidation_target=best_cluster.price,
            estimated_profit_percent=profit_percent,
            risk_reward_ratio=risk_reward
        )

        logger.info(f"Generated {action} signal for {symbol} @ ${entry_price:.2f}")
        logger.info(f"Target: ${take_profit:.2f} ({profit_percent:.2f}%), R/R: {risk_reward:.2f}")

        return signal

    def monitor_and_alert(
        self,
        symbol: str,
        market_data: Dict
    ) -> Dict:
        """
        Continuously monitor for liquidation opportunities.

        Args:
            symbol: Trading pair to monitor
            market_data: Current market data

        Returns:
            Alert dict with signals and cascade detection
        """
        # Analyze liquidation levels
        clusters = self.analyze_liquidation_levels(
            symbol=symbol,
            current_price=market_data['price'],
            open_interest=market_data['open_interest'],
            long_short_ratio=market_data['long_short_ratio'],
            funding_rate=market_data['funding_rate']
        )

        # Generate signal
        signal = self.generate_signal(symbol, market_data['price'], clusters)

        # Detect cascade
        cascade = self.detect_liquidation_cascade(
            symbol,
            market_data.get('recent_liquidations', [])
        )

        alert = {
            'timestamp': market_data.get('timestamp'),
            'symbol': symbol,
            'price': market_data['price'],
            'clusters_found': len(clusters),
            'top_cluster': clusters[0] if clusters else None,
            'signal': signal,
            'cascade_detected': cascade is not None,
            'cascade_details': cascade,
            'recommendation': self._get_recommendation(signal, cascade)
        }

        return alert

    def _get_recommendation(
        self,
        signal: Optional[LiquidationSignal],
        cascade: Optional[Dict]
    ) -> str:
        """Get trading recommendation based on analysis."""
        if cascade and signal:
            return "🔴 HIGH PRIORITY: Cascade detected + Signal generated"
        elif cascade:
            return "⚠️  CAUTION: Cascade detected, wait for signal"
        elif signal:
            return "🟡 MODERATE: Signal generated, monitor for cascade"
        else:
            return "🟢 NORMAL: No immediate opportunities"


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🎯 Liquidation Hunter Demo")
    print("=" * 60)

    # Initialize hunter
    hunter = LiquidationHunter(
        min_liquidation_size=1_000_000,
        min_confidence=0.6,
        max_leverage=5.0
    )

    # Simulate market data
    market_data = {
        'timestamp': 1234567890,
        'price': 45000.0,
        'open_interest': 2_000_000_000,  # $2B OI
        'long_short_ratio': 1.5,  # 60% long, 40% short
        'funding_rate': 0.0002,  # 0.02% positive (shorts pay longs)
        'recent_liquidations': [
            {'timestamp': 1234567890 - i, 'amount_usd': 100000 * (1 + i/10), 'side': 'long'}
            for i in range(20)
        ]
    }

    print(f"\n📊 Market Data:")
    print(f"   Symbol: BTC/USDT")
    print(f"   Price: ${market_data['price']:,.0f}")
    print(f"   Open Interest: ${market_data['open_interest']:,.0f}")
    print(f"   Long/Short Ratio: {market_data['long_short_ratio']:.2f}")
    print(f"   Funding Rate: {market_data['funding_rate']:.4f}%")

    # Monitor for opportunities
    print(f"\n🔍 Analyzing liquidation levels...")
    alert = hunter.monitor_and_alert('BTC/USDT', market_data)

    print(f"\n✅ Analysis Results:")
    print(f"   Clusters Found: {alert['clusters_found']}")

    if alert['top_cluster']:
        cluster = alert['top_cluster']
        print(f"\n🎯 Top Liquidation Cluster:")
        print(f"   Price: ${cluster.price:,.2f}")
        print(f"   Amount: ${cluster.liquidation_amount_usd:,.0f}")
        print(f"   Side: {cluster.side}")
        print(f"   Leverage: {cluster.leverage_avg}x")
        print(f"   Urgency: {cluster.urgency_score:.2%}")
        print(f"   Cascade Potential: ${cluster.estimated_cascade:,.0f}")

    if alert['signal']:
        signal = alert['signal']
        print(f"\n🚨 TRADING SIGNAL GENERATED:")
        print(f"   Action: {signal.action.upper()}")
        print(f"   Entry: ${signal.entry_price:,.2f}")
        print(f"   Take Profit: ${signal.take_profit:,.2f}")
        print(f"   Stop Loss: ${signal.stop_loss:,.2f}")
        print(f"   Est. Profit: {signal.estimated_profit_percent:.2f}%")
        print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}:1")
        print(f"   Confidence: {signal.confidence:.0%}")

    if alert['cascade_detected']:
        cascade = alert['cascade_details']
        print(f"\n💥 LIQUIDATION CASCADE DETECTED:")
        print(f"   Total Liquidated: ${cascade['total_liquidated_usd']:,.0f}")
        print(f"   Number of Events: {cascade['num_liquidations']}")
        print(f"   Dominant Side: {cascade['dominant_side']}")
        print(f"   Severity: {cascade['cascade_severity']:.2f}x")

    print(f"\n📋 Recommendation: {alert['recommendation']}")

    print(f"\n💡 Strategy Tips:")
    print(f"   ✅ Use tight stop losses (1-2%)")
    print(f"   ✅ Take profits quickly (2-5%)")
    print(f"   ✅ High leverage can amplify returns")
    print(f"   ⚠️  High risk - only for experienced traders")
    print(f"   ⚠️  Requires fast execution (< 1 minute)")
    print(f"   ⚠️  Monitor funding rates for pressure")
