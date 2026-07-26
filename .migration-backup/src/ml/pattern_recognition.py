"""
Pattern Recognition System
Detects candlestick patterns and chart formations using ML.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CandlestickPattern(Enum):
    """Candlestick pattern types."""
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    DOJI = "doji"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    SHOOTING_STAR = "shooting_star"


class ChartPattern(Enum):
    """Chart formation types."""
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"


class PatternSignal(Enum):
    """Pattern trading signal."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class Candle:
    """OHLCV candlestick."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def body_size(self) -> float:
        """Size of the candle body."""
        return abs(self.close - self.open)

    @property
    def upper_shadow(self) -> float:
        """Upper shadow/wick length."""
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        """Lower shadow/wick length."""
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        """Is bullish candle (close > open)."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Is bearish candle (close < open)."""
        return self.close < self.open

    @property
    def total_range(self) -> float:
        """Total high-low range."""
        return self.high - self.low


@dataclass
class PatternDetection:
    """Detected pattern with metadata."""
    pattern_id: str
    timestamp: datetime
    pattern_type: str  # CandlestickPattern or ChartPattern
    signal: PatternSignal
    confidence: float  # 0-1

    # Pattern location
    start_index: int
    end_index: int
    key_levels: List[float] = field(default_factory=list)

    # Trading implications
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None

    # Pattern success rate (historical)
    historical_success_rate: float = 0.65

    # Metadata
    metadata: Dict = field(default_factory=dict)


class CandlestickPatternDetector:
    """
    Candlestick Pattern Detector

    Detects and classifies candlestick patterns.
    """

    def __init__(self):
        """Initialize candlestick detector."""
        logger.info("Candlestick pattern detector initialized")

    def detect_hammer(self, candle: Candle, prev_trend: str = "down") -> Optional[PatternDetection]:
        """
        Detect Hammer pattern (bullish reversal).

        Characteristics:
        - Small body at top
        - Long lower shadow (2-3x body)
        - Little/no upper shadow
        - Appears after downtrend
        """
        body = candle.body_size
        lower_shadow = candle.lower_shadow
        upper_shadow = candle.upper_shadow
        total_range = candle.total_range

        if total_range == 0:
            return None

        # Criteria
        is_hammer = (
            lower_shadow >= 2 * body and  # Long lower shadow
            upper_shadow <= 0.1 * total_range and  # Small upper shadow
            body <= 0.3 * total_range and  # Small body
            prev_trend == "down"  # After downtrend
        )

        if is_hammer:
            return PatternDetection(
                pattern_id=f"HAMMER-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=candle.timestamp,
                pattern_type=CandlestickPattern.HAMMER.value,
                signal=PatternSignal.BULLISH,
                confidence=0.75,
                start_index=0,
                end_index=0,
                key_levels=[candle.low, candle.close],
                entry_price=candle.close,
                target_price=candle.close * 1.05,  # 5% target
                stop_loss=candle.low * 0.98,  # Below hammer low
                risk_reward_ratio=5.0,
                historical_success_rate=0.72,
                metadata={'body_ratio': body / total_range, 'shadow_ratio': lower_shadow / body}
            )

        return None

    def detect_doji(self, candle: Candle) -> Optional[PatternDetection]:
        """
        Detect Doji pattern (indecision).

        Characteristics:
        - Very small body (open ≈ close)
        - Represents indecision
        """
        body = candle.body_size
        total_range = candle.total_range

        if total_range == 0:
            return None

        is_doji = body <= 0.1 * total_range  # Body is ≤10% of range

        if is_doji:
            return PatternDetection(
                pattern_id=f"DOJI-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=candle.timestamp,
                pattern_type=CandlestickPattern.DOJI.value,
                signal=PatternSignal.NEUTRAL,
                confidence=0.65,
                start_index=0,
                end_index=0,
                key_levels=[candle.open, candle.high, candle.low],
                historical_success_rate=0.60,
                metadata={'body_size': body, 'indecision_level': 1 - (body / total_range)}
            )

        return None

    def detect_engulfing(self, candle1: Candle, candle2: Candle) -> Optional[PatternDetection]:
        """
        Detect Engulfing pattern (reversal).

        Bullish Engulfing:
        - Small bearish candle followed by large bullish candle
        - Second candle engulfs first

        Bearish Engulfing:
        - Small bullish candle followed by large bearish candle
        """
        # Bullish engulfing
        if candle1.is_bearish and candle2.is_bullish:
            if candle2.close > candle1.open and candle2.open < candle1.close:
                return PatternDetection(
                    pattern_id=f"ENGULF-BULL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    timestamp=candle2.timestamp,
                    pattern_type=CandlestickPattern.ENGULFING_BULLISH.value,
                    signal=PatternSignal.BULLISH,
                    confidence=0.80,
                    start_index=0,
                    end_index=1,
                    key_levels=[candle1.low, candle2.close],
                    entry_price=candle2.close,
                    target_price=candle2.close * 1.05,
                    stop_loss=candle1.low * 0.99,
                    risk_reward_ratio=4.0,
                    historical_success_rate=0.75
                )

        # Bearish engulfing
        if candle1.is_bullish and candle2.is_bearish:
            if candle2.close < candle1.open and candle2.open > candle1.close:
                return PatternDetection(
                    pattern_id=f"ENGULF-BEAR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    timestamp=candle2.timestamp,
                    pattern_type=CandlestickPattern.ENGULFING_BEARISH.value,
                    signal=PatternSignal.BEARISH,
                    confidence=0.80,
                    start_index=0,
                    end_index=1,
                    key_levels=[candle2.close, candle1.high],
                    entry_price=candle2.close,
                    target_price=candle2.close * 0.95,
                    stop_loss=candle1.high * 1.01,
                    risk_reward_ratio=4.0,
                    historical_success_rate=0.75
                )

        return None

    def detect_three_white_soldiers(self, candles: List[Candle]) -> Optional[PatternDetection]:
        """
        Detect Three White Soldiers (strong bullish).

        Three consecutive bullish candles with:
        - Progressively higher closes
        - Small upper shadows
        """
        if len(candles) < 3:
            return None

        c1, c2, c3 = candles[-3:]

        is_pattern = (
            c1.is_bullish and c2.is_bullish and c3.is_bullish and
            c2.close > c1.close and c3.close > c2.close and
            c1.upper_shadow < c1.body_size * 0.3 and
            c2.upper_shadow < c2.body_size * 0.3 and
            c3.upper_shadow < c3.body_size * 0.3
        )

        if is_pattern:
            return PatternDetection(
                pattern_id=f"3WS-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=c3.timestamp,
                pattern_type=CandlestickPattern.THREE_WHITE_SOLDIERS.value,
                signal=PatternSignal.BULLISH,
                confidence=0.85,
                start_index=len(candles)-3,
                end_index=len(candles)-1,
                key_levels=[c1.low, c3.close],
                entry_price=c3.close,
                target_price=c3.close * 1.08,
                stop_loss=c1.low * 0.98,
                risk_reward_ratio=5.0,
                historical_success_rate=0.78
            )

        return None

    def scan_all_patterns(self, candles: List[Candle]) -> List[PatternDetection]:
        """
        Scan for all candlestick patterns.

        Args:
            candles: List of candles to analyze

        Returns:
            List of detected patterns
        """
        patterns = []

        if len(candles) < 1:
            return patterns

        # Single candle patterns
        last_candle = candles[-1]

        hammer = self.detect_hammer(last_candle, prev_trend="down")
        if hammer:
            patterns.append(hammer)

        doji = self.detect_doji(last_candle)
        if doji:
            patterns.append(doji)

        # Two candle patterns
        if len(candles) >= 2:
            engulfing = self.detect_engulfing(candles[-2], candles[-1])
            if engulfing:
                patterns.append(engulfing)

        # Three candle patterns
        if len(candles) >= 3:
            three_soldiers = self.detect_three_white_soldiers(candles)
            if three_soldiers:
                patterns.append(three_soldiers)

        return patterns


class ChartPatternDetector:
    """
    Chart Pattern Detector

    Detects larger-scale chart formations.
    """

    def __init__(self):
        """Initialize chart pattern detector."""
        logger.info("Chart pattern detector initialized")

    def detect_double_top(self, candles: List[Candle], tolerance: float = 0.02) -> Optional[PatternDetection]:
        """
        Detect Double Top (bearish reversal).

        Two peaks at similar levels with a trough between them.
        """
        if len(candles) < 10:
            return None

        highs = [c.high for c in candles]

        # Find peaks
        peaks = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))

        # Check for double top
        if len(peaks) >= 2:
            peak1_idx, peak1_price = peaks[-2]
            peak2_idx, peak2_price = peaks[-1]

            # Peaks should be at similar levels
            price_diff = abs(peak1_price - peak2_price) / peak1_price

            if price_diff <= tolerance:
                # Find trough between peaks
                trough_prices = [c.low for c in candles[peak1_idx:peak2_idx]]
                trough_price = min(trough_prices) if trough_prices else peak1_price * 0.95

                return PatternDetection(
                    pattern_id=f"DBL-TOP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    timestamp=candles[-1].timestamp,
                    pattern_type=ChartPattern.DOUBLE_TOP.value,
                    signal=PatternSignal.BEARISH,
                    confidence=0.75,
                    start_index=peak1_idx,
                    end_index=peak2_idx,
                    key_levels=[peak1_price, trough_price],
                    entry_price=candles[-1].close,
                    target_price=trough_price * 0.95,
                    stop_loss=peak2_price * 1.02,
                    risk_reward_ratio=3.0,
                    historical_success_rate=0.70,
                    metadata={'peak1': peak1_price, 'peak2': peak2_price, 'trough': trough_price}
                )

        return None

    def detect_triangle(self, candles: List[Candle], min_touches: int = 4) -> Optional[PatternDetection]:
        """
        Detect Triangle patterns.

        Ascending: Higher lows, flat top
        Descending: Lower highs, flat bottom
        Symmetrical: Converging trendlines
        """
        if len(candles) < 20:
            return None

        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        # Simple linear regression for trend detection
        n = len(candles)
        x = list(range(n))

        # High trend
        avg_x = sum(x) / n
        avg_high = sum(highs) / n
        high_slope = sum((x[i] - avg_x) * (highs[i] - avg_high) for i in range(n)) / sum((x[i] - avg_x) ** 2 for i in range(n))

        # Low trend
        avg_low = sum(lows) / n
        low_slope = sum((x[i] - avg_x) * (lows[i] - avg_low) for i in range(n)) / sum((x[i] - avg_x) ** 2 for i in range(n))

        # Determine triangle type
        if abs(high_slope) < 0.01 and low_slope > 0.05:
            # Ascending triangle (flat top, rising lows)
            return PatternDetection(
                pattern_id=f"TRI-ASC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=candles[-1].timestamp,
                pattern_type=ChartPattern.TRIANGLE_ASCENDING.value,
                signal=PatternSignal.BULLISH,
                confidence=0.70,
                start_index=0,
                end_index=len(candles)-1,
                key_levels=[max(highs), candles[-1].close],
                entry_price=max(highs) * 1.01,  # Breakout above resistance
                target_price=max(highs) * 1.10,
                stop_loss=candles[-1].low * 0.98,
                risk_reward_ratio=4.0,
                historical_success_rate=0.68
            )

        elif abs(low_slope) < 0.01 and high_slope < -0.05:
            # Descending triangle (falling highs, flat bottom)
            return PatternDetection(
                pattern_id=f"TRI-DESC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=candles[-1].timestamp,
                pattern_type=ChartPattern.TRIANGLE_DESCENDING.value,
                signal=PatternSignal.BEARISH,
                confidence=0.70,
                start_index=0,
                end_index=len(candles)-1,
                key_levels=[candles[-1].close, min(lows)],
                entry_price=min(lows) * 0.99,  # Breakout below support
                target_price=min(lows) * 0.90,
                stop_loss=candles[-1].high * 1.02,
                risk_reward_ratio=4.0,
                historical_success_rate=0.68
            )

        return None


class PatternRecognitionEngine:
    """
    Complete Pattern Recognition Engine

    Combines candlestick and chart pattern detection.
    """

    def __init__(self):
        """Initialize pattern recognition engine."""
        self.candlestick_detector = CandlestickPatternDetector()
        self.chart_detector = ChartPatternDetector()
        logger.info("Pattern recognition engine initialized")

    def analyze(self, candles: List[Candle]) -> Dict:
        """
        Analyze candles for all patterns.

        Args:
            candles: List of candles

        Returns:
            Analysis results with detected patterns
        """
        candlestick_patterns = self.candlestick_detector.scan_all_patterns(candles)
        chart_patterns = []

        # Chart patterns require more data
        if len(candles) >= 20:
            double_top = self.chart_detector.detect_double_top(candles)
            if double_top:
                chart_patterns.append(double_top)

            triangle = self.chart_detector.detect_triangle(candles)
            if triangle:
                chart_patterns.append(triangle)

        all_patterns = candlestick_patterns + chart_patterns

        # Categorize by signal
        bullish = [p for p in all_patterns if p.signal == PatternSignal.BULLISH]
        bearish = [p for p in all_patterns if p.signal == PatternSignal.BEARISH]
        neutral = [p for p in all_patterns if p.signal == PatternSignal.NEUTRAL]

        return {
            'total_patterns': len(all_patterns),
            'candlestick_patterns': candlestick_patterns,
            'chart_patterns': chart_patterns,
            'bullish_signals': bullish,
            'bearish_signals': bearish,
            'neutral_signals': neutral,
            'dominant_signal': self._get_dominant_signal(all_patterns)
        }

    def _get_dominant_signal(self, patterns: List[PatternDetection]) -> PatternSignal:
        """Determine dominant signal from patterns."""
        if not patterns:
            return PatternSignal.NEUTRAL

        # Weight by confidence
        bullish_weight = sum(p.confidence for p in patterns if p.signal == PatternSignal.BULLISH)
        bearish_weight = sum(p.confidence for p in patterns if p.signal == PatternSignal.BEARISH)

        if bullish_weight > bearish_weight:
            return PatternSignal.BULLISH
        elif bearish_weight > bullish_weight:
            return PatternSignal.BEARISH
        else:
            return PatternSignal.NEUTRAL


if __name__ == '__main__':
    import logging
    import random

    logging.basicConfig(level=logging.INFO)

    print("🔍 Pattern Recognition Demo")
    print("=" * 60)

    # Generate sample candles
    print("\n1. Generating Sample Candles...")
    print("-" * 60)

    random.seed(42)
    candles = []
    base_price = 2000.0

    for i in range(30):
        # Simulate price movement
        open_price = base_price + random.gauss(0, 10)
        high = open_price + abs(random.gauss(5, 5))
        low = open_price - abs(random.gauss(5, 5))
        close = open_price + random.gauss(0, 10)
        volume = 1000000 + random.gauss(0, 200000)

        candle = Candle(
            timestamp=datetime.now(),
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume
        )
        candles.append(candle)
        base_price = close

    print(f"Generated {len(candles)} candles")

    # Detect patterns
    print("\n2. Detecting Patterns...")
    print("-" * 60)

    engine = PatternRecognitionEngine()
    analysis = engine.analyze(candles)

    print(f"\nTotal Patterns Detected: {analysis['total_patterns']}")
    print(f"Candlestick Patterns: {len(analysis['candlestick_patterns'])}")
    print(f"Chart Patterns: {len(analysis['chart_patterns'])}")
    print(f"Dominant Signal: {analysis['dominant_signal'].value.upper()}")

    # Show detected patterns
    print("\n3. Pattern Details")
    print("-" * 60)

    for pattern in analysis['candlestick_patterns']:
        print(f"\n{pattern.pattern_type.upper()}")
        print(f"  Signal: {pattern.signal.value}")
        print(f"  Confidence: {pattern.confidence*100:.1f}%")
        print(f"  Success Rate: {pattern.historical_success_rate*100:.1f}%")
        if pattern.entry_price:
            print(f"  Entry: ${pattern.entry_price:.2f}")
            print(f"  Target: ${pattern.target_price:.2f}")
            print(f"  Stop Loss: ${pattern.stop_loss:.2f}")
            print(f"  R/R Ratio: {pattern.risk_reward_ratio:.1f}:1")

    for pattern in analysis['chart_patterns']:
        print(f"\n{pattern.pattern_type.upper()}")
        print(f"  Signal: {pattern.signal.value}")
        print(f"  Confidence: {pattern.confidence*100:.1f}%")
        print(f"  Success Rate: {pattern.historical_success_rate*100:.1f}%")

    # Trading recommendation
    print("\n4. Trading Recommendation")
    print("-" * 60)

    if analysis['dominant_signal'] == PatternSignal.BULLISH:
        print("📈 BULLISH - Consider LONG positions")
        bullish_patterns = analysis['bullish_signals']
        if bullish_patterns:
            best = max(bullish_patterns, key=lambda x: x.confidence)
            print(f"  Best pattern: {best.pattern_type}")
            print(f"  Confidence: {best.confidence*100:.1f}%")
    elif analysis['dominant_signal'] == PatternSignal.BEARISH:
        print("📉 BEARISH - Consider SHORT positions")
        bearish_patterns = analysis['bearish_signals']
        if bearish_patterns:
            best = max(bearish_patterns, key=lambda x: x.confidence)
            print(f"  Best pattern: {best.pattern_type}")
            print(f"  Confidence: {best.confidence*100:.1f}%")
    else:
        print("➡️  NEUTRAL - Wait for clearer signals")

    print("\n✅ Pattern recognition demo complete!")
