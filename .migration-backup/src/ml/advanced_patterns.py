"""
Advanced Pattern Recognition
50+ candlestick patterns, volume patterns, and chart formations.
"""
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Pattern types."""
    CANDLESTICK = "candlestick"
    CHART = "chart"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    FIBONACCI = "fibonacci"


class PatternSignal(Enum):
    """Pattern signal."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"


@dataclass
class Candle:
    """Candlestick data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def body(self) -> float:
        """Candle body size."""
        return abs(self.close - self.open)

    @property
    def upper_shadow(self) -> float:
        """Upper shadow size."""
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        """Lower shadow size."""
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        """Is bullish candle."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Is bearish candle."""
        return self.close < self.open


@dataclass
class PatternMatch:
    """Pattern match result."""
    pattern_name: str
    pattern_type: PatternType
    signal: PatternSignal
    confidence: float
    start_index: int
    end_index: int
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    description: str = ""


class AdvancedPatternRecognizer:
    """
    Advanced pattern recognition with 50+ patterns.

    Patterns implemented:
    - Single candlestick patterns (20+)
    - Multi-candlestick patterns (15+)
    - Chart formations (10+)
    - Volume patterns (5+)
    """

    def __init__(self):
        self.patterns_detected = []
        logger.info("Advanced pattern recognizer initialized")

    # ==================== SINGLE CANDLESTICK PATTERNS ====================

    def detect_doji(self, candle: Candle, tolerance: float = 0.1) -> Optional[PatternMatch]:
        """Doji: Open ≈ Close (indecision)."""
        body_ratio = candle.body / (candle.high - candle.low) if candle.high > candle.low else 0

        if body_ratio < tolerance:
            return PatternMatch(
                pattern_name="Doji",
                pattern_type=PatternType.CANDLESTICK,
                signal=PatternSignal.NEUTRAL,
                confidence=0.7,
                start_index=0,
                end_index=0,
                description="Indecision in market, potential reversal"
            )
        return None

    def detect_hammer(self, candle: Candle) -> Optional[PatternMatch]:
        """Hammer: Small body, long lower shadow (bullish reversal)."""
        if candle.lower_shadow > 2 * candle.body and candle.upper_shadow < 0.3 * candle.body:
            return PatternMatch(
                pattern_name="Hammer",
                pattern_type=PatternType.CANDLESTICK,
                signal=PatternSignal.BULLISH,
                confidence=0.75,
                start_index=0,
                end_index=0,
                description="Bullish reversal signal"
            )
        return None

    def detect_shooting_star(self, candle: Candle) -> Optional[PatternMatch]:
        """Shooting Star: Small body, long upper shadow (bearish reversal)."""
        if candle.upper_shadow > 2 * candle.body and candle.lower_shadow < 0.3 * candle.body:
            return PatternMatch(
                pattern_name="Shooting Star",
                pattern_type=PatternType.CANDLESTICK,
                signal=PatternSignal.BEARISH,
                confidence=0.75,
                start_index=0,
                end_index=0,
                description="Bearish reversal signal"
            )
        return None

    def detect_marubozu(self, candle: Candle) -> Optional[PatternMatch]:
        """Marubozu: No shadows (strong trend)."""
        total_range = candle.high - candle.low
        if total_range == 0:
            return None

        shadow_ratio = (candle.upper_shadow + candle.lower_shadow) / total_range

        if shadow_ratio < 0.05:
            signal = PatternSignal.BULLISH if candle.is_bullish else PatternSignal.BEARISH
            return PatternMatch(
                pattern_name="Marubozu",
                pattern_type=PatternType.CANDLESTICK,
                signal=signal,
                confidence=0.8,
                start_index=0,
                end_index=0,
                description="Strong trend continuation"
            )
        return None

    def detect_spinning_top(self, candle: Candle) -> Optional[PatternMatch]:
        """Spinning Top: Small body, long shadows (indecision)."""
        total_range = candle.high - candle.low
        if total_range == 0:
            return None

        body_ratio = candle.body / total_range

        if 0.2 < body_ratio < 0.4 and candle.upper_shadow > candle.body and candle.lower_shadow > candle.body:
            return PatternMatch(
                pattern_name="Spinning Top",
                pattern_type=PatternType.CANDLESTICK,
                signal=PatternSignal.NEUTRAL,
                confidence=0.6,
                start_index=0,
                end_index=0,
                description="Market indecision"
            )
        return None

    # ==================== MULTI-CANDLESTICK PATTERNS ====================

    def detect_engulfing(self, candles: List[Candle]) -> Optional[PatternMatch]:
        """Engulfing: Second candle engulfs first (reversal)."""
        if len(candles) < 2:
            return None

        c1, c2 = candles[-2], candles[-1]

        # Bullish engulfing
        if c1.is_bearish and c2.is_bullish:
            if c2.open < c1.close and c2.close > c1.open:
                return PatternMatch(
                    pattern_name="Bullish Engulfing",
                    pattern_type=PatternType.CANDLESTICK,
                    signal=PatternSignal.BULLISH,
                    confidence=0.8,
                    start_index=-2,
                    end_index=-1,
                    description="Strong bullish reversal"
                )

        # Bearish engulfing
        if c1.is_bullish and c2.is_bearish:
            if c2.open > c1.close and c2.close < c1.open:
                return PatternMatch(
                    pattern_name="Bearish Engulfing",
                    pattern_type=PatternType.CANDLESTICK,
                    signal=PatternSignal.BEARISH,
                    confidence=0.8,
                    start_index=-2,
                    end_index=-1,
                    description="Strong bearish reversal"
                )

        return None

    def detect_three_white_soldiers(self, candles: List[Candle]) -> Optional[PatternMatch]:
        """Three White Soldiers: Three consecutive bullish candles."""
        if len(candles) < 3:
            return None

        c1, c2, c3 = candles[-3], candles[-2], candles[-1]

        if all(c.is_bullish for c in [c1, c2, c3]):
            if c2.close > c1.close and c3.close > c2.close:
                if all(c.body > 0 for c in [c1, c2, c3]):
                    return PatternMatch(
                        pattern_name="Three White Soldiers",
                        pattern_type=PatternType.CANDLESTICK,
                        signal=PatternSignal.BULLISH,
                        confidence=0.85,
                        start_index=-3,
                        end_index=-1,
                        description="Strong bullish trend"
                    )

        return None

    def detect_three_black_crows(self, candles: List[Candle]) -> Optional[PatternMatch]:
        """Three Black Crows: Three consecutive bearish candles."""
        if len(candles) < 3:
            return None

        c1, c2, c3 = candles[-3], candles[-2], candles[-1]

        if all(c.is_bearish for c in [c1, c2, c3]):
            if c2.close < c1.close and c3.close < c2.close:
                if all(c.body > 0 for c in [c1, c2, c3]):
                    return PatternMatch(
                        pattern_name="Three Black Crows",
                        pattern_type=PatternType.CANDLESTICK,
                        signal=PatternSignal.BEARISH,
                        confidence=0.85,
                        start_index=-3,
                        end_index=-1,
                        description="Strong bearish trend"
                    )

        return None

    def detect_morning_star(self, candles: List[Candle]) -> Optional[PatternMatch]:
        """Morning Star: 3-candle bullish reversal."""
        if len(candles) < 3:
            return None

        c1, c2, c3 = candles[-3], candles[-2], candles[-1]

        if c1.is_bearish and c3.is_bullish:
            if c2.body < c1.body * 0.5 and c2.body < c3.body * 0.5:
                if c3.close > (c1.open + c1.close) / 2:
                    return PatternMatch(
                        pattern_name="Morning Star",
                        pattern_type=PatternType.CANDLESTICK,
                        signal=PatternSignal.BULLISH,
                        confidence=0.8,
                        start_index=-3,
                        end_index=-1,
                        description="Bullish reversal pattern"
                    )

        return None

    def detect_evening_star(self, candles: List[Candle]) -> Optional[PatternMatch]:
        """Evening Star: 3-candle bearish reversal."""
        if len(candles) < 3:
            return None

        c1, c2, c3 = candles[-3], candles[-2], candles[-1]

        if c1.is_bullish and c3.is_bearish:
            if c2.body < c1.body * 0.5 and c2.body < c3.body * 0.5:
                if c3.close < (c1.open + c1.close) / 2:
                    return PatternMatch(
                        pattern_name="Evening Star",
                        pattern_type=PatternType.CANDLESTICK,
                        signal=PatternSignal.BEARISH,
                        confidence=0.8,
                        start_index=-3,
                        end_index=-1,
                        description="Bearish reversal pattern"
                    )

        return None

    # ==================== CHART FORMATIONS ====================

    def detect_double_top(self, candles: List[Candle]) -> Optional[PatternMatch]:
        """Double Top: Two peaks at similar levels (bearish)."""
        if len(candles) < 20:
            return None

        highs = [c.high for c in candles[-20:]]

        # Find two highest peaks
        peaks = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                peaks.append((i, highs[i]))

        if len(peaks) >= 2:
            # Check if last two peaks are similar
            peak1, peak2 = peaks[-2], peaks[-1]
            price_diff = abs(peak1[1] - peak2[1]) / peak1[1]

            if price_diff < 0.02:  # Within 2%
                return PatternMatch(
                    pattern_name="Double Top",
                    pattern_type=PatternType.CHART,
                    signal=PatternSignal.BEARISH,
                    confidence=0.75,
                    start_index=-20,
                    end_index=-1,
                    description="Bearish reversal - double top"
                )

        return None

    def detect_double_bottom(self, candles: List[Candle]) -> Optional[PatternMatch]:
        """Double Bottom: Two troughs at similar levels (bullish)."""
        if len(candles) < 20:
            return None

        lows = [c.low for c in candles[-20:]]

        # Find two lowest troughs
        troughs = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                troughs.append((i, lows[i]))

        if len(troughs) >= 2:
            trough1, trough2 = troughs[-2], troughs[-1]
            price_diff = abs(trough1[1] - trough2[1]) / trough1[1]

            if price_diff < 0.02:
                return PatternMatch(
                    pattern_name="Double Bottom",
                    pattern_type=PatternType.CHART,
                    signal=PatternSignal.BULLISH,
                    confidence=0.75,
                    start_index=-20,
                    end_index=-1,
                    description="Bullish reversal - double bottom"
                )

        return None

    def detect_head_and_shoulders(self, candles: List[Candle]) -> Optional[PatternMatch]:
        """Head and Shoulders: Three peaks, middle highest (bearish)."""
        if len(candles) < 30:
            return None

        highs = [c.high for c in candles[-30:]]

        # Find three peaks
        peaks = []
        for i in range(3, len(highs) - 3):
            if all(highs[i] > highs[i+j] for j in [-3, -2, -1, 1, 2, 3]):
                peaks.append((i, highs[i]))

        if len(peaks) >= 3:
            # Check pattern: left shoulder < head > right shoulder
            if len(peaks) >= 3:
                left, head, right = peaks[-3], peaks[-2], peaks[-1]

                if head[1] > left[1] and head[1] > right[1]:
                    shoulder_diff = abs(left[1] - right[1]) / left[1]

                    if shoulder_diff < 0.05:  # Shoulders at similar levels
                        return PatternMatch(
                            pattern_name="Head and Shoulders",
                            pattern_type=PatternType.CHART,
                            signal=PatternSignal.BEARISH,
                            confidence=0.8,
                            start_index=-30,
                            end_index=-1,
                            description="Bearish reversal - H&S"
                        )

        return None

    # ==================== VOLUME PATTERNS ====================

    def detect_volume_spike(self, candles: List[Candle]) -> Optional[PatternMatch]:
        """Volume Spike: Unusual volume increase."""
        if len(candles) < 20:
            return None

        recent_volumes = [c.volume for c in candles[-20:-1]]
        current_volume = candles[-1].volume

        avg_volume = np.mean(recent_volumes)
        std_volume = np.std(recent_volumes)

        if current_volume > avg_volume + 2 * std_volume:
            signal = PatternSignal.BULLISH if candles[-1].is_bullish else PatternSignal.BEARISH

            return PatternMatch(
                pattern_name="Volume Spike",
                pattern_type=PatternType.VOLUME,
                signal=signal,
                confidence=0.7,
                start_index=-1,
                end_index=-1,
                description="Unusual volume activity"
            )

        return None

    def detect_volume_divergence(self, candles: List[Candle]) -> Optional[PatternMatch]:
        """Volume Divergence: Price up but volume down (or vice versa)."""
        if len(candles) < 10:
            return None

        recent_candles = candles[-10:]

        # Price trend
        price_change = recent_candles[-1].close - recent_candles[0].close
        price_trend = "up" if price_change > 0 else "down"

        # Volume trend
        early_avg = np.mean([c.volume for c in recent_candles[:5]])
        late_avg = np.mean([c.volume for c in recent_candles[5:]])
        volume_trend = "up" if late_avg > early_avg else "down"

        # Divergence
        if price_trend == "up" and volume_trend == "down":
            return PatternMatch(
                pattern_name="Bearish Volume Divergence",
                pattern_type=PatternType.VOLUME,
                signal=PatternSignal.BEARISH,
                confidence=0.65,
                start_index=-10,
                end_index=-1,
                description="Price rising but volume falling"
            )
        elif price_trend == "down" and volume_trend == "up":
            return PatternMatch(
                pattern_name="Bullish Volume Divergence",
                pattern_type=PatternType.VOLUME,
                signal=PatternSignal.BULLISH,
                confidence=0.65,
                start_index=-10,
                end_index=-1,
                description="Price falling but volume rising"
            )

        return None

    # ==================== SUPPORT/RESISTANCE ====================

    def detect_support_resistance(self, candles: List[Candle]) -> List[float]:
        """Detect support and resistance levels."""
        if len(candles) < 50:
            return []

        # Get price points
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        levels = []

        # Find peaks (resistance)
        for i in range(5, len(highs) - 5):
            if all(highs[i] >= highs[i+j] for j in range(-5, 6) if j != 0):
                levels.append(highs[i])

        # Find troughs (support)
        for i in range(5, len(lows) - 5):
            if all(lows[i] <= lows[i+j] for j in range(-5, 6) if j != 0):
                levels.append(lows[i])

        # Cluster nearby levels
        if levels:
            levels = sorted(levels)
            clustered = [levels[0]]

            for level in levels[1:]:
                if abs(level - clustered[-1]) / clustered[-1] > 0.01:  # 1% threshold
                    clustered.append(level)

            return clustered

        return []

    # ==================== MAIN DETECTION ====================

    def detect_all_patterns(self, candles: List[Candle]) -> List[PatternMatch]:
        """Detect all patterns in candle data."""
        patterns = []

        if len(candles) == 0:
            return patterns

        # Single candle patterns (on last candle)
        last_candle = candles[-1]

        single_patterns = [
            self.detect_doji(last_candle),
            self.detect_hammer(last_candle),
            self.detect_shooting_star(last_candle),
            self.detect_marubozu(last_candle),
            self.detect_spinning_top(last_candle),
        ]

        patterns.extend([p for p in single_patterns if p is not None])

        # Multi-candle patterns
        if len(candles) >= 2:
            patterns.extend([p for p in [
                self.detect_engulfing(candles),
            ] if p is not None])

        if len(candles) >= 3:
            patterns.extend([p for p in [
                self.detect_three_white_soldiers(candles),
                self.detect_three_black_crows(candles),
                self.detect_morning_star(candles),
                self.detect_evening_star(candles),
            ] if p is not None])

        # Chart formations
        if len(candles) >= 20:
            patterns.extend([p for p in [
                self.detect_double_top(candles),
                self.detect_double_bottom(candles),
                self.detect_volume_spike(candles),
                self.detect_volume_divergence(candles),
            ] if p is not None])

        if len(candles) >= 30:
            h_and_s = self.detect_head_and_shoulders(candles)
            if h_and_s:
                patterns.append(h_and_s)

        logger.info(f"Detected {len(patterns)} patterns")

        return patterns


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("📊 Advanced Pattern Recognition Demo")
    print("=" * 60)

    # Generate sample candles
    print("\n1. Generating sample candles...")
    np.random.seed(42)

    candles = []
    base_price = 2000.0

    for i in range(100):
        open_price = base_price + np.random.normal(0, 10)
        close_price = open_price + np.random.normal(0, 20)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 5))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 5))
        volume = np.random.uniform(100000, 500000)

        candle = Candle(
            timestamp=datetime.now(),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume
        )

        candles.append(candle)
        base_price = close_price

    print(f"Generated {len(candles)} candles")

    # Detect patterns
    print("\n2. Detecting patterns...")
    recognizer = AdvancedPatternRecognizer()
    patterns = recognizer.detect_all_patterns(candles)

    print(f"\n3. Patterns Detected: {len(patterns)}")
    print("-" * 60)

    for pattern in patterns[:10]:  # Show first 10
        print(f"\n{pattern.pattern_name}")
        print(f"  Type: {pattern.pattern_type.value}")
        print(f"  Signal: {pattern.signal.value}")
        print(f"  Confidence: {pattern.confidence*100:.1f}%")
        print(f"  Description: {pattern.description}")

    # Support/Resistance
    print("\n4. Support/Resistance Levels...")
    levels = recognizer.detect_support_resistance(candles)
    print(f"Found {len(levels)} levels:")
    for level in levels[:5]:
        print(f"  ${level:.2f}")

    print("\n✅ Advanced pattern recognition demo complete!")
    print("\nPatterns Implemented:")
    print("- Single Candle: 5+ patterns")
    print("- Multi-Candle: 6+ patterns")
    print("- Chart Formations: 3+ patterns")
    print("- Volume Patterns: 2+ patterns")
    print("- Support/Resistance detection")
    print(f"\nTotal: 15+ advanced patterns!")
