"""
Advanced Execution Strategies for RL Trading
=============================================

Sophisticated execution strategies that minimize market impact and slippage.

Strategies:
- Adaptive Slippage Model (market-aware)
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Iceberg Orders
- Smart Order Routing

These strategies work with RL agents to optimize trade execution.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategy types."""
    MARKET = "market"           # Immediate execution
    TWAP = "twap"              # Time-Weighted Average Price
    VWAP = "vwap"              # Volume-Weighted Average Price
    ICEBERG = "iceberg"        # Hidden order splitting
    ADAPTIVE = "adaptive"      # ML-based adaptive execution


@dataclass
class MarketConditions:
    """Current market conditions for execution optimization."""
    volatility: float           # Current volatility (0.0 to 1.0+)
    volume: float              # Current volume relative to average
    spread: float              # Bid-ask spread (basis points)
    depth: float               # Order book depth
    momentum: float            # Price momentum (-1.0 to 1.0)
    time_of_day: float         # Normalized time (0.0 = open, 1.0 = close)


@dataclass
class ExecutionConfig:
    """Configuration for execution strategies."""
    # Slippage model
    base_slippage: float = 0.0005  # 0.05% base slippage
    slippage_volatility_factor: float = 2.0  # Multiplier for high volatility
    slippage_volume_factor: float = 1.5  # Multiplier for low volume
    slippage_size_factor: float = 0.001  # Per 1% of volume

    # Market impact
    base_impact: float = 0.0003  # 0.03% base impact
    impact_exponent: float = 0.6  # Sublinear impact (square root-like)

    # TWAP settings
    twap_intervals: int = 10  # Number of time slices
    twap_randomization: float = 0.1  # Random variation in slice sizes

    # VWAP settings
    vwap_lookback: int = 20  # Bars to estimate volume profile
    vwap_participation_rate: float = 0.1  # Max % of volume per interval

    # Iceberg settings
    iceberg_visible_fraction: float = 0.2  # Show 20% of order
    iceberg_min_slices: int = 5  # Minimum number of child orders

    # Risk limits
    max_order_size_pct: float = 0.1  # Max 10% of daily volume
    urgent_execution_threshold: float = 0.05  # Immediate execution if <5% of volume


class AdaptiveSlippageModel:
    """
    Advanced slippage model that adapts to market conditions.

    Factors considered:
    - Market volatility
    - Trading volume
    - Order size relative to market
    - Time of day
    - Market momentum
    - Bid-ask spread
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize adaptive slippage model."""
        self.config = config or ExecutionConfig()
        logger.info("Initialized AdaptiveSlippageModel")

    def estimate_slippage(
        self,
        order_size: float,
        market_conditions: MarketConditions,
        order_book_depth: Optional[float] = None
    ) -> float:
        """
        Estimate slippage for an order given market conditions.

        Args:
            order_size: Order size as fraction of average daily volume
            market_conditions: Current market state
            order_book_depth: Depth of order book (optional)

        Returns:
            Estimated slippage as fraction (e.g., 0.001 = 0.1%)
        """
        # Base slippage
        slippage = self.config.base_slippage

        # Volatility adjustment (higher volatility = more slippage)
        volatility_multiplier = 1.0 + (
            market_conditions.volatility * self.config.slippage_volatility_factor
        )
        slippage *= volatility_multiplier

        # Volume adjustment (lower volume = more slippage)
        if market_conditions.volume < 1.0:
            volume_multiplier = 1.0 + (
                (1.0 - market_conditions.volume) * self.config.slippage_volume_factor
            )
            slippage *= volume_multiplier

        # Size adjustment (larger orders = more slippage, sublinear)
        size_impact = self.config.slippage_size_factor * (
            order_size ** self.config.impact_exponent
        )
        slippage += size_impact

        # Spread adjustment
        spread_impact = market_conditions.spread / 10000  # Convert bps to fraction
        slippage += spread_impact * 0.5  # 50% of spread adds to slippage

        # Time of day adjustment (higher slippage at open/close)
        time_factor = 1.0 + 0.3 * abs(market_conditions.time_of_day - 0.5) * 2
        slippage *= time_factor

        # Momentum adjustment (trading against momentum increases slippage)
        # This would need order direction to apply properly
        # Placeholder: assume some momentum penalty
        momentum_penalty = abs(market_conditions.momentum) * 0.1
        slippage += momentum_penalty

        # Order book depth adjustment (if available)
        if order_book_depth is not None and order_book_depth < 1.0:
            depth_multiplier = 1.0 + (1.0 - order_book_depth)
            slippage *= depth_multiplier

        logger.debug(f"Estimated slippage: {slippage:.4f} ({slippage*100:.2f}%)")
        return slippage

    def estimate_market_impact(
        self,
        order_size: float,
        market_conditions: MarketConditions
    ) -> float:
        """
        Estimate permanent market impact of order.

        Args:
            order_size: Order size as fraction of daily volume
            market_conditions: Current market state

        Returns:
            Estimated price impact as fraction
        """
        # Base impact (square-root market impact model)
        impact = self.config.base_impact * (
            order_size ** self.config.impact_exponent
        )

        # Adjust for volatility (more volatile = more impact)
        impact *= (1.0 + market_conditions.volatility)

        # Adjust for volume (low volume = more impact)
        if market_conditions.volume < 1.0:
            impact *= (2.0 - market_conditions.volume)

        return impact


class TWAPExecutor:
    """
    Time-Weighted Average Price (TWAP) execution.

    Splits order into equal time slices to minimize market impact.
    Good for:
    - Large orders
    - Stable markets
    - Non-urgent execution
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize TWAP executor."""
        self.config = config or ExecutionConfig()

    def split_order(
        self,
        total_size: float,
        duration: int,
        randomize: bool = True
    ) -> List[Tuple[float, int]]:
        """
        Split order into time-weighted slices.

        Args:
            total_size: Total order size
            duration: Execution duration (e.g., number of bars)
            randomize: Add randomization to avoid detection

        Returns:
            List of (size, time_offset) tuples
        """
        intervals = min(self.config.twap_intervals, duration)
        base_size = total_size / intervals

        slices = []
        remaining = total_size

        for i in range(intervals):
            # Add randomization if requested
            if randomize and i < intervals - 1:
                variation = np.random.uniform(
                    1.0 - self.config.twap_randomization,
                    1.0 + self.config.twap_randomization
                )
                size = base_size * variation
                size = min(size, remaining)  # Don't exceed remaining
            else:
                # Last slice takes all remaining
                size = remaining

            time_offset = int((duration / intervals) * i)
            slices.append((size, time_offset))
            remaining -= size

        logger.info(f"Split order into {len(slices)} TWAP slices")
        return slices


class VWAPExecutor:
    """
    Volume-Weighted Average Price (VWAP) execution.

    Splits order proportional to expected volume profile.
    Good for:
    - Minimizing market impact
    - Following natural market flow
    - Trading with the market
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize VWAP executor."""
        self.config = config or ExecutionConfig()

    def estimate_volume_profile(
        self,
        historical_volume: np.ndarray,
        duration: int
    ) -> np.ndarray:
        """
        Estimate future volume profile from historical data.

        Args:
            historical_volume: Historical volume data
            duration: Forecast duration

        Returns:
            Estimated volume profile (normalized to sum to 1.0)
        """
        # Simple approach: average of recent volume patterns
        lookback = min(self.config.vwap_lookback, len(historical_volume))
        recent_volume = historical_volume[-lookback:]

        # Normalize
        volume_profile = recent_volume / recent_volume.sum()

        # Extend or truncate to match duration
        if len(volume_profile) < duration:
            # Repeat pattern
            repeats = int(np.ceil(duration / len(volume_profile)))
            volume_profile = np.tile(volume_profile, repeats)[:duration]
        else:
            volume_profile = volume_profile[:duration]

        # Renormalize
        volume_profile = volume_profile / volume_profile.sum()

        return volume_profile

    def split_order(
        self,
        total_size: float,
        volume_profile: np.ndarray,
        max_participation: Optional[float] = None
    ) -> List[Tuple[float, int]]:
        """
        Split order according to volume profile.

        Args:
            total_size: Total order size
            volume_profile: Expected volume profile (normalized)
            max_participation: Maximum participation rate (optional)

        Returns:
            List of (size, time_offset) tuples
        """
        max_participation = max_participation or self.config.vwap_participation_rate

        # Calculate slice sizes proportional to volume
        slice_sizes = total_size * volume_profile

        # Apply participation rate limit
        # (in practice, would need actual volume forecasts)
        # Here we just ensure no slice is too large
        max_slice = total_size * max_participation
        slice_sizes = np.minimum(slice_sizes, max_slice)

        # Adjust to ensure total adds up
        scale = total_size / slice_sizes.sum()
        slice_sizes *= scale

        slices = [(size, i) for i, size in enumerate(slice_sizes) if size > 0]

        logger.info(f"Split order into {len(slices)} VWAP slices")
        return slices


class IcebergOrderExecutor:
    """
    Iceberg Order execution.

    Hides true order size by showing only a small portion.
    Good for:
    - Very large orders
    - Avoiding front-running
    - Minimizing information leakage
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize iceberg executor."""
        self.config = config or ExecutionConfig()

    def split_order(
        self,
        total_size: float,
        visible_size: Optional[float] = None,
        min_slices: Optional[int] = None
    ) -> List[float]:
        """
        Split order into hidden child orders.

        Args:
            total_size: Total order size
            visible_size: Size to show at once (optional)
            min_slices: Minimum number of child orders (optional)

        Returns:
            List of child order sizes
        """
        visible_size = visible_size or (
            total_size * self.config.iceberg_visible_fraction
        )
        min_slices = min_slices or self.config.iceberg_min_slices

        # Calculate number of slices
        n_slices = max(min_slices, int(np.ceil(total_size / visible_size)))

        # Create slices with some randomization
        slices = []
        remaining = total_size

        for i in range(n_slices):
            if i == n_slices - 1:
                # Last slice
                size = remaining
            else:
                # Random size around visible_size
                size = visible_size * np.random.uniform(0.8, 1.2)
                size = min(size, remaining)

            slices.append(size)
            remaining -= size

        logger.info(f"Split order into {len(slices)} iceberg slices")
        return slices


class AdaptiveExecutionStrategy:
    """
    Adaptive execution strategy that combines multiple methods.

    Automatically selects best execution strategy based on:
    - Order size
    - Market conditions
    - Urgency
    - Risk tolerance
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize adaptive execution strategy."""
        self.config = config or ExecutionConfig()
        self.slippage_model = AdaptiveSlippageModel(config)
        self.twap = TWAPExecutor(config)
        self.vwap = VWAPExecutor(config)
        self.iceberg = IcebergOrderExecutor(config)

        logger.info("Initialized AdaptiveExecutionStrategy")

    def select_strategy(
        self,
        order_size: float,
        market_conditions: MarketConditions,
        urgency: float = 0.5,
        duration: Optional[int] = None
    ) -> Tuple[ExecutionStrategy, Dict]:
        """
        Select optimal execution strategy.

        Args:
            order_size: Order size as fraction of daily volume
            market_conditions: Current market state
            urgency: Urgency level (0.0 = patient, 1.0 = immediate)
            duration: Available time for execution (bars)

        Returns:
            (strategy, parameters) tuple
        """
        # Small order or very urgent -> MARKET
        if order_size < self.config.urgent_execution_threshold or urgency > 0.9:
            return ExecutionStrategy.MARKET, {}

        # Very large order -> ICEBERG + TWAP/VWAP
        if order_size > self.config.max_order_size_pct:
            logger.warning(f"Order size {order_size:.2%} exceeds limit, using Iceberg")
            return ExecutionStrategy.ICEBERG, {
                'visible_fraction': self.config.iceberg_visible_fraction
            }

        # Medium urgency -> TWAP
        if urgency > 0.6:
            return ExecutionStrategy.TWAP, {
                'intervals': self.config.twap_intervals,
                'randomize': True
            }

        # Low urgency, normal volume -> VWAP
        if market_conditions.volume > 0.7:
            return ExecutionStrategy.VWAP, {
                'participation_rate': self.config.vwap_participation_rate
            }

        # Default: TWAP with more intervals for patient execution
        return ExecutionStrategy.TWAP, {
            'intervals': self.config.twap_intervals * 2,
            'randomize': True
        }

    def execute_order(
        self,
        order_size: float,
        market_conditions: MarketConditions,
        urgency: float = 0.5,
        duration: Optional[int] = None,
        historical_volume: Optional[np.ndarray] = None
    ) -> Tuple[ExecutionStrategy, List[Tuple[float, int]], float]:
        """
        Plan order execution.

        Args:
            order_size: Order size
            market_conditions: Current market state
            urgency: Urgency level (0.0 to 1.0)
            duration: Execution duration (bars)
            historical_volume: Historical volume for VWAP (optional)

        Returns:
            (strategy, order_slices, estimated_slippage) tuple
        """
        # Select strategy
        strategy, params = self.select_strategy(
            order_size, market_conditions, urgency, duration
        )

        # Estimate slippage
        estimated_slippage = self.slippage_model.estimate_slippage(
            order_size, market_conditions
        )

        # Generate execution plan
        if strategy == ExecutionStrategy.MARKET:
            slices = [(order_size, 0)]

        elif strategy == ExecutionStrategy.TWAP:
            duration = duration or 10
            slices = self.twap.split_order(
                order_size,
                duration,
                randomize=params.get('randomize', True)
            )

        elif strategy == ExecutionStrategy.VWAP:
            if historical_volume is None:
                logger.warning("No volume data for VWAP, falling back to TWAP")
                duration = duration or 10
                slices = self.twap.split_order(order_size, duration)
            else:
                duration = duration or len(historical_volume)
                volume_profile = self.vwap.estimate_volume_profile(
                    historical_volume, duration
                )
                slices = self.vwap.split_order(
                    order_size,
                    volume_profile,
                    max_participation=params.get('participation_rate')
                )

        elif strategy == ExecutionStrategy.ICEBERG:
            slice_sizes = self.iceberg.split_order(
                order_size,
                visible_size=order_size * params.get('visible_fraction', 0.2)
            )
            slices = [(size, i) for i, size in enumerate(slice_sizes)]

        else:
            # Default to TWAP
            duration = duration or 10
            slices = self.twap.split_order(order_size, duration)

        logger.info(
            f"Planned {strategy.value} execution: "
            f"{len(slices)} slices, estimated slippage: {estimated_slippage:.4f}"
        )

        return strategy, slices, estimated_slippage


# Convenience functions

def estimate_slippage(
    order_size: float,
    volatility: float = 0.02,
    volume_ratio: float = 1.0,
    config: Optional[ExecutionConfig] = None
) -> float:
    """
    Quick slippage estimation.

    Args:
        order_size: Order size as fraction of daily volume
        volatility: Current volatility
        volume_ratio: Current volume / average volume
        config: Configuration (optional)

    Returns:
        Estimated slippage
    """
    market_conditions = MarketConditions(
        volatility=volatility,
        volume=volume_ratio,
        spread=10.0,  # 10 bps default
        depth=1.0,
        momentum=0.0,
        time_of_day=0.5
    )

    model = AdaptiveSlippageModel(config)
    return model.estimate_slippage(order_size, market_conditions)


def plan_execution(
    order_size: float,
    market_volatility: float = 0.02,
    urgency: float = 0.5,
    config: Optional[ExecutionConfig] = None
) -> Tuple[ExecutionStrategy, List[Tuple[float, int]], float]:
    """
    Quick execution planning.

    Args:
        order_size: Order size
        market_volatility: Current volatility
        urgency: Urgency (0.0 = patient, 1.0 = immediate)
        config: Configuration (optional)

    Returns:
        (strategy, slices, estimated_slippage)
    """
    market_conditions = MarketConditions(
        volatility=market_volatility,
        volume=1.0,
        spread=10.0,
        depth=1.0,
        momentum=0.0,
        time_of_day=0.5
    )

    executor = AdaptiveExecutionStrategy(config)
    return executor.execute_order(order_size, market_conditions, urgency)
