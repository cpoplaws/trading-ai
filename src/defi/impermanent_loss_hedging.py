"""
Impermanent Loss Hedging Strategy
==================================

Hedges impermanent loss (IL) risk when providing liquidity to AMMs.

Impermanent Loss occurs when:
- You provide liquidity to an AMM (e.g., Uniswap)
- Token prices diverge from entry point
- Your LP position underperforms vs holding tokens

Hedging Strategies:
1. **Options Hedging**: Buy put options on volatile token
2. **Perpetual Futures**: Short perpetual to hedge price movement
3. **Dynamic Rebalancing**: Adjust LP range based on price
4. **Correlated Pairs**: Provide liquidity to highly correlated pairs

Example:
- Provide $10K liquidity to ETH/USDC on Uniswap
- If ETH drops 20%, you lose ~2% to IL
- Hedge: Short $5K of ETH perpetuals
- Net: Hedge gains offset IL losses
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
import math

logger = logging.getLogger(__name__)


class HedgeType(str, Enum):
    """Hedging instrument types."""
    OPTIONS = "options"           # Put options
    PERPETUALS = "perpetuals"     # Perpetual futures
    DYNAMIC_RANGE = "dynamic_range"  # Uniswap V3 range adjustment
    CORRELATED_PAIR = "correlated_pair"  # Choose correlated assets


class LPPosition:
    """Liquidity Provider position."""

    def __init__(
        self,
        token_a: str,
        token_b: str,
        amount_a: float,
        amount_b: float,
        entry_price: float,  # Price of token_a in terms of token_b
        pool_address: str = "",
        fee_tier: float = 0.003  # 0.3% default
    ):
        self.token_a = token_a
        self.token_b = token_b
        self.amount_a = amount_a
        self.amount_b = amount_b
        self.entry_price = entry_price
        self.pool_address = pool_address
        self.fee_tier = fee_tier

        # Track fees earned
        self.fees_earned_a = 0.0
        self.fees_earned_b = 0.0

    def calculate_value(self, current_price: float) -> float:
        """
        Calculate current LP position value.

        Args:
            current_price: Current price of token_a in token_b

        Returns:
            Position value in token_b
        """
        # For constant product AMM: x * y = k
        # When price changes, amounts rebalance automatically

        price_ratio = current_price / self.entry_price

        # New amounts after price change
        # k = x * y, and price = y / x
        k = self.amount_a * self.amount_b
        new_amount_a = math.sqrt(k / current_price)
        new_amount_b = k / new_amount_a

        # Value in token_b
        value = new_amount_a * current_price + new_amount_b

        return value

    def calculate_impermanent_loss(self, current_price: float) -> Tuple[float, float]:
        """
        Calculate impermanent loss.

        Args:
            current_price: Current price of token_a

        Returns:
            (IL percentage, IL value in token_b)
        """
        # Value if held tokens
        hold_value = self.amount_a * current_price + self.amount_b

        # Value in LP
        lp_value = self.calculate_value(current_price)

        # IL is the difference
        il_value = hold_value - lp_value
        il_percentage = (il_value / hold_value) * 100 if hold_value > 0 else 0

        return il_percentage, il_value

    def calculate_total_return(
        self,
        current_price: float,
        days_elapsed: int = 30
    ) -> Dict[str, float]:
        """
        Calculate total return including fees.

        Args:
            current_price: Current price
            days_elapsed: Days since position opened

        Returns:
            Return breakdown
        """
        # Initial value
        initial_value = self.amount_a * self.entry_price + self.amount_b

        # Current LP value
        lp_value = self.calculate_value(current_price)

        # Fees earned (in token_b equivalent)
        fees_value = self.fees_earned_a * current_price + self.fees_earned_b

        # IL
        il_pct, il_value = self.calculate_impermanent_loss(current_price)

        # Total value
        total_value = lp_value + fees_value

        # Returns
        il_return = (il_value / initial_value) * 100 if initial_value > 0 else 0
        fee_return = (fees_value / initial_value) * 100 if initial_value > 0 else 0
        total_return = ((total_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0

        return {
            'initial_value': initial_value,
            'lp_value': lp_value,
            'fees_earned': fees_value,
            'total_value': total_value,
            'il_percentage': il_pct,
            'il_value': il_value,
            'il_return': il_return,
            'fee_return': fee_return,
            'total_return': total_return
        }


@dataclass
class HedgePosition:
    """Hedge position details."""
    hedge_type: HedgeType
    instrument: str  # "ETH-PERP", "ETH-PUT-3000", etc.
    size: float  # Size of hedge
    entry_price: float
    current_price: float = 0.0
    pnl: float = 0.0
    cost: float = 0.0  # Cost to establish hedge (premium, fees)
    opened_at: datetime = None

    def __post_init__(self):
        if self.opened_at is None:
            self.opened_at = datetime.utcnow()

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate hedge P&L."""
        if self.hedge_type == HedgeType.PERPETUALS:
            # Short perpetual: profit when price drops
            self.pnl = self.size * (self.entry_price - current_price) - self.cost
        elif self.hedge_type == HedgeType.OPTIONS:
            # Put option: profit when price drops below strike
            strike_price = self.entry_price
            if current_price < strike_price:
                self.pnl = self.size * (strike_price - current_price) - self.cost
            else:
                self.pnl = -self.cost  # Lose premium

        self.current_price = current_price
        return self.pnl


class ILHedgingStrategy:
    """
    Impermanent Loss Hedging Strategy.

    Manages LP positions with hedges to reduce IL risk.
    """

    def __init__(
        self,
        lp_position: LPPosition,
        hedge_type: HedgeType = HedgeType.PERPETUALS,
        hedge_ratio: float = 0.5  # Hedge 50% of position
    ):
        """
        Initialize IL hedging strategy.

        Args:
            lp_position: LP position to hedge
            hedge_type: Type of hedge to use
            hedge_ratio: Fraction of position to hedge (0-1)
        """
        self.lp_position = lp_position
        self.hedge_type = hedge_type
        self.hedge_ratio = hedge_ratio
        self.hedge_position: Optional[HedgePosition] = None

        logger.info(f"Initialized IL hedging for {lp_position.token_a}/{lp_position.token_b}")

    def calculate_hedge_size(self) -> float:
        """
        Calculate optimal hedge size.

        Returns:
            Hedge size in token_a
        """
        # Simple approach: hedge a fraction of token_a exposure
        hedge_size = self.lp_position.amount_a * self.hedge_ratio

        # More sophisticated: calculate delta and hedge accordingly
        # Delta = d(LP_value) / d(price)

        return hedge_size

    def open_hedge(self, current_price: float, hedging_cost: float = 0.0) -> HedgePosition:
        """
        Open hedge position.

        Args:
            current_price: Current price of token_a
            hedging_cost: Cost to establish hedge (fees, premium)

        Returns:
            Hedge position
        """
        hedge_size = self.calculate_hedge_size()

        instrument = f"{self.lp_position.token_a}-PERP"
        if self.hedge_type == HedgeType.OPTIONS:
            instrument = f"{self.lp_position.token_a}-PUT-{int(current_price)}"

        self.hedge_position = HedgePosition(
            hedge_type=self.hedge_type,
            instrument=instrument,
            size=hedge_size,
            entry_price=current_price,
            cost=hedging_cost
        )

        logger.info(f"Opened hedge: {instrument}, size={hedge_size:.4f}, cost=${hedging_cost:.2f}")
        return self.hedge_position

    def calculate_hedged_return(self, current_price: float) -> Dict[str, float]:
        """
        Calculate total return including LP and hedge.

        Args:
            current_price: Current price of token_a

        Returns:
            Complete return breakdown
        """
        # LP return
        lp_returns = self.lp_position.calculate_total_return(current_price)

        # Hedge P&L
        hedge_pnl = 0.0
        if self.hedge_position:
            hedge_pnl = self.hedge_position.calculate_pnl(current_price)

        # Combined
        initial_value = lp_returns['initial_value']
        hedged_value = lp_returns['total_value'] + hedge_pnl
        hedged_return = ((hedged_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0

        # Effectiveness: how much IL was offset
        il_value = lp_returns['il_value']
        hedge_effectiveness = (hedge_pnl / abs(il_value)) * 100 if il_value != 0 else 0

        return {
            **lp_returns,
            'hedge_pnl': hedge_pnl,
            'hedged_value': hedged_value,
            'hedged_return': hedged_return,
            'hedge_effectiveness': hedge_effectiveness
        }

    def should_rebalance_hedge(self, current_price: float, threshold: float = 0.10) -> bool:
        """
        Check if hedge should be rebalanced.

        Args:
            current_price: Current price
            threshold: Rebalance if price moves >10%

        Returns:
            Whether to rebalance
        """
        if not self.hedge_position:
            return False

        price_change = abs(current_price - self.hedge_position.entry_price) / self.hedge_position.entry_price

        return price_change > threshold

    def generate_report(self, current_price: float) -> str:
        """Generate detailed hedging report."""
        returns = self.calculate_hedged_return(current_price)

        report = []
        report.append("=" * 70)
        report.append("IMPERMANENT LOSS HEDGING REPORT")
        report.append("=" * 70)

        # LP Position
        report.append(f"\nLP Position: {self.lp_position.token_a}/{self.lp_position.token_b}")
        report.append(f"Entry Price: {self.lp_position.entry_price:.2f}")
        report.append(f"Current Price: {current_price:.2f}")
        price_change = ((current_price - self.lp_position.entry_price) / self.lp_position.entry_price) * 100
        report.append(f"Price Change: {price_change:+.2f}%")

        # Returns
        report.append("\n" + "-" * 70)
        report.append("RETURNS")
        report.append("-" * 70)
        report.append(f"Initial Value: ${returns['initial_value']:,.2f}")
        report.append(f"LP Value: ${returns['lp_value']:,.2f}")
        report.append(f"Fees Earned: ${returns['fees_earned']:,.2f}")
        report.append(f"\nImpermanent Loss: {returns['il_percentage']:.2f}% (${returns['il_value']:,.2f})")
        report.append(f"Fee Return: {returns['fee_return']:+.2f}%")
        report.append(f"Unhedged LP Return: {returns['total_return']:+.2f}%")

        # Hedge
        if self.hedge_position:
            report.append("\n" + "-" * 70)
            report.append("HEDGE POSITION")
            report.append("-" * 70)
            report.append(f"Type: {self.hedge_type.value.upper()}")
            report.append(f"Instrument: {self.hedge_position.instrument}")
            report.append(f"Size: {self.hedge_position.size:.4f} {self.lp_position.token_a}")
            report.append(f"Hedge Ratio: {self.hedge_ratio:.1%}")
            report.append(f"Hedge P&L: ${returns['hedge_pnl']:+,.2f}")
            report.append(f"Hedge Effectiveness: {returns['hedge_effectiveness']:.1f}%")

            report.append("\n" + "-" * 70)
            report.append("HEDGED RETURNS")
            report.append("-" * 70)
            report.append(f"Total Hedged Value: ${returns['hedged_value']:,.2f}")
            report.append(f"Hedged Return: {returns['hedged_return']:+.2f}%")

            # Comparison
            improvement = returns['hedged_return'] - returns['total_return']
            report.append(f"\nImprovement vs Unhedged: {improvement:+.2f}%")

        return "\n".join(report)


def simulate_il_scenarios(
    token_a: str = "ETH",
    token_b: str = "USDC",
    initial_amount_a: float = 5.0,
    entry_price: float = 2000.0
) -> None:
    """Simulate IL under various price scenarios."""

    print("\n" + "=" * 70)
    print("IMPERMANENT LOSS SIMULATION")
    print("=" * 70)

    # Create LP position
    initial_amount_b = initial_amount_a * entry_price  # Equal value
    lp_position = LPPosition(
        token_a=token_a,
        token_b=token_b,
        amount_a=initial_amount_a,
        amount_b=initial_amount_b,
        entry_price=entry_price
    )

    # Simulate different price changes
    price_changes = [-0.50, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.50]

    print(f"\nInitial Position: {initial_amount_a} {token_a} + ${initial_amount_b:.2f} {token_b}")
    print(f"Entry Price: ${entry_price:.2f}")
    print(f"\nPrice Scenario Analysis:")
    print("-" * 70)
    print(f"{'Price Change':<15} {'New Price':<12} {'IL %':<10} {'IL Value':<12} {'Hold Value':<12} {'LP Value':<12}")
    print("-" * 70)

    for price_change_pct in price_changes:
        new_price = entry_price * (1 + price_change_pct)
        il_pct, il_value = lp_position.calculate_impermanent_loss(new_price)

        hold_value = initial_amount_a * new_price + initial_amount_b
        lp_value = lp_position.calculate_value(new_price)

        print(f"{price_change_pct:+.0%}".ljust(15) +
              f"${new_price:.2f}".ljust(12) +
              f"{il_pct:.2f}%".ljust(10) +
              f"${il_value:.2f}".ljust(12) +
              f"${hold_value:.2f}".ljust(12) +
              f"${lp_value:.2f}".ljust(12))

    # Show hedging benefits
    print("\n" + "=" * 70)
    print("HEDGING COMPARISON (-30% Price Drop Scenario)")
    print("=" * 70)

    crash_price = entry_price * 0.70  # 30% drop

    # Unhedged
    strategy_unhedged = ILHedgingStrategy(lp_position, hedge_ratio=0.0)
    returns_unhedged = strategy_unhedged.calculate_hedged_return(crash_price)

    # 50% hedged with perpetuals
    strategy_hedged = ILHedgingStrategy(
        lp_position,
        hedge_type=HedgeType.PERPETUALS,
        hedge_ratio=0.5
    )
    strategy_hedged.open_hedge(entry_price, hedging_cost=50.0)  # $50 cost
    returns_hedged = strategy_hedged.calculate_hedged_return(crash_price)

    print(f"\nUnhedged Position:")
    print(f"  IL: {returns_unhedged['il_percentage']:.2f}%")
    print(f"  Total Return: {returns_unhedged['total_return']:+.2f}%")

    print(f"\n50% Hedged Position (Perpetuals):")
    print(f"  IL: {returns_hedged['il_percentage']:.2f}%")
    print(f"  Hedge P&L: ${returns_hedged['hedge_pnl']:+,.2f}")
    print(f"  Hedged Return: {returns_hedged['hedged_return']:+.2f}%")
    print(f"  Improvement: {returns_hedged['hedged_return'] - returns_unhedged['total_return']:+.2f}%")


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("\nImpermanent Loss Hedging Example")
    print("=" * 70)

    # Create LP position: 5 ETH + $10,000 USDC at $2000/ETH
    lp_position = LPPosition(
        token_a="ETH",
        token_b="USDC",
        amount_a=5.0,
        amount_b=10000.0,
        entry_price=2000.0,
        fee_tier=0.003
    )

    # Add some fees earned (simulate 30 days)
    lp_position.fees_earned_a = 0.1  # 0.1 ETH
    lp_position.fees_earned_b = 100.0  # $100 USDC

    # Create hedging strategy (hedge 50% with perpetuals)
    strategy = ILHedgingStrategy(
        lp_position,
        hedge_type=HedgeType.PERPETUALS,
        hedge_ratio=0.5
    )

    # Open hedge at entry
    strategy.open_hedge(current_price=2000.0, hedging_cost=50.0)

    # Simulate price drop to $1400 (30% down)
    print("\nScenario: ETH drops from $2000 to $1400 (-30%)")
    print("=" * 70)

    report = strategy.generate_report(current_price=1400.0)
    print(report)

    # Run IL simulation
    print("\n")
    simulate_il_scenarios()

    print("\n✅ Impermanent Loss Hedging Example Complete!")
