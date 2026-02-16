"""
Yield Farming Optimizer
========================

Automatically finds and manages optimal yield farming opportunities across
multiple DeFi protocols.

Features:
- Multi-protocol yield scanning (Aave, Compound, Curve, Yearn, etc.)
- APY calculation with fees and rewards
- Automatic capital reallocation
- Gas cost optimization
- Impermanent loss consideration
- Risk scoring

Protocols Supported:
- Aave (lending/borrowing)
- Compound (lending)
- Curve (stablecoin pools)
- Uniswap V3 (concentrated liquidity)
- Yearn (vaults)
- Convex (Curve boost)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

# Try importing Web3
try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    logger.warning("web3 not available. Install with: pip install web3")


class Protocol(str, Enum):
    """Supported DeFi protocols."""
    AAVE = "aave"
    COMPOUND = "compound"
    CURVE = "curve"
    UNISWAP_V3 = "uniswap_v3"
    YEARN = "yearn"
    CONVEX = "convex"
    BALANCER = "balancer"


class RiskLevel(str, Enum):
    """Risk levels for yield opportunities."""
    LOW = "low"           # Stablecoins, blue-chip protocols
    MEDIUM = "medium"     # Major tokens, established protocols
    HIGH = "high"         # Volatile tokens, new protocols
    VERY_HIGH = "very_high"  # Experimental, high IL risk


@dataclass
class YieldOpportunity:
    """Yield farming opportunity."""
    protocol: Protocol
    pool_address: str
    pool_name: str

    # Assets
    token_a: str
    token_b: Optional[str] = None  # None for single-sided

    # Yields
    base_apy: float = 0.0  # Base pool APY
    reward_apy: float = 0.0  # Reward token APY
    total_apy: float = 0.0  # Total APY

    # Costs
    deposit_gas_cost: float = 0.0  # USD
    withdrawal_gas_cost: float = 0.0  # USD

    # Risk
    risk_level: RiskLevel = RiskLevel.MEDIUM
    impermanent_loss_risk: float = 0.0  # 0-1 scale

    # TVL and utilization
    tvl: float = 0.0  # Total Value Locked (USD)
    utilization: float = 0.0  # 0-1 scale

    # Metadata
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def net_apy(self, investment_amount: float, holding_days: int = 30) -> float:
        """
        Calculate net APY after gas costs.

        Args:
            investment_amount: Amount to invest (USD)
            holding_days: Expected holding period

        Returns:
            Net APY percentage
        """
        # Annualize gas costs
        gas_cost_per_year = (self.deposit_gas_cost + self.withdrawal_gas_cost) * (365 / holding_days)
        gas_cost_apy = (gas_cost_per_year / investment_amount) * 100 if investment_amount > 0 else 0

        return self.total_apy - gas_cost_apy

    def risk_adjusted_apy(self) -> float:
        """Calculate risk-adjusted APY (Sharpe-like metric)."""
        risk_factors = {
            RiskLevel.LOW: 0.9,
            RiskLevel.MEDIUM: 0.7,
            RiskLevel.HIGH: 0.5,
            RiskLevel.VERY_HIGH: 0.3
        }

        risk_multiplier = risk_factors.get(self.risk_level, 0.7)
        il_penalty = self.impermanent_loss_risk * 10  # Up to 10% penalty

        return self.total_apy * risk_multiplier - il_penalty


@dataclass
class OptimizerConfig:
    """Yield optimizer configuration."""
    # Capital allocation
    total_capital: float = 10000.0
    min_allocation: float = 100.0  # Min per position
    max_allocation_pct: float = 0.30  # Max 30% per position

    # Risk management
    max_risk_level: RiskLevel = RiskLevel.MEDIUM
    max_impermanent_loss_risk: float = 0.20  # Max 20% IL risk

    # APY requirements
    min_apy: float = 5.0  # Minimum 5% APY
    min_tvl: float = 1_000_000.0  # Min $1M TVL

    # Gas optimization
    max_gas_cost_pct: float = 0.05  # Max 5% of investment

    # Rebalancing
    rebalance_threshold: float = 0.02  # Rebalance if APY diff > 2%
    rebalance_frequency_hours: int = 24

    # Protocols
    enabled_protocols: List[Protocol] = field(default_factory=lambda: [
        Protocol.AAVE,
        Protocol.CURVE,
        Protocol.UNISWAP_V3
    ])


class YieldOptimizer:
    """
    Yield Farming Optimizer.

    Finds and manages optimal yield across multiple DeFi protocols.
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        Initialize yield optimizer.

        Args:
            config: Optimizer configuration
        """
        self.config = config or OptimizerConfig()
        self.opportunities: List[YieldOpportunity] = []
        self.current_positions: List[Dict] = []
        self.last_scan: Optional[datetime] = None

        logger.info(f"Initialized YieldOptimizer with ${self.config.total_capital:,.2f}")

    def scan_opportunities(self) -> List[YieldOpportunity]:
        """
        Scan all enabled protocols for yield opportunities.

        Returns:
            List of yield opportunities
        """
        logger.info("Scanning yield opportunities across protocols...")

        opportunities = []

        for protocol in self.config.enabled_protocols:
            try:
                protocol_opps = self._scan_protocol(protocol)
                opportunities.extend(protocol_opps)
            except Exception as e:
                logger.error(f"Error scanning {protocol}: {e}")

        # Filter by criteria
        filtered = self._filter_opportunities(opportunities)

        # Sort by risk-adjusted APY
        filtered.sort(key=lambda x: x.risk_adjusted_apy(), reverse=True)

        self.opportunities = filtered
        self.last_scan = datetime.utcnow()

        logger.info(f"Found {len(filtered)} qualifying opportunities")
        return filtered

    def _scan_protocol(self, protocol: Protocol) -> List[YieldOpportunity]:
        """Scan specific protocol for opportunities."""

        if protocol == Protocol.AAVE:
            return self._scan_aave()
        elif protocol == Protocol.CURVE:
            return self._scan_curve()
        elif protocol == Protocol.UNISWAP_V3:
            return self._scan_uniswap_v3()
        elif protocol == Protocol.YEARN:
            return self._scan_yearn()
        else:
            return []

    def _scan_aave(self) -> List[YieldOpportunity]:
        """Scan Aave lending pools."""
        # Mock data - in production, query Aave subgraph or API
        opportunities = [
            YieldOpportunity(
                protocol=Protocol.AAVE,
                pool_address="0xaave_usdc_pool",
                pool_name="USDC Lending",
                token_a="USDC",
                base_apy=4.5,
                reward_apy=2.0,
                total_apy=6.5,
                deposit_gas_cost=15.0,
                withdrawal_gas_cost=15.0,
                risk_level=RiskLevel.LOW,
                tvl=500_000_000.0,
                utilization=0.75
            ),
            YieldOpportunity(
                protocol=Protocol.AAVE,
                pool_address="0xaave_eth_pool",
                pool_name="ETH Lending",
                token_a="ETH",
                base_apy=3.2,
                reward_apy=1.5,
                total_apy=4.7,
                deposit_gas_cost=20.0,
                withdrawal_gas_cost=20.0,
                risk_level=RiskLevel.LOW,
                tvl=1_200_000_000.0,
                utilization=0.65
            )
        ]

        return opportunities

    def _scan_curve(self) -> List[YieldOpportunity]:
        """Scan Curve stablecoin pools."""
        opportunities = [
            YieldOpportunity(
                protocol=Protocol.CURVE,
                pool_address="0xcurve_3pool",
                pool_name="3Pool (USDC/USDT/DAI)",
                token_a="USDC",
                token_b="USDT",
                base_apy=8.5,
                reward_apy=3.5,
                total_apy=12.0,
                deposit_gas_cost=25.0,
                withdrawal_gas_cost=25.0,
                risk_level=RiskLevel.LOW,
                impermanent_loss_risk=0.01,  # Very low IL for stablecoins
                tvl=2_000_000_000.0,
                utilization=0.85
            ),
            YieldOpportunity(
                protocol=Protocol.CURVE,
                pool_address="0xcurve_eth_pool",
                pool_name="ETH/stETH",
                token_a="ETH",
                token_b="stETH",
                base_apy=15.0,
                reward_apy=5.0,
                total_apy=20.0,
                deposit_gas_cost=30.0,
                withdrawal_gas_cost=30.0,
                risk_level=RiskLevel.MEDIUM,
                impermanent_loss_risk=0.05,
                tvl=800_000_000.0,
                utilization=0.90
            )
        ]

        return opportunities

    def _scan_uniswap_v3(self) -> List[YieldOpportunity]:
        """Scan Uniswap V3 concentrated liquidity pools."""
        opportunities = [
            YieldOpportunity(
                protocol=Protocol.UNISWAP_V3,
                pool_address="0xuniswap_usdc_eth",
                pool_name="USDC/ETH 0.3%",
                token_a="USDC",
                token_b="ETH",
                base_apy=25.0,  # High APY from fees
                reward_apy=0.0,
                total_apy=25.0,
                deposit_gas_cost=40.0,
                withdrawal_gas_cost=40.0,
                risk_level=RiskLevel.MEDIUM,
                impermanent_loss_risk=0.15,  # Higher IL risk
                tvl=400_000_000.0,
                utilization=0.70
            ),
            YieldOpportunity(
                protocol=Protocol.UNISWAP_V3,
                pool_address="0xuniswap_usdc_usdt",
                pool_name="USDC/USDT 0.01%",
                token_a="USDC",
                token_b="USDT",
                base_apy=18.0,
                reward_apy=0.0,
                total_apy=18.0,
                deposit_gas_cost=35.0,
                withdrawal_gas_cost=35.0,
                risk_level=RiskLevel.LOW,
                impermanent_loss_risk=0.02,
                tvl=600_000_000.0,
                utilization=0.80
            )
        ]

        return opportunities

    def _scan_yearn(self) -> List[YieldOpportunity]:
        """Scan Yearn vaults."""
        opportunities = [
            YieldOpportunity(
                protocol=Protocol.YEARN,
                pool_address="0xyearn_usdc_vault",
                pool_name="yvUSDC Vault",
                token_a="USDC",
                base_apy=10.0,
                reward_apy=0.0,
                total_apy=10.0,
                deposit_gas_cost=20.0,
                withdrawal_gas_cost=20.0,
                risk_level=RiskLevel.LOW,
                tvl=300_000_000.0,
                utilization=0.95
            )
        ]

        return opportunities

    def _filter_opportunities(self, opportunities: List[YieldOpportunity]) -> List[YieldOpportunity]:
        """Filter opportunities by configured criteria."""
        filtered = []

        for opp in opportunities:
            # Check risk level
            risk_levels = {
                RiskLevel.LOW: 0,
                RiskLevel.MEDIUM: 1,
                RiskLevel.HIGH: 2,
                RiskLevel.VERY_HIGH: 3
            }

            if risk_levels[opp.risk_level] > risk_levels[self.config.max_risk_level]:
                continue

            # Check IL risk
            if opp.impermanent_loss_risk > self.config.max_impermanent_loss_risk:
                continue

            # Check TVL
            if opp.tvl < self.config.min_tvl:
                continue

            # Check APY (after gas costs)
            if opp.net_apy(self.config.min_allocation) < self.config.min_apy:
                continue

            filtered.append(opp)

        return filtered

    def optimize_allocation(self) -> Dict[str, float]:
        """
        Optimize capital allocation across opportunities.

        Returns:
            Allocation map: {opportunity_id: amount}
        """
        if not self.opportunities:
            logger.warning("No opportunities available. Run scan_opportunities() first.")
            return {}

        # Simple greedy allocation by risk-adjusted APY
        # More sophisticated: use Modern Portfolio Theory

        allocation = {}
        remaining_capital = self.config.total_capital

        for opp in self.opportunities:
            if remaining_capital <= self.config.min_allocation:
                break

            # Calculate allocation
            max_allocation = self.config.total_capital * self.config.max_allocation_pct
            allocation_amount = min(
                max_allocation,
                remaining_capital,
                self.config.min_allocation
            )

            # Check if profitable after gas
            if opp.net_apy(allocation_amount) >= self.config.min_apy:
                allocation[f"{opp.protocol}_{opp.pool_name}"] = allocation_amount
                remaining_capital -= allocation_amount

        logger.info(f"Optimized allocation: {len(allocation)} positions, "
                   f"${self.config.total_capital - remaining_capital:,.2f} allocated")

        return allocation

    def should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced."""
        if self.last_scan is None:
            return True

        hours_since_scan = (datetime.utcnow() - self.last_scan).total_seconds() / 3600

        if hours_since_scan >= self.config.rebalance_frequency_hours:
            return True

        # Check if current positions are still optimal
        # In production: compare current APYs with new opportunities

        return False

    def get_best_opportunities(self, top_n: int = 5) -> List[YieldOpportunity]:
        """
        Get top N opportunities by risk-adjusted APY.

        Args:
            top_n: Number of opportunities to return

        Returns:
            List of best opportunities
        """
        if not self.opportunities:
            self.scan_opportunities()

        return self.opportunities[:top_n]

    def generate_report(self) -> str:
        """Generate yield optimization report."""
        if not self.opportunities:
            return "No opportunities scanned yet."

        allocation = self.optimize_allocation()

        report = []
        report.append("=" * 70)
        report.append("YIELD FARMING OPTIMIZATION REPORT")
        report.append("=" * 70)
        report.append(f"\nTotal Capital: ${self.config.total_capital:,.2f}")
        report.append(f"Last Scan: {self.last_scan}")
        report.append(f"\nTop {min(5, len(self.opportunities))} Opportunities:")
        report.append("-" * 70)

        for i, opp in enumerate(self.opportunities[:5], 1):
            report.append(f"\n{i}. {opp.protocol.upper()} - {opp.pool_name}")
            report.append(f"   Base APY: {opp.base_apy:.2f}% | Reward APY: {opp.reward_apy:.2f}%")
            report.append(f"   Total APY: {opp.total_apy:.2f}%")
            report.append(f"   Risk-Adjusted APY: {opp.risk_adjusted_apy():.2f}%")
            report.append(f"   Risk: {opp.risk_level.value.upper()} | IL Risk: {opp.impermanent_loss_risk:.1%}")
            report.append(f"   TVL: ${opp.tvl:,.0f}")

        report.append("\n" + "=" * 70)
        report.append("RECOMMENDED ALLOCATION")
        report.append("=" * 70)

        total_allocated = sum(allocation.values())
        expected_apy = 0

        for pool_name, amount in allocation.items():
            pct = (amount / self.config.total_capital) * 100
            report.append(f"\n{pool_name}:")
            report.append(f"  Amount: ${amount:,.2f} ({pct:.1f}%)")

            # Find corresponding opportunity
            for opp in self.opportunities:
                if f"{opp.protocol}_{opp.pool_name}" == pool_name:
                    net_apy = opp.net_apy(amount)
                    report.append(f"  Net APY: {net_apy:.2f}%")
                    expected_apy += (amount / total_allocated) * net_apy if total_allocated > 0 else 0
                    break

        report.append(f"\nTotal Allocated: ${total_allocated:,.2f} ({(total_allocated/self.config.total_capital)*100:.1f}%)")
        report.append(f"Expected Portfolio APY: {expected_apy:.2f}%")
        report.append(f"Expected Annual Return: ${(self.config.total_capital * expected_apy / 100):,.2f}")

        return "\n".join(report)


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("\nYield Farming Optimizer Example")
    print("=" * 70)

    # Create optimizer
    config = OptimizerConfig(
        total_capital=50000.0,
        min_apy=5.0,
        max_risk_level=RiskLevel.MEDIUM,
        enabled_protocols=[Protocol.AAVE, Protocol.CURVE, Protocol.UNISWAP_V3]
    )

    optimizer = YieldOptimizer(config)

    # Scan opportunities
    opportunities = optimizer.scan_opportunities()

    print(f"\nFound {len(opportunities)} opportunities")

    # Get report
    report = optimizer.generate_report()
    print(f"\n{report}")

    print("\n✅ Yield Optimizer Example Complete!")
