"""
DeFi Yield Optimizer
Automatically find and allocate capital to highest-yield DeFi opportunities.

Features:
- Compare yields across protocols (Aave, Compound, Curve, etc.)
- Auto-compound rewards
- Risk-adjusted yield calculations
- Gas-cost aware rebalancing
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
class YieldOpportunity:
    """Represents a yield-generating opportunity."""
    protocol: str  # 'aave', 'compound', 'curve', etc.
    pool_name: str
    token: str
    apy: float  # Annual percentage yield
    tvl: float  # Total value locked
    risk_score: float  # 0-1, higher = riskier
    il_risk: bool  # Impermanent loss risk
    lockup_period: int  # Days (0 = no lockup)
    auto_compound: bool
    rewards_token: Optional[str]  # Additional reward tokens
    gas_cost_usd: float  # Est. gas cost to enter/exit


@dataclass
class YieldAllocation:
    """Represents capital allocation recommendation."""
    opportunity: YieldOpportunity
    allocation_percent: float
    expected_apy: float  # Risk-adjusted
    monthly_return_usd: float
    gas_breakeven_days: int  # Days to break even on gas costs


@dataclass
class YieldPortfolio:
    """Current yield farming portfolio."""
    total_capital: float
    allocations: List[YieldAllocation]
    total_monthly_return: float
    total_apy: float
    avg_risk_score: float


class YieldOptimizer:
    """
    DeFi Yield Optimizer

    Finds best yield opportunities across DeFi protocols and
    optimizes capital allocation for maximum risk-adjusted returns.

    Considers:
    - APY rates
    - TVL (liquidity risk)
    - Protocol risk
    - Impermanent loss risk
    - Gas costs
    - Lockup periods
    """

    # Protocol risk ratings (lower = safer)
    PROTOCOL_RISKS = {
        'aave': 0.15,
        'compound': 0.15,
        'curve': 0.2,
        'convex': 0.25,
        'yearn': 0.3,
        'pancakeswap': 0.35,
        'unknown': 0.5
    }

    def __init__(
        self,
        min_apy: float = 5.0,
        max_risk_score: float = 0.5,
        min_tvl: float = 1_000_000,
        gas_price_gwei: float = 30
    ):
        """
        Initialize yield optimizer.

        Args:
            min_apy: Minimum APY to consider (%)
            max_risk_score: Maximum risk score (0-1)
            min_tvl: Minimum TVL to consider
            gas_price_gwei: Gas price for cost calculations
        """
        self.min_apy = min_apy
        self.max_risk_score = max_risk_score
        self.min_tvl = min_tvl
        self.gas_price_gwei = gas_price_gwei

        # Available opportunities
        self.opportunities: List[YieldOpportunity] = []

        logger.info(f"Yield optimizer initialized (min APY: {min_apy}%)")

    def add_opportunity(self, opportunity: YieldOpportunity):
        """Add yield opportunity to consideration."""
        if opportunity.apy >= self.min_apy and opportunity.tvl >= self.min_tvl:
            self.opportunities.append(opportunity)
            logger.info(f"Added: {opportunity.protocol}/{opportunity.pool_name} - {opportunity.apy:.2f}% APY")

    def calculate_risk_adjusted_apy(self, opportunity: YieldOpportunity) -> float:
        """
        Calculate risk-adjusted APY.

        Risk factors:
        - Protocol risk
        - TVL (liquidity) risk
        - Impermanent loss risk
        - Lockup risk

        Returns:
            Risk-adjusted APY
        """
        base_apy = opportunity.apy

        # Protocol risk adjustment
        protocol_risk = self.PROTOCOL_RISKS.get(opportunity.protocol, 0.5)
        risk_factor = 1 - protocol_risk

        # TVL risk (lower TVL = higher risk)
        tvl_factor = min(1.0, opportunity.tvl / 100_000_000)  # $100M baseline

        # Impermanent loss risk
        il_factor = 0.85 if opportunity.il_risk else 1.0

        # Lockup risk
        lockup_factor = max(0.9, 1 - (opportunity.lockup_period / 365))

        # Combined risk adjustment
        adjusted_apy = base_apy * risk_factor * tvl_factor * il_factor * lockup_factor

        return adjusted_apy

    def find_best_opportunities(
        self,
        capital: float,
        top_n: int = 5,
        diversification: bool = True
    ) -> List[YieldAllocation]:
        """
        Find best yield opportunities for given capital.

        Args:
            capital: Total capital to allocate
            top_n: Number of opportunities to return
            diversification: Whether to diversify across protocols

        Returns:
            List of recommended allocations
        """
        if not self.opportunities:
            logger.warning("No opportunities available")
            return []

        # Calculate risk-adjusted APY for all
        scored_opportunities = []
        for opp in self.opportunities:
            if opp.risk_score <= self.max_risk_score:
                adjusted_apy = self.calculate_risk_adjusted_apy(opp)
                scored_opportunities.append((opp, adjusted_apy))

        # Sort by adjusted APY
        scored_opportunities.sort(key=lambda x: x[1], reverse=True)

        # Select top opportunities
        selected = scored_opportunities[:top_n]

        # Allocate capital
        allocations = []

        if diversification:
            # Diversified allocation (weighted by APY)
            total_apy = sum(adj_apy for _, adj_apy in selected)

            for opp, adj_apy in selected:
                allocation_percent = (adj_apy / total_apy) * 100
                allocation_amount = capital * (allocation_percent / 100)

                monthly_return = allocation_amount * (adj_apy / 100) / 12

                # Calculate gas breakeven
                gas_breakeven_days = (opp.gas_cost_usd * 2) / (monthly_return / 30) if monthly_return > 0 else 999

                allocation = YieldAllocation(
                    opportunity=opp,
                    allocation_percent=allocation_percent,
                    expected_apy=adj_apy,
                    monthly_return_usd=monthly_return,
                    gas_breakeven_days=int(gas_breakeven_days)
                )
                allocations.append(allocation)

        else:
            # Concentrated allocation (all in best)
            best_opp, best_apy = selected[0]

            monthly_return = capital * (best_apy / 100) / 12

            gas_breakeven_days = (best_opp.gas_cost_usd * 2) / (monthly_return / 30) if monthly_return > 0 else 999

            allocation = YieldAllocation(
                opportunity=best_opp,
                allocation_percent=100.0,
                expected_apy=best_apy,
                monthly_return_usd=monthly_return,
                gas_breakeven_days=int(gas_breakeven_days)
            )
            allocations.append(allocation)

        return allocations

    def create_portfolio(
        self,
        capital: float,
        diversification: bool = True
    ) -> YieldPortfolio:
        """
        Create optimized yield portfolio.

        Args:
            capital: Total capital
            diversification: Whether to diversify

        Returns:
            Optimized yield portfolio
        """
        allocations = self.find_best_opportunities(capital, diversification=diversification)

        if not allocations:
            return YieldPortfolio(
                total_capital=capital,
                allocations=[],
                total_monthly_return=0,
                total_apy=0,
                avg_risk_score=0
            )

        # Calculate portfolio metrics
        total_monthly_return = sum(a.monthly_return_usd for a in allocations)
        total_apy = (total_monthly_return * 12 / capital) * 100 if capital > 0 else 0

        avg_risk = sum(a.opportunity.risk_score * (a.allocation_percent / 100) for a in allocations)

        portfolio = YieldPortfolio(
            total_capital=capital,
            allocations=allocations,
            total_monthly_return=total_monthly_return,
            total_apy=total_apy,
            avg_risk_score=avg_risk
        )

        logger.info(f"Created portfolio: {total_apy:.2f}% APY, ${total_monthly_return:.2f}/month")

        return portfolio

    def should_rebalance(
        self,
        current_portfolio: YieldPortfolio,
        new_opportunities: List[YieldOpportunity],
        gas_cost_usd: float = 50.0
    ) -> bool:
        """
        Determine if portfolio should be rebalanced.

        Args:
            current_portfolio: Current portfolio
            new_opportunities: New opportunities available
            gas_cost_usd: Gas cost for rebalancing

        Returns:
            True if rebalancing recommended
        """
        # Update opportunities
        old_opportunities = self.opportunities.copy()
        self.opportunities = new_opportunities

        # Calculate new optimal portfolio
        new_portfolio = self.create_portfolio(current_portfolio.total_capital)

        # Restore old opportunities
        self.opportunities = old_opportunities

        # Calculate improvement
        apy_improvement = new_portfolio.total_apy - current_portfolio.total_apy
        monthly_improvement = new_portfolio.total_monthly_return - current_portfolio.total_monthly_return

        # Calculate months to break even on gas
        months_to_breakeven = gas_cost_usd / monthly_improvement if monthly_improvement > 0 else 999

        # Rebalance if improvement is significant and gas-efficient
        should_rebalance = (
            apy_improvement > 2.0 and  # At least 2% APY improvement
            months_to_breakeven < 3  # Break even within 3 months
        )

        if should_rebalance:
            logger.info(f"Rebalancing recommended: +{apy_improvement:.2f}% APY")
        else:
            logger.info(f"Rebalancing not worth it: +{apy_improvement:.2f}% APY, {months_to_breakeven:.1f} months to break even")

        return should_rebalance

    def estimate_il(
        self,
        token_a_change_percent: float,
        token_b_change_percent: float
    ) -> float:
        """
        Estimate impermanent loss for an LP position.

        Args:
            token_a_change_percent: Price change of token A (%)
            token_b_change_percent: Price change of token B (%)

        Returns:
            Impermanent loss (%)
        """
        # Simplified IL calculation
        # IL = (2 * sqrt(price_ratio) / (1 + price_ratio)) - 1

        price_ratio_a = 1 + (token_a_change_percent / 100)
        price_ratio_b = 1 + (token_b_change_percent / 100)

        price_ratio = price_ratio_a / price_ratio_b if price_ratio_b != 0 else 1

        il = (2 * (price_ratio ** 0.5) / (1 + price_ratio)) - 1

        return abs(il) * 100  # Return as positive percentage


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🌾 DeFi Yield Optimizer Demo")
    print("=" * 60)

    # Initialize optimizer
    optimizer = YieldOptimizer(
        min_apy=5.0,
        max_risk_score=0.5,
        min_tvl=1_000_000
    )

    # Add opportunities
    print("\n📊 Adding yield opportunities...")

    opportunities = [
        YieldOpportunity(
            protocol='aave',
            pool_name='USDC',
            token='USDC',
            apy=3.5,
            tvl=500_000_000,
            risk_score=0.15,
            il_risk=False,
            lockup_period=0,
            auto_compound=True,
            rewards_token='AAVE',
            gas_cost_usd=25
        ),
        YieldOpportunity(
            protocol='curve',
            pool_name='3pool',
            token='3CRV',
            apy=8.2,
            tvl=1_200_000_000,
            risk_score=0.2,
            il_risk=False,
            lockup_period=0,
            auto_compound=False,
            rewards_token='CRV',
            gas_cost_usd=30
        ),
        YieldOpportunity(
            protocol='convex',
            pool_name='cvxCRV',
            token='cvxCRV',
            apy=12.5,
            tvl=800_000_000,
            risk_score=0.3,
            il_risk=False,
            lockup_period=0,
            auto_compound=True,
            rewards_token='CVX',
            gas_cost_usd=35
        ),
        YieldOpportunity(
            protocol='pancakeswap',
            pool_name='CAKE-BNB',
            token='CAKE-BNB LP',
            apy=45.0,
            tvl=150_000_000,
            risk_score=0.45,
            il_risk=True,
            lockup_period=0,
            auto_compound=False,
            rewards_token='CAKE',
            gas_cost_usd=5  # BSC is cheaper
        ),
        YieldOpportunity(
            protocol='yearn',
            pool_name='yvUSDC',
            token='yvUSDC',
            apy=6.8,
            tvl=300_000_000,
            risk_score=0.25,
            il_risk=False,
            lockup_period=0,
            auto_compound=True,
            rewards_token=None,
            gas_cost_usd=28
        ),
    ]

    for opp in opportunities:
        optimizer.add_opportunity(opp)

    # Create portfolio
    capital = 10000  # $10k
    print(f"\n💰 Optimizing portfolio for ${capital:,.0f}...")

    portfolio = optimizer.create_portfolio(capital, diversification=True)

    print(f"\n✅ Optimized Portfolio:")
    print(f"   Total Capital: ${portfolio.total_capital:,.2f}")
    print(f"   Expected APY: {portfolio.total_apy:.2f}%")
    print(f"   Monthly Return: ${portfolio.total_monthly_return:.2f}")
    print(f"   Avg Risk Score: {portfolio.avg_risk_score:.2f}")

    print(f"\n📊 Allocations:")
    for i, allocation in enumerate(portfolio.allocations, 1):
        opp = allocation.opportunity
        print(f"\n   {i}. {opp.protocol.upper()} - {opp.pool_name}")
        print(f"      Allocation: {allocation.allocation_percent:.1f}% (${capital * (allocation.allocation_percent/100):,.2f})")
        print(f"      APY: {opp.apy:.2f}% (Risk-adjusted: {allocation.expected_apy:.2f}%)")
        print(f"      Monthly Return: ${allocation.monthly_return_usd:.2f}")
        print(f"      Risk Score: {opp.risk_score:.2f}")
        print(f"      TVL: ${opp.tvl:,.0f}")
        print(f"      IL Risk: {'Yes' if opp.il_risk else 'No'}")
        print(f"      Gas Breakeven: {allocation.gas_breakeven_days} days")

    # Calculate IL
    print(f"\n💸 Impermanent Loss Calculator:")
    print(f"   Scenario: Token A +20%, Token B -10%")
    il = optimizer.estimate_il(20, -10)
    print(f"   Estimated IL: {il:.2f}%")

    print(f"\n💡 Optimization Tips:")
    print(f"   ✅ Diversify across protocols to reduce risk")
    print(f"   ✅ Consider gas costs - high fees eat returns")
    print(f"   ✅ Stablecoin pools avoid IL risk")
    print(f"   ✅ Auto-compounding boosts returns")
    print(f"   ✅ Higher APY often means higher risk")
    print(f"   ⚠️  Always verify APYs - they change frequently")
    print(f"   ⚠️  Smart contract risk exists for all protocols")
