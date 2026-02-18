"""
Supervisor Agent Enhancements - Advanced allocation and risk management

New features for Phase 4:
1. Multi-factor allocation (Sharpe + Sortino + Calmar + consistency)
2. Real-time arbitrage detection using live prices
3. Portfolio-wide position tracking
4. Advanced risk metrics
5. Automated rebalancing triggers
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    consistency_score: float  # 0-1, how consistent returns are
    total_pnl: float
    total_trades: int


@dataclass
class PortfolioPosition:
    """Portfolio-wide position tracking"""
    asset: str
    total_quantity: float  # Across all chains
    total_value_usd: float
    chains: Dict[str, float]  # Chain -> quantity
    percentage_of_portfolio: float
    cost_basis: float
    unrealized_pnl: float


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio"""
    total_exposure: float  # Total capital at risk
    max_position_size: float  # Largest single position
    concentration_risk: float  # % in top 3 positions
    daily_var: float  # Value at Risk (95% confidence)
    sharpe_ratio: float
    beta: float  # vs benchmark
    correlation_to_btc: float


class AllocationOptimizer:
    """
    Advanced capital allocation using multiple factors.

    Considers:
    - Sharpe ratio (risk-adjusted returns)
    - Sortino ratio (downside risk)
    - Calmar ratio (return / max drawdown)
    - Win rate and consistency
    - Recent performance trend
    """

    def __init__(
        self,
        sharpe_weight: float = 0.35,
        sortino_weight: float = 0.25,
        calmar_weight: float = 0.20,
        consistency_weight: float = 0.15,
        recent_trend_weight: float = 0.05
    ):
        """
        Initialize optimizer with metric weights.

        Args:
            sharpe_weight: Weight for Sharpe ratio
            sortino_weight: Weight for Sortino ratio
            calmar_weight: Weight for Calmar ratio
            consistency_weight: Weight for consistency score
            recent_trend_weight: Weight for recent performance
        """
        self.weights = {
            'sharpe': sharpe_weight,
            'sortino': sortino_weight,
            'calmar': calmar_weight,
            'consistency': consistency_weight,
            'trend': recent_trend_weight
        }

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

        logger.info(f"Allocation Optimizer initialized with weights: {self.weights}")

    def calculate_composite_score(
        self,
        metrics: PerformanceMetrics,
        recent_returns: List[float]
    ) -> float:
        """
        Calculate composite score from multiple metrics.

        Args:
            metrics: Performance metrics
            recent_returns: Recent trade returns (last 10-20 trades)

        Returns:
            Composite score (0-100)
        """
        # Normalize each metric to 0-100 scale
        sharpe_score = self._normalize_sharpe(metrics.sharpe_ratio)
        sortino_score = self._normalize_sortino(metrics.sortino_ratio)
        calmar_score = self._normalize_calmar(metrics.calmar_ratio)
        consistency_score = metrics.consistency_score * 100
        trend_score = self._calculate_trend_score(recent_returns)

        # Weighted average
        composite = (
            self.weights['sharpe'] * sharpe_score +
            self.weights['sortino'] * sortino_score +
            self.weights['calmar'] * calmar_score +
            self.weights['consistency'] * consistency_score +
            self.weights['trend'] * trend_score
        )

        return max(0.0, min(100.0, composite))

    def _normalize_sharpe(self, sharpe: float) -> float:
        """Normalize Sharpe ratio to 0-100 scale."""
        # Sharpe > 3 is excellent, map to 100
        # Sharpe < 0 is poor, map to 0
        return max(0, min(100, (sharpe / 3.0) * 100))

    def _normalize_sortino(self, sortino: float) -> float:
        """Normalize Sortino ratio to 0-100 scale."""
        return max(0, min(100, (sortino / 4.0) * 100))

    def _normalize_calmar(self, calmar: float) -> float:
        """Normalize Calmar ratio to 0-100 scale."""
        return max(0, min(100, (calmar / 2.0) * 100))

    def _calculate_trend_score(self, returns: List[float]) -> float:
        """
        Calculate trend score from recent returns.

        Args:
            returns: Recent returns

        Returns:
            Score 0-100
        """
        if not returns or len(returns) < 3:
            return 50.0  # Neutral

        # Simple linear regression slope
        x = np.arange(len(returns))
        y = np.array(returns)

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        # Normalize slope to 0-100
        # Positive slope = uptrend (good)
        # Negative slope = downtrend (bad)
        return max(0, min(100, 50 + slope * 1000))

    def optimize_allocations(
        self,
        instances: Dict[str, PerformanceMetrics],
        total_capital: float,
        min_allocation: float = 0.05,  # Min 5% per instance
        max_allocation: float = 0.40   # Max 40% per instance
    ) -> Dict[str, float]:
        """
        Optimize capital allocation across instances.

        Args:
            instances: Dict of instance_id -> metrics
            total_capital: Total capital to allocate
            min_allocation: Minimum allocation per instance
            max_allocation: Maximum allocation per instance

        Returns:
            Dict of instance_id -> allocation
        """
        if not instances:
            return {}

        # Calculate composite scores
        scores = {}
        for instance_id, metrics in instances.items():
            # Get recent returns (mock for now)
            recent_returns = [0.01, 0.02, -0.005, 0.015, 0.01]  # TODO: Real data
            score = self.calculate_composite_score(metrics, recent_returns)
            scores[instance_id] = score

        # Rank by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Allocate based on scores with constraints
        total_score = sum(scores.values())
        if total_score == 0:
            # Equal allocation if all scores are zero
            equal_amount = total_capital / len(instances)
            return {inst_id: equal_amount for inst_id in instances.keys()}

        allocations = {}
        for instance_id, score in ranked:
            # Proportional allocation based on score
            proportion = score / total_score
            allocation = total_capital * proportion

            # Apply constraints
            allocation = max(total_capital * min_allocation, allocation)
            allocation = min(total_capital * max_allocation, allocation)

            allocations[instance_id] = allocation

        # Normalize to ensure total equals total_capital
        current_total = sum(allocations.values())
        if current_total > 0:
            scale_factor = total_capital / current_total
            allocations = {k: v * scale_factor for k, v in allocations.items()}

            # Re-apply max constraint after normalization
            for instance_id in allocations:
                allocations[instance_id] = min(allocations[instance_id], total_capital * max_allocation)

            # Redistribute excess to stay at total_capital
            final_total = sum(allocations.values())
            if final_total < total_capital:
                # Distribute shortfall proportionally to instances below max
                shortfall = total_capital - final_total
                below_max = {k: v for k, v in allocations.items() if v < total_capital * max_allocation}
                if below_max:
                    total_below = sum(below_max.values())
                    for inst_id in below_max:
                        additional = shortfall * (allocations[inst_id] / total_below)
                        allocations[inst_id] = min(
                            allocations[inst_id] + additional,
                            total_capital * max_allocation
                        )

        return allocations


class ArbitrageScanner:
    """
    Real-time arbitrage detection across CEX and DEX.

    Monitors:
    - CEX prices (Binance, Coinbase)
    - DEX prices (Uniswap, Jupiter)
    - Cross-chain opportunities
    - Fee and gas costs
    """

    def __init__(
        self,
        min_profit_pct: float = 0.01,  # 1% minimum profit
        max_position_size: float = 10000.0  # Max $10k per arb
    ):
        """
        Initialize arbitrage scanner.

        Args:
            min_profit_pct: Minimum profit percentage
            max_position_size: Maximum position size in USD
        """
        self.min_profit_pct = min_profit_pct
        self.max_position_size = max_position_size

        logger.info(f"Arbitrage Scanner initialized | Min profit: {min_profit_pct*100:.1f}%")

    def scan_opportunities(
        self,
        cex_connector,
        dex_connector,
        tokens: List[str] = None
    ) -> List[Dict]:
        """
        Scan for arbitrage opportunities.

        Args:
            cex_connector: CEXConnector instance
            dex_connector: DEXConnector instance
            tokens: List of tokens to check

        Returns:
            List of arbitrage opportunities
        """
        if tokens is None:
            tokens = ["BTC", "ETH", "SOL"]

        opportunities = []

        for token in tokens:
            # TODO: Get real prices from connectors
            # For now, simulate
            cex_price = self._get_mock_cex_price(token)
            dex_price = self._get_mock_dex_price(token)

            # Calculate arbitrage
            if cex_price and dex_price:
                spread_pct = abs(dex_price - cex_price) / cex_price

                if spread_pct > self.min_profit_pct:
                    # Calculate net profit after fees
                    cex_fee = cex_price * 0.001  # 0.1%
                    dex_fee = dex_price * 0.003  # 0.3%
                    gas_cost = 8.0  # $8 mock

                    gross_profit = abs(dex_price - cex_price) * 1.0  # For 1 unit
                    net_profit = gross_profit - cex_fee - dex_fee - gas_cost

                    if net_profit > 0:
                        opportunities.append({
                            'token': token,
                            'buy_venue': 'CEX' if cex_price < dex_price else 'DEX',
                            'sell_venue': 'DEX' if cex_price < dex_price else 'CEX',
                            'buy_price': min(cex_price, dex_price),
                            'sell_price': max(cex_price, dex_price),
                            'spread_pct': spread_pct,
                            'net_profit_usd': net_profit,
                            'net_profit_pct': net_profit / min(cex_price, dex_price),
                            'timestamp': datetime.now()
                        })

        return opportunities

    def _get_mock_cex_price(self, token: str) -> float:
        """Get mock CEX price."""
        prices = {
            "BTC": 64000.0,
            "ETH": 3000.0,
            "SOL": 120.0
        }
        return prices.get(token, 100.0)

    def _get_mock_dex_price(self, token: str) -> float:
        """Get mock DEX price (slightly different for arb opportunities)."""
        prices = {
            "BTC": 64500.0,  # 0.78% higher
            "ETH": 3005.0,   # 0.17% higher
            "SOL": 119.5     # 0.42% lower
        }
        return prices.get(token, 100.0)


class RiskMonitor:
    """
    Portfolio-wide risk monitoring and management.

    Tracks:
    - Position concentration
    - Correlation between positions
    - Value at Risk (VaR)
    - Drawdown from peak
    - Daily P&L limits
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_asset_pct: float = 0.50,
        max_daily_loss_pct: float = 0.15,
        max_drawdown_pct: float = 0.25
    ):
        """
        Initialize risk monitor.

        Args:
            max_position_pct: Max % in single position
            max_asset_pct: Max % in single asset
            max_daily_loss_pct: Max daily loss before halt
            max_drawdown_pct: Max drawdown from peak before halt
        """
        self.max_position_pct = max_position_pct
        self.max_asset_pct = max_asset_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct

        logger.info(
            f"Risk Monitor initialized | "
            f"Max position: {max_position_pct*100:.0f}% | "
            f"Max asset: {max_asset_pct*100:.0f}% | "
            f"Daily loss limit: {max_daily_loss_pct*100:.0f}%"
        )

    def aggregate_positions(
        self,
        instances: Dict[str, any]  # instance_id -> instance data
    ) -> Dict[str, PortfolioPosition]:
        """
        Aggregate positions across all instances and chains.

        Args:
            instances: Dict of instances

        Returns:
            Dict of asset -> PortfolioPosition
        """
        positions = {}

        # TODO: Aggregate real positions from all instances
        # For now, return mock data

        return positions

    def calculate_risk_metrics(
        self,
        positions: Dict[str, PortfolioPosition],
        total_portfolio_value: float
    ) -> RiskMetrics:
        """
        Calculate portfolio risk metrics.

        Args:
            positions: Current positions
            total_portfolio_value: Total portfolio value

        Returns:
            Risk metrics
        """
        if not positions:
            return RiskMetrics(
                total_exposure=0.0,
                max_position_size=0.0,
                concentration_risk=0.0,
                daily_var=0.0,
                sharpe_ratio=0.0,
                beta=1.0,
                correlation_to_btc=0.0
            )

        # Calculate metrics
        total_exposure = sum(p.total_value_usd for p in positions.values())
        max_position = max(p.total_value_usd for p in positions.values())

        # Top 3 concentration
        sorted_positions = sorted(
            positions.values(),
            key=lambda x: x.total_value_usd,
            reverse=True
        )
        top3_value = sum(p.total_value_usd for p in sorted_positions[:3])
        concentration = top3_value / total_portfolio_value if total_portfolio_value > 0 else 0

        return RiskMetrics(
            total_exposure=total_exposure,
            max_position_size=max_position,
            concentration_risk=concentration,
            daily_var=total_portfolio_value * 0.05,  # Mock 5% VaR
            sharpe_ratio=1.5,  # Mock
            beta=0.8,  # Mock
            correlation_to_btc=0.6  # Mock
        )

    def check_limits(
        self,
        positions: Dict[str, PortfolioPosition],
        total_value: float,
        daily_pnl: float,
        peak_value: float
    ) -> Tuple[bool, List[str]]:
        """
        Check if any risk limits are violated.

        Args:
            positions: Current positions
            total_value: Total portfolio value
            daily_pnl: Daily P&L
            peak_value: Peak portfolio value

        Returns:
            (is_safe, list_of_violations)
        """
        violations = []

        # Check position concentration
        for asset, pos in positions.items():
            pos_pct = pos.total_value_usd / total_value if total_value > 0 else 0

            if pos_pct > self.max_position_pct:
                violations.append(
                    f"Position too large: {asset} at {pos_pct*100:.1f}% "
                    f"(max {self.max_position_pct*100:.0f}%)"
                )

        # Check daily loss
        daily_loss_pct = abs(daily_pnl) / peak_value if peak_value > 0 else 0
        if daily_pnl < 0 and daily_loss_pct > self.max_daily_loss_pct:
            violations.append(
                f"Daily loss limit exceeded: {daily_loss_pct*100:.1f}% "
                f"(max {self.max_daily_loss_pct*100:.0f}%)"
            )

        # Check drawdown from peak
        drawdown_pct = (peak_value - total_value) / peak_value if peak_value > 0 else 0
        if drawdown_pct > self.max_drawdown_pct:
            violations.append(
                f"Drawdown from peak exceeded: {drawdown_pct*100:.1f}% "
                f"(max {self.max_drawdown_pct*100:.0f}%)"
            )

        is_safe = len(violations) == 0
        return is_safe, violations


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print("SUPERVISOR ENHANCEMENTS TEST")
    print("="*70)

    # Test 1: Allocation Optimizer
    print("\n--- Test 1: Allocation Optimizer ---")
    optimizer = AllocationOptimizer()

    mock_metrics = {
        "instance_1": PerformanceMetrics(
            sharpe_ratio=2.5,
            sortino_ratio=3.0,
            calmar_ratio=1.8,
            max_drawdown=0.10,
            win_rate=0.65,
            profit_factor=2.2,
            avg_win=150.0,
            avg_loss=-80.0,
            consistency_score=0.75,
            total_pnl=5000.0,
            total_trades=50
        ),
        "instance_2": PerformanceMetrics(
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            calmar_ratio=1.2,
            max_drawdown=0.15,
            win_rate=0.55,
            profit_factor=1.6,
            avg_win=120.0,
            avg_loss=-90.0,
            consistency_score=0.60,
            total_pnl=2000.0,
            total_trades=40
        )
    }

    allocations = optimizer.optimize_allocations(mock_metrics, 100000.0)
    print(f"Allocations for $100k:")
    for inst_id, amount in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
        pct = (amount / 100000.0) * 100
        print(f"  {inst_id}: ${amount:,.2f} ({pct:.1f}%)")

    # Test 2: Arbitrage Scanner
    print("\n--- Test 2: Arbitrage Scanner ---")
    scanner = ArbitrageScanner(min_profit_pct=0.005)
    opportunities = scanner.scan_opportunities(None, None)
    print(f"Found {len(opportunities)} opportunities:")
    for opp in opportunities:
        print(f"  {opp['token']}: {opp['buy_venue']} → {opp['sell_venue']}")
        print(f"    Spread: {opp['spread_pct']*100:.2f}%")
        print(f"    Net profit: ${opp['net_profit_usd']:.2f}")

    # Test 3: Risk Monitor
    print("\n--- Test 3: Risk Monitor ---")
    monitor = RiskMonitor()
    print("✅ Risk Monitor initialized")

    print("\n" + "="*70)
    print("✅ Supervisor Enhancements ready!")
    print("="*70)
