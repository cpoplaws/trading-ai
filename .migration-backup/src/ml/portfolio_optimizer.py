"""
ML-Powered Portfolio Optimization
Optimal asset allocation using modern portfolio theory enhanced with ML.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Portfolio optimization objectives."""
    MAX_SHARPE = "max_sharpe"  # Maximize Sharpe ratio
    MIN_RISK = "min_risk"  # Minimize volatility
    MAX_RETURN = "max_return"  # Maximize expected return
    RISK_PARITY = "risk_parity"  # Equal risk contribution


@dataclass
class Asset:
    """Asset in portfolio."""
    symbol: str
    name: str
    current_price: float

    # Historical metrics
    expected_return: float  # Annual expected return
    volatility: float  # Annual volatility (std dev)
    sharpe_ratio: float

    # ML predictions
    predicted_return: Optional[float] = None
    confidence: float = 0.5


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result."""
    allocation_id: str
    timestamp: datetime
    objective: OptimizationObjective

    # Allocations
    weights: Dict[str, float]  # symbol -> weight (0-1)
    assets: List[Asset]

    # Portfolio metrics
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float

    # Risk metrics
    var_95: float  # Value at Risk (95%)
    max_drawdown: float

    # Diversification
    diversification_score: float  # 0-1

    # Constraints applied
    constraints: Dict = field(default_factory=dict)


class ModernPortfolioTheory:
    """
    Modern Portfolio Theory (MPT) implementation.

    Finds optimal asset allocation based on risk-return tradeoff.
    """

    def __init__(self, risk_free_rate: float = 0.03):
        """
        Initialize MPT optimizer.

        Args:
            risk_free_rate: Risk-free rate (e.g., Treasury yield)
        """
        self.risk_free_rate = risk_free_rate
        logger.info(f"MPT optimizer initialized (risk_free_rate={risk_free_rate})")

    def calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        assets: List[Asset],
        correlation_matrix: Optional[Dict[Tuple[str, str], float]] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate portfolio expected return, volatility, and Sharpe ratio.

        Args:
            weights: Asset weights
            assets: List of assets
            correlation_matrix: Correlation between assets (simplified if None)

        Returns:
            (expected_return, volatility, sharpe_ratio) tuple
        """
        # Expected return (weighted average)
        expected_return = sum(
            weights.get(asset.symbol, 0) * asset.expected_return
            for asset in assets
        )

        # Portfolio variance (simplified without full covariance matrix)
        if correlation_matrix is None:
            # Simplified: assume 0.5 correlation between all assets
            variance = 0
            for asset in assets:
                w = weights.get(asset.symbol, 0)
                variance += (w ** 2) * (asset.volatility ** 2)

            # Add correlation term (simplified)
            for i, asset1 in enumerate(assets):
                for asset2 in assets[i+1:]:
                    w1 = weights.get(asset1.symbol, 0)
                    w2 = weights.get(asset2.symbol, 0)
                    variance += 2 * w1 * w2 * asset1.volatility * asset2.volatility * 0.5
        else:
            # Full covariance calculation
            variance = 0
            for asset1 in assets:
                for asset2 in assets:
                    w1 = weights.get(asset1.symbol, 0)
                    w2 = weights.get(asset2.symbol, 0)
                    corr = correlation_matrix.get((asset1.symbol, asset2.symbol), 0.5)
                    variance += w1 * w2 * asset1.volatility * asset2.volatility * corr

        volatility = math.sqrt(max(0, variance))

        # Sharpe ratio
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        return expected_return, volatility, sharpe_ratio

    def optimize_max_sharpe(
        self,
        assets: List[Asset],
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Find allocation that maximizes Sharpe ratio.

        Args:
            assets: List of assets to allocate
            constraints: Allocation constraints

        Returns:
            Optimal weights
        """
        if not assets:
            return {}

        constraints = constraints or {}
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)

        # Simplified optimization (in production, use scipy.optimize)
        # Grid search over possible allocations

        best_sharpe = -999
        best_weights = None

        # Generate candidate allocations
        n_assets = len(assets)
        step = 0.1

        # Try equal weight baseline
        equal_weight = 1.0 / n_assets
        if equal_weight >= min_weight and equal_weight <= max_weight:
            weights = {asset.symbol: equal_weight for asset in assets}
            _, _, sharpe = self.calculate_portfolio_metrics(weights, assets)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = weights

        # Try weighted by Sharpe ratio
        total_sharpe = sum(max(0.01, asset.sharpe_ratio) for asset in assets)
        weights = {}
        for asset in assets:
            w = max(0.01, asset.sharpe_ratio) / total_sharpe
            w = max(min_weight, min(max_weight, w))
            weights[asset.symbol] = w

        # Normalize
        total_weight = sum(weights.values())
        weights = {s: w/total_weight for s, w in weights.items()}

        _, _, sharpe = self.calculate_portfolio_metrics(weights, assets)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights

        # Try risk-adjusted weights
        if best_weights is None:
            best_weights = {asset.symbol: 1.0/n_assets for asset in assets}

        return best_weights

    def optimize_min_risk(
        self,
        assets: List[Asset],
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Find allocation that minimizes portfolio volatility.

        Args:
            assets: List of assets
            constraints: Allocation constraints

        Returns:
            Optimal weights
        """
        if not assets:
            return {}

        constraints = constraints or {}
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 1.0)

        # Weight by inverse volatility
        inv_vols = {asset.symbol: 1.0 / max(0.01, asset.volatility) for asset in assets}
        total_inv_vol = sum(inv_vols.values())

        weights = {}
        for symbol, inv_vol in inv_vols.items():
            w = inv_vol / total_inv_vol
            w = max(min_weight, min(max_weight, w))
            weights[symbol] = w

        # Normalize
        total_weight = sum(weights.values())
        weights = {s: w/total_weight for s, w in weights.items()}

        return weights

    def optimize_risk_parity(
        self,
        assets: List[Asset]
    ) -> Dict[str, float]:
        """
        Risk parity: each asset contributes equally to portfolio risk.

        Args:
            assets: List of assets

        Returns:
            Risk parity weights
        """
        if not assets:
            return {}

        # Weight by inverse volatility (simplified risk parity)
        inv_vols = {asset.symbol: 1.0 / max(0.01, asset.volatility) for asset in assets}
        total_inv_vol = sum(inv_vols.values())

        weights = {symbol: inv_vol / total_inv_vol for symbol, inv_vol in inv_vols.items()}

        return weights


class MLPortfolioOptimizer:
    """
    ML-Enhanced Portfolio Optimizer

    Combines MPT with ML predictions for optimal allocation.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.03,
        use_ml_predictions: bool = True
    ):
        """
        Initialize ML portfolio optimizer.

        Args:
            risk_free_rate: Risk-free rate
            use_ml_predictions: Use ML predicted returns
        """
        self.mpt = ModernPortfolioTheory(risk_free_rate)
        self.use_ml_predictions = use_ml_predictions
        self.allocation_counter = 0

        logger.info(
            f"ML portfolio optimizer initialized "
            f"(ML predictions={'enabled' if use_ml_predictions else 'disabled'})"
        )

    def optimize(
        self,
        assets: List[Asset],
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        constraints: Optional[Dict] = None
    ) -> PortfolioAllocation:
        """
        Optimize portfolio allocation.

        Args:
            assets: List of assets
            objective: Optimization objective
            constraints: Allocation constraints

        Returns:
            Optimal allocation
        """
        if not assets:
            raise ValueError("Assets list cannot be empty")

        # Enhance assets with ML predictions if enabled
        if self.use_ml_predictions:
            assets = self._enhance_with_ml(assets)

        # Optimize based on objective
        if objective == OptimizationObjective.MAX_SHARPE:
            weights = self.mpt.optimize_max_sharpe(assets, constraints)
        elif objective == OptimizationObjective.MIN_RISK:
            weights = self.mpt.optimize_min_risk(assets, constraints)
        elif objective == OptimizationObjective.RISK_PARITY:
            weights = self.mpt.optimize_risk_parity(assets)
        else:
            # Default to max Sharpe
            weights = self.mpt.optimize_max_sharpe(assets, constraints)

        # Calculate portfolio metrics
        expected_return, volatility, sharpe = self.mpt.calculate_portfolio_metrics(
            weights, assets
        )

        # Estimate VaR (95% confidence)
        var_95 = expected_return - 1.645 * volatility  # 95% confidence interval

        # Diversification score (entropy-based)
        diversification_score = self._calculate_diversification(weights)

        # Generate allocation
        self.allocation_counter += 1
        allocation = PortfolioAllocation(
            allocation_id=f"ALLOC-{self.allocation_counter:06d}",
            timestamp=datetime.now(),
            objective=objective,
            weights=weights,
            assets=assets,
            expected_return=expected_return,
            expected_volatility=volatility,
            sharpe_ratio=sharpe,
            var_95=var_95,
            max_drawdown=volatility * 2,  # Simplified
            diversification_score=diversification_score,
            constraints=constraints or {}
        )

        logger.info(
            f"Portfolio optimized: {objective.value} | "
            f"Return: {expected_return*100:.2f}% | "
            f"Risk: {volatility*100:.2f}% | "
            f"Sharpe: {sharpe:.2f}"
        )

        return allocation

    def _enhance_with_ml(self, assets: List[Asset]) -> List[Asset]:
        """Enhance assets with ML predictions."""
        # In production, this would use actual ML models
        # For now, simulate ML enhancement

        enhanced = []
        for asset in assets:
            # ML prediction adjusts expected return based on confidence
            if asset.predicted_return is not None:
                # Blend historical and predicted returns
                blended_return = (
                    asset.expected_return * (1 - asset.confidence) +
                    asset.predicted_return * asset.confidence
                )

                # Create enhanced asset
                enhanced_asset = Asset(
                    symbol=asset.symbol,
                    name=asset.name,
                    current_price=asset.current_price,
                    expected_return=blended_return,
                    volatility=asset.volatility,
                    sharpe_ratio=(blended_return - 0.03) / asset.volatility if asset.volatility > 0 else 0,
                    predicted_return=asset.predicted_return,
                    confidence=asset.confidence
                )
                enhanced.append(enhanced_asset)
            else:
                enhanced.append(asset)

        return enhanced

    def _calculate_diversification(self, weights: Dict[str, float]) -> float:
        """
        Calculate diversification score (0-1).

        Uses Shannon entropy to measure diversification.
        """
        if not weights:
            return 0.0

        # Shannon entropy
        entropy = -sum(w * math.log(w) if w > 0 else 0 for w in weights.values())

        # Normalize to 0-1
        max_entropy = math.log(len(weights))
        diversification = entropy / max_entropy if max_entropy > 0 else 0

        return diversification

    def rebalance(
        self,
        current_allocation: PortfolioAllocation,
        current_values: Dict[str, float],
        threshold: float = 0.05
    ) -> Optional[PortfolioAllocation]:
        """
        Check if rebalancing is needed.

        Args:
            current_allocation: Current portfolio allocation
            current_values: Current value of each asset
            threshold: Rebalancing threshold (5% = 0.05)

        Returns:
            New allocation if rebalancing needed, None otherwise
        """
        # Calculate current weights
        total_value = sum(current_values.values())
        current_weights = {
            symbol: value / total_value
            for symbol, value in current_values.items()
        }

        # Check if drift exceeds threshold
        max_drift = 0.0
        for symbol, target_weight in current_allocation.weights.items():
            current_weight = current_weights.get(symbol, 0)
            drift = abs(current_weight - target_weight)
            max_drift = max(max_drift, drift)

        if max_drift > threshold:
            logger.info(f"Rebalancing needed: max drift = {max_drift:.2%}")
            # Reoptimize
            return self.optimize(
                current_allocation.assets,
                current_allocation.objective,
                current_allocation.constraints
            )

        return None


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    print("📊 ML Portfolio Optimization Demo")
    print("=" * 60)

    # Create sample assets
    print("\n1. Creating Asset Universe...")
    print("-" * 60)

    assets = [
        Asset(
            symbol="BTC",
            name="Bitcoin",
            current_price=45000.0,
            expected_return=0.50,  # 50% annual return
            volatility=0.80,  # 80% volatility
            sharpe_ratio=0.59,
            predicted_return=0.55,  # ML predicts 55%
            confidence=0.7
        ),
        Asset(
            symbol="ETH",
            name="Ethereum",
            current_price=2500.0,
            expected_return=0.60,
            volatility=0.90,
            sharpe_ratio=0.63,
            predicted_return=0.65,
            confidence=0.6
        ),
        Asset(
            symbol="SOL",
            name="Solana",
            current_price=100.0,
            expected_return=0.70,
            volatility=1.20,
            sharpe_ratio=0.56,
            predicted_return=0.60,
            confidence=0.5
        ),
        Asset(
            symbol="USDC",
            name="USD Coin",
            current_price=1.0,
            expected_return=0.05,  # 5% (staking yield)
            volatility=0.01,  # Very low volatility
            sharpe_ratio=2.00,
            predicted_return=0.05,
            confidence=0.9
        )
    ]

    for asset in assets:
        print(f"{asset.symbol}: Return={asset.expected_return*100:.1f}%, "
              f"Risk={asset.volatility*100:.1f}%, "
              f"Sharpe={asset.sharpe_ratio:.2f}")

    # Optimize portfolio
    print("\n2. Optimizing Portfolio (Max Sharpe)...")
    print("-" * 60)

    optimizer = MLPortfolioOptimizer(risk_free_rate=0.03, use_ml_predictions=True)

    allocation = optimizer.optimize(
        assets=assets,
        objective=OptimizationObjective.MAX_SHARPE,
        constraints={'min_weight': 0.05, 'max_weight': 0.50}
    )

    print(f"\nAllocation ID: {allocation.allocation_id}")
    print(f"Objective: {allocation.objective.value}")
    print(f"\nPortfolio Metrics:")
    print(f"  Expected Return: {allocation.expected_return*100:.2f}%")
    print(f"  Expected Volatility: {allocation.expected_volatility*100:.2f}%")
    print(f"  Sharpe Ratio: {allocation.sharpe_ratio:.2f}")
    print(f"  VaR (95%): {allocation.var_95*100:.2f}%")
    print(f"  Diversification: {allocation.diversification_score*100:.1f}%")

    print(f"\nAsset Allocation:")
    for symbol, weight in sorted(allocation.weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {symbol}: {weight*100:.2f}%")

    # Compare objectives
    print("\n3. Comparing Optimization Objectives...")
    print("-" * 60)

    objectives = [
        OptimizationObjective.MAX_SHARPE,
        OptimizationObjective.MIN_RISK,
        OptimizationObjective.RISK_PARITY
    ]

    for obj in objectives:
        alloc = optimizer.optimize(assets, objective=obj)

        print(f"\n{obj.value.upper()}:")
        print(f"  Return: {alloc.expected_return*100:.2f}%")
        print(f"  Risk: {alloc.expected_volatility*100:.2f}%")
        print(f"  Sharpe: {alloc.sharpe_ratio:.2f}")
        print(f"  Top holdings: ", end="")

        top_3 = sorted(alloc.weights.items(), key=lambda x: x[1], reverse=True)[:3]
        print(", ".join(f"{s} ({w*100:.0f}%)" for s, w in top_3))

    print("\n✅ Portfolio optimization demo complete!")
