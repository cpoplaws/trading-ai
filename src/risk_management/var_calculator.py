"""
Value at Risk (VaR) and Conditional VaR (CVaR) Calculator
Measures potential losses in portfolio value.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class VaRMethod(Enum):
    """VaR calculation methods."""
    HISTORICAL = "historical"  # Historical simulation
    PARAMETRIC = "parametric"  # Variance-covariance
    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation


@dataclass
class VaRResult:
    """VaR calculation result."""
    var: float  # Value at Risk
    cvar: float  # Conditional VaR (Expected Shortfall)
    confidence_level: float
    method: VaRMethod
    time_horizon_days: int
    calculation_time: datetime
    percentile: float
    num_scenarios: int = 0
    mean_return: float = 0.0
    volatility: float = 0.0


class VaRCalculator:
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR).

    VaR: Maximum expected loss at a given confidence level
    CVaR (ES): Expected loss given that VaR threshold is exceeded

    Methods:
    - Historical: Uses historical returns distribution
    - Parametric: Assumes normal distribution
    - Monte Carlo: Simulates future scenarios

    Typical Values:
    - Confidence Level: 95% or 99%
    - Time Horizon: 1 day
    - VaR interpretation: "95% confident we won't lose more than $X in 1 day"
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1
    ):
        """
        Initialize VaR calculator.

        Args:
            confidence_level: Confidence level (0.95 = 95%)
            time_horizon_days: Time horizon in days
        """
        self.confidence_level = confidence_level
        self.time_horizon_days = time_horizon_days

        logger.info(f"VaR calculator initialized: {confidence_level*100}% confidence, "
                   f"{time_horizon_days}d horizon")

    def calculate_var(
        self,
        returns: np.ndarray,
        method: VaRMethod = VaRMethod.HISTORICAL,
        portfolio_value: float = 100000.0
    ) -> VaRResult:
        """
        Calculate VaR using specified method.

        Args:
            returns: Array of historical returns (as decimals, e.g., 0.01 = 1%)
            method: Calculation method (VaRMethod enum or string: 'historical', 'parametric', 'monte_carlo')
            portfolio_value: Current portfolio value

        Returns:
            VaR result with VaR and CVaR values
        """
        if len(returns) < 30:
            logger.warning(f"Only {len(returns)} data points, need at least 30 for reliable VaR")

        # Convert string to enum if needed
        if isinstance(method, str):
            method_map = {
                'historical': VaRMethod.HISTORICAL,
                'parametric': VaRMethod.PARAMETRIC,
                'monte_carlo': VaRMethod.MONTE_CARLO
            }
            method = method_map.get(method.lower())
            if method is None:
                raise ValueError(f"Unknown VaR method: {method}. Use 'historical', 'parametric', or 'monte_carlo'")

        if method == VaRMethod.HISTORICAL:
            return self._historical_var(returns, portfolio_value)
        elif method == VaRMethod.PARAMETRIC:
            return self._parametric_var(returns, portfolio_value)
        elif method == VaRMethod.MONTE_CARLO:
            return self._monte_carlo_var(returns, portfolio_value)
        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def _historical_var(
        self,
        returns: np.ndarray,
        portfolio_value: float
    ) -> VaRResult:
        """
        Calculate VaR using historical simulation.

        Uses actual historical returns distribution.
        No distribution assumptions.
        """
        # Scale returns to time horizon (sqrt of time rule)
        scaled_returns = returns * np.sqrt(self.time_horizon_days)

        # Calculate VaR (loss is negative return)
        percentile = (1 - self.confidence_level) * 100
        var_percentile = np.percentile(scaled_returns, percentile)
        var = -var_percentile * portfolio_value  # Convert to positive loss

        # Calculate CVaR (average of losses beyond VaR)
        losses_beyond_var = scaled_returns[scaled_returns <= var_percentile]
        cvar_percentile = np.mean(losses_beyond_var) if len(losses_beyond_var) > 0 else var_percentile
        cvar = -cvar_percentile * portfolio_value

        return VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=self.confidence_level,
            method=VaRMethod.HISTORICAL,
            time_horizon_days=self.time_horizon_days,
            calculation_time=datetime.now(),
            percentile=percentile,
            num_scenarios=len(returns),
            mean_return=np.mean(returns),
            volatility=np.std(returns)
        )

    def _parametric_var(
        self,
        returns: np.ndarray,
        portfolio_value: float
    ) -> VaRResult:
        """
        Calculate VaR using parametric method (variance-covariance).

        Assumes returns are normally distributed.
        Fast but may underestimate tail risk.
        """
        # Calculate statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Scale to time horizon
        scaled_mean = mean_return * self.time_horizon_days
        scaled_std = std_return * np.sqrt(self.time_horizon_days)

        # Z-score for confidence level (e.g., 1.65 for 95%)
        z_score = stats.norm.ppf(1 - self.confidence_level)

        # VaR = -(mean + z * std) * portfolio_value
        var_return = -(scaled_mean + z_score * scaled_std)
        var = var_return * portfolio_value

        # CVaR for normal distribution
        # E[L | L > VaR] = mean + (pdf(z) / (1-confidence)) * std
        phi_z = stats.norm.pdf(z_score)
        cvar_return = -(scaled_mean + (phi_z / (1 - self.confidence_level)) * scaled_std)
        cvar = cvar_return * portfolio_value

        return VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=self.confidence_level,
            method=VaRMethod.PARAMETRIC,
            time_horizon_days=self.time_horizon_days,
            calculation_time=datetime.now(),
            percentile=(1 - self.confidence_level) * 100,
            num_scenarios=len(returns),
            mean_return=mean_return,
            volatility=std_return
        )

    def _monte_carlo_var(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        num_simulations: int = 10000
    ) -> VaRResult:
        """
        Calculate VaR using Monte Carlo simulation.

        Simulates future returns based on historical distribution.
        More flexible than parametric, captures non-normality.
        """
        # Fit distribution to returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Generate simulated returns
        simulated_returns = np.random.normal(
            mean_return * self.time_horizon_days,
            std_return * np.sqrt(self.time_horizon_days),
            num_simulations
        )

        # Calculate VaR
        percentile = (1 - self.confidence_level) * 100
        var_percentile = np.percentile(simulated_returns, percentile)
        var = -var_percentile * portfolio_value

        # Calculate CVaR
        losses_beyond_var = simulated_returns[simulated_returns <= var_percentile]
        cvar_percentile = np.mean(losses_beyond_var)
        cvar = -cvar_percentile * portfolio_value

        return VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=self.confidence_level,
            method=VaRMethod.MONTE_CARLO,
            time_horizon_days=self.time_horizon_days,
            calculation_time=datetime.now(),
            percentile=percentile,
            num_scenarios=num_simulations,
            mean_return=mean_return,
            volatility=std_return
        )

    def calculate_portfolio_var(
        self,
        positions: Dict[str, float],  # {symbol: value}
        returns_data: Dict[str, np.ndarray],  # {symbol: returns}
        correlation_matrix: Optional[np.ndarray] = None,
        method: VaRMethod = VaRMethod.HISTORICAL
    ) -> VaRResult:
        """
        Calculate VaR for multi-asset portfolio.

        Args:
            positions: Portfolio positions {symbol: value}
            returns_data: Historical returns for each asset
            correlation_matrix: Asset correlation matrix (computed if None)
            method: Calculation method

        Returns:
            Portfolio VaR result
        """
        symbols = list(positions.keys())
        weights = np.array([positions[s] for s in symbols])
        total_value = np.sum(weights)

        if total_value == 0:
            return VaRResult(
                var=0.0,
                cvar=0.0,
                confidence_level=self.confidence_level,
                method=method,
                time_horizon_days=self.time_horizon_days,
                calculation_time=datetime.now(),
                percentile=0.0
            )

        # Normalize weights
        weights = weights / total_value

        # Build returns matrix
        returns_list = [returns_data[s] for s in symbols]
        min_length = min(len(r) for r in returns_list)
        returns_matrix = np.array([r[-min_length:] for r in returns_list]).T

        if method == VaRMethod.PARAMETRIC:
            # Parametric approach using correlation
            if correlation_matrix is None:
                correlation_matrix = np.corrcoef(returns_matrix.T)

            # Mean and covariance
            mean_returns = np.mean(returns_matrix, axis=0)
            cov_matrix = np.cov(returns_matrix.T)

            # Portfolio return statistics
            portfolio_mean = np.dot(weights, mean_returns)
            portfolio_var_matrix = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_var_matrix)

            # Scale to time horizon
            scaled_mean = portfolio_mean * self.time_horizon_days
            scaled_std = portfolio_std * np.sqrt(self.time_horizon_days)

            # Calculate VaR
            z_score = stats.norm.ppf(1 - self.confidence_level)
            var_return = -(scaled_mean + z_score * scaled_std)
            var = var_return * total_value

            # CVaR
            phi_z = stats.norm.pdf(z_score)
            cvar_return = -(scaled_mean + (phi_z / (1 - self.confidence_level)) * scaled_std)
            cvar = cvar_return * total_value

            return VaRResult(
                var=var,
                cvar=cvar,
                confidence_level=self.confidence_level,
                method=VaRMethod.PARAMETRIC,
                time_horizon_days=self.time_horizon_days,
                calculation_time=datetime.now(),
                percentile=(1 - self.confidence_level) * 100,
                num_scenarios=len(returns_matrix),
                mean_return=portfolio_mean,
                volatility=portfolio_std
            )

        else:
            # Historical or Monte Carlo: calculate portfolio returns
            portfolio_returns = np.dot(returns_matrix, weights)
            return self.calculate_var(portfolio_returns, method, total_value)

    def backtest_var(
        self,
        returns: np.ndarray,
        portfolio_values: np.ndarray,
        var_results: List[VaRResult]
    ) -> Dict:
        """
        Backtest VaR model accuracy.

        Args:
            returns: Actual returns
            portfolio_values: Actual portfolio values
            var_results: VaR predictions for each period

        Returns:
            Backtest statistics
        """
        actual_losses = -returns * portfolio_values
        predicted_vars = np.array([v.var for v in var_results])

        # Count exceedances (actual loss > VaR)
        exceedances = actual_losses > predicted_vars
        num_exceedances = np.sum(exceedances)
        exceedance_rate = num_exceedances / len(returns)

        # Expected exceedance rate
        expected_rate = 1 - self.confidence_level

        # Kupiec test (likelihood ratio test)
        n = len(returns)
        x = num_exceedances
        p = expected_rate

        if x > 0 and x < n:
            lr = -2 * np.log(
                ((1-p)**(n-x) * p**x) /
                ((1-x/n)**(n-x) * (x/n)**x)
            )
            p_value = 1 - stats.chi2.cdf(lr, 1)
        else:
            lr = np.nan
            p_value = np.nan

        return {
            'num_observations': len(returns),
            'num_exceedances': num_exceedances,
            'exceedance_rate': exceedance_rate,
            'expected_rate': expected_rate,
            'kupiec_lr': lr,
            'kupiec_p_value': p_value,
            'model_accurate': 0.01 < p_value < 0.99 if not np.isnan(p_value) else False
        }


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    print("📊 VaR & CVaR Calculator Demo")
    print("=" * 60)

    # Generate sample returns (daily returns for 1 year)
    np.random.seed(42)
    days = 252
    mean_return = 0.001  # 0.1% daily
    volatility = 0.02  # 2% daily volatility
    returns = np.random.normal(mean_return, volatility, days)

    portfolio_value = 100000  # $100k portfolio

    print(f"\n📈 Portfolio: ${portfolio_value:,}")
    print(f"   Data: {days} days of returns")
    print(f"   Mean Return: {np.mean(returns)*100:.3f}%")
    print(f"   Volatility: {np.std(returns)*100:.2f}%")

    # Calculate VaR using different methods
    calculator = VaRCalculator(confidence_level=0.95, time_horizon_days=1)

    print("\n" + "=" * 60)
    print("VaR Calculations (95% confidence, 1-day horizon)")
    print("=" * 60)

    for method in VaRMethod:
        result = calculator.calculate_var(returns, method, portfolio_value)

        print(f"\n{method.value.upper()} Method:")
        print(f"   VaR: ${result.var:,.2f}")
        print(f"   CVaR: ${result.cvar:,.2f}")
        print(f"   Interpretation: 95% confident we won't lose more than ${result.var:,.2f} in 1 day")
        print(f"   If loss exceeds VaR, expected loss is ${result.cvar:,.2f}")

    # Compare confidence levels
    print("\n" + "=" * 60)
    print("VaR at Different Confidence Levels")
    print("=" * 60)

    for confidence in [0.90, 0.95, 0.99]:
        calc = VaRCalculator(confidence_level=confidence, time_horizon_days=1)
        result = calc.calculate_var(returns, VaRMethod.HISTORICAL, portfolio_value)
        print(f"\n{confidence*100}% Confidence:")
        print(f"   VaR: ${result.var:,.2f}")
        print(f"   CVaR: ${result.cvar:,.2f}")

    # Multi-asset portfolio
    print("\n" + "=" * 60)
    print("Multi-Asset Portfolio VaR")
    print("=" * 60)

    # Simulate 3-asset portfolio
    positions = {
        'BTC': 50000,
        'ETH': 30000,
        'SOL': 20000
    }

    returns_data = {
        'BTC': np.random.normal(0.002, 0.03, days),
        'ETH': np.random.normal(0.0015, 0.025, days),
        'SOL': np.random.normal(0.001, 0.04, days)
    }

    calculator = VaRCalculator(confidence_level=0.95, time_horizon_days=1)
    result = calculator.calculate_portfolio_var(
        positions,
        returns_data,
        method=VaRMethod.PARAMETRIC
    )

    print(f"\nPortfolio Composition:")
    for symbol, value in positions.items():
        pct = value / sum(positions.values()) * 100
        print(f"   {symbol}: ${value:,} ({pct:.1f}%)")

    print(f"\nPortfolio VaR (95%, 1-day):")
    print(f"   VaR: ${result.var:,.2f}")
    print(f"   CVaR: ${result.cvar:,.2f}")

    # Backtest
    print("\n" + "=" * 60)
    print("VaR Backtest")
    print("=" * 60)

    # Simulate actual returns
    test_days = 100
    actual_returns = np.random.normal(mean_return, volatility, test_days)
    actual_values = np.full(test_days, portfolio_value)

    # Calculate VaR for each day
    var_results = []
    for i in range(test_days):
        result = calculator.calculate_var(
            returns[:-(test_days-i)] if i < test_days-1 else returns,
            VaRMethod.HISTORICAL,
            portfolio_value
        )
        var_results.append(result)

    backtest = calculator.backtest_var(actual_returns, actual_values, var_results)

    print(f"\nBacktest Results:")
    print(f"   Observations: {backtest['num_observations']}")
    print(f"   Exceedances: {backtest['num_exceedances']}")
    print(f"   Exceedance Rate: {backtest['exceedance_rate']*100:.2f}%")
    print(f"   Expected Rate: {backtest['expected_rate']*100:.2f}%")
    print(f"   Kupiec P-Value: {backtest['kupiec_p_value']:.4f}")
    print(f"   Model Accurate: {'✅ Yes' if backtest['model_accurate'] else '❌ No'}")

    print("\n✅ VaR calculator demo complete!")
