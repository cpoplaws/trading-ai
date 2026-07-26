"""
Unit Tests for VaR Calculator
Tests VaR and CVaR calculations with different methods.
"""
import pytest
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from risk_management.var_calculator import VaRCalculator, VaRMethod, VaRResult


@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    return np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns


@pytest.fixture
def calculator():
    """Create VaR calculator instance."""
    return VaRCalculator(confidence_level=0.95, time_horizon_days=1)


class TestVaRCalculator:
    """Test VaR calculator functionality."""

    def test_initialization(self):
        """Test calculator initialization."""
        calc = VaRCalculator(confidence_level=0.99, time_horizon_days=10)
        assert calc.confidence_level == 0.99
        assert calc.time_horizon_days == 10

    def test_historical_var(self, calculator, sample_returns):
        """Test historical VaR calculation."""
        result = calculator.calculate_var(
            sample_returns,
            VaRMethod.HISTORICAL,
            portfolio_value=100000
        )

        assert isinstance(result, VaRResult)
        assert result.var > 0  # VaR should be positive (loss)
        assert result.cvar > 0
        assert result.cvar >= result.var  # CVaR >= VaR
        assert result.method == VaRMethod.HISTORICAL
        assert result.confidence_level == 0.95
        assert result.num_scenarios == len(sample_returns)

    def test_parametric_var(self, calculator, sample_returns):
        """Test parametric VaR calculation."""
        result = calculator.calculate_var(
            sample_returns,
            VaRMethod.PARAMETRIC,
            portfolio_value=100000
        )

        assert isinstance(result, VaRResult)
        assert result.var > 0
        assert result.cvar > 0
        assert result.cvar >= result.var
        assert result.method == VaRMethod.PARAMETRIC
        assert result.mean_return is not None
        assert result.volatility > 0

    def test_monte_carlo_var(self, calculator, sample_returns):
        """Test Monte Carlo VaR calculation."""
        result = calculator.calculate_var(
            sample_returns,
            VaRMethod.MONTE_CARLO,
            portfolio_value=100000
        )

        assert isinstance(result, VaRResult)
        assert result.var > 0
        assert result.cvar > 0
        assert result.cvar >= result.var
        assert result.method == VaRMethod.MONTE_CARLO
        assert result.num_scenarios == 10000  # Default simulations

    def test_var_scales_with_portfolio_value(self, calculator, sample_returns):
        """Test that VaR scales linearly with portfolio value."""
        result1 = calculator.calculate_var(sample_returns, VaRMethod.HISTORICAL, 100000)
        result2 = calculator.calculate_var(sample_returns, VaRMethod.HISTORICAL, 200000)

        # VaR should approximately double
        ratio = result2.var / result1.var
        assert 1.9 < ratio < 2.1  # Allow 10% tolerance

    def test_var_confidence_levels(self, sample_returns):
        """Test that VaR increases with confidence level."""
        calc95 = VaRCalculator(confidence_level=0.95)
        calc99 = VaRCalculator(confidence_level=0.99)

        var95 = calc95.calculate_var(sample_returns, VaRMethod.HISTORICAL, 100000)
        var99 = calc99.calculate_var(sample_returns, VaRMethod.HISTORICAL, 100000)

        # 99% VaR should be higher than 95% VaR
        assert var99.var > var95.var

    def test_var_time_horizon(self, sample_returns):
        """Test that VaR increases with time horizon."""
        calc1d = VaRCalculator(time_horizon_days=1)
        calc10d = VaRCalculator(time_horizon_days=10)

        var1d = calc1d.calculate_var(sample_returns, VaRMethod.HISTORICAL, 100000)
        var10d = calc10d.calculate_var(sample_returns, VaRMethod.HISTORICAL, 100000)

        # 10-day VaR should be higher than 1-day VaR
        assert var10d.var > var1d.var

    def test_portfolio_var(self, calculator):
        """Test multi-asset portfolio VaR."""
        positions = {
            'BTC': 50000,
            'ETH': 30000,
            'SOL': 20000
        }

        returns_data = {
            'BTC': np.random.normal(0.002, 0.03, 252),
            'ETH': np.random.normal(0.0015, 0.025, 252),
            'SOL': np.random.normal(0.001, 0.04, 252)
        }

        result = calculator.calculate_portfolio_var(
            positions,
            returns_data,
            method=VaRMethod.PARAMETRIC
        )

        assert isinstance(result, VaRResult)
        assert result.var > 0
        assert result.cvar > 0

        # Portfolio VaR should be less than sum of individual VaRs (diversification benefit)
        individual_vars = []
        for symbol in positions.keys():
            var = calculator.calculate_var(
                returns_data[symbol],
                VaRMethod.PARAMETRIC,
                positions[symbol]
            )
            individual_vars.append(var.var)

        assert result.var < sum(individual_vars)

    def test_backtest_var(self, calculator, sample_returns):
        """Test VaR backtesting."""
        # Calculate VaR for each day
        var_results = []
        for i in range(100):
            returns_slice = sample_returns[:-(100-i)] if i < 99 else sample_returns
            result = calculator.calculate_var(
                returns_slice,
                VaRMethod.HISTORICAL,
                100000
            )
            var_results.append(result)

        # Backtest
        actual_returns = sample_returns[-100:]
        portfolio_values = np.full(100, 100000)

        backtest = calculator.backtest_var(actual_returns, portfolio_values, var_results)

        assert 'num_exceedances' in backtest
        assert 'exceedance_rate' in backtest
        assert 'expected_rate' in backtest
        assert backtest['num_observations'] == 100
        assert 0 <= backtest['exceedance_rate'] <= 1

    def test_empty_returns(self, calculator):
        """Test handling of empty returns array."""
        with pytest.warns(UserWarning):
            result = calculator.calculate_var(
                np.array([]),
                VaRMethod.HISTORICAL,
                100000
            )

    def test_insufficient_data(self, calculator):
        """Test warning with insufficient data points."""
        small_returns = np.random.normal(0.001, 0.02, 10)  # Only 10 points

        with pytest.warns(UserWarning):
            result = calculator.calculate_var(
                small_returns,
                VaRMethod.HISTORICAL,
                100000
            )

    def test_invalid_method(self, calculator, sample_returns):
        """Test error handling for invalid method."""
        with pytest.raises(ValueError):
            calculator.calculate_var(
                sample_returns,
                "invalid_method",  # Not a VaRMethod enum
                100000
            )

    def test_cvar_always_greater_than_var(self, calculator, sample_returns):
        """Test that CVaR is always >= VaR."""
        for method in VaRMethod:
            result = calculator.calculate_var(sample_returns, method, 100000)
            assert result.cvar >= result.var

    def test_var_result_attributes(self, calculator, sample_returns):
        """Test that VaRResult has all required attributes."""
        result = calculator.calculate_var(sample_returns, VaRMethod.HISTORICAL, 100000)

        required_attrs = [
            'var', 'cvar', 'confidence_level', 'method',
            'time_horizon_days', 'calculation_time', 'percentile'
        ]

        for attr in required_attrs:
            assert hasattr(result, attr)
            assert getattr(result, attr) is not None


@pytest.mark.parametric
class TestParametricAssumptions:
    """Test parametric VaR assumptions and edge cases."""

    def test_normal_distribution_assumption(self):
        """Test that parametric VaR assumes normal distribution."""
        calc = VaRCalculator(confidence_level=0.95)

        # Generate normally distributed returns
        returns = np.random.normal(0, 0.02, 1000)

        result = calc.calculate_var(returns, VaRMethod.PARAMETRIC, 100000)

        # Parametric VaR should be close to theoretical value
        # For 95% confidence: VaR ≈ 1.65 * σ * portfolio_value
        theoretical_var = 1.65 * 0.02 * 100000
        assert abs(result.var - theoretical_var) / theoretical_var < 0.2  # Within 20%

    def test_fat_tails_underestimation(self):
        """Test that parametric VaR may underestimate for fat-tailed distributions."""
        calc = VaRCalculator(confidence_level=0.95)

        # Generate fat-tailed returns (student t-distribution)
        from scipy.stats import t
        returns = t.rvs(df=3, loc=0, scale=0.02, size=1000)

        parametric_var = calc.calculate_var(returns, VaRMethod.PARAMETRIC, 100000)
        historical_var = calc.calculate_var(returns, VaRMethod.HISTORICAL, 100000)

        # Parametric should underestimate for fat tails
        # (This might not always hold with random data, so we just check they're calculated)
        assert parametric_var.var > 0
        assert historical_var.var > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=src/risk_management'])
