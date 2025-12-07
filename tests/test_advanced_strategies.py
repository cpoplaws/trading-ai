"""
Comprehensive tests for advanced strategies modules.
"""
import numpy as np
import pandas as pd
import pytest

from advanced_strategies.portfolio_optimizer import PortfolioOptimizer
from advanced_strategies.sentiment_analyzer import SentimentAnalyzer
from advanced_strategies.options_strategies import OptionsStrategy
from advanced_strategies.enhanced_ml_models import EnhancedMLModels
from advanced_strategies.multi_timeframe import MultiTimeframeAnalyzer


class TestPortfolioOptimizer:
    """Tests for portfolio optimization."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        return returns

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        prices = pd.Series(100 * (1 + np.random.normal(0, 0.02, 252)).cumprod(), index=dates)
        return prices

    def test_kelly_criterion_positive_returns(self, sample_returns):
        """Test Kelly Criterion with positive expected returns."""
        optimizer = PortfolioOptimizer(["AAPL"])
        kelly_size = optimizer.calculate_kelly_criterion(sample_returns, confidence=0.6)
        assert isinstance(kelly_size, float)
        assert 0 <= kelly_size <= 1  # Kelly size should be between 0 and 100%

    def test_kelly_criterion_zero_confidence(self, sample_returns):
        """Test Kelly Criterion with zero confidence."""
        optimizer = PortfolioOptimizer(["AAPL"])
        kelly_size = optimizer.calculate_kelly_criterion(sample_returns, confidence=0)
        assert kelly_size == 0  # Zero confidence should result in no position

    def test_mean_reversion_detection(self, sample_prices):
        """Test mean reversion detection."""
        optimizer = PortfolioOptimizer(["AAPL"])
        opportunities = optimizer.detect_mean_reversion_opportunities(sample_prices)
        assert isinstance(opportunities, dict)
        assert "z_score" in opportunities
        assert "signal" in opportunities

    def test_mean_reversion_extreme_z_score(self):
        """Test mean reversion with extreme Z-scores."""
        optimizer = PortfolioOptimizer(["AAPL"])
        # Create data that should trigger mean reversion signal
        prices = pd.Series([100, 100, 100, 100, 120])  # Sudden spike
        opportunities = optimizer.detect_mean_reversion_opportunities(prices)
        assert abs(opportunities["z_score"]) > 1  # Should have high Z-score


class TestSentimentAnalyzer:
    """Tests for sentiment analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create sentiment analyzer instance."""
        return SentimentAnalyzer()

    def test_sentiment_analyzer_init(self, analyzer):
        """Test sentiment analyzer initialization."""
        assert analyzer is not None
        assert hasattr(analyzer, "aggregate_sentiment_signals")

    def test_simulated_sentiment_scores(self, analyzer):
        """Test that simulated sentiment scores are reasonable."""
        sentiment = analyzer.aggregate_sentiment_signals("AAPL")
        assert isinstance(sentiment, dict)
        assert "overall_sentiment" in sentiment
        assert "confidence" in sentiment
        assert -1 <= sentiment["overall_sentiment"] <= 1
        assert 0 <= sentiment["confidence"] <= 1

    def test_sentiment_signal_generation(self, analyzer):
        """Test sentiment signal generation logic."""
        sentiment = analyzer.aggregate_sentiment_signals("AAPL")
        assert "signal" in sentiment
        assert sentiment["signal"] in ["BUY", "SELL", "HOLD"]


class TestOptionsStrategy:
    """Tests for options strategies."""

    @pytest.fixture
    def options(self):
        """Create options strategy instance."""
        return OptionsStrategy(risk_free_rate=0.05)

    def test_black_scholes_call_price(self, options):
        """Test Black-Scholes call option pricing."""
        price = options.black_scholes_price(
            S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call"
        )
        assert price > 0
        assert price < 100  # Call price should be less than stock price

    def test_black_scholes_put_price(self, options):
        """Test Black-Scholes put option pricing."""
        price = options.black_scholes_price(
            S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put"
        )
        assert price > 0
        assert price < 100  # Put price should be less than stock price

    def test_call_delta(self, options):
        """Test call option delta calculation."""
        greeks = options.calculate_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
        assert 0 <= greeks["delta"] <= 1  # Call delta should be between 0 and 1

    def test_put_delta(self, options):
        """Test put option delta calculation."""
        greeks = options.calculate_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
        assert -1 <= greeks["delta"] <= 0  # Put delta should be between -1 and 0

    def test_gamma_positive(self, options):
        """Test that gamma is always positive."""
        greeks = options.calculate_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2)
        assert greeks["gamma"] >= 0  # Gamma should always be non-negative

    def test_bull_call_spread(self, options):
        """Test bull call spread strategy."""
        spread = options.bull_call_spread(
            current_price=150, lower_strike=145, upper_strike=155, time_to_expiry=30 / 365, volatility=0.25
        )
        assert isinstance(spread, dict)
        assert "net_debit" in spread
        assert "max_profit" in spread
        assert "breakeven" in spread
        assert spread["max_profit"] > 0

    def test_long_straddle(self, options):
        """Test long straddle strategy."""
        straddle = options.long_straddle(current_price=150, strike=150, time_to_expiry=30 / 365, volatility=0.25)
        assert isinstance(straddle, dict)
        assert "total_premium" in straddle
        assert "lower_breakeven" in straddle
        assert "upper_breakeven" in straddle
        assert straddle["total_premium"] > 0


class TestEnhancedMLModels:
    """Tests for enhanced ML models."""

    @pytest.fixture
    def sample_ml_data(self):
        """Create sample data for ML models."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "close": 100 + np.random.randn(100).cumsum(),
                "volume": np.random.randint(1000000, 10000000, 100),
                "high": 100 + np.random.randn(100).cumsum() + 1,
                "low": 100 + np.random.randn(100).cumsum() - 1,
            },
            index=dates,
        )
        return data

    def test_enhanced_ml_init(self):
        """Test enhanced ML models initialization."""
        ml_models = EnhancedMLModels()
        assert ml_models is not None

    def test_ensemble_model_creation(self, sample_ml_data):
        """Test ensemble model creation."""
        ml_models = EnhancedMLModels()
        # Create simple features and target
        features = sample_ml_data[["volume"]].iloc[:-1]
        target = (sample_ml_data["close"].shift(-1) > sample_ml_data["close"]).iloc[:-1].dropna()
        
        # Align features and target
        common_idx = features.index.intersection(target.index)
        features = features.loc[common_idx]
        target = target.loc[common_idx]
        
        if len(features) > 10:  # Only test if we have enough data
            model, score = ml_models.create_ensemble_model(features, target)
            assert model is not None
            assert isinstance(score, float)


class TestMultiTimeframeAnalyzer:
    """Tests for multi-timeframe analysis."""

    @pytest.fixture
    def mtf_analyzer(self):
        """Create multi-timeframe analyzer instance."""
        return MultiTimeframeAnalyzer("AAPL")

    def test_mtf_analyzer_init(self, mtf_analyzer):
        """Test multi-timeframe analyzer initialization."""
        assert mtf_analyzer is not None
        assert mtf_analyzer.symbol == "AAPL"

    def test_signal_aggregation(self, mtf_analyzer):
        """Test signal aggregation across timeframes."""
        # Create mock signals
        signals = {
            "1min": {"signal": "BUY", "confidence": 0.7},
            "5min": {"signal": "BUY", "confidence": 0.8},
            "1h": {"signal": "SELL", "confidence": 0.6},
            "1d": {"signal": "BUY", "confidence": 0.9},
        }
        
        aggregated = mtf_analyzer.aggregate_timeframe_signals(signals)
        assert isinstance(aggregated, dict)
        assert "final_signal" in aggregated
        assert aggregated["final_signal"] in ["BUY", "SELL", "HOLD"]


class TestIntegration:
    """Integration tests for advanced strategies."""

    def test_all_strategies_can_be_imported(self):
        """Test that all strategy modules can be imported."""
        try:
            from advanced_strategies import AdvancedTradingStrategies
            assert AdvancedTradingStrategies is not None
        except ImportError as e:
            pytest.fail(f"Failed to import AdvancedTradingStrategies: {e}")

    def test_strategy_weights_configurable(self):
        """Test that strategy weights are configurable."""
        from advanced_strategies import AdvancedTradingStrategies

        strategies = AdvancedTradingStrategies(["AAPL"])
        original_weights = strategies.strategy_weights.copy()
        
        # Modify weights
        strategies.strategy_weights["ml_models"] = 0.5
        assert strategies.strategy_weights["ml_models"] == 0.5
        assert strategies.strategy_weights != original_weights


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
