"""
Comprehensive tests for all trading strategies.
"""
import pytest
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crypto_strategies.dca_bot import DCABot, DCAConfig, DCAFrequency, DCAMode
from crypto_strategies.market_making import MarketMakingStrategy, MarketMakingConfig
from crypto_strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from crypto_strategies.momentum import MomentumStrategy, MomentumConfig
from crypto_strategies.grid_trading_bot import GridTradingBot, GridConfig


class TestDCABot:
    """Tests for DCA Bot strategy."""

    def test_initialization(self):
        """Test DCA Bot initialization."""
        config = DCAConfig(
            symbol='BTC',
            frequency=DCAFrequency.DAILY,
            mode=DCAMode.FIXED_AMOUNT,
            base_amount=100.0
        )
        bot = DCABot(config)
        assert bot.config.symbol == 'BTC'
        assert bot.config.base_amount == 100.0
        assert len(bot.purchase_history) == 0

    def test_purchase_execution(self):
        """Test executing a purchase."""
        config = DCAConfig(
            symbol='BTC',
            frequency=DCAFrequency.DAILY,
            mode=DCAMode.FIXED_AMOUNT,
            base_amount=100.0
        )
        bot = DCABot(config)

        current_price = 40000.0
        current_time = datetime.now()

        result = bot.execute_purchase(current_price, current_time)

        assert result['success'] is True
        assert result['amount_usd'] == 100.0
        assert result['quantity'] > 0
        assert len(bot.purchase_history) == 1

    def test_dip_detection(self):
        """Test dip detection logic."""
        config = DCAConfig(
            symbol='BTC',
            frequency=DCAFrequency.DAILY,
            mode=DCAMode.DYNAMIC,
            base_amount=100.0
        )
        bot = DCABot(config)

        # Set up price history
        for price in [45000, 44000, 43000]:
            bot.execute_purchase(price, datetime.now())

        # Test with significant dip
        is_dip = bot._is_significant_dip(38000.0)
        assert is_dip is True

    def test_metrics_calculation(self):
        """Test DCA metrics calculation."""
        config = DCAConfig(
            symbol='BTC',
            frequency=DCAFrequency.DAILY,
            mode=DCAMode.FIXED_AMOUNT,
            base_amount=100.0
        )
        bot = DCABot(config)

        # Make some purchases
        for price in [40000, 42000, 38000]:
            bot.execute_purchase(price, datetime.now())

        metrics = bot.calculate_metrics(40000.0)
        assert metrics.total_invested > 0
        assert metrics.total_quantity > 0
        assert metrics.avg_cost > 0


class TestMarketMaking:
    """Tests for Market Making strategy."""

    def test_initialization(self):
        """Test Market Making initialization."""
        config = MarketMakingConfig(
            symbol='BTC-USDC',
            base_spread_bps=10.0,
            order_size_usd=1000.0,
            max_inventory_usd=5000.0
        )
        strategy = MarketMakingStrategy(config)
        assert strategy.config.symbol == 'BTC-USDC'
        assert strategy.config.base_spread_bps == 10.0

    def test_quote_generation(self):
        """Test quote generation."""
        config = MarketMakingConfig(
            symbol='BTC-USDC',
            base_spread_bps=10.0,
            order_size_usd=1000.0,
            max_inventory_usd=5000.0
        )
        strategy = MarketMakingStrategy(config)

        mid_price = 40000.0
        quotes = strategy.generate_quotes(mid_price)

        assert quotes['bid'] < mid_price
        assert quotes['ask'] > mid_price
        assert quotes['ask'] > quotes['bid']

    def test_inventory_management(self):
        """Test inventory management."""
        config = MarketMakingConfig(
            symbol='BTC-USDC',
            base_spread_bps=10.0,
            order_size_usd=1000.0,
            max_inventory_usd=5000.0
        )
        strategy = MarketMakingStrategy(config)

        # Build inventory
        strategy.current_inventory = 0.1  # BTC

        skew = strategy.calculate_inventory_skew()
        assert skew != 0  # Should adjust quotes based on inventory

    def test_spread_calculation(self):
        """Test dynamic spread calculation."""
        config = MarketMakingConfig(
            symbol='BTC-USDC',
            base_spread_bps=10.0,
            order_size_usd=1000.0,
            max_inventory_usd=5000.0
        )
        strategy = MarketMakingStrategy(config)

        mid_price = 40000.0
        spread = strategy.calculate_spread(mid_price)

        assert spread >= config.base_spread_bps
        assert spread > 0

    def test_simulation(self):
        """Test market making simulation."""
        config = MarketMakingConfig(
            symbol='BTC-USDC',
            base_spread_bps=10.0,
            order_size_usd=1000.0,
            max_inventory_usd=5000.0
        )
        strategy = MarketMakingStrategy(config)

        # Generate price series
        prices = np.random.uniform(39000, 41000, 100)

        results = strategy.simulate_market_making(prices, hit_probability=0.25)

        assert 'total_pnl' in results
        assert 'total_trades' in results
        assert 'total_volume_usd' in results


class TestMeanReversion:
    """Tests for Mean Reversion strategy."""

    def test_initialization(self):
        """Test Mean Reversion initialization."""
        config = MeanReversionConfig(
            symbol='BTC',
            lookback_period=50,
            bb_period=20,
            bb_std=2.0,
            rsi_period=14
        )
        strategy = MeanReversionStrategy(config)
        assert strategy.config.symbol == 'BTC'
        assert strategy.config.bb_period == 20

    def test_indicator_calculation(self):
        """Test technical indicator calculations."""
        config = MeanReversionConfig(
            symbol='BTC',
            lookback_period=50,
            bb_period=20,
            bb_std=2.0,
            rsi_period=14
        )
        strategy = MeanReversionStrategy(config)

        # Generate price data
        prices = np.random.uniform(39000, 41000, 100)

        # Calculate Bollinger Bands
        bb = strategy.calculate_bollinger_bands(prices, period=20, std_dev=2.0)
        assert 'upper' in bb
        assert 'middle' in bb
        assert 'lower' in bb
        assert bb['upper'] > bb['middle'] > bb['lower']

    def test_signal_generation(self):
        """Test signal generation."""
        config = MeanReversionConfig(
            symbol='BTC',
            lookback_period=50,
            bb_period=20,
            bb_std=2.0,
            rsi_period=14
        )
        strategy = MeanReversionStrategy(config)

        # Price near lower Bollinger Band (buy signal)
        prices = np.linspace(42000, 38000, 50)  # Downtrend
        current_price = 37500  # Below trend

        signal = strategy.generate_signal(current_price)
        # Signal should be generated when multiple indicators agree


class TestMomentum:
    """Tests for Momentum strategy."""

    def test_initialization(self):
        """Test Momentum initialization."""
        config = MomentumConfig(
            symbol='BTC',
            adx_threshold=25.0,
            use_trailing_stop=True
        )
        strategy = MomentumStrategy(config)
        assert strategy.config.symbol == 'BTC'
        assert strategy.config.adx_threshold == 25.0

    def test_macd_calculation(self):
        """Test MACD calculation."""
        config = MomentumConfig(
            symbol='BTC',
            adx_threshold=25.0,
            use_trailing_stop=True
        )
        strategy = MomentumStrategy(config)

        prices = np.linspace(38000, 42000, 100)  # Uptrend

        macd = strategy.calculate_macd(prices)
        assert 'macd_line' in macd
        assert 'signal_line' in macd
        assert 'histogram' in macd

    def test_adx_calculation(self):
        """Test ADX calculation."""
        config = MomentumConfig(
            symbol='BTC',
            adx_threshold=25.0,
            use_trailing_stop=True
        )
        strategy = MomentumStrategy(config)

        highs = np.linspace(39000, 43000, 50)
        lows = np.linspace(37000, 41000, 50)
        closes = np.linspace(38000, 42000, 50)

        adx = strategy.calculate_adx(highs, lows, closes)
        assert adx >= 0
        assert adx <= 100


class TestGridTrading:
    """Tests for Grid Trading strategy."""

    def test_initialization(self):
        """Test Grid Trading initialization."""
        config = GridConfig(
            symbol='BTC',
            price_range=(38000, 42000),
            num_grids=10,
            grid_spacing_pct=2.0,
            order_size_usd=500.0
        )
        bot = GridTradingBot(config)
        assert bot.config.symbol == 'BTC'
        assert bot.config.num_grids == 10

    def test_grid_level_generation(self):
        """Test grid level generation."""
        config = GridConfig(
            symbol='BTC',
            price_range=(38000, 42000),
            num_grids=10,
            grid_spacing_pct=2.0,
            order_size_usd=500.0
        )
        bot = GridTradingBot(config)

        levels = bot.calculate_grid_levels()
        assert len(levels) == config.num_grids
        assert all(38000 <= level <= 42000 for level in levels)
        # Levels should be evenly spaced
        spacing = levels[1] - levels[0]
        assert all(abs((levels[i+1] - levels[i]) - spacing) < 1 for i in range(len(levels)-1))

    def test_trade_execution_at_grid(self):
        """Test trade execution at grid levels."""
        config = GridConfig(
            symbol='BTC',
            price_range=(38000, 42000),
            num_grids=10,
            grid_spacing_pct=2.0,
            order_size_usd=500.0
        )
        bot = GridTradingBot(config)

        # Price crosses grid level
        result = bot.check_grid_level(39000.0)
        # Should generate buy or sell signal based on grid logic


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    np.random.seed(42)
    base = 40000
    returns = np.random.normal(0, 0.02, 100)
    return base * np.exp(np.cumsum(returns))


def test_all_strategies_have_backtest_method(sample_prices):
    """Verify all strategies implement backtest method."""
    strategies = [
        (DCABot, DCAConfig(symbol='BTC', frequency=DCAFrequency.DAILY, mode=DCAMode.FIXED_AMOUNT, base_amount=100)),
        (MarketMakingStrategy, MarketMakingConfig(symbol='BTC-USDC', base_spread_bps=10, order_size_usd=1000, max_inventory_usd=5000)),
    ]

    for StrategyClass, config in strategies:
        strategy = StrategyClass(config)
        # Each strategy should have some form of backtesting capability
        assert hasattr(strategy, 'simulate_market_making') or hasattr(strategy, 'backtest') or hasattr(strategy, 'execute_purchase')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
