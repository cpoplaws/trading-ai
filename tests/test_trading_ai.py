"""
Basic test suite for the trading AI system.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import data_ingestion.fetch_data as fetch_data_module
from data_ingestion.fetch_data import fetch_data
from feature_engineering.feature_generator import FeatureGenerator
from modeling.train_model import train_model
from strategy.simple_strategy import generate_signals
from utils.logger import setup_logger

class TestDataIngestion:
    REFERENCE_DATE = pd.Timestamp('2024-01-15')
    BASE_OFFSET_DAYS = 10
    END_OFFSET_DAYS = 9
    SAMPLE_PERIODS = 7

    def _date_bounds(self):
        start_date = (self.REFERENCE_DATE - pd.tseries.offsets.BDay(self.BASE_OFFSET_DAYS)).date()
        end_date = (pd.Timestamp(start_date) + pd.tseries.offsets.BDay(self.END_OFFSET_DAYS)).date()
        return start_date, end_date

    def _sample_df(self):
        start_date, _ = self._date_bounds()
        dates = pd.date_range(start=start_date, periods=self.SAMPLE_PERIODS, freq='B')
        return pd.DataFrame(
            {
                'Open': [100 + i for i in range(len(dates))],
                'High': [101 + i for i in range(len(dates))],
                'Low': [99 + i for i in range(len(dates))],
                'Close': [100 + i for i in range(len(dates))],
                'Adj Close': [100 + i for i in range(len(dates))],
                'Volume': [1000 for _ in dates],
            },
            index=dates,
        )

    def test_fetch_data_success(self, monkeypatch, tmp_path):
        """Test successful data fetching with missing-date fill."""
        sample_df = self._sample_df()

        def fake_download(*args, **kwargs):
            return sample_df

        monkeypatch.setattr(fetch_data_module.yf, "download", fake_download)

        start_date, end_date = self._date_bounds()
        target_dir = tmp_path / "raw"
        success = fetch_data(
            ['AAPL'],
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            str(target_dir),
        )

        saved_file = target_dir / "AAPL.csv"
        assert success is True
        assert saved_file.exists()

        saved_df = pd.read_csv(saved_file, index_col=0, parse_dates=True)
        expected_index = pd.date_range(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            freq='B',
        )
        assert list(saved_df.index) == list(expected_index)
        assert saved_df.isna().sum().sum() == 0

    def test_fetch_data_invalid_ticker(self, monkeypatch, tmp_path):
        """Test behavior with invalid ticker."""

        def failing_download(*args, **kwargs):
            raise ValueError("Ticker not found")

        monkeypatch.setattr(fetch_data_module.yf, "download", failing_download)

        result = fetch_data(['INVALID_TICKER_XYZ'], '2023-01-01', '2023-01-31', str(tmp_path))
        assert result is False

class TestFeatureEngineering:
    def create_sample_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        price = 100
        prices = [price]
        
        for i in range(len(dates) - 1):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            price = price * (1 + change)
            prices.append(price)
        
        df = pd.DataFrame({
            'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': [np.random.randint(1000000, 10000000) for _ in prices]
        }, index=dates)
        
        return df
    
    def test_feature_generator_init(self):
        """Test FeatureGenerator initialization."""
        df = self.create_sample_data()
        fg = FeatureGenerator(df)
        assert fg.data is not None
        assert 'close' in fg.data.columns
        
    def test_feature_generation(self):
        """Test feature generation."""
        df = self.create_sample_data()
        fg = FeatureGenerator(df)
        features_df = fg.generate_features()
        
        # Check if features were generated
        expected_features = ['SMA_10', 'SMA_30', 'RSI_14', 'Volatility_20']
        for feature in expected_features:
            assert feature in features_df.columns
            
    def test_sma_calculation(self):
        """Test SMA calculation."""
        df = self.create_sample_data()
        fg = FeatureGenerator(df)
        sma_10 = fg.calculate_sma(10)
        
        # SMA should not have NaN for the last values (after window)
        assert not pd.isna(sma_10.iloc[-1])
        
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        df = self.create_sample_data()
        fg = FeatureGenerator(df)
        rsi = fg.calculate_rsi(14)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert all(0 <= val <= 100 for val in valid_rsi)

class TestModeling:
    def create_sample_features_data(self):
        """Create sample data with features for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        n_samples = len(dates)
        df = pd.DataFrame({
            'Close': np.random.uniform(90, 110, n_samples),
            'SMA_10': np.random.uniform(85, 115, n_samples),
            'SMA_30': np.random.uniform(80, 120, n_samples),
            'RSI_14': np.random.uniform(20, 80, n_samples),
            'Volatility_20': np.random.uniform(0.01, 0.05, n_samples),
        }, index=dates)
        
        return df
    
    def test_train_model(self):
        """Test model training."""
        df = self.create_sample_features_data()
        success, metrics = train_model(df=df, save_path='./test_models/')
        
        assert success is True
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        
        # Clean up
        if os.path.exists('./test_models/'):
            import shutil
            shutil.rmtree('./test_models/')

class TestStrategy:
    def test_signal_generation_placeholder(self):
        """Placeholder test for signal generation."""
        # This would require a trained model and processed data
        # For now, just test that the function exists
        assert callable(generate_signals)

class TestLogger:
    def test_logger_setup(self):
        """Test logger setup."""
        logger = setup_logger("test_logger", "INFO")
        assert logger is not None
        assert logger.name == "test_logger"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
