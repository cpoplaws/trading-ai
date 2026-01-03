"""
Basic test suite for the trading AI system.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import joblib

# Simple picklable model for signal generation tests
class DummyModel:
    def predict(self, X):
        return np.array(['UP', 'DOWN', 'UP'])

    def predict_proba(self, X):
        # Probabilities align with predictions (P(SELL), P(BUY))
        return np.array([
            [0.2, 0.8],
            [0.7, 0.3],
            [0.1, 0.9],
        ])

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_ingestion.fetch_data import fetch_data
from feature_engineering.feature_generator import FeatureGenerator
from modeling.train_model import train_model
from strategy.simple_strategy import generate_signals
from utils.logger import setup_logger

class TestDataIngestion:
    def test_fetch_data_success(self):
        """Test successful data fetching."""
        # Create test directory
        test_dir = './test_data/'
        os.makedirs(test_dir, exist_ok=True)
        
        # Test with a simple ticker
        success = fetch_data(['AAPL'], '2023-01-01', '2023-01-31', test_dir)
        
        # Check if file was created
        assert os.path.exists(f'{test_dir}AAPL.csv')
        
        # Clean up
        if os.path.exists(f'{test_dir}AAPL.csv'):
            os.remove(f'{test_dir}AAPL.csv')
        os.rmdir(test_dir)
        
    def test_fetch_data_invalid_ticker(self):
        """Test behavior with invalid ticker."""
        test_dir = './test_data/'
        os.makedirs(test_dir, exist_ok=True)
        
        # This should not crash, but may return False
        result = fetch_data(['INVALID_TICKER_XYZ'], '2023-01-01', '2023-01-31', test_dir)
        
        # Clean up
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)

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
    def test_generate_signals_up_down_mapping(self, tmp_path):
        """Ensure UP/DOWN predictions map to BUY/SELL signals."""
        model_path = tmp_path / "model_test.joblib"
        joblib.dump(DummyModel(), model_path)

        # Save feature list alongside model for loader
        feature_path = tmp_path / "features_test.joblib"
        joblib.dump(['SMA_10', 'SMA_30', 'RSI_14', 'Volatility_20'], feature_path)

        dates = pd.date_range(start='2024-01-01', periods=3, freq='D')
        data = pd.DataFrame({
            'SMA_10': [1, 2, 3],
            'SMA_30': [1, 2, 3],
            'RSI_14': [30, 50, 70],
            'Volatility_20': [0.1, 0.2, 0.3],
            'Close': [100, 101, 102],
        }, index=dates)

        data_path = tmp_path / "data.csv"
        data.to_csv(data_path)

        success = generate_signals(str(model_path), str(data_path), save_path=str(tmp_path))
        assert success is True

        signals_path = tmp_path / "data_signals.csv"
        assert signals_path.exists()

        signals_df = pd.read_csv(signals_path, index_col=0, parse_dates=True)
        assert list(signals_df['Signal']) == ['BUY', 'SELL', 'BUY']
        assert set(signals_df['Prediction'].unique()) == {'UP', 'DOWN'}

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
