"""Feature engineering module for generating technical indicators and multimodal features."""
import os
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

from utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureGenerator:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the FeatureGenerator with price data.
        :param data: DataFrame with OHLCV columns (Close, High, Low, Volume)
        """
        # Standardize column names to lowercase
        data.columns = data.columns.str.lower()
        
        if 'close' not in data.columns:
            raise ValueError("Input data must contain a 'close' column.")
        self.data = data.copy()
        self.multimodal_data = {}  # Store macro, news, sentiment data

    def calculate_sma(self, window: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA)."""
        close_prices = pd.to_numeric(self.data['close'], errors='coerce')
        return close_prices.rolling(window=window).mean()

    def calculate_ema(self, window: int) -> pd.Series:
        """Calculate Exponential Moving Average (EMA)."""
        close_prices = pd.to_numeric(self.data['close'], errors='coerce')
        return close_prices.ewm(span=window).mean()

    def calculate_rsi(self, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        close_prices = pd.to_numeric(self.data['close'], errors='coerce')
        delta = close_prices.diff()
        
        # Calculate gains and losses using pandas methods
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Avoid division by zero
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_volatility(self, window: int) -> pd.Series:
        """Calculate Rolling Standard Deviation (Volatility)."""
        close_prices = pd.to_numeric(self.data['close'], errors='coerce')
        return close_prices.rolling(window=window).std()
    
    def calculate_bollinger_bands(self, window: int = 20, num_std: float = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = self.calculate_sma(window)
        std = self.calculate_volatility(window)
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = self.calculate_ema(fast)
        ema_slow = self.calculate_ema(slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def generate_features(self) -> pd.DataFrame:
        """
        Generate all features and return a DataFrame aligned with prices.
        :return: DataFrame with technical indicators.
        """
        try:
            # Moving averages
            self.data['SMA_10'] = self.calculate_sma(10)
            self.data['SMA_30'] = self.calculate_sma(30)
            self.data['EMA_20'] = self.calculate_ema(20)
            
            # Momentum indicators
            self.data['RSI_14'] = self.calculate_rsi(14)
            
            # Volatility
            self.data['Volatility_20'] = self.calculate_volatility(20)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands()
            self.data['BB_Upper'] = bb_upper
            self.data['BB_Lower'] = bb_lower
            self.data['BB_Width'] = (bb_upper - bb_lower) / bb_middle
            
            # MACD
            macd, macd_signal, macd_hist = self.calculate_macd()
            self.data['MACD'] = macd
            self.data['MACD_Signal'] = macd_signal
            self.data['MACD_Histogram'] = macd_hist
            
            # Price-based features
            if 'high' in self.data.columns and 'low' in self.data.columns:
                self.data['Price_Range'] = self.data['high'] - self.data['low']
                self.data['Price_Position'] = (self.data['close'] - self.data['low']) / (self.data['high'] - self.data['low'])
            
            # Volume features (if available)
            if 'volume' in self.data.columns:
                self.data['Volume_SMA'] = self.data['volume'].rolling(window=20).mean()
                self.data['Volume_Ratio'] = self.data['volume'] / self.data['Volume_SMA']
            
            logger.info(f"Generated {len([c for c in self.data.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} features")
            
            # Drop rows with NaN values caused by rolling calculations
            return self.data.dropna()
            
        except Exception as e:
            logger.error(f"Error generating features: {str(e)}")
            raise
            
    def add_macro_features(self, macro_data: Dict[str, float]) -> None:
        """
        Add macroeconomic indicators as features.
        
        Args:
            macro_data: Dictionary of macro indicators (e.g., {'fed_funds_rate': 5.5, 'inflation_cpi': 3.2})
        """
        try:
            for key, value in macro_data.items():
                self.data[f'macro_{key}'] = value
                
            logger.info(f"Added {len(macro_data)} macro features")
            self.multimodal_data['macro'] = macro_data
            
        except Exception as e:
            logger.error(f"Error adding macro features: {str(e)}")
            
    def add_news_sentiment(self, news_sentiment: float, news_volume: int) -> None:
        """
        Add news sentiment features.
        
        Args:
            news_sentiment: Sentiment score (-1 to 1)
            news_volume: Number of news articles
        """
        try:
            self.data['news_sentiment'] = news_sentiment
            self.data['news_volume'] = news_volume
            self.data['news_volume_ma'] = news_volume  # Can be enhanced with rolling avg
            
            logger.info(f"Added news sentiment features: sentiment={news_sentiment:.3f}, volume={news_volume}")
            self.multimodal_data['news'] = {'sentiment': news_sentiment, 'volume': news_volume}
            
        except Exception as e:
            logger.error(f"Error adding news sentiment: {str(e)}")
            
    def add_social_sentiment(self, reddit_sentiment: float, reddit_mentions: int) -> None:
        """
        Add social media sentiment features.
        
        Args:
            reddit_sentiment: Reddit sentiment score (-1 to 1)
            reddit_mentions: Number of Reddit mentions
        """
        try:
            self.data['reddit_sentiment'] = reddit_sentiment
            self.data['reddit_mentions'] = reddit_mentions
            self.data['reddit_buzz'] = np.log1p(reddit_mentions)  # Log transform for better distribution
            
            logger.info(f"Added social sentiment features: sentiment={reddit_sentiment:.3f}, mentions={reddit_mentions}")
            self.multimodal_data['social'] = {'sentiment': reddit_sentiment, 'mentions': reddit_mentions}
            
        except Exception as e:
            logger.error(f"Error adding social sentiment: {str(e)}")
            
    def add_market_regime(self, regime: str) -> None:
        """
        Add market regime indicator.
        
        Args:
            regime: Market regime ('expansion', 'recession', 'stagflation', 'recovery', 'neutral')
        """
        try:
            # One-hot encode the regime
            regimes = ['expansion', 'recession', 'stagflation', 'recovery', 'neutral']
            for r in regimes:
                self.data[f'regime_{r}'] = 1 if regime == r else 0
                
            logger.info(f"Added market regime: {regime}")
            self.multimodal_data['regime'] = regime
            
        except Exception as e:
            logger.error(f"Error adding market regime: {str(e)}")

    def save_features(self, save_path: str) -> bool:
        """
        Save the generated features to a CSV file.
        :param save_path: Path to save the processed DataFrame.
        :return: True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.data.dropna().to_csv(save_path)
            logger.info(f"Saved engineered features to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving features to {save_path}: {str(e)}")
            return False

# Example usage:
if __name__ == "__main__":
    # df = pd.read_csv('./data/raw/AAPL.csv', index_col=0, parse_dates=True)
    # df.columns = df.columns.str.lower()  # Standardize column names
    # feature_generator = FeatureGenerator(df)
    # features = feature_generator.generate_features()
    # feature_generator.save_features('./data/processed/AAPL_features.csv')
    pass