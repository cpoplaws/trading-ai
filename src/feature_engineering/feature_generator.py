import pandas as pd
import numpy as np
import talib
import os

class FeatureGenerator:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the FeatureGenerator with price data.
        :param data: DataFrame with a 'close' column containing closing prices.
        """
        if 'close' not in data.columns:
            raise ValueError("Input data must contain a 'close' column.")
        self.data = data

    def calculate_sma(self, window: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA)."""
        return self.data['close'].rolling(window=window).mean()

    def calculate_ema(self, window: int) -> pd.Series:
        """Calculate Exponential Moving Average (EMA)."""
        return talib.EMA(self.data['close'], timeperiod=window)

    def calculate_rsi(self, window: int) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_volatility(self, window: int) -> pd.Series:
        """Calculate Rolling Standard Deviation (Volatility)."""
        return self.data['close'].rolling(window=window).std()

    def generate_features(self) -> pd.DataFrame:
        """
        Generate all features and return a DataFrame aligned with prices.
        :return: DataFrame with SMA, EMA, RSI, and Volatility features.
        """
        self.data['SMA_10'] = self.calculate_sma(10)
        self.data['SMA_30'] = self.calculate_sma(30)
        self.data['EMA_20'] = self.calculate_ema(20)
        self.data['RSI_14'] = self.calculate_rsi(14)
        self.data['Volatility_20'] = self.calculate_volatility(20)

        # Drop rows with NaN values caused by rolling calculations
        return self.data.dropna().reset_index(drop=True)

    def save_features(self, save_path: str):
        """
        Save the generated features to a CSV file.
        :param save_path: Path to save the processed DataFrame.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.data.dropna().to_csv(save_path)
        print(f"Saved engineered features to {save_path}")

# Example usage:
# df = pd.read_csv('./data/raw/AAPL.csv', index_col=0, parse_dates=True)
# feature_generator = FeatureGenerator(df.rename(columns={'Close': 'close'}))
# features = feature_generator.generate_features()
# feature_generator.save_features('./data/processed/AAPL_features.csv')