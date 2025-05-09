import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# daily_retrain.py - Automates daily data fetch ➔ feature ➔ train
from data_ingestion.fetch_data import fetch_data
import pandas as pd
from feature_engineering.feature_generator import FeatureGenerator
from modeling.train_model import train_model
from strategy.simple_strategy import generate_signals
from utils.logger import setup_logger
from backtesting.backtest import backtest

logger = setup_logger()

def daily_pipeline():
    tickers = ['AAPL', 'MSFT', 'SPY']
    fetch_data(tickers, '2020-01-01', None)
    for ticker in tickers:
        raw_path = f'./data/raw/{ticker}.csv'
        df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        fg = FeatureGenerator(df)
        features_df = fg.generate_features()
        processed_path = f'./data/processed/{ticker}.csv'
        fg.save_features(processed_path)
        train_model(df=features_df)
        model_path = f'./models/model_{ticker}.joblib'
        generate_signals(model_path, processed_path)
        signal_file = f'./signals/{ticker}_signals.csv'
        backtest(signal_file)
        logger.info(f"Completed pipeline for {ticker}")

if __name__ == "__main__":
    daily_pipeline()