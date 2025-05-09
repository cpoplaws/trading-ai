import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# daily_retrain.py - Automates daily data fetch ➔ feature ➔ train
from data_ingestion.fetch_data import fetch_data
from feature_engineering.feature_generator import generate_features
from modeling.train_model import train_model
from strategy.simple_strategy import generate_signals

def daily_pipeline():
    tickers = ['AAPL', 'MSFT', 'SPY']
    fetch_data(tickers, '2020-01-01', None)
    for ticker in tickers:
        raw_path = f'./data/raw/{ticker}.csv'
        generate_features(raw_path)
        processed_path = f'./data/processed/{ticker}.csv'
        train_model(processed_path)
        model_path = f'./models/model_{ticker}.joblib'
        generate_signals(model_path, processed_path)

if __name__ == "__main__":
    daily_pipeline()