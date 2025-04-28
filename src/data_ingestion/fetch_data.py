# fetch_data.py - Pulls daily OHLCV data for selected tickers
import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def fetch_data(tickers, start_date, end_date, save_path='./data/raw/'):
    os.makedirs(save_path, exist_ok=True)
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty:
            file_path = os.path.join(save_path, f"{ticker}.csv")
            df.to_csv(file_path)
            print(f"Saved {ticker} data to {file_path}")
        else:
            print(f"No data for {ticker}")

if __name__ == "__main__":
    fetch_data(['AAPL', 'MSFT', 'SPY'], '2020-01-01', datetime.today().strftime('%Y-%m-%d'))
