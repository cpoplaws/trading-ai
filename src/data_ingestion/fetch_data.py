# fetch_data.py - Pulls daily OHLCV data for selected tickers
import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_data(tickers: List[str], start_date: str, end_date: Optional[str] = None, save_path: str = './data/raw/') -> bool:
    """
    Fetch OHLCV data for given tickers and save to CSV files.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (optional, defaults to today)
        save_path: Directory to save the data files
        
    Returns:
        bool: True if all tickers were successfully fetched, False otherwise
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
        
    os.makedirs(save_path, exist_ok=True)
    success_count = 0
    
    for ticker in tickers:
        try:
            logger.info(f"Fetching data for {ticker}...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df is not None and not df.empty:
                file_path = os.path.join(save_path, f"{ticker}.csv")
                df.to_csv(file_path)
                logger.info(f"Saved {ticker} data to {file_path}")
                success_count += 1
            else:
                logger.warning(f"No data returned for {ticker}")
                
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
    
    logger.info(f"Successfully fetched data for {success_count}/{len(tickers)} tickers")
    return success_count == len(tickers)

if __name__ == "__main__":
    success = fetch_data(['AAPL', 'MSFT', 'SPY'], '2020-01-01')
    if not success:
        logger.error("Failed to fetch all ticker data")
