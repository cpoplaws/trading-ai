"""Data ingestion module for fetching OHLCV market data."""
import os
import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from requests import exceptions as requests_exceptions
import yfinance as yf

from utils.logger import setup_logger

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_FILE_PATH = os.path.join(BASE_DIR, "logs", "data_fetch.log")
DEFAULT_SAVE_PATH = os.path.join(BASE_DIR, "data", "raw")
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0
MAX_FILL_DAYS = 5

logger = setup_logger(__name__, log_file=LOG_FILE_PATH)


def _download_with_retries(
    ticker: str,
    start_date: str,
    end_date: str,
    max_retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> pd.DataFrame:
    """Download ticker data with retry logic.

    Falls back to a generic exception handler to capture unexpected provider errors.
    """
    last_exception: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if df is not None and not df.empty:
                return df

            logger.warning(
                f"No data returned for {ticker} on attempt {attempt}/{max_retries}"
            )
        except requests_exceptions.RequestException as exc:
            last_exception = exc
            logger.error(
                f"Network error fetching data for {ticker} on attempt {attempt}/{max_retries}: {exc}"
            )
        except (ValueError, KeyError) as exc:  # captures provider data issues
            last_exception = exc
            logger.error(
                f"Error fetching data for {ticker} on attempt {attempt}/{max_retries}: {exc}"
            )
        except Exception:
            # Re-raise unexpected exceptions to avoid masking programming errors.
            raise

        if attempt < max_retries:
            time.sleep(retry_delay)

    if last_exception:
        logger.error(f"Failed to fetch data for {ticker} after {max_retries} attempts")

    return pd.DataFrame()


def _ensure_no_missing_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Ensure there are no missing business days in the returned dataset.

    Uses pandas business-day frequency ('B'), which excludes weekends.
    """
    if df.empty:
        return df

    cleaned = df.copy()
    cleaned.index = pd.to_datetime(cleaned.index).normalize()
    cleaned = cleaned[~cleaned.index.duplicated(keep="first")].sort_index()

    expected_index = pd.date_range(start=start_date, end=end_date, freq="B")
    missing_dates = expected_index.difference(cleaned.index)

    if not missing_dates.empty:
        logger.warning(f"Filling {len(missing_dates)} missing dates for range {start_date} to {end_date}")
        cleaned = cleaned.reindex(expected_index)

        price_columns = [col for col in ["Open", "High", "Low", "Close", "Adj Close"] if col in cleaned.columns]
        if price_columns:
            # Allow limited fills from both directions (up to 2 * MAX_FILL_DAYS) to patch small gaps.
            cleaned[price_columns] = cleaned[price_columns].ffill(limit=MAX_FILL_DAYS).bfill(limit=MAX_FILL_DAYS)
        if "Volume" in cleaned.columns:
            cleaned["Volume"] = cleaned["Volume"].fillna(0)

    return cleaned


def fetch_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    save_path: str = DEFAULT_SAVE_PATH,
    max_retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> bool:
    """
    Fetch OHLCV data for given tickers and save to CSV files.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (optional, defaults to today)
        save_path: Directory to save the data files
        max_retries: Maximum retry attempts per ticker
        retry_delay: Base delay (seconds) between retries
        
    Returns:
        bool: True if all tickers were successfully fetched, False otherwise
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
        
    os.makedirs(save_path, exist_ok=True)
    success_count = 0
    
    for ticker in tickers:
        logger.info(f"Fetching data for {ticker}...")
        df = _download_with_retries(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        if df is None or df.empty:
            logger.warning(f"No data returned for {ticker}")
            continue

        df = _ensure_no_missing_dates(df, start_date, end_date)
        file_path = os.path.join(save_path, f"{ticker}.csv")
        df.to_csv(file_path)
        logger.info(f"Saved {ticker} data to {file_path} ({len(df)} rows)")
        success_count += 1
    
    logger.info(f"Successfully fetched data for {success_count}/{len(tickers)} tickers")
    return success_count == len(tickers)

if __name__ == "__main__":
    default_start = (datetime.today() - timedelta(days=365 * 3)).strftime('%Y-%m-%d')
    success = fetch_data(['AAPL', 'MSFT'], default_start)
    if not success:
        logger.error("Failed to fetch all ticker data")
