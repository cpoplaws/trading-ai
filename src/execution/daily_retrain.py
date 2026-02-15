"""
Daily automated trading pipeline - Fetch data, engineer features, train models, and generate signals.
"""
import sys
import os
import shutil
import time
from datetime import datetime, timedelta
from typing import List, Optional
try:
    import schedule  # type: ignore
except ImportError:
    schedule = None
SCHEDULE_AVAILABLE = schedule is not None

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import with relative paths
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_ingestion.fetch_data import fetch_data
from feature_engineering.feature_generator import FeatureGenerator
from modeling.train_model import train_model
from strategy.simple_strategy import generate_signals, analyze_signals
from utils.logger import setup_logger
import pandas as pd

# Set up logging
logger = setup_logger("daily_pipeline", "INFO")
DEFAULT_LOOKBACK_DAYS = 365
SCHEDULE_DEFAULT_SLEEP = 5
SCHEDULE_MIN_SLEEP = 1
SCHEDULE_MAX_SLEEP = 60
MODEL_PATH_TEMPLATE = "./models/model_{ticker}.joblib"
SIGNAL_FILE_TEMPLATE = "./signals/{ticker}_signals.csv"


def resolve_start_date(start_date: Optional[str], window_days: int, current_time: Optional[datetime] = None) -> str:
    """Return an explicit start_date using a rolling lookback window when not provided."""
    now = current_time or datetime.utcnow()
    if start_date:
        return start_date
    return (now - timedelta(days=window_days)).strftime('%Y-%m-%d')


def _calculate_sleep_duration(next_run: Optional[float]) -> float:
    """
    Clamp the schedule idle time to a reasonable sleep duration.
    
    Args:
        next_run: Seconds until the next scheduled task as reported by `schedule.idle_seconds()`.
    
    Returns:
        Sleep duration bounded between SCHEDULE_MIN_SLEEP and SCHEDULE_MAX_SLEEP.
    """
    if next_run is None:
        return float(SCHEDULE_DEFAULT_SLEEP)
    return float(max(SCHEDULE_MIN_SLEEP, min(next_run, SCHEDULE_MAX_SLEEP)))


def archive_model(model_path: str, ticker: str, run_date: Optional[datetime] = None, archive_dir: Optional[str] = None) -> Optional[str]:
    """
    Save a dated copy of the latest model so each day's retrain is preserved.

    Args:
        model_path: Path to the most recent model file.
        ticker: Ticker symbol used for naming.
        run_date: Optional date to embed in the archived filename.
        archive_dir: Optional directory for archived models (defaults to ./models/daily/).

    Returns:
        The path to the archived model, or None if the source model is missing.
    """
    if not os.path.exists(model_path):
        logger.warning(f"Model path not found for archiving: {model_path}")
        return None

    archive_dir = archive_dir or os.path.join(os.path.dirname(model_path), "daily")
    os.makedirs(archive_dir, exist_ok=True)

    run_date = run_date or datetime.utcnow()
    dated_name = f"model_{ticker}_{run_date.strftime('%Y%m%d_%H%M%S')}.joblib"
    archived_path = os.path.join(archive_dir, dated_name)

    shutil.copy2(model_path, archived_path)
    logger.info(f"Archived daily model for {ticker} to {archived_path}")
    return archived_path


def schedule_daily_retrain(run_time: str = "09:00", tickers: Optional[List[str]] = None, window_days: int = DEFAULT_LOOKBACK_DAYS) -> None:
    """
    Schedule the daily retrain job to run once per day at a specific time.

    This function uses the `schedule` library to register `daily_pipeline` as a
    recurring job and then enters a loop that keeps the scheduler running until
    the process is interrupted.

    Args:
        run_time:
            Time of day (in 24-hour ``"HH:MM"`` format) when the daily retrain
            should run, e.g. ``"09:00"`` for 9:00 AM or ``"17:30"`` for 5:30 PM.
            The string is passed directly to ``schedule.every().day.at(run_time)``.
        tickers:
            Optional list of ticker symbols to process in the daily pipeline.
            If ``None``, the pipeline will fall back to its internal defaults
            (currently ``["AAPL", "MSFT", "SPY"]`` as reflected in the log
            message).
        window_days:
            Size of the rolling lookback window, in days, that controls how much
            recent historical data is used when fetching data and retraining
            models in ``daily_pipeline``.

    Behavior:
        After scheduling the job, this function enters an infinite loop that:

        * runs any pending scheduled jobs via ``schedule.run_pending()``, and
        * sleeps until the next scheduled run (with a bounded sleep duration).

        It will continue running until the process receives a ``KeyboardInterrupt``
        (for example, by pressing ``Ctrl+C`` in the terminal), at which point the
        loop exits and a shutdown message is logged. This provides a graceful way
        to stop the scheduler.

    Example:
        Run the daily retrain at 09:30 UTC for a custom set of tickers, using a
        180-day lookback window:

        .. code-block:: python

            from execution.daily_retrain import schedule_daily_retrain

            if __name__ == "__main__":
                schedule_daily_retrain(
                    run_time="09:30",
                    tickers=["AAPL", "MSFT", "SPY"],
                    window_days=180,
                )
    """
    if not SCHEDULE_AVAILABLE:
        raise ImportError(
            "The 'schedule' library is required for scheduled retraining. "
            "Install it with: 'pip install schedule' or 'pip install -r requirements.txt'."
        )

    logger.info(f"Scheduling daily retrain at {run_time} for tickers: {tickers or ['AAPL', 'MSFT', 'SPY']}")
    schedule.every().day.at(run_time).do(daily_pipeline, tickers=tickers, window_days=window_days)

    try:
        while True:
            schedule.run_pending()
            next_run = schedule.idle_seconds()
            time.sleep(_calculate_sleep_duration(next_run))
    except KeyboardInterrupt:
        logger.info("Stopping scheduled daily retrain loop")


def daily_pipeline(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    window_days: int = DEFAULT_LOOKBACK_DAYS,
) -> bool:
    """
    Execute the complete daily trading pipeline.
    
    Args:
        tickers: List of ticker symbols to process
        start_date: Optional start date override (defaults to rolling window)
        window_days: Rolling window lookback (in days) for retraining
        
    Returns:
        bool: True if pipeline completed successfully
    """
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'SPY']
    
    start_date = resolve_start_date(start_date, window_days)
    logger.info(f"Starting daily pipeline for tickers: {tickers} (rolling window start: {start_date})")
    
    # Step 1: Fetch latest data
    logger.info("Step 1: Fetching market data...")
    try:
        success = fetch_data(tickers, start_date)
        if not success:
            logger.error("Failed to fetch all ticker data")
            return False
    except Exception as e:
        logger.error(f"Error in data fetching: {str(e)}")
        return False
    
    # Step 2: Process each ticker
    successful_tickers = []
    
    for ticker in tickers:
        logger.info(f"Processing {ticker}...")
        
        try:
            # Load raw data
            raw_path = f'./data/raw/{ticker}.csv'
            if not os.path.exists(raw_path):
                logger.warning(f"No data file found for {ticker}")
                continue
                
            # Read CSV and skip problematic header rows
            df = pd.read_csv(raw_path, index_col=0, parse_dates=True, skiprows=[1, 2])
            
            # Ensure all price columns are numeric
            price_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
            for col in price_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows with NaN values from conversion
            df = df.dropna()
            
            logger.info(f"Loaded {len(df)} data points for {ticker}")
            
            # Step 3: Generate features
            logger.info(f"Generating features for {ticker}...")
            df.columns = df.columns.str.lower()  # Standardize column names
            fg = FeatureGenerator(df)
            features_df = fg.generate_features()
            
            # Save processed data
            processed_path = f'./data/processed/{ticker}.csv'
            model_path = MODEL_PATH_TEMPLATE.format(ticker=ticker)
            os.makedirs('./data/processed', exist_ok=True)
            success = fg.save_features(processed_path)
            if not success:
                logger.error(f"Failed to save features for {ticker}")
                continue
            
            # Step 4: Train model
            logger.info(f"Training model for {ticker}...")
            train_success, metrics = train_model(df=features_df, file_path=processed_path)
            if not train_success:
                logger.error(f"Failed to train model for {ticker}: {metrics}")
                continue
                
            logger.info(f"Model training metrics for {ticker}: {metrics}")
            # Training succeeded; preserve the newly trained model for daily history
            if os.path.exists(model_path):
                archive_model(model_path, ticker, run_date=datetime.utcnow())
            else:
                logger.warning(f"Expected model not found for {ticker} after training at {model_path}")
            
            # Step 5: Generate signals
            logger.info(f"Generating signals for {ticker}...")
            os.makedirs('./signals', exist_ok=True)
            
            signal_success = generate_signals(model_path, processed_path)
            if not signal_success:
                logger.error(f"Failed to generate signals for {ticker}")
                continue
            
            # Step 6: Analyze signals
            signal_file = SIGNAL_FILE_TEMPLATE.format(ticker=ticker)
            if os.path.exists(signal_file):
                analysis = analyze_signals(signal_file)
                logger.info(f"Signal analysis for {ticker}: {analysis}")
            
            successful_tickers.append(ticker)
            logger.info(f"Successfully completed pipeline for {ticker}")
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            continue
    
    # Pipeline summary
    logger.info(f"Pipeline completed. Successful tickers: {successful_tickers}")
    logger.info(f"Success rate: {len(successful_tickers)}/{len(tickers)}")
    
    return len(successful_tickers) > 0

def run_backtest(ticker: str) -> bool:
    """
    Run backtest for a specific ticker (placeholder for future implementation).
    
    Args:
        ticker: Ticker symbol
        
    Returns:
        bool: True if backtest successful
    """
    signal_file = f'./signals/{ticker}_signals.csv'
    if not os.path.exists(signal_file):
        logger.error(f"No signals file found for {ticker}")
        return False
    
    # TODO: Implement actual backtesting logic
    logger.info(f"Backtest placeholder executed for {ticker}")
    return True

if __name__ == "__main__":
    if "--schedule" in sys.argv:
        schedule_time = os.getenv("DAILY_RETRAIN_TIME", "09:00")
        schedule_daily_retrain(run_time=schedule_time)
    else:
        success = daily_pipeline()
        
        if success:
            logger.info("Daily pipeline completed successfully!")
        else:
            logger.error("Daily pipeline failed!")
            sys.exit(1)
