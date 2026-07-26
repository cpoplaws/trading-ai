"""
Daily automated trading pipeline - Fetch data, engineer features, train models, and generate signals.
"""
import sys
import os
from datetime import datetime
from typing import List, Optional

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import with relative paths
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .data_ingestion.fetch_data import fetch_data
from .feature_engineering.feature_generator import FeatureGenerator
from .modeling.train_model import train_model
from .strategy.simple_strategy import generate_signals, analyze_signals
from ..utils.logger import setup_logger
import pandas as pd

# Set up logging
logger = setup_logger("daily_pipeline", "INFO")

def daily_pipeline(tickers: Optional[List[str]] = None, start_date: str = '2020-01-01') -> bool:
    """
    Execute the complete daily trading pipeline.
    
    Args:
        tickers: List of ticker symbols to process
        start_date: Start date for data fetching
        
    Returns:
        bool: True if pipeline completed successfully
    """
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'SPY']
    
    logger.info(f"Starting daily pipeline for tickers: {tickers}")
    
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
            
            # Step 5: Generate signals
            logger.info(f"Generating signals for {ticker}...")
            model_path = f'./models/model_{ticker}.joblib'
            os.makedirs('./signals', exist_ok=True)
            
            signal_success = generate_signals(model_path, processed_path)
            if not signal_success:
                logger.error(f"Failed to generate signals for {ticker}")
                continue
            
            # Step 6: Analyze signals
            signal_file = f'./signals/{ticker}_signals.csv'
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
    
    # Implement backtesting logic
    try:
        # Read signals
        signals_df = pd.read_csv(signal_file)
        if 'date' in signals_df.columns:
            signals_df['date'] = pd.to_datetime(signals_df['date'])
            signals_df.set_index('date', inplace=True)

        # Get historical price data
        from ..data_ingestion.fetch_data import download_data
        start_date = signals_df.index.min()
        end_date = signals_df.index.max()

        price_data = download_data(
            ticker,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        if price_data.empty:
            logger.warning(f"No price data for {ticker} backtest")
            return False

        # Run backtest
        from ..backtesting.backtest import backtest_strategy

        results = backtest_strategy(
            price_data,
            signals_df,
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005
        )

        if results:
            logger.info(f"Backtest for {ticker}: Return={results.get('total_return', 0):.2%}, Sharpe={results.get('sharpe_ratio', 0):.2f}")

            # Save results
            os.makedirs('./backtests', exist_ok=True)
            results_file = f'./backtests/{ticker}_backtest.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            return True
        return False
    except Exception as e:
        logger.error(f"Backtest error for {ticker}: {e}")
        return False

if __name__ == "__main__":
    # Run the daily pipeline
    success = daily_pipeline()
    
    if success:
        logger.info("Daily pipeline completed successfully!")
    else:
        logger.error("Daily pipeline failed!")
        sys.exit(1)